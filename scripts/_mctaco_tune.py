import argparse
import functools
import inspect
import itertools
import pathlib
from typing import List, Union, Callable

from datasets import load_dataset
import pytorch_lightning as pl
from torchmetrics import MetricCollection, Accuracy, F1Score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import wandb

SEED_DEFAULT = None


class MCTACODataset(Dataset):
    def __init__(self, split: str, tokenizer, sequence_length: int):
        self.dataset = load_dataset("mc_taco")[split]
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.dataset)

    def truncate_pair(self, tokens_a, tokens_b, max_length):
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    # TODO(shwang): Would make more sense to tokenize the entire thing first and
    #   just return little slices.
    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx): 
        item = self.dataset[idx] 
        tokenize = self.tokenizer.tokenize
        sequence = tokenize(item['sentence'] + " " + item['question'])
        answer = tokenize(item['answer']) 
        label = item['label']
        # Truncate excess tokens 
        if answer: 
            self.truncate_pair(sequence, answer, self.sequence_length - 3)
        else: 
            if len(sequence) > self.sequence_length - 2:
                sequence = sequence[0:(self.sequence_length - 2)]
        # Compute tokens, ids, mask 
        tokens = ['<s>'] + sequence + ['</s></s>'] + answer + ['</s>']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        # Pad with 0 
        while len(input_ids) < self.sequence_length:
            input_ids.append(0)
            input_mask.append(0)
        return torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(label)


class MCTACODatamodule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        batch_size: int,
        sequence_length: int 
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.dataset_train = None
        self.dataset_valid = None

    def setup(self, stage = None):
        self.dataset_train = MCTACODataset(
            split='validation', 
            tokenizer=self.tokenizer, 
            sequence_length=self.sequence_length
        )
        self.dataset_valid = MCTACODataset(
            split='test', 
            tokenizer=self.tokenizer, 
            sequence_length=self.sequence_length
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset_valid,
            batch_size=self.batch_size,
            shuffle=False,
        )


def inf_norm(x):
    return torch.norm(x, p=float('inf'), dim=-1, keepdim=True)


def default(val, d):
    if val is not None:
        return val
    return d() if inspect.isfunction(d) else d


class SMARTLoss(nn.Module):
    def __init__(
        self,
        eval_fn: Callable,
        loss_fn: Callable,
        loss_last_fn: Callable = None, 
        norm_fn: Callable = inf_norm, 
        num_steps: int = 1,
        step_size: float = 1e-3,   # I.E learning rate
        epsilon: float = 1e-6,
        noise_var: float = 1e-5
    ) -> None:
        super().__init__()
        self.eval_fn = eval_fn 
        self.loss_fn = loss_fn
        self.loss_last_fn = default(loss_last_fn, loss_fn)
        self.norm_fn = norm_fn
        self.num_steps = num_steps 
        self.step_size = step_size
        self.epsilon = epsilon 
        self.noise_var = noise_var
        
    def forward(self, embed: th.Tensor, state: Union[th.Tensor, List[th.Tensor]]) -> th.Tensor:
        noise = torch.randn_like(embed, requires_grad=True) * self.noise_var

        # Indefinite loop with counter 
        for i in itertools.count():
            # Compute perturbed embed and states 
            embed_perturbed = embed + noise 
            state_perturbed = self.eval_fn(embed_perturbed)
            # Return final loss if last step (undetached state)
            if i == self.num_steps: 
                return self.loss_last_fn(state_perturbed, state)
                # Need to be able to back-propgate through model output in the return value.

            # Compute perturbation loss (detached state)
            loss = self.loss_fn(state_perturbed, state.detach())
            # Compute noise gradient ∂loss/∂noise
            noise_gradient, = torch.autograd.grad(loss, noise)
            # Move noise towards gradient to change state as much as possible
            step = noise + self.step_size * noise_gradient 
            # Normalize new noise step into norm induced ball 
            step_norm = self.norm_fn(step)
            noise = step / (step_norm + self.epsilon)

            # Reset noise gradients for next step
            noise = noise.detach().requires_grad_()


def kl_loss(input, target, reduction='batchmean'):
    return F.kl_div(
        F.log_softmax(input, dim=-1),
        F.softmax(target, dim=-1),
        reduction=reduction,
    )


def sym_kl_loss(input, target, reduction='sum', alpha=1.0):
    return alpha * F.kl_div(
        F.log_softmax(input, dim=-1),
        F.softmax(target.detach(), dim=-1),
        reduction=reduction,
    ) + F.kl_div(
        F.log_softmax(target, dim=-1),
        F.softmax(input.detach(), dim=-1),
        reduction=reduction,
    )


def js_loss(input, target, reduction='sum', alpha=1.0):
    # NOTE(shwang): Seems to pass muster.
    mean_proba = 0.5 * (F.softmax(input.detach(), dim=-1) + F.softmax(target.detach(), dim=-1))
    return alpha * (F.kl_div(
        F.log_softmax(input, dim=-1),
        mean_proba,
        reduction=reduction
    ) + F.kl_div(
        F.log_softmax(target, dim=-1),
        mean_proba,
        reduction=reduction
    ))


class SMARTClassificationModel(nn.Module):
    """
    Forward, in addition to returning the model output, also returns the
    """
    # b: batch_size, s: sequence_length, d: hidden_size , n: num_labels

    def __init__(self, model, smart_weight, radius, js_finish):
        super().__init__()
        self.model = model
        assert smart_weight >= 0, f"Negative value {smart_weight} not allowed"
        self.smart_weight = smart_weight
        self.radius = radius
        self.js_finish = js_finish

    def forward(self, input_ids, attention_mask, labels):
        # input_ids: (b, s), attention_mask: (b, s), labels: (b,)

        # Allow bert and deberta?
        embed = self.model.roberta.embeddings(input_ids)  # (b, s, d)

        def eval(embed):
            outputs = self.model.roberta(inputs_embeds=embed, attention_mask=attention_mask)
            # pooled = outputs['last_hidden_state']  # (b, d)
            pooled = outputs[0]  # (b, d)
            logits = self.model.classifier(pooled)  # (b, n)
            return logits

        def norm(x):
            return torch.norm(x, p=float('inf'), dim=-1, keepdim=True) / self.radius

        if self.js_finish:
            loss_last_fn = js_loss
        else:
            loss_last_fn = sym_kl_loss

        smart_loss_fn = SMARTLoss(
            eval_fn=eval,
            loss_fn=kl_loss, loss_last_fn=loss_last_fn, norm_fn=norm)
        state = eval(embed)
        loss = F.cross_entropy(state.view(-1, 2), labels.view(-1))
        # if embed.requires_grad and self.smart_weight > 0:
            # Embed should always require_grad. Otherwise this would be a silent skip of the SMART Loss, which would
            # be very bad for us.
        # TODO(shwang): Return a stats_dict here so that we can log smart_loss
        #    as well.
        if embed.requires_grad and self.smart_weight > 0:
            smart_loss = smart_loss_fn(embed, state)
            loss += self.smart_weight * smart_loss
        else:
            smart_loss = 0.0

        stats_dict = dict()
        stats_dict["smart_loss"] = float(smart_loss)
        return state, loss, stats_dict


class TextClassificationModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-5,
        weight_decay: float = 0,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        assert weight_decay >= 0
        self.weight_decay = weight_decay
        metrics = MetricCollection([Accuracy(), F1Score()])
        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='val/')

    def configure_optimizers(self, selective_decay=False):
        if selective_decay:
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters()
                            if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.weight_decay},  # TODO(shwang): Or whatever weight decay.
                {'params': [p for n, p in self.model.named_parameters()
                            if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
        else:
            optimizer_grouped_parameters = [
                {'params': self.model.parameters(), 'weight_decay': self.weight_decay}
            ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr,
                                      weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        input_ids, attention_masks, labels = batch
        # Compute output
        outputs, loss, stats = self.model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
        labels_pred = torch.argmax(outputs, dim=1)
        # Compute metrics
        metrics = self.train_metrics(labels, labels_pred)

        # Log loss and metrics
        def l2_squared(parameters):
            result = 0
            with th.no_grad():
                for p in parameters:
                    result += th.norm(p) ** 2
            return result

        self.log("train/loss", loss, on_step=True)
        est_weight_decay = self.weight_decay * l2_squared(self.model.parameters())
        self.log("train/est_weight_decay", est_weight_decay, on_step=True)
        self.log_dict(metrics, on_step=True, on_epoch=True)
        self.log_dict(stats, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_masks, labels = batch
        # Compute output
        outputs, loss, stats = self.model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
        labels_pred = torch.argmax(outputs, dim=1)
        # Compute metrics
        metrics = self.valid_metrics(labels, labels_pred)

        # Log loss and metrics
        self.log("valid_loss", loss, on_step=True)
        self.log_dict(metrics, on_step=True, on_epoch=True)
        self.log_dict(stats, on_step=True, on_epoch=True)
        return loss


DEFAULT_NAME = None
DEFAULT_GROUP = "NO_GROUP"

LOG_ROOT = pathlib.Path("logs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default=DEFAULT_NAME)
    parser.add_argument("-g", "--group", default=DEFAULT_GROUP)
    parser.add_argument("-p", "--project", default="cs4nlp-shwang-weight-decay")
    parser.add_argument(
        "--gpus", type=int, default=1,
        help="Use None to automatically detect GPU",
    )
    parser.add_argument("-l", "--learning-rate", "--lr", type=float, default=3e-5)
    parser.add_argument("-e", "--epochs", type=int, default=15)
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("--js_finish", action="store_true")
    parser.add_argument("--entity", default="sh-wang")
    parser.add_argument("--smart-loss-weight", "--smart-weight", type=float, default=0.0)
    parser.add_argument("--smart-loss-radius", "--smart-radius", type=float, default=0.25)
    parser.add_argument("--weight-decay", "-w", type=float, default=0.0)
    parser.add_argument("--pretrained-model", type=str, default='roberta-base')
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = parser.parse_args()

    wandb.login()
    pl.seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    if 'deberta-v3' in tokenizer.name_or_path:
        # This wasn't configured by the maintainers as of May 15 2022, leading to errors.
        # Adding it in for them.
        tokenizer.max_length = 512
        tokenizer.model_max_length = 512
    architecture = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model)

    datamodule = MCTACODatamodule(tokenizer, batch_size=args.batch_size, sequence_length=args.sequence_length)
    datamodule.setup()

    smart_architecture = SMARTClassificationModel(
        architecture,
        smart_weight=args.smart_loss_weight,
        radius=args.smart_loss_radius,
        js_finish=args.js_finish,
    )
    model = TextClassificationModel(smart_architecture, lr=args.learning_rate, weight_decay=args.weight_decay)
    # input_ids, input_mask, labels = next(iter(datamodule.train_dataloader()))
    # output, loss = smart_architecture(input_ids, input_mask, labels)

    # Wandb Logger
    pl_root_dir = LOG_ROOT / "pl"
    wandb_root_dir = LOG_ROOT / "wandb"
    pl_root_dir.mkdir(parents=True, exist_ok=True)
    wandb_root_dir.mkdir(parents=True, exist_ok=True)
    logger = pl.loggers.wandb.WandbLogger(
        project=args.project,
        entity=args.entity,
        save_dir=wandb_root_dir,
        name=args.name,
        group=args.group,
    )
    # Callbacks
    cb_progress_bar = pl.callbacks.RichProgressBar()
    cb_model_summary = pl.callbacks.RichModelSummary()

    # Train
    n_gpus = args.gpus if args.gpus is not None else int(th.cuda.is_available())
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[cb_progress_bar, cb_model_summary],
        max_epochs=args.epochs,
        gpus=n_gpus,
        default_root_dir=pl_root_dir,
        enable_checkpointing=False,
    )
    trainer.logger.log_hyperparams(args)
    trainer.fit(model=model, datamodule=datamodule)
    wandb.finish()


if __name__ == "__main__":
    main()
