import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import argparse
from torchmetrics import MetricCollection, Accuracy, F1Score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from typing import List, Union, Callable
from torch import Tensor
from itertools import count
from vat_pytorch import ALICEPPLoss, ALICELoss, SMARTLoss, kl_loss, sym_kl_loss
from pytorch_lightning.callbacks import GradientAccumulationScheduler

DEFAULT_NAME = "unamed_mctaco_tune_run"
DEFAULT_GROUP = "NO_GROUP"


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default=DEFAULT_NAME)
    parser.add_argument("-g", "--group", default=DEFAULT_GROUP)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--acc-grad", type=int, default=1)
    parser.add_argument(
        "--acc-grad-schedule",
        type=str,
        default="none",
        choices=["none", "increasing", "decreasing"],
    )
    parser.add_argument("--vat-loss-weight", type=float, default=1.0)
    parser.add_argument("--vat-loss-radius", type=float, default=1.0)
    parser.add_argument("--pretrained-model", type=str, default="roberta-base")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--vat", type=str, default="ALICE", choices=["SMART", "ALICE", "ALICEPP"]
    )
    parser.add_argument("--step-size", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--noise-var", type=float, default=1e-5)
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    args = parser.parse_args()

    # Use wandb login directly in the terminal before running the script
    # wandb.login()
    wandb.init(config=args)
    config = wandb.config

    if config.precision == 16:
        print(
            "WARNING: half precision used. This takes up to 1.5x GPU's memory, it is thus recommanded to use this option only in combination with gradient accumulation."
        )
    pl.seed_everything(config.seed)

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

        def __getitem__(self, idx):
            item = self.dataset[idx]
            tokenize = self.tokenizer.tokenize
            sequence = tokenize(item["sentence"] + " " + item["question"])
            answer = tokenize(item["answer"])
            label = item["label"]
            # Truncate excess tokens
            if answer:
                self.truncate_pair(sequence, answer, self.sequence_length - 3)
            else:
                if len(sequence) > self.sequence_length - 2:
                    sequence = sequence[0 : (self.sequence_length - 2)]
            # Compute tokens, ids, mask
            tokens = ["<s>"] + sequence + ["</s></s>"] + answer + ["</s>"]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            # Pad with 0
            while len(input_ids) < self.sequence_length:
                input_ids.append(0)
                input_mask.append(0)
            return (
                torch.tensor(input_ids),
                torch.tensor(input_mask),
                torch.tensor(label),
            )

    class MCTACODatamodule(pl.LightningDataModule):
        def __init__(
            self, tokenizer, batch_size: int, sequence_length: int, num_workers: int = 8
        ):
            super().__init__()
            self.tokenizer = tokenizer
            self.batch_size = batch_size
            self.sequence_length = sequence_length
            self.dataset_train = None
            self.dataset_valid = None
            self.num_workers = num_workers

        def setup(self, stage=None):
            self.dataset_train = MCTACODataset(
                split="validation",
                tokenizer=self.tokenizer,
                sequence_length=self.sequence_length,
            )
            self.dataset_valid = MCTACODataset(
                split="test",
                tokenizer=self.tokenizer,
                sequence_length=self.sequence_length,
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

    class ExtractedRoBERTa(nn.Module):
        def __init__(self):
            super().__init__()
            model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
            self.roberta = model.roberta
            self.layers = model.roberta.encoder.layer
            self.classifier = model.classifier
            self.attention_mask = None
            self.num_layers = len(self.layers) - 1

        def forward(self, hidden, with_hidden_states=False, start_layer=0):
            """Forwards the hidden value from self.start_layer layer to the logits."""
            hidden_states = [hidden]

            for layer in self.layers[start_layer:]:
                hidden = layer(hidden, attention_mask=self.attention_mask)[0]
                hidden_states += [hidden]

            logits = self.classifier(hidden)

            return (logits, hidden_states) if with_hidden_states else logits

        def get_embeddings(self, input_ids):
            """Computes first embedding layer given inputs_ids"""
            return self.roberta.embeddings(input_ids)

        def set_attention_mask(self, attention_mask):
            """Sets the correct mask on all subsequent forward passes"""
            self.attention_mask = self.roberta.get_extended_attention_mask(
                attention_mask,
                input_shape=attention_mask.shape,
                device=attention_mask.device,
            )  # (b, 1, 1, s)

    class SMARTClassificationModel(nn.Module):
        # b: batch_size, s: sequence_length, d: hidden_size , n: num_labels

        def __init__(self, extracted_model, weight=1.0):
            super().__init__()
            self.model = extracted_model
            self.weight = weight
            self.vat_loss = SMARTLoss(
                model=extracted_model,
                loss_fn=kl_loss,
                loss_last_fn=sym_kl_loss,
                step_size=config.step_size,
                epsilon=config.epsilon,
                noise_var=config.noise_var,
            )

        def forward(self, input_ids, attention_mask, labels):
            """input_ids: (b, s), attention_mask: (b, s), labels: (b,)"""
            # Get input embeddings
            embeddings = self.model.get_embeddings(input_ids)
            # Set mask and compute logits
            self.model.set_attention_mask(attention_mask)
            logits = self.model(embeddings)
            # Compute CE loss
            ce_loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1))
            # Compute VAT loss
            vat_loss = self.vat_loss(embeddings, logits)
            # Merge losses
            loss = ce_loss + self.weight * vat_loss
            return logits, loss

    class ALICEClassificationModel(nn.Module):
        # b: batch_size, s: sequence_length, d: hidden_size , n: num_labels

        def __init__(self, extracted_model):
            super().__init__()
            self.model = extracted_model
            self.vat_loss = ALICELoss(
                model=extracted_model,
                loss_fn=kl_loss,
                num_classes=2,
                alpha=config.vat_loss_weight,
                step_size=config.step_size,
                epsilon=config.epsilon,
                noise_var=config.noise_var,
            )

        def forward(self, input_ids, attention_mask, labels):
            """input_ids: (b, s), attention_mask: (b, s), labels: (b,)"""
            # Get input embeddings
            embeddings = self.model.get_embeddings(input_ids)
            # Set iteration specific data (e.g. attention mask)
            self.model.set_attention_mask(attention_mask)
            # Compute logits
            logits = self.model(embeddings)
            # Compute VAT loss
            loss = self.vat_loss(embeddings, logits, labels)
            return logits, loss

    class ALICEPPClassificationModel(nn.Module):
        # b: batch_size, s: sequence_length, d: hidden_size , n: num_labels

        def __init__(self, extracted_model):
            super().__init__()
            self.model = extracted_model
            self.vat_loss = ALICEPPLoss(
                model=extracted_model,
                loss_fn=kl_loss,
                num_classes=2,
                num_layers=self.model.num_layers,
                alpha=config.vat_loss_weight,
                step_size=config.step_size,
                epsilon=config.epsilon,
                noise_var=config.noise_var,
            )

        def forward(self, input_ids, attention_mask, labels):
            """input_ids: (b, s), attention_mask: (b, s), labels: (b,)"""
            # Get input embeddings
            embeddings = self.model.get_embeddings(input_ids)
            # Set iteration specific data (e.g. attention mask)
            self.model.set_attention_mask(attention_mask)
            # Compute logits
            logits, hidden_states = self.model(embeddings, with_hidden_states=True)
            # Compute VAT loss
            loss = self.vat_loss(hidden_states, logits, labels)
            return logits, loss

    class TextClassificationModel(pl.LightningModule):
        def __init__(self, model: nn.Module, lr: float = 3e-5):
            super().__init__()
            self.model = model
            self.lr = lr
            metrics = MetricCollection([Accuracy(), F1Score()])
            self.train_metrics = metrics.clone(prefix="train_")
            self.valid_metrics = metrics.clone(prefix="val_")

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
            return optimizer

        def training_step(self, batch, batch_idx):
            input_ids, attention_masks, labels = batch
            # Compute output
            outputs, loss = self.model(
                input_ids=input_ids, attention_mask=attention_masks, labels=labels
            )
            labels_pred = torch.argmax(outputs, dim=1)
            # Compute metrics
            metrics = self.train_metrics(labels, labels_pred)
            # Log loss and metrics
            self.log("train_loss", loss, on_step=True)
            self.log_dict(metrics, on_step=True, on_epoch=True)
            return loss

        def validation_step(self, batch, batch_idx):
            input_ids, attention_masks, labels = batch
            # Compute output
            outputs, loss = self.model(
                input_ids=input_ids, attention_mask=attention_masks, labels=labels
            )
            labels_pred = torch.argmax(outputs, dim=1)
            # Compute metrics
            metrics = self.valid_metrics(labels, labels_pred)
            # Log loss and metrics
            self.log("valid_loss", loss, on_step=True)
            self.log_dict(metrics, on_step=True, on_epoch=True)
            return loss

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
    datamodule = MCTACODatamodule(
        tokenizer, batch_size=config.batch_size, sequence_length=config.sequence_length
    )
    datamodule.setup()

    extracted_model = ExtractedRoBERTa()
    if config.vat == "SMART":
        vat_architecture = SMARTClassificationModel(
            extracted_model,
            weight=config.vat_loss_weight,
            radius=config.vat_loss_radius,
        )
    elif config.vat == "ALICE":
        vat_architecture = ALICEClassificationModel(extracted_model)
    else:
        vat_architecture = ALICEPPClassificationModel(extracted_model)
    model = TextClassificationModel(vat_architecture, lr=config.lr)

    # Wandb Logger
    logger = pl.loggers.wandb.WandbLogger(
        project="cs4nlp-vat-hp",
        entity="nextmachina",
        name=config.name,
        group=config.group,
    )
    # Callbacks
    cb_progress_bar = pl.callbacks.RichProgressBar()
    cb_model_summary = pl.callbacks.RichModelSummary()
    if config.acc_grad_schedule == "none":
        accumulator = GradientAccumulationScheduler(scheduling={0: config.acc_grad})
    elif config.acc_grad_schedule == "decreasing":
        accumulator = GradientAccumulationScheduler(
            scheduling={0: config.acc_grad, 8: 1}
        )
    else:
        accumulator = GradientAccumulationScheduler(scheduling={8: config.acc_grad})
    # Train
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[cb_progress_bar, cb_model_summary, accumulator],
        max_epochs=config.epochs,
        gpus=config.gpus,
        precision=config.precision,
    )
    trainer.logger.log_hyperparams(config)
    trainer.fit(model=model, datamodule=datamodule)
    wandb.finish()


if __name__ == "__main__":
    train()
