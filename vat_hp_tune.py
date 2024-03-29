import argparse
import csv
import dataclasses
import os
import platform
import sklearn.model_selection
from typing import Optional, Type

import datasets
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection, Accuracy, F1Score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from vat_pytorch import ALICEPPLoss, ALICELoss, SMARTLoss, kl_loss, sym_kl_loss, js_loss
from pytorch_lightning.callbacks import (
    GradientAccumulationScheduler,
    StochasticWeightAveraging,
)

DEFAULT_NAME = "unamed_mctaco_tune_run"
DEFAULT_GROUP = "NO_GROUP"

# Multithreading is not well supported outside of Linux
if platform.system() == "Linux":
    DEFAULT_CPU_WORKERS = 8
else:
    DEFAULT_CPU_WORKERS = 0


def default(value, default):
    if value is not None:
        return value
    return default


@dataclasses.dataclass
class InputExample:
    """A single training/test example for simple sequence classification."""

    guid: str
    text: str
    label: int  # 0: Duration less than one day. 1: Duration more than one day.


class TemporalVerbProcessor:
    def get_train_examples(self, data_dir):
        """See base class."""
        f = open(os.path.join(data_dir, "train.formatted.txt"), "r")
        lines = [x.strip() for x in f.readlines()]
        examples = self._create_examples(lines, "train")
        return examples

    def get_dev_examples(self, data_dir):
        f = open(os.path.join(data_dir, "test.formatted.txt"), "r")
        lines = [x.strip() for x in f.readlines()]
        return self._create_examples(lines, "dev")

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i in range(0, len(lines)):
            guid = "%s-%s" % (set_type, i)

            groups = lines[i].split("\t")
            text = groups[0]
            text_tokens = text.split()
            target_idx_a = int(groups[1]) + 1
            target_idx_b = int(groups[3]) + 2 + len(groups[0].split())
            text_tokens.insert(target_idx_a, "[unused510]")
            # text_tokens.insert(target_idx_b + 1, "[unused510]")
            text = " ".join(text_tokens)

            # Steven: cleaning some of the data to make it more palatable to
            #   the huggingface tokenizers.
            text = text.replace(" ##", "")  # into ##ler ##ant ==> intolerant
            text = text.replace(
                " ,", ","
            )  # of the world , this  ==> of the world, this
            text = text.replace(
                " .", "."
            )  # of the world . this  ==> of the world, this
            text = text.replace(
                " ' ", "'"
            )  # of the world . this  ==> of the world, this
            text = text.replace(
                " '", "'"
            )  # of the world . this  ==> of the world, this
            # text = text.replace(" [SEP] [unused510]", "[SEP]")  # of the world , this  ==> of the world, this
            label = groups[4]

            examples.append(InputExample(guid=guid, text=text, label=int(label)))
        return examples


def load_time_ml_dataset(split: str):
    data_dir = "dataset_timeml"
    processor = TemporalVerbProcessor()
    if split == "train":
        examples = processor.get_train_examples(data_dir)
    elif split == "test":
        examples = processor.get_dev_examples(data_dir)
    else:
        raise ValueError(split)
    return examples


class TimeMLDataset(Dataset):
    def __init__(self, split: str, tokenizer, sequence_length: int):
        # I'll try to load this as a simple list.
        self.dataset = load_time_ml_dataset(split)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        output = self.tokenizer(
            item.text,
            padding="max_length",
            max_length=self.sequence_length,
            truncation=True,
        )
        input_ids = output["input_ids"]
        input_mask = output["attention_mask"]
        return (
            torch.tensor(input_ids),
            torch.tensor(input_mask),
            torch.tensor(item.label),
            {},
        )


class MCTACODataset(Dataset):
    def __init__(self, split: str, tokenizer, sequence_length: int):
        mapping = {"train": "validation", "test": "test"}
        split_name = mapping[split]
        self.dataset = datasets.load_dataset("mc_taco")[split_name]
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
        item = self.dataset[int(idx)]
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
            dict(
                sentence=item["sentence"],
                question=item["question"],
                answer=item["answer"],
            ),
        )


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        batch_size: int,
        sequence_length: int,
        multithreading: bool,
        gpus: int,
        ds_constructor: Type,
        *,
        validation_mode: bool,
    ):
        super().__init__()
        self.ds_constructor = ds_constructor
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.dataset_train = None
        self.dataset_valid = None
        # If validation_mode is True, then use the training dataset only.
        # Randomly reserve 10% of the training dataset as the validation set.
        # If validation is True, then train with the full training dataset
        # and use the test dataset for evaluation.
        self.validation_mode = validation_mode
        if multithreading:
            self.num_workers = DEFAULT_CPU_WORKERS
        else:
            self.num_workers = 0
        if gpus >= 1:
            self.pin_memory = True
        else:
            self.pin_memory = False

    def setup(self, stage=None):
        dataset_train = self.ds_constructor(
            split="train",
            tokenizer=self.tokenizer,
            sequence_length=self.sequence_length,
        )
        if self.validation_mode:
            # Create a validation set split by splitting the training set.
            # We reserve 10% of the data as the validation set.
            VALIDATION_PROP = 0.1
            os.environ[
                "TOKENIZERS_PARALLELISM"
            ] = "false"  # Avoids repeated fork warning in combo with PytorchLightning
            (
                self.dataset_train,
                self.dataset_valid,
            ) = sklearn.model_selection.train_test_split(
                dataset_train, test_size=VALIDATION_PROP
            )
            print(
                f"[SETUP]: loaded training set with {len(self.dataset_train)} examples"
            )
            print(
                f"[SETUP]: loaded validation set with {len(self.dataset_valid)} examples"
            )
        else:
            self.dataset_train = dataset_train
            self.dataset_valid = self.ds_constructor(
                split="test",
                tokenizer=self.tokenizer,
                sequence_length=self.sequence_length,
            )
            print(
                f"[SETUP]: loaded training set with {len(self.dataset_train)} examples"
            )
            print(f"[SETUP]: loaded TEST set with {len(self.dataset_valid)} examples")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset_valid,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


def train():
    # Use wandb login directly in the terminal before running the script
    # wandb.login()
    args = parse_args()
    wandb.init(config=args)
    config = wandb.config

    if config.seed is not None:
        pl.seed_everything(config.seed)

    class ExtractedRoBERTa(nn.Module):
        def __init__(self):
            super().__init__()
            model = AutoModelForSequenceClassification.from_pretrained(
                config.pretrained_model, classifier_dropout=config.classifier_dropout
            )
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

    class ClassificationModel(nn.Module):
        # b: batch_size, s: sequence_length, d: hidden_size , n: num_labels

        def __init__(self, extracted_model):
            super().__init__()
            self.model = extracted_model

        def forward(self, input_ids, attention_mask, labels):
            """input_ids: (b, s), attention_mask: (b, s), labels: (b,)"""
            # Get input embeddings
            embeddings = self.model.get_embeddings(input_ids)
            # Set mask and compute logits
            self.model.set_attention_mask(attention_mask)
            logits = self.model(embeddings)
            # Compute CE loss
            loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1))
            return logits, loss

    class SMARTClassificationModel(nn.Module):
        # b: batch_size, s: sequence_length, d: hidden_size, n: num_labels

        def __init__(self, extracted_model):
            super().__init__()
            self.model = extracted_model
            self.weight = config.vat_loss_weight
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
                loss_fn=js_loss,
                num_classes=2,
                num_layers=self.model.num_layers,
                alpha=config.vat_loss_weight,
                step_size=config.step_size,
                epsilon=config.epsilon,
                noise_var=config.noise_var,
                max_layer=default(config.max_layer, self.model.num_layers),
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
            metrics = MetricCollection([Accuracy()])
            self.train_metrics = metrics.clone(prefix="train_")
            self.valid_metrics = metrics.clone(prefix="val_")

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
            return optimizer

        def training_step(self, batch, batch_idx):
            input_ids, attention_masks, labels, metadata = batch
            # Compute output
            outputs, loss = self.model(
                input_ids=input_ids, attention_mask=attention_masks, labels=labels
            )
            labels_pred = torch.argmax(outputs, dim=1)
            # Compute metrics
            metrics = self.train_metrics(labels, labels_pred)
            # Log loss and metrics
            batch_size = len(labels)
            self.log("train_loss", loss, on_step=True, batch_size=batch_size)
            self.log_dict(metrics, on_step=True, on_epoch=True, batch_size=batch_size)
            return {"loss": loss}

        def validation_step(self, batch, batch_idx):
            input_ids, attention_masks, labels, metadata = batch
            # Compute output
            outputs, loss = self.model(
                input_ids=input_ids, attention_mask=attention_masks, labels=labels
            )
            labels_pred = torch.argmax(outputs, dim=1)
            # Compute metrics
            metrics = self.valid_metrics(labels, labels_pred)
            # Log loss and metrics
            batch_size = len(labels)
            self.log("valid_loss", loss, on_step=True, batch_size=batch_size)
            self.log_dict(metrics, on_step=True, on_epoch=True, batch_size=batch_size)
            return {
                "loss": loss,
                "labels_pred": labels_pred,
                "labels": labels,
                **metadata,
            }

    class MCTACOTextClassifierModel(TextClassificationModel):
        def validation_epoch_end(self, validation_step_outputs):
            # print(validation_step_outputs[0])
            sentences = []
            questions = []
            predictions = []
            labels = []

            def _itemize(x):
                return [e.item() for e in x]

            for batch in validation_step_outputs:
                sentences.extend(batch["sentence"])
                questions.extend(batch["question"])
                predictions.extend(_itemize(batch["labels_pred"]))
                labels.extend(_itemize(batch["labels"]))
                assert len(sentences) == len(questions)
                assert len(labels) == len(predictions)
                assert len(sentences) == len(labels)
            keys = [(s, q) for s, q in zip(sentences, questions)]
            assert set(labels) == {0, 1}

            # Intermediate calculations for F1 and EM scores
            result_map = (
                {}
            )  # key => list[successful bool]  (EM if all(result_map[key]) is True)
            prediction_count_map = {}  # key => (int) num positive predictions
            gold_count_map = {}  # key => (int) num of positive ground truth labels

            prediction_map = {}  # key => list(predictions (1 or 0))
            label_map = {}  # key => list(labels (1 or 0))  1 is yes
            for i, key in enumerate(keys):
                if key not in result_map:
                    result_map[key] = []
                    prediction_count_map[key] = 0.0
                    gold_count_map[key] = 0.0
                    prediction_map[key] = []
                    label_map[key] = []
                prediction_map[key].append(predictions[i])
                label_map[key].append(labels[i])
                if predictions[i] == 1:
                    prediction_count_map[key] += 1.0
                if labels[i] == 1:
                    gold_count_map[key] += 1.0
                result_map[key].append(predictions[i] == labels[i])

            total = 0.0  # total number of keys
            exact_match_count = 0.0
            f1 = 0.0
            for key in result_map.keys():
                em_so_far = True
                total += 1.0
                true_positives = 0.0
                for i, pred_is_correct in enumerate(result_map[key]):
                    em_so_far = em_so_far and pred_is_correct
                    # prediction_map[key][i] == 1 ==> The answer is "yes"
                    # So cur_correct counts the number of true positives
                    if pred_is_correct and prediction_map[key][i] == 1:
                        assert label_map[key][i] == prediction_map[key][i]
                        true_positives += 1.0
                if em_so_far:
                    exact_match_count += 1.0
                p = 1.0
                if prediction_count_map[key] > 0.0:
                    p = true_positives / prediction_count_map[key]
                r = 1.0
                if gold_count_map[key] > 0.0:
                    r = true_positives / gold_count_map[key]
                # print("len result_map key:", len(result_map[key]))
                # print("p", p, "r", r, "tp", true_positives, "tp+fp", prediction_count_map[key],
                #       "positive labels", gold_count_map[key])
                if p + r > 0.0:
                    curr_f1 = 2 * p * r / (p + r)
                    # print(curr_f1, "\n")
                    f1 += curr_f1

            em_score = exact_match_count / total
            f1_score = f1 / total
            # if abs(em_score - f1_score) < 1e-6:
            #     breakpoint()
            self.log("val_custom_EM_score", em_score, on_epoch=True)
            self.log("val_custom_F1_score", f1_score, on_epoch=True)

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
    if config.dataset == "mctaco":
        ds_constructor = MCTACODataset
    elif config.dataset == "timeml":
        ds_constructor = TimeMLDataset
    else:
        raise ValueError(config.dataset)
    datamodule = DataModule(
        tokenizer,
        ds_constructor=ds_constructor,
        batch_size=config.batch_size,
        sequence_length=config.sequence_length,
        gpus=config.gpus,
        multithreading=config.multithreading,
        validation_mode=(args.train_mode == "valid"),
    )

    extracted_model = ExtractedRoBERTa()
    if config.vat == "None":
        architecture = ClassificationModel(extracted_model)
    elif config.vat == "SMART":
        architecture = SMARTClassificationModel(
            extracted_model,
        )
    elif config.vat == "ALICE":
        architecture = ALICEClassificationModel(extracted_model)
    elif config.vat == "ALICEPP":
        architecture = ALICEPPClassificationModel(extracted_model)
    else:
        raise ValueError(config.vat)

    if config.dataset == "timeml":
        model = TextClassificationModel(architecture, lr=config.lr)
    elif config.dataset == "mctaco":
        model = MCTACOTextClassifierModel(architecture, lr=config.lr)
    else:
        raise ValueError(config.dataset)

    # Wandb Logger
    logger = pl.loggers.wandb.WandbLogger(
        project="CS4NLP_Project2022",
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
        accumulator = GradientAccumulationScheduler(
            scheduling={0: 1, 8: config.acc_grad}
        )

    if config.swa:
        swa = StochasticWeightAveraging(swa_lrs=5e-6, annealing_strategy="linear")
    else:
        swa = None

    # Train
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[cb_progress_bar, cb_model_summary, accumulator],
        max_epochs=config.epochs,
        gpus=config.gpus,
        precision=config.precision,
        enable_checkpointing=config.enable_checkpointing,
        gradient_clip_val=config.grad_clip_value,
    )
    trainer.logger.log_hyperparams(config)
    trainer.fit(model=model, datamodule=datamodule)
    wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default=DEFAULT_NAME)
    parser.add_argument("-g", "--group", default=DEFAULT_GROUP)
    parser.add_argument("--dataset", default="mctaco", choices=["mctaco", "timeml"])
    parser.add_argument(
        "--train-mode",
        default="valid",
        choices=["valid", "test"],
        help=(
            "If valid, then reserve 10% of training set as validation set."
            "If test, then use full training set for training, and evaluate on test set."
        ),
    )
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--vat",
        type=str,
        default="ALICE",
        choices=["None", "SMART", "ALICE", "ALICEPP"],
    )
    parser.add_argument("--step-size", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--noise-var", type=float, default=1e-5)
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--enable-checkpointing", type=bool, default=False)
    parser.add_argument("--multithreading", type=bool, default=True)
    parser.add_argument("--max-layer", type=int, default=None)
    parser.add_argument("--grad-clip-value", type=float, default=None)
    parser.add_argument("--classifier-dropout", type=float, default=None)
    parser.add_argument("--swa", type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    train()
