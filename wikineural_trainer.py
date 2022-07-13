# trainer.py
from cgi import test
import torch
import mlflow
import pandas as pd
import numpy as np
import argparse
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

from datasets import load_dataset, Dataset, concatenate_datasets
import json

from utils import *

parser = argparse.ArgumentParser(
    description="Train using Trainer Class from Huggingface!"
)
parser.add_argument(
    "output",
    type=str,
    metavar="OUTPUT",
    help="Directory to store the model.",
    #    default="betoNER-finetuned-CONLL",
)
parser.add_argument(
    "--max_len",
    type=int,
    metavar="MAX LEN",
    help="Maximum length of tokens",
    default=512,
)
parser.add_argument(
    "--epochs",
    type=int,
    metavar="EPOCHS",
    help="Epochs of training.",
    default=3,
)
parser.add_argument(
    "--lr",
    type=float,
    metavar="LEARNING_RATE",
    help="learning rate to the AdamW optimizer.",
    default=2e-5,
)
parser.add_argument(
    "--batch_size",
    type=int,
    metavar="BATCH_SIZE",
    help="Batch size",
    default=8,
)

args = parser.parse_args()

OUTPUT_DIR = f"results/{args.output}"
MAX_LEN = args.max_len
PERC = 100
EPOCHS = args.epochs
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}")

checkpoint = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, num_labels=len(LABEL_LIST))
model = AutoModelForTokenClassification.from_pretrained(
    checkpoint, num_labels=len(LABEL_LIST)
)


from transformers.tokenization_utils_base import BatchEncoding


def tokenize_and_align_labels(examples) -> BatchEncoding:
    """
    Funcion para tokenizar las sentencias, ademas de alinear palabras con los
    labels. Usada junto al map de los Dataset, retorna otro Dataset, que
    ahora contiene labels, token_ids y attention_mask.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        return_token_type_ids=False,
        # max_length=MAX_LEN,
        # padding="max_length",
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(
            batch_index=i
        )  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(IGNORE_INDEX)
            elif (
                word_idx != previous_word_idx
            ):  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                ##label_ids.append(IGNORE_INDEX)
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def output_phrase(phrase: str, trainer: Trainer) -> str:
    tokenized_input = tokenizer([phrase], return_token_type_ids=False)
    ds = Dataset.from_dict(tokenized_input)
    pred = trainer.predict(ds)
    labels = pred.predictions.argmax(-1)[0]
    res = ""
    for i, s in enumerate(tokenized_input["input_ids"][0]):
        # res += (s, tokenizer.decode(s), labels[i], TOKEN_MAP[labels[i]])
        res += f"{tokenizer.decode(s)} {TOKEN_MAP[labels[i]]}\n"
    return res


def main():

    mlflow.set_experiment(f"{OUTPUT_DIR}")
    with mlflow.start_run():
        mlflow.log_param("a", 1)
        mlflow.log_metric("b", 2)

    train_ds, test_ds, val_ds = load_dataset(
        "Babelscape/wikineural", split=[f"train_es", f"test_es", f"val_es"]
    )
    # los agrupo a los 3 datasets por ahora, puedo intentar hacer un cross validation o algo sino
    train_ds = concatenate_datasets([train_ds, test_ds, val_ds])

    test_ds, valid_ds = load_dataset(
        "conll2002",
        "es",
        split=["test", "validation"],
    )

    train_ds = train_ds.filter(lambda ex: ex["ner_tags"] != [0] * len(ex["ner_tags"]))
    test_ds = test_ds.filter(lambda ex: ex["ner_tags"] != [0] * len(ex["ner_tags"]))
    valid_ds = valid_ds.filter(lambda ex: ex["ner_tags"] != [0] * len(ex["ner_tags"]))

    train_ds = train_ds.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=["lang", "ner_tags"],
    )
    test_ds = test_ds.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=["id", "pos_tags", "ner_tags", "tokens"],
    )
    valid_ds = valid_ds.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=["id", "pos_tags", "ner_tags", "tokens"],
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        save_strategy="no",  # esto para no hacer checkpointing
        logging_steps=50,
        evaluation_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.1,
        warmup_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(f"{OUTPUT_DIR}/trained_model/")

    dump_log(f"{OUTPUT_DIR}/logs.txt", trainer)

    with open(f"{OUTPUT_DIR}/results.txt", "w+") as f:
        f.write(f"Evaluation on train data:\n{evaluate(trainer, train_ds)}\n")
        f.write(f"Evaluation on test data:\n{evaluate(trainer, test_ds)}\n")
        f.write(f"Evaluation on validation data:\n{evaluate(trainer, valid_ds)}\n")


if __name__ == "__main__":
    main()
