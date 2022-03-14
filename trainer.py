# trainer.py
import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertForTokenClassification,
)

# TODO
# -> Parametros para guardar el modelo final
# -> seguir insistiendo con el padding para que se haga dinamico
# -> probar diferentes batch sizes, tambien por parametros


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)

MAX_LEN = 512  # el maximo de BERT
AMOUNT_TAGS = 9

checkpoint = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, num_labels=AMOUNT_TAGS)
model = BertForTokenClassification.from_pretrained(
    checkpoint, num_labels=AMOUNT_TAGS
)
# model = AutoModelForMaskedLM.from_pretrained(checkpoint, num_labels=AMOUNT_TAGS)


from datasets import load_dataset, load_metric

raw_datasets = load_dataset("conll2002", "es")

raw_datasets["train"] = raw_datasets["train"].rename_column(
    "ner_tags", "labels"
)
raw_datasets["test"] = raw_datasets["test"].rename_column("ner_tags", "labels")
raw_datasets["validation"] = raw_datasets["validation"].rename_column(
    "ner_tags", "labels"
)


def tokenize_function(example):
    return tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )


def extend_labels(example):
    k = len(example["labels"])
    if k > MAX_LEN:  # truncation
        example["labels"] = example["labels"][:MAX_LEN]
        return example
    else:  # padding with zeros
        example["labels"].extend([0 for _ in range(MAX_LEN - k)])
        return example


def compute_metrics(eval_pred):
    metric = load_metric("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


encoded_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["id", "tokens", "pos_tags"],
)
finish_datasets = encoded_datasets.map(extend_labels)

from transformers import DataCollatorWithPadding, TrainingArguments, Trainer

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer, padding="max_length", max_length=MAX_LEN
)
# Supuestamente con el dataCollator deberia de hacerse dinamico y mas liviano
# en memoria, pero no funcionaba. La idea es postergar lo mas posible el
# padding.

training_args = TrainingArguments(
    "betoNER-finetuned-CONLL",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=1e-5,
    group_by_length=True,
)


trainer = Trainer(
    model,
    training_args,
    train_dataset=finish_datasets["train"],
    eval_dataset=finish_datasets["test"],
    optimizers=(torch.optim.AdamW(model.parameters()), None),
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model("betoNER-finetuned-CONLL/trained_model/")

metrics=trainer.evaluate()
print(metrics)

