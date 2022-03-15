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
# -> intentar overfittear para un cjto del train
# -> Parametros para guardar el modelo final
# -> seguir insistiendo con el padding para que se haga dinamico
# -> probar diferentes batch sizes, tambien por parametros


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)

MAX_LEN = 512  # el maximo de BERT
AMOUNT_TAGS = 9
PERC = 5  # porcentaje del dataset que vamos a utilizar

checkpoint = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, num_labels=AMOUNT_TAGS)
model = BertForTokenClassification.from_pretrained(
    checkpoint, num_labels=AMOUNT_TAGS
)
# model = AutoModelForMaskedLM.from_pretrained(checkpoint, num_labels=AMOUNT_TAGS)

from datasets import load_dataset, load_metric

metric = load_metric("f1")


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


from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
)

def compute_metrics(pred: EvalPrediction):
  """Funcion que ejecuta el Trainer al evaluar"""

  # Aca tengo que ver como usar la mascara para poder sacar todo lo que metio
  # el padding

  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)

  #mini solucion, funciona pero hay que hacerlo mejor
  flat_labels = [item for label_list in labels for item in label_list]
  flat_preds = [item for label_list in preds for item in label_list]

  f1 = f1_score(flat_labels, flat_preds, average="macro")
  acc = accuracy_score(flat_labels, flat_preds)
  return {"accuracy": acc,"f1": f1,}

train_ds, test_ds = load_dataset(
    "conll2002", "es", split=[f"train[:{PERC}%]", f"test[:{PERC}%]"]
)
train_ds, test_ds = train_ds.rename_column(
    "ner_tags", "labels"
), test_ds.rename_column("ner_tags", "labels")

train_ds = train_ds.map(
    tokenize_function,
    batched=True,
    remove_columns=["id", "tokens", "pos_tags"],
)
test_ds = test_ds.map(
    tokenize_function,
    batched=True,
    remove_columns=["id", "tokens", "pos_tags"],
)

train_ds = train_ds.map(extend_labels)  # , batched=True)
test_ds = test_ds.map(extend_labels)  # , batched=True)

from transformers import DataCollatorWithPadding, TrainingArguments, Trainer

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer, padding="max_length", max_length=MAX_LEN
)
# Supuestamente con el dataCollator deberia de hacerse dinamico y mas liviano
# en memoria, pero no funcionaba. La idea es postergar lo mas posible el
# padding.

training_args = TrainingArguments(
    f"betoNER-finetuned-CONLL-{PERC}",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=1e-5,
    weight_decay=0.1,
    group_by_length=True,
)


trainer = Trainer(
    model,
    training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    optimizers=(torch.optim.AdamW(model.parameters()), None),
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model(f"betoNER-finetuned-CONLL-{PERC}/trained_model/")

metrics = trainer.evaluate()
print(metrics)
