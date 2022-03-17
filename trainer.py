# trainer.py
#import torch
import pandas as pd
import numpy as np
import argparse
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertForTokenClassification,
)

parser = argparse.ArgumentParser(description='Train using Trainer Class from Huggingface!')
parser.add_argument("output",
                    type=str,
                    metavar="OUTPUT",
                    help="Directory to store the model.",
                    default="betoNER-finetuned-CONLL")
parser.add_argument("--maxlength",
                    type=int,
                    metavar="MAX LEN",
                    help="Maximum length of tokens",
                    default=512)
parser.add_argument("--percentage",
                    type=int,
                    metavar="PERCENTAGE",
                    help="Percentage of training samples",
                    default=10)
parser.add_argument("--epochs",
                    type=int,
                    metavar="EPOCHS",
                    help="number of epochs of training.",
                    default=3)
parser.add_argument("--lr",
                    type=float,
                    metavar="LEARNING_RATE",
                    help="learning rate.",
                    default=2e-5)
parser.add_argument("--train_batch_size",
                    type=int,
                    metavar="TRAIN_BACTH_SIZE",
                    help="Batch size in training stage",
                    default=8)
parser.add_argument("--test_batch_size",
                    type=int,
                    metavar="TEST_BACTH_SIZE",
                    help="Batch size in test stage",
                    default=8)

args = parser.parse_args()

OUTPUT_DIR = args.output
MAX_LEN = args.maxlength
AMOUNT_TAGS = 9
PERC = args.percentage
EPOCHS = args.epochs
LEARNING_RATE = args.lr
TRAIN_BACTH_SIZE = args.train_batch_size
TEST_BACTH_SIZE = args.test_batch_size


# TODO
# -> intentar overfittear para un cjto del train
# -> Parametros para guardar el modelo final
# -> seguir insistiendo con el padding para que se haga dinamico
# -> probar diferentes batch sizes, tambien por parametros


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)

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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def compute_metrics(pred : EvalPrediction):
    """Funcion que ejecuta el Trainer al evaluar"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels,
        preds,
        average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


train_ds, test_ds = load_dataset('conll2002', 'es', split=[f'train[:{PERC}%]', f'test[:{PERC}%]'])
train_ds, test_ds = train_ds.rename_column('ner_tags', 'labels'), test_ds.rename_column('ner_tags', 'labels')

train_ds = train_ds.map(tokenize_function, batched=True, remove_columns=['id', 'pos_tags']) #Dejar tokens por el len
test_ds = test_ds.map(tokenize_function, batched=True, remove_columns=['id', 'pos_tags']) #Dejar tokens por el len

train_ds = train_ds.map(extend_labels) #, batched=True)
test_ds = test_ds.map(extend_labels) #, batched=True)

from transformers import DataCollatorWithPadding, TrainingArguments, Trainer

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer, padding="max_length", max_length=MAX_LEN
)
# Supuestamente con el dataCollator deberia de hacerse dinamico y mas liviano
# en memoria, pero no funcionaba. La idea es postergar lo mas posible el
# padding.

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_accumulation_steps=1,
    evaluation_strategy="epoch",
    per_device_train_batch_size=TRAIN_BACTH_SIZE,
    per_device_eval_batch_size=TEST_BACTH_SIZE,
    num_train_epochs=EPOCH,
    learning_rate=LEARNING_RATE,
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

#El tema de los metrics, no anda para nada, vemos por otro lado
#metrics=trainer.evaluate()
#print(metrics)


from sklearn.metrics import accuracy_score, f1_score
def evaluate(ds : Dataset):
    predictions = trainer.predict(ds)
    preds = predictions.predictions.argmax(-1)
    labels = np.array(ds['labels'], dtype=object)
    unpadding_labels, unpadding_preds = [], []
    for i,s in enumerate(ds['tokens']):
        k = len(s)
        unpadding_labels.append(labels[i][:k])
        unpadding_preds.append(preds[i][:k])

    flat_preds = [item for label_list in unpadding_preds for item in label_list]
    flat_labels = [item for label_list in unpadding_labels for item in label_list]
    print(f"Length predictions:{len(flat_preds)}, Length labels: {len(flat_labels)}")
    
    f1 = f1_score(flat_labels, flat_preds, average="micro")
    acc = accuracy_score(flat_labels, flat_preds)
    return {"accuracy": acc, "f1": f1}

evaluate(train_ds)
