# trainer.py
import torch
import pandas as pd
import numpy as np
import argparse
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

from datasets import load_dataset, load_metric, Dataset
from transformers.trainer_utils import EvalPrediction
import json

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
    "--percentage",
    type=int,
    metavar="PERCENTAGE",
    help="Percentage of training dataset",
    default=10,
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

IGNORE_INDEX = nn.CrossEntropyLoss().ignore_index

LABEL_LIST = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
    "PAD",
]

LABEL_MAP = {label: i for i, label in enumerate(LABEL_LIST)}

OUTPUT_DIR = f"results/{args.output}"
MAX_LEN = args.max_len
PERC = args.percentage
EPOCHS = args.epochs
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size


# TODO
# -> intentar overfittear para un cjto del train
# -> seguir insistiendo con el padding para que se haga dinamico ✓
# -> probar diferentes batch sizes, tambien por parametros ✓
# -> Parametros para guardar el modelo final ✓


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

checkpoint = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, num_labels=len(LABEL_LIST))
model = AutoModelForTokenClassification.from_pretrained(
    checkpoint, num_labels=len(LABEL_LIST)
)


def tokenize_and_align_labels(examples):
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
    for i, label in enumerate(examples[f"ner_tags"]):
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
                label_ids.append(IGNORE_INDEX)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def flat_list(t: list):
    return [item for sublist in t for item in sublist]


def correct_pad(labels, preds):
    # detect pad
    unpad_labels, unpad_preds = [], []
    for idx, label in enumerate(labels):
        a, i = label[1], 1
        while a != -100 and i < (len(label) - 1):
            i += 1
            a = label[i]
        unpad_labels.append(label[1:i])
        unpad_preds.append(preds[idx][1:i])

    assert len(unpad_labels) == len(unpad_preds)

    return flat_list(unpad_labels), flat_list(unpad_preds)


def compute_metrics(pred: EvalPrediction):
    """
    Funcion que ejecuta el Trainer al evaluar, retorna un diccionario con la
    precision y el f1-score. La 2da metrica es mejor cuando los datos tienen
    mas desiguladad en las labels.
    """
    metric = load_metric("f1")
    # aca podria ignorar los que tengan el -100
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    labels, predictions = correct_pad(labels, predictions)

    return metric.compute(predictions=predictions, references=labels, average="micro")


from sklearn.metrics import accuracy_score, f1_score


def evaluate(ds: Dataset):
    """
    Para poder evaluar en base a un Dataset con el mismo formato que fue
    entrenado este modelo
    """
    predictions = trainer.predict(ds)
    preds = predictions.predictions.argmax(-1)
    # labels = predictions.label_ids
    y_pred, y_true = [], []
    labels = np.array(ds["labels"], dtype=object)
    for i in range(len(preds)):
        k = len(labels[i])
        # Saco el primero y el ultimo que son el [CLS] y [SEP]
        y_pred.append(preds[i][1 : k - 1])
        y_true.append(labels[i][1 : k - 1])

    flat_preds = flat_list(y_pred)
    flat_labels = flat_list(y_true)
    assert len(flat_preds) == len(flat_labels)

    f1_macro = f1_score(flat_labels, flat_preds, average="macro")
    acc = accuracy_score(flat_labels, flat_preds)
    return {
        "accuracy": acc,
        # "f1_micro": f1_micro,
        "f1_macro": f1_macro,
    }


def dump_log(filename, trainer):
    with open(f"{filename}", "+a") as f:
        for obj in trainer.state.log_history:
            json.dump(obj, f, indent=2)


def main():
    train_ds, test_ds, valid_ds = load_dataset(
        "conll2002",
        "es",
        split=[f"train[:{PERC}%]", f"test[:{PERC}%]", f"validation[:{PERC}%]"],
    )

    train_ds = train_ds.filter(lambda ex: ex["ner_tags"] != [0] * len(ex["ner_tags"]))
    test_ds = test_ds.filter(lambda ex: ex["ner_tags"] != [0] * len(ex["ner_tags"]))
    valid_ds = valid_ds.filter(lambda ex: ex["ner_tags"] != [0] * len(ex["ner_tags"]))

    train_ds = train_ds.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=["id", "pos_tags", "ner_tags", "tokens"],
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
        eval_dataset=test_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(f"{OUTPUT_DIR}/trained_model/")

    dump_log(f"{OUTPUT_DIR}/logs.txt", trainer)

    print(evaluate(train_ds))
    print(evaluate(test_ds))
    print(evaluate(valid_ds))


if __name__ == "__main__":
    main()
