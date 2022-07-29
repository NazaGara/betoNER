# utils.py
import torch
import pandas as pd
import numpy as np
from transformers import Trainer
from datasets import load_metric, Dataset
from transformers.trainer_utils import EvalPrediction
import json


IGNORE_INDEX = torch.nn.CrossEntropyLoss().ignore_index

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
    #    "PAD",
]

LABEL_MAP = {label: i for i, label in enumerate(LABEL_LIST)}
TOKEN_MAP = {i: label for i, label in enumerate(LABEL_LIST)}


def flat_list(t: list) -> list:
    return [item for sublist in t for item in sublist]


def correct_pad(labels, preds):
    """
    Function that removes the pad elements present in the labels and in the
    predictions. Retorns a tuple with the flatten unpadded lists, ready to be
    used in the metric function.
    """
    # detect pad
    unpad_labels, unpad_preds = [], []
    for idx, label in enumerate(labels):
        elem, i = label[1], 1
        while elem != -100 and i < (len(label) - 1):
            i += 1
            elem = label[i]
        unpad_labels.append(label[1:i])
        unpad_preds.append(preds[idx][1:i])

    assert len(unpad_labels) == len(unpad_preds)

    return flat_list(unpad_labels), flat_list(unpad_preds)


def compute_metrics(pred: EvalPrediction) -> dict:
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

    return metric.compute(predictions=predictions, references=labels, average="macro")


from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


def evaluate(trainer: Trainer, ds: Dataset) -> dict:
    """
    Para poder evaluar en base a un Dataset con el mismo formato que fue
    entrenado este modelo
    """
    predictions = trainer.predict(ds)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    # flat_preds, flat_labels = correct_pad(labels, preds)
    flat_labels, flat_preds = correct_pad(labels, preds)
    # assert len(flat_preds) == len(flat_labels)

    # f1_macro = f1_score(flat_labels, flat_preds, average="macro")
    # acc = accuracy_score(flat_labels, flat_preds)
    return classification_report(
        flat_labels,
        flat_preds,
        target_names=LABEL_LIST,
        output_dict=True,
    )


def evaluate_and_save(filename, trainer: Trainer, ds: Dataset):
    predictions = trainer.predict(ds)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids

    # flat_preds, flat_labels = correct_pad(labels, preds)
    flat_labels, flat_preds = correct_pad(labels, preds)
    # assert len(flat_preds) == len(flat_labels)

    report = classification_report(
        flat_labels,
        flat_preds,
        target_names=LABEL_LIST,
        output_dict=True,
    )

    y_true = [*map(TOKEN_MAP.get, flat_labels)]
    y_pred = [*map(TOKEN_MAP.get, flat_preds)]
    fig, ax = plt.subplots(figsize=(15, 15))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="gist_ncar_r")

    plt.savefig(f"{filename}_heatmap.png")

    df = pd.DataFrame(report).transpose()
    df.to_csv(filename)
    return report


def dump_log(filename, trainer: Trainer):
    """
    Save the log from the training into a filename on the OUTPUT DIR directory
    """
    with open(f"{filename}", "w+") as f:
        for obj in trainer.state.log_history:
            json.dump(obj, f, indent=2)
