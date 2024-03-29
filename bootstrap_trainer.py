# trainer.py
import numpy as np
import torch
import mlflow
import argparse
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

from datasets import load_dataset, Dataset, concatenate_datasets

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
EPOCHS = args.epochs
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

checkpoint = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, num_labels=len(LABEL_LIST))

model = AutoModelForTokenClassification.from_pretrained(
    # "results/conll_baseline/trained_model/",
    # si hay un modelo ya, cambiar path y descomentar.
    checkpoint,
    num_labels=9,
    id2label=TOKEN_MAP,
    label2id=LABEL_MAP,
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
    )

    labels, w_ids = [], []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(
            batch_index=i
        )  # Mapea tokens a cada palabra
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Setea tokens especiales a -100
            if word_idx is None:
                label_ids.append(IGNORE_INDEX)
            elif word_idx != previous_word_idx:  # pone le label del primer token
                label_ids.append(label[word_idx])
            else:
                # Replica el token. Si se cambia, entonces se pone -100 en
                # las subpalabras que no sea la primera.
                # label_ids.append(IGNORE_INDEX)
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
        w_ids.append(word_ids)

    tokenized_inputs["labels"] = labels
    tokenized_inputs["word_ids"] = w_ids
    return tokenized_inputs


def main():

    mlflow.set_experiment(f"{OUTPUT_DIR}")
    with mlflow.start_run():
        mlflow.log_param("a", 1)
        mlflow.log_metric("b", 2)

    test_ds = load_dataset("conll2002", "es", split="test")

    removable_columns_conll = ["id", "pos_tags", "ner_tags"]
    removable_columns_wikineural = ["lang", "ner_tags"]

    train_ds, test_ds, valid_ds = load_dataset(
        "conll2002", "es", split=["train", "test", "validation"]
    )

    train_ds = train_ds.filter(lambda ex: ex["ner_tags"] != [0] * len(ex["ner_tags"]))

    train_ds = train_ds.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=removable_columns_conll,
    )
    valid_ds = valid_ds.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=removable_columns_conll,
    )
    test_ds = test_ds.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=removable_columns_conll,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        save_strategy="no",
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

    trainer.save_model(f"{OUTPUT_DIR}/pre_boot/trained_model/")

    evaluate_and_save(f"{OUTPUT_DIR}/pre_boot/test.csv", trainer, test_ds)

    dump_log(f"{OUTPUT_DIR}/pre_boot/logs.txt", trainer)

    ## bootstrapping

    wneural_train_ds = load_dataset("Babelscape/wikineural", split="train_es[:100%]")
    wneural_train_ds = wneural_train_ds.filter(
        lambda ex: ex["ner_tags"] != [0] * len(ex["ner_tags"])
    )
    wneural_train_ds = wneural_train_ds.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=removable_columns_wikineural,
    )

    # wneural_bootstraped_ds = bootstrap_dataset(wneural_train_ds, trainer)
    wneural_bootstraped_ds = bootstrap_fine_grained(wneural_train_ds, trainer, 0.95)

    conll_ents = train_coverage(train_ds)
    wneural_ents = train_coverage(wneural_bootstraped_ds)
    with open(f"{OUTPUT_DIR}/coverage.txt", "+w") as f:
        f.write(f"Conll entities: {len(conll_ents)}\n")
        f.write(f"wikineural entities: {len(wneural_ents)}\n")
        f.write(f"Total entities: {len(conll_ents|wneural_ents)}\n")

    wneural_bootstraped_ds = wneural_bootstraped_ds.remove_columns(["tokens"])
    # wneural_bootstraped_ds.save_to_disk(f"{OUTPUT_DIR}/wikineural_bootstrapped_ds")

    trainer_b = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer_b.train()
    dump_log(f"{OUTPUT_DIR}/logs.txt", trainer_b)
    evaluate_and_save(f"{OUTPUT_DIR}/test.csv", trainer_b, test_ds)


if __name__ == "__main__":
    main()
