# trainer.py
import torch
import pandas as pd
import numpy as np
import argparse
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertForTokenClassification,
)

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
    "--maxlength",
    type=int,
    metavar="MAX LEN",
    help="Maximum length of tokens",
    default=512,
)
parser.add_argument(
    "--percentage",
    type=int,
    metavar="PERCENTAGE",
    help="Percentage of training samples",
    default=5,
)
parser.add_argument(
    "--epochs",
    type=int,
    metavar="EPOCHS",
    help="number of epochs of training.",
    default=3,
)
parser.add_argument(
    "--lr",
    type=float,
    metavar="LEARNING_RATE",
    help="learning rate.",
    default=2e-5,
)
parser.add_argument(
    "--train_batch_size",
    type=int,
    metavar="TRAIN_BACTH_SIZE",
    help="Batch size in training stage",
    default=8,
)
parser.add_argument(
    "--test_batch_size",
    type=int,
    metavar="TEST_BACTH_SIZE",
    help="Batch size in test stage",
    default=8,
)

args = parser.parse_args()

OUTPUT_DIR = f"results/{args.output}"
MAX_LEN = args.maxlength
PERC = args.percentage
EPOCHS = args.epochs
LEARNING_RATE = args.lr
TRAIN_BACTH_SIZE = args.train_batch_size
TEST_BACTH_SIZE = args.test_batch_size


# TODO
# -> intentar overfittear para un cjto del train
# -> seguir insistiendo con el padding para que se haga dinamico
# -> probar diferentes batch sizes, tambien por parametros
# -> Parametros para guardar el modelo final ✓

TAGS = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
    "INWORD_TOKEN",
]
tag2id = {t: i for i, t in enumerate(TAGS)}
id2tag = {i: t for i, t in enumerate(TAGS)}


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)

checkpoint = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, num_labels=len(TAGS))
model = BertForTokenClassification.from_pretrained(
    checkpoint, num_labels=len(TAGS)
)

from datasets import load_dataset, load_metric, Dataset

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
                label_ids.append(-100)
            elif (
                word_idx != previous_word_idx
            ):  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(tag2id["INWORD_TOKEN"])
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


from transformers.trainer_utils import EvalPrediction

def flat_list(t):
    return [item for sublist in t for item in sublist]


def compute_metrics(pred: EvalPrediction):
    """
    Funcion que ejecuta el Trainer al evaluar, retorna un diccionario con la
    precision y el f1-score. La 2da metrica es mejor cuando los datos tienen
    mas desiguladad en las labels.
    """
    metric = load_metric("f1")
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    flat_labels, flat_preds = flat_list(labels), flat_list(predictions)
    return metric.compute(
        predictions=flat_preds, references=flat_labels, average="micro"
    )


train_ds, test_ds, valid_ds = load_dataset(
    "conll2002",
    "es",
    split=[f"train[:{PERC}%]", f"test[:{PERC}%]", f"validation[:{PERC}%]"],
)

train_ds = train_ds.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=["id", "pos_tags", "ner_tags"],
)
test_ds = test_ds.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=["id", "pos_tags", "ner_tags"],
)

from transformers import (
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

# data_collator = DataCollatorWithPadding(
#    tokenizer=tokenizer, padding="max_length", max_length=MAX_LEN
# )
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_accumulation_steps=1,
    evaluation_strategy="epoch",
    per_device_train_batch_size=TRAIN_BACTH_SIZE,
    per_device_eval_batch_size=TEST_BACTH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.1,
    # group_by_length=True,
)


trainer = Trainer(
    model,
    training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    optimizers=(torch.optim.AdamW(model.parameters()), None),
    data_collator=data_collator,
    tokenizer=tokenizer,
    # compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model(f"{OUTPUT_DIR}/trained_model/")

# El tema de los metrics, no anda para nada, vemos por otro lado
# metrics=trainer.evaluate()
# print(metrics)


from sklearn.metrics import accuracy_score, f1_score


def evaluate(ds: Dataset):
    """
    Para poder evaluar en base a un Dataset con el mismo formato que fue
    entrenado este modelo
    """
    predictions = trainer.predict(ds)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids #np.array(ds["labels"], dtype=object)
    flat_preds = flat_list(preds)
    flat_labels =flat_list(labels)

    assert(len(flat_preds)==len(flat_labels))

    f1 = f1_score(flat_labels, flat_preds, average="micro")
    acc = accuracy_score(flat_labels, flat_preds)
    metric = load_metric('f1')
    f1_hf =  metric.compute(predictions=flat_preds, references=flat_labels, average='micro')
    return {"accuracy": acc, "f1": f1, "f1_HF" : f1_hf}


#results = evaluate(test_ds)
#print(f"Results obtained: {results}")

print('\n\n')
predictions = trainer.predict(train_ds)
preds = predictions.predictions.argmax(-1)

print(preds[0], preds[1])

with open('results/prueba.txt', '+a') as f:
    f.write(preds)

