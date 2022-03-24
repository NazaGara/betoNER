import torch
import pandas as pd
import numpy as np
import argparse
from conllu import parse_incr
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    AutoModelForTokenClassification,
)
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
from classes import CustomDataset, SentenceGetter, BERTClass

import torch.nn as nn

IGNORE_INDEX = nn.CrossEntropyLoss().ignore_index
INWORD_PAD_LABEL = "PAD"
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
]

LABEL_MAP = {label: i for i, label in enumerate(LABEL_LIST)}
LABEL_MAP[INWORD_PAD_LABEL] = IGNORE_INDEX

parser = argparse.ArgumentParser(description="Train using Pytorch tensors!")
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
    "--batch_size",
    type=int,
    metavar="BATCH SIZE",
    help="Batch size for train and evaluation",
    default=8,
)
args = parser.parse_args()
OUTPUT_DIR = f"results/{args.output}"
MAX_LEN = args.maxlength
BATCH_SIZE = args.batch_size


# Pre Processing
checkpoint = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, num_labels=len(TAGS))
model = AutoModelForTokenClassification.from_pretrained(
    checkpoint, finetuning_task="conll2002", num_labels=len(TAGS)
)


def read_conllu(file_name: str):
    f = open(f"{file_name}.conllu", "r", encoding="utf-8")
    token_lists = list(parse_incr(f, fields=["id", "word", "tag"]))
    f.close()
    words, words_tags, labels = [], [], []
    sen_idx, c = [], 0
    for i, tl in enumerate(token_lists):
        for j, t in enumerate(tl):
            words.append(token_lists[i][j]["word"])
            words_tags.append(token_lists[i][j]["tag"])

            tokenized = tokenizer.tokenize(token_lists[i][j]["word"])
            labels += [
                [LABEL_MAP[token_lists[i][j]["tag"]]]
                + [LABEL_MAP[INWORD_PAD_LABEL]] * (len(tokenized) - 1)
            ]

            if token_lists[i][j]["id"] == 0:
                sen_idx.append(f"sentence {c}")
                c += 1
            else:
                sen_idx.append(np.NaN)

    data = {
        "sentence_id": sen_idx,
        "word": words,
        "tag": words_tags,
        "labels": labels,
    }
    df = pd.DataFrame(data=data)
    df = df.fillna(method="ffill")
    return df


def flat_list(t):
    return [item for sublist in t for item in sublist]


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)

df_train = read_conllu("data/conll/train")
df_test = read_conllu("data/conll/test")

train_getter = SentenceGetter(df_train)
test_getter = SentenceGetter(df_test)

tags_vals = list(set(df_train["tag"].values))
train_sentences = [" ".join([s[0] for s in sent]) for sent in train_getter.sentences]
test_sentences = [" ".join([s[0] for s in sent]) for sent in test_getter.sentences]
train_labels_from_df = [[s[1] for s in sent] for sent in train_getter.sentences]
test_labels_from_df = [[s[1] for s in sent] for sent in test_getter.sentences]
train_labels = list(map(flat_list, train_labels_from_df))
test_labels = list(map(flat_list, test_labels_from_df))

# Configuration

# el original estaba en 200, pero uso el max de bert 512

WEIGHT_DECAY = 0.1
EPOCHS = 3
LEARNING_RATE = 2e-05

# test_labels = train_labels  # afirmo que usan las mismas labels

print(f"TRAIN Dataset: {len(train_sentences)} sentences")
print(f"TEST Dataset: {len(test_sentences)} sentences")

training_set = CustomDataset(tokenizer, train_sentences, train_labels, MAX_LEN)
testing_set = CustomDataset(tokenizer, test_sentences, test_labels, MAX_LEN)

train_params = {
    "batch_size": BATCH_SIZE,
    "shuffle": True,
    "num_workers": 0,
}

test_params = {
    "batch_size": BATCH_SIZE,
    "shuffle": True,
    "num_workers": 0,
}

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

# model = BERTClass(len(TAGS))

model.to(device)
WARMUP_STEPS = 100
total_steps = len(training_loader) * EPOCHS
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(WARMUP_STEPS * total_steps),  # warmup is a %
    num_training_steps=total_steps,
)


def train(epoch):
    model.train()
    for _, data in enumerate(training_loader):
        ids = data["ids"].to(device, dtype=torch.long)
        mask = data["mask"].to(device, dtype=torch.long)
        targets = data["tags"].to(device, dtype=torch.long)

        model.zero_grad()
        optimizer.zero_grad()

        loss = model(ids, mask, labels=targets)[0]

        loss.backward()
        optimizer.step()
        scheduler.step()

        if _ % 500 == 0:
            print(f"Epoch: {epoch}, Loss:  {loss.item()}")


for e in tqdm(range(EPOCHS), desc="Training Model"):
    train(e)


model.save_pretrained(OUTPUT_DIR)

# Evaluation

from seqeval.metrics import f1_score


def flat_accuracy(preds, labels):
    flat_preds = np.argmax(preds, axis=2).flatten()
    flat_labels = labels.flatten()

    with open(f"{OUTPUT_DIR}/preds", "+a") as f:
        for item in flat_preds:
            f.write("%s\n" % item)

    with open(f"{OUTPUT_DIR}/labels", "+a") as f:
        for item in flat_labels:
            f.write("%s\n" % item)

    return np.sum(flat_preds == flat_labels) / len(flat_labels)


model.eval()
eval_loss = 0
eval_accuracy = 0
n_correct = 0
n_wrong = 0
total = 0
predictions, true_labels = [], []
nb_eval_steps, nb_eval_examples = 0, 0
with torch.no_grad():
    for data in tqdm(training_loader, desc="Evaluation"):
        ids = data["ids"].to(device, dtype=torch.long)
        mask = data["mask"].to(device, dtype=torch.long)
        targets = data["tags"].to(device, dtype=torch.long)

        output = model(ids, mask, labels=targets)
        loss, logits = output[:2]
        logits = logits.detach().cpu().numpy()
        label_ids = targets.to("cpu").numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)
        accuracy = flat_accuracy(logits, label_ids)
        eval_loss += loss.mean().item()
        eval_accuracy += accuracy
        nb_eval_examples += ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    print(f"Validation loss: {format(eval_loss)}")
    print(f"Validation Accuracy: {format(eval_accuracy/nb_eval_steps)}")


def valid(model, testing_loader):
    model.eval()
    eval_loss = 0
    eval_accuracy = 0
    n_correct = 0
    n_wrong = 0
    total = 0
    predictions, true_labels = [], []
    nb_eval_steps, nb_eval_examples = 0, 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            targets = data["tags"].to(device, dtype=torch.long)

            output = model(ids, mask, labels=targets)
            loss, logits = output[:2]
            logits = logits.detach().cpu().numpy()
            label_ids = targets.to("cpu").numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)
            accuracy = flat_accuracy(logits, label_ids)
            eval_loss += loss.mean().item()
            eval_accuracy += accuracy
            nb_eval_examples += ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss / nb_eval_steps
        print(f"Validation loss: {format(eval_loss)}")
        print(f"Validation Accuracy: {format(eval_accuracy/nb_eval_steps)}")
        y_pred = []
        for p in predictions:
            s = []
            for p_i in p:
                s.append(tags_vals[p_i])
            y_pred.append(s)

        y_true = []
        for l in true_labels:
            for l_i in l:
                s = []
                for l_ii in l_i:
                    s.append(tags_vals[l_ii])
                y_true.append(s)

        # pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
        # valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
        print(f"F1-Score: {f1_score(y_true, y_pred)}")
