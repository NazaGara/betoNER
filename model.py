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
    TensorDataset,
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
EPOCHS = 5
WEIGHT_DECAY = 0.0
LEARNING_RATE = 2e-05

from collections import namedtuple
from torch.optim.lr_scheduler import LambdaLR


tokenizer = AutoTokenizer.from_pretrained(
    "dccuchile/bert-base-spanish-wwm-cased", num_labels=len(LABEL_LIST)
)
model = BertForTokenClassification.from_pretrained(
    "dccuchile/bert-base-spanish-wwm-cased", num_labels=len(LABEL_LIST)
)
Example = namedtuple("Example", ["tokens", "labels", "prediction_mask"])


def read_conllu(file_name: str):
    f = open(f"{file_name}.conllu", "r", encoding="utf-8")
    token_lists = list(parse_incr(f, fields=["id", "word", "tag"]))
    f.close()
    # words, words_tags = [], []
    tokens, labels = [], []
    for i, tl in tqdm(enumerate(token_lists), desc="parsing input"):
        for j, _ in enumerate(tl):

            id = token_lists[i][j]
            # words.append(id['word'])
            # words_tags.append(id['tag'])

            tokenized = tokenizer.tokenize(id["word"])
            tokens += tokenized
            labels += [id["tag"]] + [INWORD_PAD_LABEL] * (len(tokenized) - 1)

    return tokens, labels


def generate_examples(tokens: list, labels: list):
    half_len = MAX_LEN // 2
    examples = []
    for i in tqdm(range(0, len(tokens) - half_len, half_len), desc="Creating examples"):
        token_window = tokens[i : i + 2 * half_len]
        label_window = labels[i : i + 2 * half_len]

        if i == 0:
            prediction_mask = [1] * 2 * half_len
        else:
            prediction_mask = [0] * half_len + [1] * (len(token_window) - half_len)

        assert len(token_window) == len(label_window) == len(prediction_mask)
        example = Example(token_window, label_window, prediction_mask)
        examples.append(example)
    return examples


def create_dataset(examples: Example):
    input_ids, labels, prediction_masks = [], [], []
    for i, (tokens, ex_labels, prediction_mask) in enumerate(
        tqdm(examples, desc="Converting examples")
    ):
        ids = tokenizer.convert_tokens_to_ids(tokens)
        label_ids = [LABEL_MAP[l] for l in ex_labels]

        # the last examples should be padded
        if i == len(examples) - 1:
            pad_length = MAX_LEN - len(tokens)
            ids += [tokenizer.pad_token_id] * pad_length
            label_ids += [IGNORE_INDEX] * pad_length
            prediction_mask += [0] * pad_length

        input_ids.append(ids)
        labels.append(label_ids)
        prediction_masks.append(prediction_mask)

    # Verify that everything is the same length
    for list_ in (input_ids, labels, prediction_masks):
        for item in list_:
            assert len(item) == MAX_LEN

    dataset = TensorDataset(
        torch.tensor(input_ids),
        torch.tensor(labels),
        torch.tensor(prediction_masks),
    )
    return dataset


from torch.optim import AdamW  # Aca cambie el Adam, por AdamW


def pre_train(dataset: TensorDataset):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    total_steps = len(dataloader) * 3

    no_decay = ["bias", "LayerNorm.weight"]
    grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": WEIGHT_DECAY,
        },
    ]
    optimizer = AdamW(grouped_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(50 * total_steps),  # warmup is a %
        num_training_steps=total_steps,
    )
    return dataloader, optimizer, scheduler


def train(dataloader: DataLoader, optimizer: AdamW, scheduler: LambdaLR):
    global_step, tr_loss, running_loss = 0, 0.0, 0.0
    model.to(device)
    model.train()
    for _ in tqdm(range(EPOCHS), desc="Epoch"):
        for step, batch in enumerate(dataloader):
            batch = [t.to(device) for t in batch]

            model.zero_grad()
            optimizer.zero_grad()

            # loss = model(batch[0], batch[2], labels = batch[1])[0]
            loss = model(
                input_ids=batch[0],
                # notar que aca no le pasa la mask
                labels=batch[1],
            )[0]
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            running_loss += loss.item()
            global_step += 1

            scheduler.step()
        # scheduler.step()

    print(global_step, tr_loss / global_step)


def main():
    train_tokens, train_labels = read_conllu("train")
    test_tokens, test_labels = read_conllu("test")

    train_examples = generate_examples(train_tokens, train_labels)
    test_examples = generate_examples(test_tokens, test_labels)

    frac = len(train_examples) // 10
    train_examples = train_examples[:frac]

    train_dataset = create_dataset(train_examples)
    test_dataset = create_dataset(test_examples)

    train_dl, optimizer, scheduler = pre_train(train_dataset)

    train(train_dl, optimizer, scheduler)

    # evaluation
    model.eval()
    eval_loss = 0
    eval_accuracy = 0
    n_correct = 0
    n_wrong = 0
    total = 0
    predictions, true_labels = [], []
    nb_eval_steps, nb_eval_examples = 0, 0
    with torch.no_grad():
        for batch in tqdm(train_dl, desc="Evaluation"):
            input_ids, labels, prediction_mask = [t.to(device) for t in batch]

            output = model(input_ids, prediction_mask, labels=labels)
            loss, logits = output[:2]
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to("cpu").numpy()

    predictions = logits.argmax(-1)
    print(predictions.shape)
    print(predictions[0])
    print(predictions[1])
    print(predictions[2])


if __name__ == "__main__":
    main()
