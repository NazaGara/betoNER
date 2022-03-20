# classes.py


class SentenceGetter(object):
    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [
            (w, t)
            for w, t in zip(
                s["word"].values.tolist(), s["labels"].values.tolist()
            )
        ]
        self.grouped = self.dataset.groupby("sentence_id").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, tokenizer, sentences, labels, max_len):
        self.len = len(sentences)
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        sentence = str(self.sentences[index])
        # esto del encode_plus medio que jode a veces, no se porque, puedo probar con usar Tokenizer de una
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            # pad_to_max_length=True,
            truncation=True,
            padding="max_length",  # esto deberia de arreglar el warning de la linea de arriba
            return_token_type_ids=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        label = self.labels[index]
        label.extend([10] * 512)
        label = label[:512]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "tags": torch.tensor(label, dtype=torch.long),
        }

    def __len__(self):
        return self.len


from transformers import (
    BertForTokenClassification,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertForTokenClassification,
    BertTokenizer,
    BertConfig,
    BertModel,
)


class BERTClass(torch.nn.Module):
    def __init__(self, amount_tags):
        super(BERTClass, self).__init__()
        # for token classification or for masked LM?
        self.l1 = AutoModelForMaskedLM.from_pretrained(
            "dccuchile/bert-base-spanish-wwm-cased", num_labels=amount_tags
        )
        # self.l1 = BertForTokenClassification.from_pretrained('dccuchile/bert-base-spanish-wwm-cased', num_labels=amount_tags)

        # self.l2 = torch.nn.Dropout(0.3)
        # self.l3 = torch.nn.Linear(768, 200)

    def forward(self, ids, mask, labels):
        output_1 = self.l1(ids, mask, labels=labels)
        # output_2 = self.l2(output_1[0])
        # output = self.l3(output_2)
        return output_1
