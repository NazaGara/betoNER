# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import datasets


_CITATION = """\
@inproceedings{,
    title = "",
    author = "Garagiola, Nazareno",
    year = "2022",
    url = ""
}
"""

_DESCRIPTION = """Dataset used to train a NER model"""

_URL = "https://raw.githubusercontent.com/NazaGara/betoNER/main/data/wikiner/"
_TRAINING_FILE = "train.conllu"


class WikinerConfig(datasets.BuilderConfig):
    """BuilderConfig"""

    def __init__(self, **kwargs):
        """BuilderConfig
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(WikinerConfig, self).__init__(**kwargs)


class Wikiner(datasets.GeneratorBasedBuilder):
    """Wikiner dataset."""

    BUILDER_CONFIGS = [
        WikinerConfig(
            name="wikiner",
            version=datasets.Version("1.0.0"),
            description="wikiner dataset",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "pos_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "ACRNM",
                                "ADJ",
                                "ADV",
                                "ALFS",
                                "ART",
                                "BACKSLASH",
                                "CARD",
                                "CC",
                                "CCAD",
                                "CCNEG",
                                "CM",
                                "CODE",
                                "COLON",
                                "CQUE",
                                "CSUBF",
                                "CSUBI",
                                "CSUBX",
                                "DM",
                                "DOTS",
                                "FS",
                                "INT",
                                "LP",
                                "NC",
                                "NEG",
                                "NMEA",
                                "NMON",
                                "NP",
                                "ORD",
                                "PAL",
                                "PDEL",
                                "PE",
                                "PERCT",
                                "PPC",
                                "PPO",
                                "PPX",
                                "PREP",
                                "QT",
                                "QU",
                                "REL",
                                "RP",
                                "SE",
                                "SEMICOLON",
                                "SLASH",
                                "SYM",
                                "UMMX",
                                "VCLIfin",
                                "VCLIger",
                                "VCLIinf",
                                "VEadj",
                                "VEfin",
                                "VEger",
                                "VEinf",
                                "VHadj",
                                "VHfin",
                                "VHger",
                                "VHinf",
                                "VLadj",
                                "VLfin",
                                "VLger",
                                "VLinf",
                                "VMadj",
                                "VMfin",
                                "VMger",
                                "VMinf",
                                "VSadj",
                                "VSfin",
                                "VSger",
                                "VSinf",
                            ]
                        )
                    ),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
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
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}{_TRAINING_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
        ]

    def _generate_examples(self, filepath):
        logging.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens, pos_tags, ner_tags = [], [], []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "pos_tags": pos_tags,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        pos_tags = []
                        ner_tags = []
                else:
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    pos_tags.append(splits[1])
                    ner_tags.append(splits[2].rstrip())
            # last example
            if tokens:
                yield guid, {
                    "id": str(guid),
                    "tokens": tokens,
                    "pos_tags": pos_tags,
                    "ner_tags": ner_tags,
                }
