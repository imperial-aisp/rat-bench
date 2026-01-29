import json
from typing import List
from pii_benchmark.anonymizers.textwash_config import Config
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from pii_benchmark.anonymizers.anonymizer import Anonymizer

import re
from copy import deepcopy

class TextWashAnonymizer(Anonymizer):
    def __init__(self, language: str = "en", attributes: List[str] = "", cpu: bool=True):
        super().__init__()
        self.config = Config(language=language)
        self.attributes = assert_entities(attributes, self.config.path_to_model) if attributes != "" else None
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.path_to_model)
        self.model = AutoModelForTokenClassification.from_pretrained(self.config.path_to_model)
        self.classifier = pipeline("ner", model=self.model, tokenizer=self.tokenizer, device=torch.device("cuda:0" if not cpu else "cpu"))
        self.anonymizer = TextWashBackend(self.config, self.classifier)

    def anonymize(self, text: str) -> str:
        anonymized_text = self.anonymizer.anonymize(
            text, selected_entities=self.attributes
        )
        return anonymized_text

class TextWashBackend:
    def __init__(self, config, classifier):
        self.config = config
        self.classifier = classifier

        with open(self.config.path_to_months_file, "r") as f:
            self.months = f.readlines()
            self.months = [m.replace("\n", "") for m in self.months]

        with open(self.config.path_to_written_numbers_file, "r") as f:
            self.written_numbers = f.readlines()
            self.written_numbers = [w.replace("\n", "") for w in self.written_numbers]

        self.valid_surrounding_chars = [
            ".",
            ",",
            ";",
            "!",
            ":",
            "\n",
            "’",
            "‘",
            "'",
            '"',
            "?",
            "-",
        ]

    def replace_identified_entities(self, entities, anon_input_seq, entity2generic):
        for phrase, _ in sorted(
            entities.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if len(phrase) > 1 or phrase.isalnum():
                try:
                    for char in self.valid_surrounding_chars:
                        anon_input_seq = re.sub(
                            "[^a-zA-Z0-9]{}[{}]".format(phrase, char),
                            " {}{}".format(entity2generic[phrase], char),
                            anon_input_seq,
                        )

                    anon_input_seq = re.sub(
                        "[^a-zA-Z0-9\n]{}[^a-zA-Z0-9\n]".format(phrase),
                        " {} ".format(entity2generic[phrase]),
                        anon_input_seq,
                    )

                    anon_input_seq = re.sub(
                        "[\n]{}".format(phrase),
                        "\n{}".format(entity2generic[phrase]),
                        anon_input_seq,
                    )

                    anon_input_seq = re.sub(
                        "{}".format(phrase),
                        "{}".format(entity2generic[phrase]),
                        anon_input_seq,
                    )
                except re.error:
                    anon_input_seq = anon_input_seq.replace(
                        "{}".format(phrase), "{}".format(entity2generic[phrase])
                    )

        return anon_input_seq

    def get_identifiable_tokens(self, text_input):
        predictions = decode_outputs(
            self.classifier(text_input), model_type=self.config.model_type
        )

        entities = {
            p["word"]: p["entity"]
            for p in predictions
            if p["entity"] != "NONE" and len(p["word"]) > 1 and p["word"].isalnum()
        }

        return entities

    def get_entity_type_mapping(self, entities):
        entity2generic_c = {v: 1 for _, v in entities.items()}
        entity2generic = {}

        for phrase, entity_type in entities.items():
            entity2generic[phrase] = "{}_{}".format(
                entity_type, entity2generic_c[entity_type]
            )

            entity2generic_c[entity_type] += 1

        return entity2generic

    def replace_numerics(self, anon_input_seq):
        # https://pythonexamples.org/python-regex-extract-find-all-the-numbers-in-string/
        all_numeric = list(set(re.findall("[0-9]+", anon_input_seq)))
        numeric_map = {k: "NUMERIC_{}".format(v + 1) for v, k in enumerate(all_numeric)}

        for k, v in sorted(numeric_map.items(), key=lambda x: int(x[0]), reverse=True):
            anon_input_seq = re.sub(
                "[^NUMERIC_0-9+]{}".format(k), " {}".format(v), anon_input_seq
            )

        return anon_input_seq

    def replace_pronouns(self, anon_input_seq):
        # https://blog.hubspot.com/marketing/gender-neutral-pronouns
        pronoun_map = {
            "he": "PRONOUN",
            "she": "PRONOUN",
            "him": "PRONOUN",
            "his": "PRONOUN",
            "her": "PRONOUN",
            "hers": "PRONOUN",
            "himself": "PRONOUN",
            "herself": "PRONOUN",
            "mr": "MR/MS",
            "mrs": "MR/MS",
            "mr.": "MR/MS",
            "mrs.": "MR/MS",
            "miss": "MR/MS",
            "ms": "MR/MS",
            "dr": "TITLE",
            "dr.": "TITLE",
            "prof": "TITLE",
            "prof.": "TITLE",
            "sir": "TITLE",
            "dame": "TITLE",
            "madam": "TITLE",
            "lady": "TITLE",
            "lord": "TITLE",
        }

        for k, v in pronoun_map.items():
            if anon_input_seq.startswith("{} ".format(k)):
                anon_input_seq = anon_input_seq.replace(
                    "{} ".format(k), "{} ".format(v), 1
                )

            if anon_input_seq.startswith("{} ".format(k.capitalize())):
                anon_input_seq = anon_input_seq.replace(
                    "{} ".format(k.capitalize()), "{} ".format(v), 1
                )

            for char in self.valid_surrounding_chars:
                anon_input_seq = re.sub(
                    "[^a-zA-Z0-9]{}[{}]".format(k, char),
                    " {}{}".format(v, char),
                    anon_input_seq,
                )
                anon_input_seq = re.sub(
                    "[^a-zA-Z0-9]{}[{}]".format(k.capitalize(), char),
                    " {}{}".format(v, char),
                    anon_input_seq,
                )

            anon_input_seq = re.sub(
                "[^a-zA-Z0-9]{}[^a-zA-Z0-9]".format(k),
                " {} ".format(v),
                anon_input_seq,
            )
            anon_input_seq = re.sub(
                "[^a-zA-Z0-9]{}[^a-zA-Z0-9]".format(k.capitalize()),
                " {} ".format(v),
                anon_input_seq,
            )

        return anon_input_seq

    def replace_numbers_and_months(self, anon_input_seq):
        entity2generic_c = {"DATE": 1, "NUMERIC": 1}
        entity2generic = {}

        spl = re.split("[ ,.-]", anon_input_seq)

        for word in spl:
            if word.lower() in self.written_numbers:
                try:
                    _ = entity2generic[word]
                except KeyError:
                    entity2generic[word] = "{}_{}".format(
                        "NUMERIC", entity2generic_c["NUMERIC"]
                    )
                    entity2generic_c["NUMERIC"] += 1

        for word in spl:
            if word.lower() in self.months:
                try:
                    _ = entity2generic[word]
                except KeyError:
                    entity2generic[word] = "{}_{}".format(
                        "DATE", entity2generic_c["DATE"]
                    )
                    entity2generic_c["DATE"] += 1

        for phrase, replacement in sorted(
            entity2generic.items(), key=lambda x: len(x[0]), reverse=True
        ):
            for char in self.valid_surrounding_chars:
                anon_input_seq = re.sub(
                    "[^a-zA-Z0-9]{}[{}]".format(phrase, char),
                    " {}{}".format(replacement, char),
                    anon_input_seq,
                )

            anon_input_seq = re.sub(
                "[^a-zA-Z0-9]{}[^a-zA-Z0-9]".format(phrase),
                " {} ".format(replacement),
                anon_input_seq,
            )

        return anon_input_seq

    def anonymize(self, input_seq, selected_entities=None):
        orig_input_seq = deepcopy(input_seq)

        entities = self.get_identifiable_tokens(deepcopy(input_seq))

        # Filter entities if necessary
        if selected_entities:
            filtered_entities = []

            for entity in entities:
                if entity[1] in selected_entities:
                    filtered_entities.append(entity)

            entities = filtered_entities

        entity2generic = self.get_entity_type_mapping(entities)

        anon_input_seq = re.sub("https*://\S+", "URL", orig_input_seq)

        anon_input_seq = self.replace_identified_entities(
            entities, orig_input_seq, entity2generic
        )

        anon_input_seq = self.replace_numerics(anon_input_seq)

        anon_input_seq = self.replace_pronouns(anon_input_seq)

        anon_input_seq = self.replace_numbers_and_months(anon_input_seq)

        return " ".join([x.strip() for x in anon_input_seq.split()])
    

def assert_entities(entities, model_path):
    with open(f"{model_path}/config.json", "r") as f:
        label_map = json.load(f)["id2label"]
    available_entities = list(label_map.keys()) + ["NUMERIC", "PRONOUN"]
    available_entities = sorted(
        list(set(available_entities).difference({"NONE", "PAD"}))
    )

    entity_list = [e.strip() for e in entities.split(",")]

    for entity in entity_list:
        if entity not in available_entities:
            raise ValueError(
                "Incorrect argument --entities provided. Please ensure that all values refer to existing entities separated by comma.\n"
                "Available entities are {}.".format(", ".join(available_entities))
            )

    return entity_list

def decode_outputs(predicted_labels, model_type="bert"):
    entities = []
    shift_idx = 2 if model_type == "bert" else 0

    for _, elem in enumerate(predicted_labels):
        attach = False

        if model_type == "bert" and elem["word"].startswith("##"):
            attach = True

        elif model_type == "roberta" and not elem["word"].startswith("Ġ"):
            attach = True

        if attach:
            entities[-1]["word"] += elem["word"][shift_idx:]
            entities[-1]["end"] = elem["end"]
        else:
            entities.append(
                {
                    "word": elem["word"],
                    "start": elem["start"],
                    "end": elem["end"],
                    "entity": elem["entity"],
                }
            )

    if model_type == "roberta":
        for elem in entities:
            elem["word"] = elem["word"][1:]

    return entities