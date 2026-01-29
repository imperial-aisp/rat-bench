from typing import List
from presidio_analyzer import AnalyzerEngine, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import GLiNERRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import json
from pprint import pprint

from pii_benchmark.anonymizers.anonymizer import Anonymizer

nlp_engine = NlpEngineProvider(
    nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }
)

entity_mapping = {
    "person": "PERSON",
    "name": "PERSON",
    "organization": "ORGANIZATION",
    "location": "LOCATION",
    "address": "ADDRESS",
    "phone number": "PHONE_NUMBER",
    "email": "EMAIL",
    "url": "URL",
    "ip": "IP_ADDRESS",
    "date_time": "DATE_TIME",
    "age": "DATE_TIME",
}

titles_list = [
    "Sir",
    "Ma'am",
    "Madam",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Miss",
    "Dr.",
    "Professor",
]
sex_indicator_list = ["male", "female", "man", "woman"]
titles_recognizer = PatternRecognizer(
    supported_entity="TITLE", deny_list=titles_list
)
sex_recognizer = PatternRecognizer(
    supported_entity="SEX", deny_list=sex_indicator_list
)


class GlinerAnonymizer(Anonymizer):
    def __init__(self, language: str = "en", attributes: List[str] = None):
        super().__init__()
        self.language = language
        self.attributes = attributes
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        gliner_recognizer = GLiNERRecognizer(
            model_name="urchade/gliner_multi_pii-v1",
            entity_mapping=entity_mapping,
            flat_ner=False,
            multi_label=True,
            map_location="cpu",
        )

        self.analyzer.registry.add_recognizer(gliner_recognizer)
        self.analyzer.registry.remove_recognizer("SpacyRecognizer")
        self.analyzer.registry.add_recognizer(titles_recognizer)
        self.analyzer.registry.add_recognizer(sex_recognizer)

    def anonymize(self, text: str, attributes: List[str] = None) -> str:
        # Analyze

        if len(text)<1000:
            analyzer_results = self.analyzer.analyze(
                text=text,
                language="en",
                entities=[
                    "CREDIT_CARD", "DATE_TIME", "EMAIL_ADDRESS", "NRP", "LOCATION",
                    "PERSON", "PHONE_NUMBER", "US_SSN"
                ],
                allow_list=["Patient:", "Doctor:"],
            )
            anonymized_results = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators={
                    "DEFAULT": OperatorConfig(
                        "mask",
                        {
                            "type": "mask",
                            "masking_char": "*",
                            "chars_to_mask": 50,
                            "from_end": True,
                        },
                    )
                },
            )
            return anonymized_results.text
        else:
            n_chunks = len(text)//1000 + 1
            redacted_text = ""

            for chunk in range(n_chunks):
               analyzer_results = self.analyzer.analyze(
                   text=text[chunk*1000:(chunk+1)*1000],
                   language="en",
                   entities=[
                       "CREDIT_CARD", "DATE_TIME", "EMAIL_ADDRESS", "NRP", "LOCATION",
                       "PERSON", "PHONE_NUMBER", "US_SSN"
                   ],
                   allow_list=["Patient:", "Doctor:"],
               )
               anonymized_results = self.anonymizer.anonymize(
                   text=text[chunk*1000:(chunk+1)*1000],
                   analyzer_results=analyzer_results,
                   operators={
                       "DEFAULT": OperatorConfig(
                           "mask",
                           {
                               "type": "mask",
                               "masking_char": "*",
                               "chars_to_mask": 50,
                               "from_end": True,
                           },
                       )
                   },
               )
               redacted_text += anonymized_results.text
            return redacted_text
