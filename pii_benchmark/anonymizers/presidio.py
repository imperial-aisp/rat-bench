from typing import List
from presidio_analyzer import AnalyzerEngine, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_analyzer.nlp_engine import SpacyNlpEngine

from pii_benchmark.anonymizers.anonymizer import Anonymizer
import spacy

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

class LoadedSpacyNlpEngine(SpacyNlpEngine):
    def __init__(self, loaded_spacy_model):
        super().__init__()
        self.nlp = {"en": loaded_spacy_model}

class PresidioAnonymizer(Anonymizer):
    def __init__(self, language: str = "en", attributes: List[str] = None) -> None:
        super().__init__()

        nlp = spacy.load("en_core_web_sm")
        loaded_nlp_engine = LoadedSpacyNlpEngine(loaded_spacy_model = nlp)
        self.analyzer = AnalyzerEngine(nlp_engine = loaded_nlp_engine)

        self.language = language
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        self.analyzer.registry.add_recognizer(titles_recognizer)
        self.analyzer.registry.add_recognizer(sex_recognizer)

    def anonymize(self, text: str) -> str:
        # Analyze
        analyzer_results = self.analyzer.analyze(
                text=text, language=self.language, entities=[
                    "CREDIT_CARD", "DATE_TIME", "EMAIL_ADDRESS", "NRP", "LOCATION",
                    "PERSON", "PHONE_NUMBER", "US_SSN"
                ]
            )
        
        # Anonymize
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
