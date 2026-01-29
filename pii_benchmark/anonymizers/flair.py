from pii_benchmark.anonymizers.anonymizer import Anonymizer
from flair.data import Sentence
from flair.nn import Classifier

entities_to_remove = [
    "PERSON",
    "CARDINAL",
    "EVENT",
    "FAC",
    "GPE",
    "LOC",
    "NORP",
    "ORDINAL",
    "ORG",
    "TIME",
]


class FlairAnonymizer(Anonymizer):
    def __init__(self, attributes=None):
        super().__init__()
        self.tagger = Classifier.load("ner-ontonotes-large")
        if attributes is None:
            self.attributes = attributes
        else:
            self.attributes = entities_to_remove

    def anonymize(self, text: str) -> str:
        sentence = Sentence(text)

        # run NER over sentence
        self.tagger.predict(sentence)
        predictions = sentence.get_labels()

        redacted_text = text

        for entity in predictions:
            if (entity.value in entities_to_remove) and (entity.score > 0.4):
                start = entity.data_point.start_position
                end = entity.data_point.end_position
                redacted_text = (
                    redacted_text[:start]
                    + ("*" * entity.length)
                    + redacted_text[end:]
                )
