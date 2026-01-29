from typing import List

import scrubadub

from pii_benchmark.anonymizers.anonymizer import Anonymizer
import nltk

nltk.download("punkt_tab")


class ScrubadubAnonymizer(Anonymizer):
    def __init__(self, locale: str = "en_US", attributes: List[str] = None):
        super().__init__()
        self.locale = locale
        self.scrubber = scrubadub.Scrubber(locale=self.locale)

        self.scrubber.add_detector(scrubadub.detectors.DateOfBirthDetector)
        self.scrubber.add_detector(scrubadub.detectors.TextBlobNameDetector)

    def anonymize(self, text: str) -> str:
        anonymized_text = self.scrubber.clean(text)
        return anonymized_text
