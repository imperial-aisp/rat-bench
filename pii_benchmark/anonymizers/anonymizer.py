from abc import ABC, abstractmethod


class Anonymizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def anonymize(self, text: str, scenario: str="medical") -> str:
        pass
