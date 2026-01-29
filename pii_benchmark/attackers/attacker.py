from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel

class Attribute(BaseModel):
    # type: str
    Guess: str
    Inference: str
    Certainty: str


class ProfileGuesses(BaseModel):
    age: Attribute
    sex: Attribute
    race: Attribute


class ProfileGuessesPUMS(BaseModel):
    AGE: Attribute
    RACE: Attribute
    CIT: Attribute
    SCHL: Attribute
    POWSP: Attribute
    JWDP: Attribute
    OCCP: Attribute
    COW: Attribute
    MAR: Attribute
    MIL: Attribute
    SEX: Attribute
    JWTRNS: Attribute
    PHONE_NUMBER: Attribute
    SSN: Attribute
    DOB: Attribute
    CREDIT_CARD: Attribute
    NAME: Attribute
    PERSONAL_EMAIL_ADDRESS: Attribute
    RESIDENTIAL_ADDRESS: Attribute

class Attacker(ABC):
    def __init__(self, model: str, task: str) -> None:
        self.model = model
        self.task = task
        pass

    @abstractmethod
    def infer(self, text: str, attributes: List[str], task: str) -> str:
        pass