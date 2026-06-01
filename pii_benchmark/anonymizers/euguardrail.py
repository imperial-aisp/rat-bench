from typing import List
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

from pii_benchmark.anonymizers.anonymizer import Anonymizer

class EUGuardrailAnonymizer(Anonymizer):
    def __init__(self) -> None:
        super().__init__()
        model_name = "bardsai/eu-pii-anonimization-multilang"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)

    def anonymize(self, text: str, scenario: str|None=None) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [self.model.config.id2label[p.item()] for p in predictions[0]]

        anonymized_text = text

        for token, label in zip(tokens, labels):
            if label != "O":
                # print(label, token.strip('▁'))
                anonymized_text = anonymized_text.replace(token.strip('▁'), "*"*len(token.strip('▁')))
        
        return anonymized_text