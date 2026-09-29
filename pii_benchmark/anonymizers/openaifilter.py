from typing import List
from transformers import pipeline
import subprocess

from pii_benchmark.anonymizers.anonymizer import Anonymizer
from transformers import AutoModelForTokenClassification, AutoTokenizer



class OpenAIAnonymizer(Anonymizer):
    def __init__(self, attributes: List[str] = None, scenario: str = "medical"):
        super().__init__()
        self.attributes = attributes
        # self.classifier = pipeline("token-classification", model="openai/privacy-filter")
        # self.tokenizer = AutoTokenizer.from_pretrained("openai/privacy-filter")
        # self.model = AutoModelForTokenClassification.from_pretrained("openai/privacy-filter", device_map="auto")


    def anonymize(self, text: str, scenario: str = "medical") -> str:
        temp_file = "/data/zexi/Projects/piiremoval/llm_anon_benchmark/tmp/temp_file.txt"
        with open(temp_file, "w") as f:
            f.write(text)
        # commandline = 'opf -f "' + temp_file + '"'
        output = subprocess.run(["opf", "-f", temp_file], encoding='utf-8', stdout=subprocess.PIPE)
        anonymized_text = output.stdout
        # print("ANONYMIZED TEXT\n", anonymized_text)
        # spans = self.classifier(text)
        # print(spans, text)
        # anonymized_text = text
        # for span in sorted(spans, key=lambda x: x["start"], reverse=True):
        #     placeholder = f"[{span['entity_group'].upper()}]"
        #     anonymized_text = anonymized_text[: span["start"]] + placeholder + anonymized_text[span["end"]:]
        # print(anonymized_text)


        return anonymized_text
