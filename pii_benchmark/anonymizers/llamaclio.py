from typing import List
from transformers import pipeline

from pii_benchmark.anonymizers.anonymizer import Anonymizer
from pii_benchmark.prompts import get_anonymization_prompt

class LlamaClioAnonymizer(Anonymizer):
    def __init__(self, prompt_type: str, attributes: List[str], model_version: str="3.1-8B-Instruct", scenario: str = "medical"):
        super().__init__()
        self.prompt_type = prompt_type
        self.attributes = attributes
        self.model_version = model_version
        self.model = pipeline("text-generation", model=f"meta-llama/Llama-{model_version}")
        self.scenario = scenario

    def anonymize(
            self, text:str
    ):
        prompt1, prompt2 = get_anonymization_prompt(method="clio", text=text, scenario=self.scenario)
        prompt1 = prompt1 + "\n{text}</conversation>"

        chat = [
            {"role": "system", "content": prompt1},
            {"role": "user", "content": text},
        ]
        response = self.model(chat, max_new_tokens=4096, do_sample=False)
        response1 = ""
        for r in response[0]["generated_text"]:
            if r["role"] == "assistant":
                response1 = r["content"]

        chat.append({
                    "role": "assistant", "content": response1
                    }
        )
        chat.append({
            "role": "user", "content": prompt2
        })

        response = self.model(chat, max_new_tokens=4096)
        anon_text = ""
        for r in response[0]["generated_text"]:
            if r["role"] == "assistant":
                anon_text = r["content"]
    
        return anon_text