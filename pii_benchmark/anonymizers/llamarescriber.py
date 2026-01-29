import json
from transformers import pipeline
from pii_benchmark.anonymizers.anonymizer import Anonymizer
from typing import List

from pii_benchmark.prompts import get_anonymization_prompt


class LlamaRescriberAnonymizer(Anonymizer):
    def __init__(self, prompt_type: str, attributes: List[str], model_version: str="3.1-8B-Instruct", scenario: str = "medical"):
        super().__init__()
        self.prompt_type = prompt_type
        self.attributes = attributes
        self.model_version = model_version
        self.model = pipeline("text-generation", model=f"meta-llama/Llama-{model_version}")
        self.scenario = scenario

    def anonymize(self, text: str) -> str:
        redacted_text = text
        entities = []

        prompt = get_anonymization_prompt(
            method="rescriber", text=text, instruct_template=True
        )

        chat = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]
        print("RESCRIBER PROMPT")
        print(prompt)
        print("TEXT")
        print(text)

        response = self.model(chat, max_new_tokens=4096)
        for r in response[0]["generated_text"]:
            if r["role"] == "assistant":
                response = r["content"]
        print("RESPONSE")
        print(response)
        entities = self.parse_results(response)
        print("ENTITIES")
        print(entities)

        for e in entities:
            entity_text = e["text"]
            while entity_text in redacted_text:
                start = redacted_text.find(entity_text)
                end = start + len(entity_text)
                redacted_text = (
                    redacted_text[:start] + ("*" * len(entity_text)) + redacted_text[end:]
                )

        return redacted_text
    
    def parse_results(self, output) -> str:
        lines = output.splitlines()

        entities = []

        for l in lines:
            line = l.strip().strip("]").strip("[").strip(",")
            if len(line)==0 or line[0] != "{":
                continue
            if "entity_type" in line and "text" in line:
                try:
                    entity = json.loads(line)
                    entities.append(entity)
                except:
                    pass
        if entities==[]:
            i = 0
            while i<len(lines):
                line = lines[i]
                if line=="{":
                    i += 1
                elif "entity_type" in line:
                    if "text" in lines[i+1]:
                        entities.append({
                            "entity_type": line.split(":")[-1].strip(",").strip("\"").strip(),
                            "text": lines[i+1].split(":")[-1].strip().strip("\"")
                        })
                    i += 2
                elif line=="}":
                    i += 1
                else:
                    i += 1


        return entities
