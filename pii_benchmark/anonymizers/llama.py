import json
from typing import List
from tqdm import tqdm
from transformers import pipeline

from pii_benchmark.anonymizers.anonymizer import Anonymizer
from pii_benchmark.prompts import get_anonymization_prompt


class LlamaAnonymizer(Anonymizer):
    def __init__(self, prompt_type: str, attributes: List[str], model_version: str="3.1-8B-Instruct", scenario:str="medical"):
        super().__init__()
        self.prompt_type = prompt_type
        self.attributes = attributes
        self.model_version = model_version
        self.model = pipeline("text-generation", model=f"meta-llama/Llama-{model_version}")
        self.scenario = scenario

    def anonymize(
        self, text: str, prompt_type: str = None, attributes: List[str] = None
    ) -> str:
        if prompt_type == None:
            pt = self.prompt_type
        elif prompt_type == "rescriber":
            return self.anonymize_rescriber(text)
        elif prompt_type=="clio":
            return self.anonymize_clio(text)
        else:
            pt = prompt_type
        if attributes == None:
            atts = self.attributes
        else:
            atts = attributes
        prompt = get_anonymization_prompt(
            pt, text, atts, instruct_template=True
        )

        chat = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]
        response = self.model(chat, max_new_tokens=4096)
        anon_text = ""
        for r in response[0]["generated_text"]:
            if r["role"] == "assistant":
                anon_text = r["content"]
        return anon_text
    
    def anonymize_rescriber(self, text: str) -> str:
        redacted_text = text
        entities = []

        prompt = get_anonymization_prompt(
            method="rescriber", text=text, instruct_template=True
        )

        chat = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]

        response = self.model(chat, max_new_tokens=4096)
        for r in response[0]["generated_text"]:
            if r["role"] == "assistant":
                response = r["content"]
        entities = self.parse_results(response)

        for e in tqdm(entities):
            entity_text = e["text"]
            redacted_text = redacted_text.replace(entity_text, "*" * len(entity_text))
            # while entity_text in redacted_text:
            #     start = redacted_text.find(entity_text)
            #     end = start + len(entity_text)
            #     redacted_text = (
            #         redacted_text[:start] + ("*" * len(entity_text)) + redacted_text[end:]
            #     )

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
                    if len(lines)> i+1 and "text" in lines[i+1]:
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
    
    def anonymize_clio(
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
