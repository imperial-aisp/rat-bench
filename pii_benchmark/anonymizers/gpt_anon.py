import json
from typing import List
from openai import OpenAI

from pii_benchmark.anonymizers.anonymizer import Anonymizer
from pii_benchmark.prompts import get_anonymization_prompt

from pii_benchmark.credentials import openai_api_key

class GPTAnonymizer(Anonymizer):
    def __init__(self, prompt_type: str, attributes: List[str], model_version: str="gpt-4.1"):
        super().__init__()
        self.prompt_type = prompt_type
        self.attributes = attributes
        self.model_version = model_version
        self.client = OpenAI(api_key=openai_api_key)

    def anonymize(
        self, text: str, prompt_type: str = None, attributes: List[str] = None
    ) -> str:
        if prompt_type == None:
            pt = self.prompt_type
        else:
            pt = prompt_type
        if attributes == None:
            atts = self.attributes
        else:
            atts = attributes
            
        prompt = get_anonymization_prompt(
            pt, text, atts, instruct_template=False
        )
        
        if pt=="clio":
            chat = [
                {"role": "user", "content": f"{prompt[0]} {text} {prompt[1]}"}
            ]
            # print(f"prompt_0: {prompt[0]}")
            # print(f"prompt_1: {prompt[1]}")
        else:
            chat = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ]
            # print(f"prompt: {prompt}")
        
        # response = self.client.chat.completions.create(model=self.model_version,
        #                                                messages=chat, temperature=0.0, max_tokens=2000)
        response = self.client.responses.create(
                model=self.model_version,
                input=chat,
                max_output_tokens=4096,
            )
        
        if pt=="rescriber":
            redacted_text = text
            entities = parse_results_rescriber(response.output_text)
            for e in entities:
                entity_text = e["text"]
                while entity_text in redacted_text:
                    start = redacted_text.find(entity_text)
                    end = start + len(entity_text)
                    redacted_text = (
                        redacted_text[:start] + ("*" * len(entity_text)) + redacted_text[end:]
                    )
            anon_text = redacted_text
        else:
            anon_text = response.output_text
        # print("type(anon_text):")
        # print(type(anon_text))
        # print(anon_text)
        
        return anon_text
    

def parse_results_rescriber(output) -> str:
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
                if i+1<len(lines) and "text" in lines[i+1]:
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