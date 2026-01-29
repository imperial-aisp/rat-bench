from typing import List
from openai import OpenAI

from pii_benchmark.anonymizers.anonymizer import Anonymizer
from pii_benchmark.credentials import openai_api_key

def prompt_template_fn(review):
    prompt = f"Review: {review}\nParaphrase of the review:"
    return prompt

class DPPromptAnonymizer(Anonymizer):
    def __init__(self, model_version: str="gpt-4o-mini", temperature: float=0.7):
        super().__init__()
        self.model_version = model_version
        self.client = OpenAI(api_key=openai_api_key)
        self.temperature = temperature

    def anonymize(self, text: str) -> str:
        
        print("Original text")
        print(text)
        
        prompt = prompt_template_fn(review=text)
        
        response = self.client.chat.completions.create(
            model=self.model_version,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=400
        )
        anon_text = response.choices[0].message.content
        
        print("Anonymized text")
        print(anon_text)
        
        print('---')
        
        return anon_text