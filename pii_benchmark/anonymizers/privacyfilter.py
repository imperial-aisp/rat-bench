from abc import ABC, abstractmethod
from huggingface_hub import InferenceClient
from pii_benchmark.credentials import hf_token
import os
import requests
from transformers import pipeline

class PrivacyFilterAnonymizer(ABC):
    def __init__(self):
        # self.client = InferenceClient(
        #     provider="hf-inference",
        #     api_key=hf_token,
        # )
        

        self.pipe = pipeline("token-classification", model="openai/privacy-filter")
        # self.API_URL = "https://router.huggingface.co/hf-inference/models/openai/privacy-filter"
        # self.headers = {
        #     "Authorization": f"Bearer {hf_token}",
        # }
        
    def anonymize(self, text: str, scenario: str="medical") -> str:
        # result = self.client.token_classification(
        #     text,
        #     model="openai/privacy-filter",
        # )
        # payload = {
        #     "inputs": text
        # }
        # response = requests.post(self.API_URL, headers=self.headers, json=payload)
        # result = response.json()
        result = self.pipe(text)
        anon_text = text
        for r in result:
            try:
                if r["score"] < 0.7:
                    continue
                word_to_replace = text[r["start"]:r["end"]].strip()
                anon_text = anon_text.replace(word_to_replace, "*"*len(word_to_replace))
            except Exception as e:
                print(f"Error processing result: {r}, error: {e}")
                print(f"Original text: {text[:10]}")
                print(f"full result: {result}")
                continue
        return anon_text