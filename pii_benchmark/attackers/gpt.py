from typing import List
from openai import OpenAI
from pii_benchmark.credentials import openai_api_key
from pii_benchmark.prompts import get_staab_prompt
from pii_benchmark.utils import parse_output_gpt
import random, time

from openai import OpenAI

MAX_DELAY = 60
BASE_DELAY = 2
MAX_RETRIES = 10

class GPTAttacker:
    def __init__(self, model_version: str = "gpt-4.1", ):
        self.model_version = model_version
        self.client = OpenAI(api_key=openai_api_key)

    def infer(
        self, text: str, attributes: List[str] = None, scenario: str = "medical"
    ):
        
        prompt = get_staab_prompt(attributes=attributes, text=text, scenario=scenario)

        chat = [{
            "role": "system",
            "content": "You are an AI Assistant that specializes in generating synthetic data. Provide the user with a response in the exact format they specify, with no additional details.",
            },
            {"role": "user", "content": prompt},
        ]

        response = self.client.responses.create(
                model=self.model_version,
                input=chat,
                max_output_tokens=4096,
        )
        model_guesses = response.output_text
        model_guesses = parse_output_gpt(model_guesses)
        print()
        return model_guesses, prompt