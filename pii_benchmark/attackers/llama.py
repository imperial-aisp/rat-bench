from transformers import pipeline
from typing import List

from pii_benchmark.prompts import get_staab_prompt_llama
from pii_benchmark.utils import parse_output

class LlamaAttacker:
    def __init__(
        self, model_version: str = "Llama-3.1-8B-Instruct"
    ) -> None:
        self.model_version = model_version
        self.model = pipeline("text-generation", model=f"meta-llama/{self.model_version}")

    def infer(
        self, text: str, attributes: List[str] = None, scenario: str = "reddit"
    ):
        system_prompt, prompt = get_staab_prompt_llama(
            attributes=attributes, text=text, scenario=scenario
        )
        
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = self.model(chat, max_new_tokens=2048)
        model_guesses = ""
        for r in response[0]["generated_text"]:
            if r["role"] == "assistant":
                model_guesses = r["content"]
        model_guesses = parse_output(model_guesses)
        return model_guesses