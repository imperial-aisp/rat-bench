from typing import List
import anthropic
from pii_benchmark.anonymizers.anonymizer import Anonymizer
from pii_benchmark.prompts import get_anonymization_prompt

from pii_benchmark.credentials import anthropic_api_key


class AnthropicAnonymizer(Anonymizer):
    def __init__(
        self,
        method: str,
        base_prompt: str,
        model_version: str = "claude-opus-4-1-20250805",
        attributes: List[str]|None = None,
        language: str = "English",
    ):
        super().__init__()
        self.method = method
        self.base_prompt = base_prompt
        self.model_version = model_version
        self.attributes = attributes
        self.language = language
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)

    def anonymize(self, text: str, scenario: str = "medical") -> str:
        base_prompt = get_anonymization_prompt(self.method, text, self.attributes, scenario=scenario, language=self.language)

        message = self.client.messages.create(
            model=self.model_version,
            max_tokens=1000,
            temperature=0,
            system=base_prompt,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": text}]}
            ],
        )

        return message.content[0].text
