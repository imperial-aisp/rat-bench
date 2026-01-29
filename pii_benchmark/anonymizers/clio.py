from typing import List
import anthropic
from pii_benchmark.anonymizers.anonymizer import Anonymizer
from pii_benchmark.prompts import get_system_prompt

from credentials import anthropic_api_key


class ClioAnonymizer(Anonymizer):
    def __init__(
        self,
        model_version: str = "claude-opus-4-1-20250805",
        attributes: List[str] = None,
    ):
        super().__init__()
        self.model_version = model_version
        self.attributes = attributes

    def anonymize(self, text: str) -> str:
        base_prompt = get_system_prompt(self.base_prompt)

        client = anthropic.Anthropic(api_key=anthropic_api_key)
        message = client.messages.create(
            model=self.model_version,
            max_tokens=1000,
            temperature=0,
            system=base_prompt,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": text}]}
            ],
        )

        return message.content
