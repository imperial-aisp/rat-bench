from typing import List

from pii_benchmark.anonymizers.anonymizer import Anonymizer
from pii_benchmark.prompts import get_anonymization_prompt
from pii_benchmark.credentials import gemini_api_key

from google import genai

import time
import random

MAX_DELAY = 60
BASE_DELAY = 2
MAX_RETRIES = 10


class GeminiAnonymizer(Anonymizer):
    def __init__(
        self,
        model_version: str,
        attributes: List[str] = None,
        prompt_type: str = "anthropic",
    ):
        """
        prompt_type defines the anonymization prompt used. It can be the generic anthropic prompt, the modified anthropic prompt that includes attribute types,
        the Clio prompt, or the Rescriber prompt.
        """
        super().__init__()
        self.prompt_type = prompt_type
        self.model_version = model_version
        self.attributes = attributes

    def anonymize(self, text: str) -> str:
        client = genai.Client(api_key=gemini_api_key)

        prompt = get_anonymization_prompt(
            self.prompt_type, text, self.attributes
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=f"gemini-{self.model_version}", contents=prompt + text
                )

                return response.text
            except Exception as e:
                if "503" in str(e) or "overloaded" in str(e).lower():
                    wait = min(MAX_DELAY, BASE_DELAY * (2 ** (attempt - 1)))
                    # Add jitter to avoid stampeding the server
                    wait += random.uniform(0, 1)
                    print(
                        f"[Retry {attempt}/{MAX_RETRIES}] {str(e)}. Waiting {wait:.1f}s..."
                    )
                    time.sleep(wait)
                else:
                    # ❌ If it's some other error, stop trying
                    raise
