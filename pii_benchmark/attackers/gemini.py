from pii_benchmark.attackers.attacker import ProfileGuessesPUMS
from pii_benchmark.prompts import get_staab_prompt
from pii_benchmark.utils import retry_with_backoff
from typing import List
from google import genai
from pii_benchmark.credentials import gemini_api_key


class GeminiAttacker:
    def __init__(self, model_version: str) -> None:
        self.model_version = model_version
        self.client = genai.Client(api_key=gemini_api_key)

    def infer(
        self,
        text: str,
        attributes: List[str] = None,
        scenario: str = "reddit",
        language: str = "English",
    ) -> str:
        prompt = get_staab_prompt(attributes=attributes, text=text, scenario=scenario, language=language)

        response = retry_with_backoff(
            self.client.models.generate_content,
            model=f"gemini-{self.model_version}",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": ProfileGuessesPUMS,
            },
        )
        return response.text, prompt

    async def infer_async(
        self, text: str, attributes: List[str] = None, scenario: str = "reddit", language: str = "English"
    ) -> str:
        prompt = get_staab_prompt(attributes=attributes, text=text, scenario=scenario, language=language)

        response = await self.client.models.generate_content(
            model=f"gemini-{self.model_version}", contents=prompt
        )

        return response.text, prompt