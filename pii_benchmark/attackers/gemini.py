from pii_benchmark.attackers.attacker import ProfileGuessesPUMS
from pii_benchmark.prompts import get_staab_prompt
from typing import List
from google import genai
from pii_benchmark.credentials import gemini_api_key
import time, random

MAX_DELAY = 60
BASE_DELAY = 2
MAX_RETRIES = 10
N_PROCESSORS = 10
RPM_LIMIT = 10
DELAY = 60 / RPM_LIMIT

class GeminiAttacker:
    def __init__(self, model_version: str) -> None:
        self.model_version = model_version

    def infer(
        self,
        text: str,
        attributes: List[str] = None,
        task: str = "reddit",
        i: int = 0,
    ) -> str:
        client = genai.Client(api_key=gemini_api_key)

        prompt = get_staab_prompt(attributes=attributes, text=text, scenario=task)

        response = None

        while response is None or response.text is None:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    response = client.models.generate_content(
                        model=f"gemini-{self.model_version}",
                        contents=prompt,
                        config={
                            "response_mime_type": "application/json",
                            "response_schema": ProfileGuessesPUMS,
                        },
                    )
                except Exception as e:
                    wait = 0
                    if "503" in str(e) or "overloaded" in str(e).lower():
                        wait = min(MAX_DELAY, BASE_DELAY * (2 ** (attempt - 1)))
                        # Add jitter to avoid stampeding the server
                        wait += random.uniform(0, 1)
                        print(
                            f"[Retry {attempt}/{MAX_RETRIES}] Model overloaded. Waiting {wait:.1f}s..."
                        )
                        time.sleep(wait)
                    else:
                        # ❌ If it's some other error, stop trying
                        print(f"exception occured: {str(e)}")
                        wait += random.uniform(0, 1)
                        print(
                            f"[Retry {attempt}/{MAX_RETRIES}] Model overloaded. Waiting {wait:.1f}s..."
                        )
                        time.sleep(wait)

        return response.text

    async def infer_async(
        self, text: str, attributes: List[str] = None, task: str = "reddit"
    ) -> str:
        client = genai.Client(api_key=gemini_api_key)

        prompt = get_staab_prompt(attributes=attributes, text=text, scenario=task)

        response = await client.models.generate_content(
            model=f"gemini-{self.model_version}", contents=prompt
        )

        return response.text