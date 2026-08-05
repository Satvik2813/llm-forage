import os

from dotenv import load_dotenv
from google import genai

from forge.config import ModelConfig


class LLMClient:
    def __init__(self, config: ModelConfig, api_key: str | None = None):
        self.config = config
        self.last_response = ""

        load_dotenv()

        api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "Gemini API key not found. "
                "Pass api_key or set GEMINI_API_KEY in .env"
            )

        self.client = genai.Client(api_key=api_key)

    def chat(self, message: str) -> str:
        response = self.client.models.generate_content(
            model=self.config.model,
            contents=message,
            config=self.config.to_gemini_config()
        )

        if not response.text:
            raise RuntimeError(
                "Gemini returned an empty response. "
                "The response may have been blocked by a safety filter."
            )

        return response.text

    def stream(self, message: str):
        chunks = []

        response = self.client.models.generate_content_stream(
            model=self.config.model,
            contents=message,
            config=self.config.to_gemini_config()
        )

        for chunk in response:
            if chunk.text is not None:
                chunks.append(chunk.text)
                yield chunk.text

        self.last_response = "".join(chunks)