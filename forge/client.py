import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors

from forge.config import ModelConfig
from forge.conversation import Conversation
from forge.metrics import UsageTracker


class LLMClient:
    RETRYABLE_CODES = {429, 500, 503}

    def __init__(
        self,
        config: ModelConfig,
        api_key: str | None = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ):
        self.config = config
        self.last_response = ""

        self.max_retries = max_retries
        self.base_delay = base_delay

        # Usage tracking
        self.usage = UsageTracker()

        load_dotenv()

        api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "Gemini API key not found. "
                "Pass api_key or set GEMINI_API_KEY in .env"
            )

        self.client = genai.Client(api_key=api_key)

    def chat(
        self,
        message: str,
        conversation: Conversation | None = None,
    ) -> str:

        # Stateless mode
        if conversation is None:
            contents = message

        # Conversation mode
        else:
            conversation.add_user(message)
            contents = conversation.to_contents()

        try:
            response = self._with_retry(
                self.client.models.generate_content,
                model=self.config.model,
                contents=contents,
                config=self.config.to_gemini_config(),
            )

            if not response.text:
                raise RuntimeError(
                    "Gemini returned an empty response. "
                    "The response may have been blocked by a safety filter."
                )

        except Exception:
            # API failed after user message was added.
            # Roll it back to avoid orphan user messages.
            if conversation is not None:
                conversation.remove_last()

            raise

        self.last_response = response.text

        # Commit model response to conversation.
        if conversation is not None:
            conversation.add_model(response.text)

        # Record token usage.
        usage = response.usage_metadata

        if usage is not None:
            self.usage.record(
                model=self.config.model,
                input_tokens=usage.prompt_token_count or 0,
                output_tokens=usage.candidates_token_count or 0,
            )

        return response.text

    def stream(
        self,
        message: str,
        conversation: Conversation | None = None,
    ):
        chunks = []
        started = False
        committed = False

        # Usage usually appears on the final chunk.
        usage_metadata = None

        # Stateless mode
        if conversation is None:
            contents = message

        # Conversation mode
        else:
            conversation.add_user(message)
            contents = conversation.to_contents()

        try:
            for retry in range(self.max_retries + 1):
                try:
                    response = (
                        self.client.models.generate_content_stream(
                            model=self.config.model,
                            contents=contents,
                            config=self.config.to_gemini_config(),
                        )
                    )

                    for chunk in response:
                        # Important:
                        # usage metadata may be present on a chunk
                        # that contains no text.
                        if chunk.usage_metadata is not None:
                            usage_metadata = chunk.usage_metadata

                        if chunk.text is not None:
                            started = True
                            chunks.append(chunk.text)

                            yield chunk.text

                    full_response = "".join(chunks)

                    if not full_response:
                        raise RuntimeError(
                            "Gemini returned an empty response. "
                            "The response may have been blocked "
                            "by a safety filter."
                        )

                    self.last_response = full_response

                    # Commit complete model response.
                    if conversation is not None:
                        conversation.add_model(full_response)

                    # Record usage after successful stream.
                    if usage_metadata is not None:
                        self.usage.record(
                            model=self.config.model,
                            input_tokens=(
                                usage_metadata.prompt_token_count or 0
                            ),
                            output_tokens=(
                                usage_metadata.candidates_token_count or 0
                            ),
                        )

                    committed = True

                    return

                except errors.APIError as e:
                    # Once output has started, never retry.
                    # Otherwise duplicate chunks could be yielded.
                    if started:
                        raise

                    if e.code not in self.RETRYABLE_CODES:
                        raise

                    if retry == self.max_retries:
                        raise

                    delay = self.base_delay * (2 ** retry)

                    print(
                        f"[retry {retry + 1}/{self.max_retries}] "
                        f"API error {e.code}, "
                        f"waiting {delay:.1f}s..."
                    )

                    time.sleep(delay)

        finally:
            # If the stream failed or caller closed the
            # generator before completion, remove the
            # orphan user message.
            if conversation is not None and not committed:
                conversation.remove_last()

    def _with_retry(self, func, *args, **kwargs):
        for retry in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)

            except errors.APIError as e:
                if e.code not in self.RETRYABLE_CODES:
                    raise

                if retry == self.max_retries:
                    raise

                delay = self.base_delay * (2 ** retry)

                print(
                    f"[retry {retry + 1}/{self.max_retries}] "
                    f"API error {e.code}, "
                    f"waiting {delay:.1f}s..."
                )

                time.sleep(delay)