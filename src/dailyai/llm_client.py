"""Unified LLM client for chat completions with structured output support."""

from __future__ import annotations

import logging
from typing import TypeVar, cast

import litellm
from litellm.types.utils import ModelResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Call LLMs via litellm with optional structured output."""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def generate(
        self, system: str, user: str, response_format: type[BaseModel] | None = None
    ) -> str | BaseModel:
        response = cast(
            ModelResponse,
            litellm.completion(
                model=self.model,
                api_key=self.api_key,
                response_format=response_format,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            ),
        )
        content = response.choices[0].message.content or ""
        if response_format:
            return response_format.model_validate_json(content)

        return content

    def generate_structured(
        self, system: str, user: str, response_format: type[T]
    ) -> T | None:
        """Like generate(), but never raises: returns None on call failure or a
        malformed response, so callers only need to handle the fallback case."""
        try:
            result = self.generate(system, user, response_format=response_format)
        except Exception:
            logger.exception("%s call failed", response_format.__name__)
            return None

        return result if isinstance(result, response_format) else None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        response = litellm.embedding(
            model=self.model, input=texts, api_key=self.api_key
        )
        return [item["embedding"] for item in response.data]
