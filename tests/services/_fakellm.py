from typing import TypeVar

from pydantic import BaseModel

from dailyai.llm_client import LLMClient

T = TypeVar("T", bound=BaseModel)


class FakeLLMBase(LLMClient):
    """Shared generate_structured for fake LLMs, mirroring LLMClient's: never
    raises, returns None on a failed call or a result of the wrong type."""

    def __init__(self):
        pass

    def generate_structured(
        self, system: str, user: str, response_format: type[T]
    ) -> T | None:
        try:
            result = self.generate(system, user, response_format=response_format)
        except Exception:
            return None
        return result if isinstance(result, response_format) else None
