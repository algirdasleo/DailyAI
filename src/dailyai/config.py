"""Load config from env vars and feeds.yaml with validation."""

from __future__ import annotations

import logging
import sys

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    openai_api_key: str

    embed_model: str = "text-embedding-3-small"
    brief_model: str = "gpt-5.4-mini"
    eval_model: str = "gpt-5.4-mini"

    db_path: str = "dailyai.db"

    window_hours: int = Field(24, gt=0)

    cluster_threshold: float = Field(0.55, ge=0, le=1)

    repeat_threshold: float = Field(
        0.55,
        ge=0,
        le=1,
        description="Skip stories matching recent clusters at or above this similarity",
    )
    stale_after_days: int = Field(14, gt=0)

    top_stories: int = Field(8, gt=0)
    rank_model: str = "gpt-5.4-mini"

    fetch_timeout: int = Field(15, gt=0)
    fetch_max_chars: int = Field(8000, gt=0)
    summarizer_model: str = "gpt-5.4-mini"

    log_level: str = "INFO"


def get_settings() -> Settings:

    return Settings()  # type: ignore - will throw if env vars are missing/invalid, which is expected.


def load_sources(path: str = "feeds.yaml") -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["sources"]


def configure_logging(level: str = "INFO") -> None:
    """Configure logging to stderr with standard format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )
