"""Finished Brief and its faithfulness EvaluationReport."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class EvaluationReport(BaseModel):
    """Faithfulness and quality evaluation results."""

    urls_cited: int = 0
    urls_grounded: int = 0
    fabricated_urls: list[str] = Field(default_factory=list)
    n_claims: int = 0
    support_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)
    notes: str = ""
    claim_check_failed: bool = False


class Brief(BaseModel):
    """Finished brief with markdown and evaluation."""

    brief_date: date
    markdown: str
    evaluation: EvaluationReport | None = None
