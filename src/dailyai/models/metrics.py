"""Per-brief evaluation metrics stored as JSON."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class PipelineMetrics(BaseModel):
    """Evaluation metrics from the pipeline (faithfulness, coherence, review)."""

    brief_date: date
    support_rate: float | None = None
    total_claims: int | None = None
    grounded_urls: int | None = None
    cluster_coherence: float | None = None
    is_scannable: bool | None = None
    stories_with_takeaway: int | None = None
    unexplained_jargon_terms: int | None = None


class ClusterCoherence(BaseModel):
    """Coherence verdict for one cluster."""

    cluster_index: int
    coherent: bool
    reason: str = ""


class EvaluationDetail(BaseModel):
    """Diagnostic payload behind the scalar metrics: the *why* of each pass.

    Stored as a JSON blob alongside PipelineMetrics so a failing rate
    (e.g. cluster_coherence=0.667) can be traced to the offending item."""

    fabricated_urls: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    cluster_verdicts: list[ClusterCoherence] = Field(default_factory=list)
    claim_check_failed: bool = False
    coherence_failed: bool = False
    review_failed: bool = False
