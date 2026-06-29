"""SQLite implementations of article, brief, and metrics repositories."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone

import numpy as np

from dailyai.models import Article, Brief, EvaluationReport, PipelineMetrics
from dailyai.repositories.base import (
    ArticleRepository,
    BriefRepository,
    MetricsRepository,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    url         TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    source      TEXT NOT NULL,
    source_type TEXT NOT NULL,
    authority   REAL NOT NULL,
    published   TEXT NOT NULL,
    summary     TEXT,
    first_seen  TEXT NOT NULL,
    embedding   BLOB
);
CREATE TABLE IF NOT EXISTS briefs (
    brief_date   TEXT PRIMARY KEY,
    markdown     TEXT NOT NULL,
    support_rate REAL,
    urls_cited   INTEGER,
    urls_grounded INTEGER,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS published_clusters (
    brief_date TEXT NOT NULL,
    centroid   BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS pipeline_metrics (
    brief_date   TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    detail_json  TEXT,
    created_at   TEXT NOT NULL
);
"""


def connect(path: str) -> sqlite3.Connection:
    """Initialize SQLite connection with schema."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)

    return conn


class SQLiteArticleRepository(ArticleRepository):
    """SQLite-backed article storage with embedding cache."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def upsert_many(self, articles: list[Article]) -> int:
        """Insert articles, skipping duplicates by URL."""
        now = datetime.now(timezone.utc).isoformat()
        added = 0
        for article in articles:
            added += self.conn.execute(
                """INSERT OR IGNORE INTO articles
                   (url, title, source, source_type, authority, published,
                    summary, first_seen)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    article.url,
                    article.title,
                    article.source,
                    article.source_type,
                    article.authority,
                    article.published.isoformat(),
                    article.summary,
                    now,
                ),
            ).rowcount
        self.conn.commit()

        return added

    def recent_cluster_centroids(self, since: date) -> list[np.ndarray]:
        """Fetch centroids from recently published clusters."""
        rows = self.conn.execute(
            "SELECT centroid FROM published_clusters WHERE brief_date >= ?",
            (since.isoformat(),),
        ).fetchall()

        centroids = [np.frombuffer(row["centroid"], dtype=np.float32) for row in rows]

        return centroids

    def add_published_clusters(
        self, centroids: list[np.ndarray], brief_date: date
    ) -> None:
        """Store cluster centroids for a published brief."""
        self.conn.executemany(
            "INSERT INTO published_clusters (brief_date, centroid) VALUES (?, ?)",
            [
                (
                    brief_date.isoformat(),
                    np.asarray(centroid, dtype=np.float32).tobytes(),
                )
                for centroid in centroids
            ],
        )
        self.conn.commit()

    def load_embeddings(self, urls: list[str]) -> dict[str, np.ndarray]:
        """Retrieve cached embeddings for given URLs."""
        if not urls:
            return {}

        placeholders = ",".join("?" * len(urls))
        rows = self.conn.execute(
            f"SELECT url, embedding FROM articles "
            f"WHERE embedding IS NOT NULL AND url IN ({placeholders})",
            urls,
        ).fetchall()

        return {
            row["url"]: np.frombuffer(row["embedding"], dtype=np.float32)
            for row in rows
        }

    def save_embeddings(self, articles: list[Article]) -> None:
        """Cache embeddings for articles."""
        self.conn.executemany(
            "UPDATE articles SET embedding = ? WHERE url = ?",
            [
                (np.asarray(article.embedding, dtype=np.float32).tobytes(), article.url)
                for article in articles
                if article.embedding is not None
            ],
        )
        self.conn.commit()


class SQLiteBriefRepository(BriefRepository):
    """SQLite-backed brief storage."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def save(self, brief: Brief) -> None:
        """Store a brief with its evaluation report."""
        evaluation = brief.evaluation or EvaluationReport()
        self.conn.execute(
            """INSERT OR REPLACE INTO briefs
               (brief_date, markdown, support_rate, urls_cited, urls_grounded, created_at)
               VALUES (?,?,?,?,?,?)""",
            (
                brief.brief_date.isoformat(),
                brief.markdown,
                evaluation.support_rate,
                evaluation.urls_cited,
                evaluation.urls_grounded,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()


class SQLiteMetricsRepository(MetricsRepository):
    """SQLite-backed metrics storage."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def save(self, metrics: PipelineMetrics, detail_json: str | None = None) -> None:
        """Store pipeline evaluation metrics and optional diagnostic detail."""
        self.conn.execute(
            "INSERT INTO pipeline_metrics "
            "(brief_date, metrics_json, detail_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                metrics.brief_date.isoformat(),
                metrics.model_dump_json(),
                detail_json,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()
