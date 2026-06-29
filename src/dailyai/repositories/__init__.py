"""Repository abstractions and SQLite implementations."""

from dailyai.repositories.base import (
    ArticleRepository,
    BriefRepository,
    MetricsRepository,
)
from dailyai.repositories.sqlite import (
    SQLiteArticleRepository,
    SQLiteBriefRepository,
    SQLiteMetricsRepository,
    connect,
)

__all__ = [
    "ArticleRepository",
    "BriefRepository",
    "MetricsRepository",
    "SQLiteArticleRepository",
    "SQLiteBriefRepository",
    "SQLiteMetricsRepository",
    "connect",
]
