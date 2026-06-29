from datetime import datetime, timezone

from dailyai.models import Article, SourceCategory
from dailyai.services.filtering import FilterService


def _article(url, title="t") -> Article:
    return Article(
        title=title,
        url=url,
        source="s",
        authority=0.5,
        published=datetime(2026, 6, 28, tzinfo=timezone.utc),
        source_type=SourceCategory.API,
    )


def test_canonical_url_strips_tracking_and_trailing_slash():
    canon = FilterService._canonical_url("https://A.com/Path/?utm_source=x&id=5")
    assert canon == "https://a.com/Path?id=5"


def test_dedupe_drops_same_canonical_url():
    svc = FilterService()
    a = _article("https://a.com/p?utm_source=x", title="one")
    b = _article("https://a.com/p", title="two-different-title")
    kept = svc.dedupe_articles([a, b])
    assert len(kept) == 1
