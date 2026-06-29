import json
from datetime import date

from dailyai.models import PipelineMetrics
from dailyai.repositories import SQLiteMetricsRepository, connect


def test_save_persists_json_payload_and_is_queryable():
    conn = connect(":memory:")
    repo = SQLiteMetricsRepository(conn)
    repo.save(PipelineMetrics(brief_date=date(2026, 6, 28), support_rate=0.9))

    row = conn.execute(
        "SELECT brief_date, metrics_json, "
        "json_extract(metrics_json, '$.support_rate') AS sr "
        "FROM pipeline_metrics"
    ).fetchone()
    assert row["brief_date"] == "2026-06-28"
    assert row["sr"] == 0.9
    payload = json.loads(row["metrics_json"])
    assert payload["support_rate"] == 0.9
    # absent metrics serialize as null, not missing schema
    assert payload["cluster_coherence"] is None
