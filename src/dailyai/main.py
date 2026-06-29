"""Build today's AI news brief and save to file."""

from __future__ import annotations

import sys
from pathlib import Path

from dailyai.config import configure_logging, get_settings
from dailyai.factory import build_pipeline
from dailyai.pipeline import EmptyBriefError


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    try:
        brief = build_pipeline(settings).run()
    except EmptyBriefError as e:
        print(e, file=sys.stderr)
        return

    out_dir = Path("briefs")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"brief-{brief.brief_date:%Y-%m-%d}.md"
    path.write_text(brief.markdown)


if __name__ == "__main__":
    main()
