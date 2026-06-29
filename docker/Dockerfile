FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY feeds.yaml ./
RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "--no-dev", "dailyai"]
