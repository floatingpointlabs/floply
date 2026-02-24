FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry>=2.0.0,<3.0.0"

WORKDIR /app

COPY pyproject.toml poetry.lock poetry.toml ./

RUN poetry install --no-interaction --no-ansi --only main,common --no-root

COPY . .

RUN poetry install --no-interaction --no-ansi --only-root

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/.streamlit/config.toml /app/.streamlit/config.toml

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]