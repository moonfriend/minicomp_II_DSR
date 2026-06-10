FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + trained tabular model artefacts
COPY app.py agent.py tools.py utils.py geo_features.py transformer_pipeline.py ./
COPY models/ ./models/

EXPOSE 8000

# Models (XGBoost + transformer) are downloaded lazily at startup,
# not during build — keeps build fast and avoids HF rate-limit failures.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
