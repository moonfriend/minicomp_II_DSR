FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app.py agent.py tools.py utils.py geo_features.py transformer_pipeline.py ./

# Pre-trained tabular model artefacts
COPY models/ ./models/

# Pre-download sentence-transformer embedding model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Pre-download fine-tuned DistilBERT from HuggingFace (baked into image layer)
# HF_TOKEN build arg needed only if the repo is private
ARG HF_TOKEN=""
RUN python -c "\
from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
import os; \
token = os.environ.get('HF_TOKEN', '') or '${HF_TOKEN}' or None; \
AutoTokenizer.from_pretrained('Li-1113/airbnb-price-tier', token=token); \
AutoModelForSequenceClassification.from_pretrained('Li-1113/airbnb-price-tier', token=token); \
print('Transformer model cached.')"

EXPOSE 8000

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
