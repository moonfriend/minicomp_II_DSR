# Mini Competition II — NYC Airbnb Price Tier Classification

Predict the `price_tier` (0–3) of NYC Airbnb listings using tabular features and listing descriptions.

**Metric:** Macro F1-Score across 4 tiers: Budget (0) · Standard (1) · Premium (2) · Ultra-Luxury (3)

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For LLM-based pipelines, [Ollama](https://ollama.com) must be running locally with `llama3.2` pulled:

```bash
ollama pull llama3.2
```

---

## Project Structure

```
data/
  train_01.csv / test_01.csv   ← dataset used by all scripts
  train.csv / test.csv         ← original files (not used directly)

utils.py                       ← shared: load_data(), split(), score(), save_predictions()

── XGBoost pipelines ──────────────────────────────────────────────
xgb_pipeline.py                ← tabular features only              (F1 ≈ 0.54)
xgb_embeddings.py              ← sentence-transformer embeddings     (F1 ≈ 0.43)
xgb_embeddings_tabular.py      ← embeddings + tabular (hybrid)       (F1 ≈ 0.51)
xgb_llm_features.py            ← LLM boolean flags + tabular         (not yet benchmarked)
compare.py                     ← run all XGBoost pipelines, print F1 table

── Transformer pipelines ───────────────────────────────────────────
transformer_pipeline.py        ← unified entry point for all transformer strategies
                                  converts every tabular column to natural language,
                                  concatenates with description, fine-tunes transformer

── LLM few-shot pipeline ───────────────────────────────────────────
pipeline.py                    ← few-shot llama3.2 classifier via Ollama (F1 ≈ 0.19, baseline)

── Deployment ──────────────────────────────────────────────────────
app.py                         ← FastAPI: POST /predict accepts CSV, returns predictions CSV
agent.py                       ← LangGraph agent orchestrating the tools
tools.py                       ← LangChain tools: column normaliser, LLM flag extractor, XGB predictor
Dockerfile                     ← for Railway deployment
.devcontainer/                 ← devcontainer config (Python 3.12 + Ollama)

outputs/                       ← generated prediction CSVs written here
```

---

## Running experiments

### XGBoost baselines

```bash
python xgb_pipeline.py                    # tabular only — fastest, best XGB baseline
python compare.py                         # compare all fast pipelines
python compare.py --llm                   # include LLM flag extraction (slow ~30 min)
```

### Transformer pipeline

Three strategies, all using text + tabular features fused into natural language:

```bash
python transformer_pipeline.py --strategy tiny    # bert-tiny 4.4M params  ~10 min CPU
python transformer_pipeline.py --strategy frozen  # distilbert frozen backbone ~8 min CPU
python transformer_pipeline.py --strategy full    # distilbert full fine-tune  ~60 min CPU
```

All predictions are saved to `outputs/`.

### FastAPI (local test before Railway)

```bash
python app.py
# then in another terminal:
curl -X POST http://localhost:8000/predict -F "file=@data/test_01.csv" -o predictions.csv
```

---

## Backlog / next steps

- [ ] Train and serialize best model to `models/xgb_model.joblib` (needed for Railway)
- [ ] Add `railway.json` and deploy with `railway up`
- [ ] Set `OPENROUTER_API_KEY` in Railway environment variables
- [ ] Convert lat/lon to natural language area description in `row_to_text()`
- [ ] Benchmark `xgb_llm_features.py`
