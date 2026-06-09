# Mini Competition II — NYC Airbnb Price Tier Classification

Predict the `price_tier` (0–3) of NYC Airbnb listings using tabular features and listing descriptions.

## Project Structure

```
.
├── data/
│   ├── train.csv            # original training data
│   ├── test.csv             # original test data
│   ├── train_01.csv         # processed training data (used by scripts)
│   ├── test_01.csv          # processed test data (used by scripts)
│   └── archive/             # earlier data versions
├── docs/
│   ├── PRD.odt              # product requirements
│   └── Prompt to Production - Student Brief.pdf
├── outputs/                 # generated prediction CSVs
├── utils.py                 # shared data loading, splitting, scoring
├── xgb_pipeline.py          # tabular-only XGBoost
├── xgb_embeddings.py        # sentence-transformer embeddings → XGBoost
├── xgb_embeddings_tabular.py# embeddings + tabular → XGBoost (hybrid)
├── xgb_llm_features.py      # LLM-extracted boolean flags + tabular → XGBoost
├── pipeline.py              # few-shot LLM pipeline (llama3.2 via Ollama)
└── compare.py               # run all pipelines and print F1 comparison table
```

## Price Tiers

| Tier | Label        |
|------|-------------|
| 0    | Budget       |
| 1    | Standard     |
| 2    | Premium      |
| 3    | Ultra-Luxury |

Metric: **Macro F1-Score**

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install xgboost scikit-learn pandas sentence-transformers langchain-ollama
```

For LLM-based pipelines, [Ollama](https://ollama.com) must be running locally with `llama3.2` pulled:

```bash
ollama pull llama3.2
```

## Usage

Run a single pipeline:

```bash
python xgb_pipeline.py              # tabular-only (fastest)
python xgb_embeddings.py            # embeddings-only
python xgb_embeddings_tabular.py    # hybrid (best ML score)
python xgb_llm_features.py          # LLM flag extraction + tabular
python pipeline.py                  # few-shot LLM classifier
```

Compare all non-LLM pipelines:

```bash
python compare.py          # tabular, embeddings, hybrid
python compare.py --llm    # include LLM-based pipeline
```

All prediction files are saved to `outputs/`.
