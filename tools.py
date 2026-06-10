"""
LangChain tools called by the LangGraph agent.
LLM is used only for high-value reasoning tasks (3 calls max per request):
  1. Schema detection  — map unknown column names to canonical ones
  2. Language + translation — detect non-English, translate batch
  3. Missing data inference — infer key tabular values from description text
Deterministic Python handles everything else.
"""

import re
import json
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate

# ── LLM factory ───────────────────────────────────────────────────────────────

OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct"
OLLAMA_MODEL     = "llama3.2"


def _get_llm():
    if os.getenv("OPENROUTER_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=OPENROUTER_MODEL,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            temperature=0,
            request_timeout=30,
        )
    from langchain_ollama import OllamaLLM
    return OllamaLLM(model=OLLAMA_MODEL, temperature=0)


def check_llm() -> dict:
    """
    Quick LLM connectivity check. Call this at startup to surface problems early.
    Returns {"ok": True, "backend": "openrouter|ollama"} or {"ok": False, "error": "..."}.
    """
    try:
        llm = _get_llm()
        backend = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "ollama"
        raw = llm.invoke("Reply with the single word: ok")
        text = raw.content if hasattr(raw, "content") else str(raw)
        return {"ok": True, "backend": backend, "response": text.strip()[:20]}
    except Exception as e:
        return {"ok": False, "backend": "unknown", "error": str(e)}


def _llm_json(prompt_text: str) -> dict:
    """Call LLM, extract first JSON object from response."""
    llm = _get_llm()
    raw = llm.invoke(prompt_text)
    raw_str = raw.content if hasattr(raw, "content") else str(raw)
    match = re.search(r"\{.*\}", raw_str, re.DOTALL)
    return json.loads(match.group()) if match else {}


# ── Canonical schema ──────────────────────────────────────────────────────────

CANONICAL_COLUMNS = {
    "property_id", "description", "neighbourhood_group", "neighbourhood",
    "latitude", "longitude", "room_type", "minimum_nights",
    "number_of_reviews", "calculated_host_listings_count", "availability_365",
}

KNOWN_ALIASES = {
    "id": "property_id", "listing_id": "property_id",
    "name": "description", "title": "description",
    "summary": "description", "neighborhood_overview": "description",
    "borough": "neighbourhood_group", "region": "neighbourhood_group",
    "area": "neighbourhood", "location": "neighbourhood",
    "type": "room_type", "listing_type": "room_type",
    "min_nights": "minimum_nights", "minimum_stay": "minimum_nights",
    "min_nights_required": "minimum_nights",   # seen in validation_full.csv
    "reviews": "number_of_reviews", "num_reviews": "number_of_reviews",
    "host_listings": "calculated_host_listings_count",
    "host_listings_count": "calculated_host_listings_count",
    "availability": "availability_365", "avail_365": "availability_365",
    "lat": "latitude", "lng": "longitude", "lon": "longitude",
}

NUMERIC_DEFAULTS = {
    "minimum_nights": 1, "number_of_reviews": 0,
    "calculated_host_listings_count": 1, "availability_365": 180,
    "latitude": 40.7128, "longitude": -74.0060,
}
CATEGORICAL_DEFAULTS = {
    "neighbourhood_group": "Manhattan",
    "neighbourhood": "Unknown",
    "room_type": "Entire home/apt",
}


@tool
def detect_schema(csv_text: str) -> str:
    """
    Parse CSV, map columns to canonical names.
    Known aliases are resolved with a dict lookup (free).
    Truly unknown columns are sent to the LLM in ONE call.
    Returns a JSON string of normalised records.
    """
    from io import StringIO
    df = pd.read_csv(StringIO(csv_text))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # resolve known aliases first
    df = df.rename(columns={k: v for k, v in KNOWN_ALIASES.items() if k in df.columns})

    # find columns still not recognised
    unknown = [c for c in df.columns if c not in CANONICAL_COLUMNS]

    if unknown:
        sample_rows = df[unknown].head(3).to_dict(orient="records")
        prompt = f"""You are mapping CSV columns from an Airbnb dataset to canonical names.

Canonical columns: {sorted(CANONICAL_COLUMNS)}

Unknown columns found: {unknown}
Sample values: {json.dumps(sample_rows, default=str)}

Return ONLY a JSON object mapping each unknown column to its canonical name,
or null if it has no match. Example: {{"listing_name": "description", "borough_name": "neighbourhood_group", "xyz": null}}"""

        mapping = _llm_json(prompt)
        rename = {k: v for k, v in mapping.items() if v and v in CANONICAL_COLUMNS}
        df = df.rename(columns=rename)

    # fill missing canonical columns with safe defaults
    for col, default in {**NUMERIC_DEFAULTS, **CATEGORICAL_DEFAULTS}.items():
        if col not in df.columns:
            df[col] = default

    for col, default in NUMERIC_DEFAULTS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    for col, default in CATEGORICAL_DEFAULTS.items():
        if col in df.columns:
            df[col] = df[col].fillna(default).astype(str)

    df["description"] = df.get("description", pd.Series([""] * len(df))).fillna("").astype(str)

    return df.to_json(orient="records")


# ── Language detection + translation ─────────────────────────────────────────

def _is_likely_non_english(text: str) -> bool:
    """Cheap heuristic: high ratio of non-ASCII chars suggests non-English."""
    if not text:
        return False
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return (non_ascii / len(text)) > 0.15


@tool
def translate_descriptions(records_json: str) -> str:
    """
    Detect non-English descriptions and translate them in ONE batched LLM call.
    English descriptions are passed through untouched.
    """
    records = json.loads(records_json)
    to_translate = {
        i: rec["description"]
        for i, rec in enumerate(records)
        if _is_likely_non_english(str(rec.get("description", "")))
    }

    if not to_translate:
        return records_json

    prompt = f"""Translate these Airbnb listing descriptions to English.
Return ONLY a JSON object mapping each index to its English translation.

Descriptions to translate:
{json.dumps(to_translate, ensure_ascii=False)}

Example output: {{"3": "Cozy studio in the heart of Paris", "7": "Luxury penthouse with sea view"}}"""

    translations = _llm_json(prompt)
    for idx_str, translated in translations.items():
        idx = int(idx_str)
        if idx < len(records):
            records[idx]["description"] = translated

    print(f"  Translated {len(to_translate)} non-English descriptions.")
    return json.dumps(records)


# ── Missing tabular data inference ────────────────────────────────────────────

INFERABLE_COLS = ["neighbourhood_group", "neighbourhood", "room_type"]


@tool
def infer_missing_tabular(records_json: str) -> str:
    """
    For rows where key tabular columns are missing/unknown, call the LLM ONCE
    with a batch of problematic rows and ask it to infer values from the description.
    """
    records = json.loads(records_json)

    missing_rows = {}
    for i, rec in enumerate(records):
        missing = [
            col for col in INFERABLE_COLS
            if not rec.get(col) or str(rec.get(col)) in ("Unknown", "nan", "")
        ]
        if missing:
            missing_rows[i] = {
                "description": rec.get("description", ""),
                "missing_cols": missing,
            }

    if not missing_rows:
        return records_json

    prompt = f"""You are filling in missing data for NYC Airbnb listings.
For each listing, infer the missing columns from the description.

Valid values:
  neighbourhood_group: Manhattan, Brooklyn, Queens, Bronx, Staten Island
  room_type: Entire home/apt, Private room, Shared room

Listings needing inference:
{json.dumps(missing_rows, indent=2)}

Return ONLY a JSON object: index → {{column: inferred_value}}.
Example: {{"2": {{"neighbourhood_group": "Manhattan", "room_type": "Entire home/apt"}}}}"""

    inferences = _llm_json(prompt)
    for idx_str, values in inferences.items():
        idx = int(idx_str)
        if idx < len(records):
            for col, val in values.items():
                if col in INFERABLE_COLS:
                    records[idx][col] = val

    print(f"  Inferred missing values for {len(missing_rows)} rows.")
    return json.dumps(records)


# ── Prediction (ensemble XGBoost + Transformer) ───────────────────────────────

MODEL_PATH = Path("models/xgb_model.joblib")
GEO_PATH   = Path("models/geo_clusterer.joblib")
HF_REPO    = "Li-1113/airbnb-price-tier"
XGB_WEIGHT = 0.3   # tuned on validation_full.csv: best F1=0.5895 at xgb_w=0.3

NUMERIC  = ["minimum_nights", "number_of_reviews", "calculated_host_listings_count",
            "availability_365", "latitude", "longitude"]
ONEHOT   = ["neighbourhood_group", "room_type"]
ORDINAL  = ["neighbourhood", "geo_label"]

# module-level cache — loaded once, reused across requests
_xgb_pipeline    = None
_geo_clusterer   = None
_transformer     = None
_tokenizer       = None


def _load_xgb():
    global _xgb_pipeline, _geo_clusterer
    if _xgb_pipeline is None:
        _xgb_pipeline  = joblib.load(MODEL_PATH)
    if _geo_clusterer is None:
        from geo_features import GeoClusterer
        _geo_clusterer = GeoClusterer.load(GEO_PATH)
    return _xgb_pipeline, _geo_clusterer


def _load_transformer():
    global _transformer, _tokenizer
    if _transformer is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        hf_token = os.getenv("HF_TOKEN")
        print(f"  Loading transformer from {HF_REPO} …")
        _tokenizer   = AutoTokenizer.from_pretrained(HF_REPO, token=hf_token)
        _transformer = AutoModelForSequenceClassification.from_pretrained(
            HF_REPO, num_labels=4, token=hf_token
        )
        _transformer.eval()
        print("  Transformer loaded.")
    return _transformer, _tokenizer


@tool
def run_ensemble_predict(records_json: str) -> str:
    """
    Ensemble: XGBoost tabular + DistilBERT text probabilities → weighted average.
    Falls back to XGBoost-only if the transformer cannot be loaded.
    Returns JSON list of {property_id, price_tier}.
    """
    import torch
    from geo_features import add_geo_label_to_df
    from transformer_pipeline import build_texts, MAX_LEN

    records  = json.loads(records_json)
    df       = pd.DataFrame(records)

    pipeline, gc = _load_xgb()
    df_geo = add_geo_label_to_df(df, gc)

    # ── XGBoost probabilities ──────────────────────────────────────────────────
    xgb_proba = pipeline.predict_proba(df_geo[NUMERIC + ONEHOT + ORDINAL])  # (N,4)

    # ── Transformer probabilities (with fallback) ──────────────────────────────
    try:
        model, tokenizer = _load_transformer()
        texts      = build_texts(df_geo.reset_index(drop=True))
        all_logits = []
        for i in range(0, len(texts), 64):
            enc = tokenizer(
                texts[i:i + 64], truncation=True,
                padding="max_length", max_length=MAX_LEN, return_tensors="pt",
            )
            with torch.no_grad():
                all_logits.append(model(**enc).logits)
        logits            = torch.cat(all_logits, dim=0)
        transformer_proba = torch.softmax(logits, dim=1).numpy()
        combined          = XGB_WEIGHT * xgb_proba + (1 - XGB_WEIGHT) * transformer_proba
        print(f"  Ensemble: XGBoost×{XGB_WEIGHT} + Transformer×{1-XGB_WEIGHT}")
    except Exception as e:
        print(f"  Transformer unavailable ({e}), using XGBoost only.")
        combined = xgb_proba

    preds = np.argmax(combined, axis=1)
    results = [
        {"property_id": int(r["property_id"]), "price_tier": int(p)}
        for r, p in zip(records, preds)
    ]
    return json.dumps(results)
