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

def _get_llm():
    if os.getenv("OPENROUTER_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="meta-llama/llama-3.1-8b-instruct:free",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            temperature=0,
        )
    from langchain_ollama import OllamaLLM
    return OllamaLLM(model="llama3.2", temperature=0)


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


# ── XGBoost prediction ────────────────────────────────────────────────────────

MODEL_PATH = Path("models/xgb_model.joblib")

NUMERIC  = ["minimum_nights", "number_of_reviews", "calculated_host_listings_count",
            "availability_365", "latitude", "longitude"]
ONEHOT   = ["neighbourhood_group", "room_type"]
ORDINAL  = ["neighbourhood"]


@tool
def run_xgb_predict(records_json: str) -> str:
    """
    Load the pre-trained XGBoost pipeline and predict price_tier.
    Returns JSON list of {property_id, price_tier}.
    """
    records = json.loads(records_json)
    df = pd.DataFrame(records)

    pipeline = joblib.load(MODEL_PATH)
    feature_cols = NUMERIC + ONEHOT + ORDINAL

    preds = pipeline.predict(df[feature_cols])
    results = [
        {"property_id": int(r["property_id"]), "price_tier": int(p)}
        for r, p in zip(records, preds)
    ]
    return json.dumps(results)
