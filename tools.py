"""
LangChain tools called by the LangGraph agent.
Each tool is a pure function: deterministic Python where possible,
LLM only where genuinely needed.
"""

import re
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from langchain_core.tools import tool

# ── Column normalisation ──────────────────────────────────────────────────────

CANONICAL_COLUMNS = {
    "property_id", "description", "neighbourhood_group", "neighbourhood",
    "latitude", "longitude", "room_type", "minimum_nights",
    "number_of_reviews", "calculated_host_listings_count", "availability_365",
}

# Common aliases the curveball set might use
ALIAS_MAP = {
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
def normalize_columns(csv_text: str) -> str:
    """
    Parse a CSV string, rename aliased columns to canonical names,
    fill missing columns with safe defaults.
    Returns a JSON string of records ready for prediction.
    """
    from io import StringIO
    df = pd.read_csv(StringIO(csv_text))

    # lowercase all column names for matching
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # rename known aliases
    df = df.rename(columns={k: v for k, v in ALIAS_MAP.items() if k in df.columns})

    # fill missing canonical columns with defaults
    for col, default in {**NUMERIC_DEFAULTS, **CATEGORICAL_DEFAULTS}.items():
        if col not in df.columns:
            df[col] = default

    # fill missing values within existing columns
    for col, default in NUMERIC_DEFAULTS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    for col, default in CATEGORICAL_DEFAULTS.items():
        if col in df.columns:
            df[col] = df[col].fillna(default).astype(str)

    df["description"] = df.get("description", pd.Series([""] * len(df))).fillna("").astype(str)

    return df.to_json(orient="records")


# ── LLM text feature extraction ───────────────────────────────────────────────

FLAG_COLS = ["has_luxury", "has_view", "has_outdoor", "is_shared",
             "needs_work", "has_amenities"]

_EXTRACT_TEMPLATE = """Analyze this Airbnb listing and return ONLY a JSON object:
  has_luxury   : true if mentions luxury, penthouse, marble, doorman, concierge
  has_view     : true if mentions view, skyline, park view, river, city view
  has_outdoor  : true if mentions rooftop, balcony, terrace, garden
  is_shared    : true if mentions shared bathroom, shared room, hostel
  needs_work   : true if mentions renovation, repair, fixer, basic, worn
  has_amenities: true if mentions pool, gym, spa, elevator, fireplace

Listing: "{description}"

Reply with ONLY the JSON. Example: {{"has_luxury":false,"has_view":true,"has_outdoor":false,"is_shared":false,"needs_work":false,"has_amenities":false}}"""


def _get_llm():
    """Return OpenRouter LLM in production, Ollama locally."""
    import os
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


@tool
def extract_text_features(records_json: str) -> str:
    """
    For each listing, call the LLM to extract boolean luxury/quality flags
    from the description. Returns the records JSON with flag columns added.
    """
    from langchain_core.prompts import PromptTemplate

    records = json.loads(records_json)
    llm = _get_llm()
    prompt = PromptTemplate(input_variables=["description"],
                            template=_EXTRACT_TEMPLATE)
    chain = prompt | llm

    for rec in records:
        desc = str(rec.get("description", ""))
        try:
            raw = chain.invoke({"description": desc})
            raw_str = raw.content if hasattr(raw, "content") else str(raw)
            match = re.search(r"\{.*\}", raw_str, re.DOTALL)
            flags = json.loads(match.group()) if match else {}
        except Exception:
            flags = {}
        for col in FLAG_COLS:
            rec[col] = int(bool(flags.get(col, False)))

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
    Load the pre-trained XGBoost pipeline and predict price_tier for each record.
    Returns JSON list of {property_id, price_tier}.
    """
    records = json.loads(records_json)
    df = pd.DataFrame(records)

    pipeline = joblib.load(MODEL_PATH)
    feature_cols = NUMERIC + ONEHOT + ORDINAL + FLAG_COLS

    # keep only columns the model knows; fill any still-missing flags with 0
    for col in FLAG_COLS:
        if col not in df.columns:
            df[col] = 0

    preds = pipeline.predict(df[feature_cols])
    results = [{"property_id": int(r["property_id"]), "price_tier": int(p)}
               for r, p in zip(records, preds)]
    return json.dumps(results)
