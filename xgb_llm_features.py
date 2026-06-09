"""
LLM feature extraction pipeline: ask llama3.2 to extract boolean flags from each
description, then train XGBoost on those flags + tabular features.
"""

import re
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from xgboost import XGBClassifier
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from utils import (load_data, split, score, save_predictions,
                   TEXT_COL, FEATURE_COLS, RANDOM_STATE)

MODEL_NAME = "llama3.2"

# ── LLM flag extraction ───────────────────────────────────────────────────────
EXTRACT_PROMPT = PromptTemplate(
    input_variables=["description"],
    template="""Analyze this Airbnb listing name/description and return ONLY a JSON object with these boolean fields:
  has_luxury      : true if mentions luxury, penthouse, marble, doorman, concierge, high-end
  has_view        : true if mentions view, skyline, park view, river, city view
  has_outdoor     : true if mentions rooftop, balcony, terrace, garden, patio
  is_shared       : true if mentions shared bathroom, shared room, hostel, dormitory
  needs_work      : true if mentions renovation, repair, fixer, worn, basic
  has_amenities   : true if mentions pool, gym, spa, elevator, fireplace

Description: "{description}"

Reply with ONLY the JSON object, nothing else. Example: {{"has_luxury": false, "has_view": true, "has_outdoor": false, "is_shared": false, "needs_work": false, "has_amenities": false}}"""
)

FLAG_COLS = ["has_luxury", "has_view", "has_outdoor", "is_shared", "needs_work", "has_amenities"]

llm   = OllamaLLM(model=MODEL_NAME, temperature=0)
chain = EXTRACT_PROMPT | llm


def extract_flags(description: str) -> dict:
    try:
        raw = chain.invoke({"description": description})
        # find the JSON object in the response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            flags = json.loads(match.group())
            return {k: int(bool(flags.get(k, False))) for k in FLAG_COLS}
    except Exception:
        pass
    return {k: 0 for k in FLAG_COLS}


def extract_all_flags(descriptions: list, label: str) -> pd.DataFrame:
    print(f"Extracting LLM flags for {label} ({len(descriptions)} rows) …")
    rows = []
    for i, desc in enumerate(descriptions, 1):
        rows.append(extract_flags(desc))
        if i % 50 == 0:
            print(f"  [{i}/{len(descriptions)}]")
    return pd.DataFrame(rows)


# ── Tabular preprocessor ─────────────────────────────────────────────────────
NUMERIC  = ["minimum_nights", "number_of_reviews", "calculated_host_listings_count",
            "availability_365", "latitude", "longitude"]
ONEHOT   = ["neighbourhood_group", "room_type"]
ORDINAL  = ["neighbourhood"]

tab_preprocessor = ColumnTransformer([
    ("onehot",   OneHotEncoder(handle_unknown="ignore"), ONEHOT),
    ("ordinal",  OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ORDINAL),
    ("passthrough", "passthrough", NUMERIC),
])


def main():
    train_df, test_df = load_data()
    X_train, X_val, y_train, y_val = split(train_df)

    # Extract LLM flags
    train_flags = extract_all_flags(X_train[TEXT_COL].tolist(), "train")
    val_flags   = extract_all_flags(X_val[TEXT_COL].tolist(), "val")
    test_flags  = extract_all_flags(test_df[TEXT_COL].tolist(), "test")

    # Process tabular features
    tab_train = tab_preprocessor.fit_transform(X_train[NUMERIC + ONEHOT + ORDINAL])
    tab_val   = tab_preprocessor.transform(X_val[NUMERIC + ONEHOT + ORDINAL])
    tab_test  = tab_preprocessor.transform(test_df[NUMERIC + ONEHOT + ORDINAL])

    # Combine
    X_train_combined = np.hstack([tab_train, train_flags.values])
    X_val_combined   = np.hstack([tab_val,   val_flags.values])
    X_test_combined  = np.hstack([tab_test,  test_flags.values])

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
    )

    print(f"\nFitting XGBoost on tabular + {len(FLAG_COLS)} LLM flags …")
    clf.fit(X_train_combined, y_train)

    val_preds = clf.predict(X_val_combined)
    score(y_val, val_preds, label="tabular+llm-flags")

    test_preds = clf.predict(X_test_combined)
    save_predictions(test_df, test_preds, "outputs/predictions_llm_features.csv")


if __name__ == "__main__":
    main()
