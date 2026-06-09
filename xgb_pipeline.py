"""
Tabular-only XGBoost pipeline for price_tier prediction.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_FILE = "data/train_01.csv"
TEST_FILE  = "data/test_01.csv"
VAL_SIZE   = 0.1
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "minimum_nights",
    "number_of_reviews",
    "calculated_host_listings_count",
    "availability_365",
    "latitude",
    "longitude",
]
ONEHOT_FEATURES  = ["neighbourhood_group", "room_type"]
ORDINAL_FEATURES = ["neighbourhood"]

# ── Preprocessor ──────────────────────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("onehot",   OneHotEncoder(handle_unknown="ignore"), ONEHOT_FEATURES),
        ("ordinal",  OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ORDINAL_FEATURES),
        ("passthrough", "passthrough", NUMERIC_FEATURES),
    ]
)

# ── Model ─────────────────────────────────────────────────────────────────────
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
    )),
])

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    train_df = pd.read_csv(TRAIN_FILE)
    test_df  = pd.read_csv(TEST_FILE)

    feature_cols = NUMERIC_FEATURES + ONEHOT_FEATURES + ORDINAL_FEATURES
    X = train_df[feature_cols]
    y = train_df["price_tier"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Train: {len(X_train)} rows  |  Val: {len(X_val)} rows")
    print("Fitting model …")
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    f1 = f1_score(y_val, val_preds, average="macro")
    print(f"\nMacro F1-Score (validation): {f1:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_val, val_preds,
                                target_names=["Budget", "Standard", "Premium", "Ultra-Luxury"]))

    print(f"Predicting test set ({len(test_df)} rows) …")
    test_preds = model.predict(test_df[feature_cols])
    test_df["price_tier"] = test_preds
    output_file = "outputs/predictions_xgb.csv"
    test_df[["property_id", "price_tier"]].to_csv(output_file, index=False)
    print(f"Saved → {output_file}")


if __name__ == "__main__":
    main()
