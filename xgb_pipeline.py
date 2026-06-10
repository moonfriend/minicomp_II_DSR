"""
Tabular-only XGBoost pipeline for price_tier prediction.
Includes geographic clustering (geo_label) as an additional ordinal feature.
Saves the fitted model to models/xgb_model.joblib for the Railway agent.
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier
from geo_features import GeoClusterer, add_geo_label_to_df

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_FILE   = "data/train_01.csv"
TEST_FILE    = "data/test_01.csv"
VAL_SIZE     = 0.1
RANDOM_STATE = 42
MODEL_PATH   = Path("models/xgb_model.joblib")

NUMERIC_FEATURES = [
    "minimum_nights", "number_of_reviews", "calculated_host_listings_count",
    "availability_365", "latitude", "longitude",
]
ONEHOT_FEATURES  = ["neighbourhood_group", "room_type"]
ORDINAL_FEATURES = ["neighbourhood", "geo_label"]


def build_pipeline():
    preprocessor = ColumnTransformer([
        ("onehot",      OneHotEncoder(handle_unknown="ignore"), ONEHOT_FEATURES),
        ("ordinal",     OrdinalEncoder(handle_unknown="use_encoded_value",
                                       unknown_value=-1), ORDINAL_FEATURES),
        ("passthrough", "passthrough", NUMERIC_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=RANDOM_STATE,
        )),
    ])


def main():
    train_df = pd.read_csv(TRAIN_FILE)
    test_df  = pd.read_csv(TEST_FILE)

    # split full rows (need price_tier for GeoClusterer.fit)
    train_rows, val_rows = train_test_split(
        train_df, test_size=VAL_SIZE, random_state=RANDOM_STATE,
        stratify=train_df["price_tier"]
    )

    # fit geographic clusters on training rows only
    gc = GeoClusterer()
    gc.fit(train_rows)
    gc.save()

    train_rows = add_geo_label_to_df(train_rows, gc)
    val_rows   = add_geo_label_to_df(val_rows,   gc)
    test_df    = add_geo_label_to_df(test_df,    gc)

    feature_cols = NUMERIC_FEATURES + ONEHOT_FEATURES + ORDINAL_FEATURES
    X_train = train_rows[feature_cols]
    y_train = train_rows["price_tier"]
    X_val   = val_rows[feature_cols]
    y_val   = val_rows["price_tier"]

    print(f"Train: {len(X_train)} rows  |  Val: {len(X_val)} rows")
    print("Fitting model …")
    model = build_pipeline()
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    f1 = f1_score(y_val, val_preds, average="macro")
    print(f"\nMacro F1-Score (validation): {f1:.4f}")
    print(classification_report(y_val, val_preds,
                                target_names=["Budget", "Standard", "Premium", "Ultra-Luxury"]))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")

    print(f"Predicting test set ({len(test_df)} rows) …")
    test_preds = model.predict(test_df[feature_cols])
    test_df["price_tier"] = test_preds
    output_file = "outputs/predictions_xgb.csv"
    test_df[["property_id", "price_tier"]].to_csv(output_file, index=False)
    print(f"Saved → {output_file}")


if __name__ == "__main__":
    main()
