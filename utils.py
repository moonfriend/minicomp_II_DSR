"""
Shared utilities: data loading, train/val split, scoring.
All experiments use the same split so F1 scores are comparable.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

TRAIN_FILE   = "data/train_01.csv"
TEST_FILE    = "data/test_01.csv"
RANDOM_STATE = 42
VAL_SIZE     = 0.1

FEATURE_COLS = [
    "minimum_nights", "number_of_reviews", "calculated_host_listings_count",
    "availability_365", "latitude", "longitude",
    "neighbourhood_group", "room_type", "neighbourhood",
]
TEXT_COL   = "description"
TARGET_COL = "price_tier"
ID_COL     = "property_id"

TIER_NAMES = ["Budget", "Standard", "Premium", "Ultra-Luxury"]


def load_data():
    train = pd.read_csv(TRAIN_FILE)
    test  = pd.read_csv(TEST_FILE)
    # fill missing descriptions with empty string
    train[TEXT_COL] = train[TEXT_COL].fillna("").astype(str)
    test[TEXT_COL]  = test[TEXT_COL].fillna("").astype(str)
    return train, test


def split(train_df):
    X = train_df.drop(columns=[TARGET_COL, ID_COL])
    y = train_df[TARGET_COL]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_val, y_train, y_val


def score(y_true, y_pred, label=""):
    f1 = f1_score(y_true, y_pred, average="macro")
    tag = f"[{label}] " if label else ""
    print(f"\n{tag}Macro F1-Score (validation): {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=TIER_NAMES))
    return f1


def save_predictions(test_df, preds, filename):
    out = test_df[[ID_COL]].copy()
    out[TARGET_COL] = preds
    out.to_csv(filename, index=False)
    print(f"Saved → {filename}")
