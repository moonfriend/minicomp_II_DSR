"""
Sweep: n_clusters × class_weight × hotspot_distances → Macro F1
Uses internal train/val split for speed, then re-evaluates best config on validation_full.csv.
"""

import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from geo_features import GeoClusterer, add_geo_label_to_df, add_hotspot_distances, HOTSPOT_FEATURES

RANDOM_STATE = 42
NUMERIC_BASE = ["minimum_nights", "number_of_reviews", "calculated_host_listings_count",
                "availability_365", "latitude", "longitude"]
ONEHOT       = ["neighbourhood_group", "room_type"]
ORDINAL      = ["neighbourhood", "geo_label"]


def build_model(numeric_features, class_weight=False):
    pre = ColumnTransformer([
        ("onehot",   OneHotEncoder(handle_unknown="ignore"), ONEHOT),
        ("ordinal",  OrdinalEncoder(handle_unknown="use_encoded_value",
                                    unknown_value=-1), ORDINAL),
        ("passthru", "passthrough", numeric_features),
    ])
    clf = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=RANDOM_STATE,
        verbosity=0,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def run_config(train_rows, val_rows, n_clusters, use_hotspots, use_class_weight):
    gc = GeoClusterer(n_clusters=n_clusters)
    gc.fit(train_rows)

    tr = add_geo_label_to_df(train_rows, gc)
    vl = add_geo_label_to_df(val_rows,   gc)

    numeric = NUMERIC_BASE.copy()
    if use_hotspots:
        tr = add_hotspot_distances(tr)
        vl = add_hotspot_distances(vl)
        numeric = numeric + HOTSPOT_FEATURES

    features = numeric + ONEHOT + ORDINAL
    X_tr, y_tr = tr[features], train_rows["price_tier"]
    X_vl, y_vl = vl[features], val_rows["price_tier"]

    model = build_model(numeric, use_class_weight)

    if use_class_weight:
        sw = compute_sample_weight("balanced", y_tr)
        model.fit(X_tr, y_tr, clf__sample_weight=sw)
    else:
        model.fit(X_tr, y_tr)

    preds = model.predict(X_vl)
    return f1_score(y_vl, preds, average="macro"), model, gc


def main():
    train_df = pd.read_csv("data/train_01.csv")
    train_rows, val_rows = train_test_split(
        train_df, test_size=0.1, random_state=RANDOM_STATE,
        stratify=train_df["price_tier"]
    )

    cluster_options  = [5, 10, 15, 20, 30]
    hotspot_options  = [False, True]
    weight_options   = [False, True]

    print(f"\n{'clusters':>8}  {'hotspots':>8}  {'balanced':>8}  {'Macro F1':>10}")
    print("-" * 44)

    results = []
    for n_cl, hot, cw in itertools.product(cluster_options, hotspot_options, weight_options):
        f1, model, gc = run_config(train_rows, val_rows, n_cl, hot, cw)
        tag = f"k={n_cl:2d}  hot={'Y' if hot else 'N'}  bal={'Y' if cw else 'N'}"
        print(f"  {n_cl:>6}  {'Y' if hot else 'N':>8}  {'Y' if cw else 'N':>8}  {f1:>10.4f}")
        results.append((f1, n_cl, hot, cw, model, gc))

    results.sort(reverse=True)
    best_f1, best_k, best_hot, best_cw, best_model, best_gc = results[0]
    print(f"\nBest: k={best_k}, hotspots={'Y' if best_hot else 'N'}, "
          f"balanced={'Y' if best_cw else 'N'}  →  F1={best_f1:.4f}")

    # ── Evaluate best config on validation_full.csv ───────────────────────────
    import pathlib
    val_path = pathlib.Path("data/validation_full.csv")
    if val_path.exists():
        val_full = pd.read_csv(val_path)
        vf = add_geo_label_to_df(val_full, best_gc)
        numeric = NUMERIC_BASE.copy()
        if best_hot:
            vf = add_hotspot_distances(vf)
            numeric = numeric + HOTSPOT_FEATURES
        features = numeric + ONEHOT + ORDINAL
        preds = best_model.predict(vf[features])
        f1_full = f1_score(val_full["price_tier"], preds, average="macro")
        print(f"\nBest config on validation_full.csv:  Macro F1 = {f1_full:.4f}")
        print(classification_report(
            val_full["price_tier"], preds,
            target_names=["Budget", "Standard", "Premium", "Ultra-Luxury"],
            labels=[0, 1, 2, 3], zero_division=0,
        ))

    # ── Also show top 5 ───────────────────────────────────────────────────────
    print("\nTop 5 configs:")
    for f1, k, hot, cw, _, _ in results[:5]:
        print(f"  k={k:2d}  hot={'Y' if hot else 'N'}  bal={'Y' if cw else 'N'}  F1={f1:.4f}")


if __name__ == "__main__":
    main()
