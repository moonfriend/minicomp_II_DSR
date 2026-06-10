"""
Geographic feature engineering: cluster NYC listings by lat/lon,
label each cluster by its dominant price tier.

Produces two things:
  1. A categorical feature `geo_tier_label` for XGBoost
     ("luxury_district", "premium_area", "standard_area", "budget_area")
  2. A text phrase for row_to_text() in transformer_pipeline.py
     "located in a luxury district"

Usage (standalone):
    python geo_features.py          # fits clusters on train, plots distribution

Usage (as module):
    from geo_features import GeoClusterer
    gc = GeoClusterer()
    gc.fit(train_df)
    train_df["geo_label"] = gc.transform(train_df)   # adds text label column
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
import joblib
from pathlib import Path

N_CLUSTERS   = 15
RANDOM_STATE = 42
MODEL_PATH   = Path("models/geo_clusterer.joblib")

# Known Ultra-Luxury hotspots in NYC (lat, lon)
LUXURY_HOTSPOTS = {
    "central_park":    (40.7829, -73.9654),
    "tribeca":         (40.7195, -74.0089),
    "upper_east_side": (40.7736, -73.9566),
    "midtown":         (40.7549, -73.9840),
    "soho":            (40.7233, -74.0030),
}


def add_hotspot_distances(df: pd.DataFrame) -> pd.DataFrame:
    """Add dist_<hotspot> columns (Euclidean degrees — scale-free for tree models)."""
    df = df.copy()
    lats = df["latitude"].values
    lons = df["longitude"].values
    for name, (hlat, hlon) in LUXURY_HOTSPOTS.items():
        df[f"dist_{name}"] = np.sqrt((lats - hlat) ** 2 + (lons - hlon) ** 2)
    return df


HOTSPOT_FEATURES = [f"dist_{name}" for name in LUXURY_HOTSPOTS]


TIER_BUCKET_LABELS = {
    (0.0, 0.75): "budget area",
    (0.75, 1.5): "standard area",
    (1.5, 2.25): "premium area",
    (2.25, 3.0): "luxury district",
}


def _tier_to_label(mean_tier: float) -> str:
    for (lo, hi), label in TIER_BUCKET_LABELS.items():
        if lo <= mean_tier < hi:
            return label
    return "luxury district"


class GeoClusterer:
    """
    Fits KMeans on (latitude, longitude) of training data.
    Labels each cluster by its mean price_tier.
    At inference time, assigns the nearest cluster's label to each row.
    """

    def __init__(self, n_clusters: int = N_CLUSTERS):
        self.n_clusters = n_clusters
        self.kmeans     = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE,
                                  n_init=10)
        self.cluster_labels: dict[int, str] = {}

    def fit(self, train_df: pd.DataFrame) -> "GeoClusterer":
        coords = train_df[["latitude", "longitude"]].values
        self.kmeans.fit(coords)

        cluster_ids = self.kmeans.labels_
        for cid in range(self.n_clusters):
            mask = cluster_ids == cid
            mean_tier = train_df.loc[mask, "price_tier"].mean()
            self.cluster_labels[cid] = _tier_to_label(mean_tier)

        self._print_cluster_summary(train_df, cluster_ids)
        return self

    def _print_cluster_summary(self, df, cluster_ids):
        print(f"\nGeo clusters ({self.n_clusters} total):")
        print(f"  {'Cluster':>8}  {'Mean tier':>10}  {'N':>6}  Label")
        for cid in range(self.n_clusters):
            mask = cluster_ids == cid
            mt   = df.loc[mask, "price_tier"].mean()
            n    = mask.sum()
            lat  = df.loc[mask, "latitude"].mean()
            lon  = df.loc[mask, "longitude"].mean()
            print(f"  {cid:>8}  {mt:>10.2f}  {n:>6}  {self.cluster_labels[cid]}"
                  f"  (lat={lat:.3f}, lon={lon:.3f})")

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Return a Series of text labels aligned to df's index."""
        coords = df[["latitude", "longitude"]].values
        cluster_ids = self.kmeans.predict(coords)
        return pd.Series(
            [self.cluster_labels[cid] for cid in cluster_ids],
            index=df.index,
            name="geo_label",
        )

    def save(self, path: Path = MODEL_PATH):
        path.parent.mkdir(exist_ok=True)
        joblib.dump(self, path)
        print(f"GeoClusterer saved → {path}")

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "GeoClusterer":
        return joblib.load(path)


# ── Integration helpers ───────────────────────────────────────────────────────

def add_geo_label_to_df(df: pd.DataFrame, clusterer: GeoClusterer) -> pd.DataFrame:
    """Add `geo_label` column to df for use as XGBoost ordinal feature."""
    df = df.copy()
    df["geo_label"] = clusterer.transform(df)
    return df


def geo_label_to_text(row: dict, clusterer: GeoClusterer) -> str:
    """
    Return a phrase for row_to_text() in transformer_pipeline.py.
    Example: "located in a luxury district"
    """
    try:
        lat = float(row.get("latitude", 40.7128))
        lon = float(row.get("longitude", -74.006))
        cid = clusterer.kmeans.predict([[lat, lon]])[0]
        return f"located in a {clusterer.cluster_labels[cid]}"
    except Exception:
        return ""


# ── Standalone: fit, evaluate, save ──────────────────────────────────────────

def main():
    from utils import load_data, split
    from sklearn.metrics import f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
    from xgboost import XGBClassifier

    train_df, test_df = load_data()
    from sklearn.model_selection import train_test_split
    X_train_df, X_val_df = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df["price_tier"]
    )

    gc = GeoClusterer()
    gc.fit(X_train_df)

    # evaluate: XGBoost with geo_label added vs without
    NUMERIC = ["minimum_nights", "number_of_reviews", "calculated_host_listings_count",
               "availability_365", "latitude", "longitude"]
    ONEHOT  = ["neighbourhood_group", "room_type"]
    ORDINAL = ["neighbourhood", "geo_label"]

    X_tr = add_geo_label_to_df(X_train_df, gc)
    X_vl = add_geo_label_to_df(X_val_df,   gc)
    y_tr = X_train_df["price_tier"]
    y_vl = X_val_df["price_tier"]

    pre = ColumnTransformer([
        ("onehot",      OneHotEncoder(handle_unknown="ignore"), ONEHOT),
        ("ordinal",     OrdinalEncoder(handle_unknown="use_encoded_value",
                                        unknown_value=-1), ORDINAL),
        ("passthrough", "passthrough", NUMERIC),
    ])
    model = Pipeline([
        ("pre", pre),
        ("clf", XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               eval_metric="mlogloss", random_state=42)),
    ])
    model.fit(X_tr[NUMERIC + ONEHOT + ORDINAL], y_tr)
    preds = model.predict(X_vl[NUMERIC + ONEHOT + ORDINAL])
    f1 = f1_score(y_vl, preds, average="macro")
    print(f"\nXGBoost + geo_label Macro F1: {f1:.4f}  (baseline without: ~0.54)")

    gc.save()


if __name__ == "__main__":
    main()
