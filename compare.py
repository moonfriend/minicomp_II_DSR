"""
Run all pipelines on the same val split and print a comparison table.
Skips LLM-based pipelines by default (slow) — pass --llm to include them.
"""

import sys
import pandas as pd
from sklearn.metrics import f1_score
from utils import load_data, split, score, TIER_NAMES

def run_tabular():
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
    from xgboost import XGBClassifier

    NUMERIC = ["minimum_nights", "number_of_reviews", "calculated_host_listings_count",
               "availability_365", "latitude", "longitude"]
    ONEHOT  = ["neighbourhood_group", "room_type"]
    ORDINAL = ["neighbourhood"]

    train_df, _ = load_data()
    X_train, X_val, y_train, y_val = split(train_df)

    preprocessor = ColumnTransformer([
        ("onehot",      OneHotEncoder(handle_unknown="ignore"), ONEHOT),
        ("ordinal",     OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ORDINAL),
        ("passthrough", "passthrough", NUMERIC),
    ])
    model = Pipeline([
        ("pre", preprocessor),
        ("clf", XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="mlogloss", random_state=42)),
    ])
    model.fit(X_train[NUMERIC + ONEHOT + ORDINAL], y_train)
    preds = model.predict(X_val[NUMERIC + ONEHOT + ORDINAL])
    return f1_score(y_val, preds, average="macro"), y_val, preds


def run_embeddings():
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from xgboost import XGBClassifier
    from utils import TEXT_COL

    train_df, _ = load_data()
    X_train, X_val, y_train, y_val = split(train_df)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    X_tr = embedder.encode(X_train[TEXT_COL].tolist(), batch_size=64, show_progress_bar=False)
    X_vl = embedder.encode(X_val[TEXT_COL].tolist(),   batch_size=64, show_progress_bar=False)

    clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric="mlogloss", random_state=42)
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_vl)
    return f1_score(y_val, preds, average="macro"), y_val, preds


def run_embeddings_tabular():
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
    from xgboost import XGBClassifier
    from utils import TEXT_COL

    NUMERIC = ["minimum_nights", "number_of_reviews", "calculated_host_listings_count",
               "availability_365", "latitude", "longitude"]
    ONEHOT  = ["neighbourhood_group", "room_type"]
    ORDINAL = ["neighbourhood"]

    train_df, _ = load_data()
    X_train, X_val, y_train, y_val = split(train_df)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    tr_emb = embedder.encode(X_train[TEXT_COL].tolist(), batch_size=64, show_progress_bar=False)
    vl_emb = embedder.encode(X_val[TEXT_COL].tolist(),   batch_size=64, show_progress_bar=False)

    tab = ColumnTransformer([
        ("onehot",      OneHotEncoder(handle_unknown="ignore"), ONEHOT),
        ("ordinal",     OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ORDINAL),
        ("passthrough", "passthrough", NUMERIC),
    ])
    tr_tab = tab.fit_transform(X_train[NUMERIC + ONEHOT + ORDINAL])
    vl_tab = tab.transform(X_val[NUMERIC + ONEHOT + ORDINAL])

    clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric="mlogloss", random_state=42)
    clf.fit(np.hstack([tr_emb, tr_tab]), y_train)
    preds = clf.predict(np.hstack([vl_emb, vl_tab]))
    return f1_score(y_val, preds, average="macro"), y_val, preds


def main():
    include_llm = "--llm" in sys.argv
    results = {}

    print("── Tabular-only XGBoost ──────────────────────")
    results["tabular"], *_ = run_tabular()

    print("── Embeddings-only XGBoost ───────────────────")
    results["embeddings"], *_ = run_embeddings()

    print("── Embeddings + Tabular XGBoost ──────────────")
    results["emb+tabular"], *_ = run_embeddings_tabular()

    if include_llm:
        print("── Tabular + LLM flags XGBoost ───────────────")
        import xgb_llm_features as llmf
        from utils import load_data, split, TEXT_COL
        import numpy as np
        train_df, _ = load_data()
        X_train, X_val, y_train, y_val = split(train_df)
        tr_flags = llmf.extract_all_flags(X_train[TEXT_COL].tolist(), "train")
        vl_flags = llmf.extract_all_flags(X_val[TEXT_COL].tolist(), "val")
        NUMERIC = ["minimum_nights","number_of_reviews","calculated_host_listings_count","availability_365","latitude","longitude"]
        ONEHOT  = ["neighbourhood_group","room_type"]
        ORDINAL = ["neighbourhood"]
        tab_train = llmf.tab_preprocessor.fit_transform(X_train[NUMERIC+ONEHOT+ORDINAL])
        tab_val   = llmf.tab_preprocessor.transform(X_val[NUMERIC+ONEHOT+ORDINAL])
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric="mlogloss", random_state=42)
        clf.fit(np.hstack([tab_train, tr_flags.values]), y_train)
        preds = clf.predict(np.hstack([tab_val, vl_flags.values]))
        results["tabular+llm-flags"] = f1_score(y_val, preds, average="macro")

    print("\n" + "═" * 45)
    print(f"{'Approach':<30} {'Macro F1':>10}")
    print("─" * 45)
    for name, f1 in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{name:<30} {f1:>10.4f}")
    print("═" * 45)


if __name__ == "__main__":
    main()
