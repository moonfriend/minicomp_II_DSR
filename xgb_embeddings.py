"""
Embeddings-only pipeline: encode descriptions with sentence-transformers → XGBoost.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
from utils import load_data, split, score, save_predictions, TEXT_COL, FEATURE_COLS

EMBED_MODEL = "all-MiniLM-L6-v2"


def encode(texts, model):
    return model.encode(texts, show_progress_bar=True, batch_size=64)


def main():
    train_df, test_df = load_data()
    X_train, X_val, y_train, y_val = split(train_df)

    print(f"Loading embedding model: {EMBED_MODEL} …")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Encoding train descriptions …")
    X_train_emb = encode(X_train[TEXT_COL].tolist(), embedder)
    print("Encoding val descriptions …")
    X_val_emb   = encode(X_val[TEXT_COL].tolist(), embedder)
    print("Encoding test descriptions …")
    X_test_emb  = encode(test_df[TEXT_COL].tolist(), embedder)

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
    )

    print(f"\nFitting XGBoost on embeddings ({X_train_emb.shape[1]} dims) …")
    clf.fit(X_train_emb, y_train)

    val_preds = clf.predict(X_val_emb)
    score(y_val, val_preds, label="embeddings-only")

    test_preds = clf.predict(X_test_emb)
    save_predictions(test_df, test_preds, "outputs/predictions_emb.csv")


if __name__ == "__main__":
    main()
