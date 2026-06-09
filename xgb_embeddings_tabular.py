"""
Hybrid pipeline: sentence-transformer embeddings + tabular features → XGBoost.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from xgboost import XGBClassifier
from utils import load_data, split, score, save_predictions, TEXT_COL, RANDOM_STATE

EMBED_MODEL = "all-MiniLM-L6-v2"

NUMERIC = ["minimum_nights", "number_of_reviews", "calculated_host_listings_count",
           "availability_365", "latitude", "longitude"]
ONEHOT  = ["neighbourhood_group", "room_type"]
ORDINAL = ["neighbourhood"]

tab_preprocessor = ColumnTransformer([
    ("onehot",      OneHotEncoder(handle_unknown="ignore"), ONEHOT),
    ("ordinal",     OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ORDINAL),
    ("passthrough", "passthrough", NUMERIC),
])


def main():
    train_df, test_df = load_data()
    X_train, X_val, y_train, y_val = split(train_df)

    print(f"Loading embedding model: {EMBED_MODEL} …")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Encoding descriptions …")
    train_emb = embedder.encode(X_train[TEXT_COL].tolist(), show_progress_bar=True, batch_size=64)
    val_emb   = embedder.encode(X_val[TEXT_COL].tolist(),   show_progress_bar=True, batch_size=64)
    test_emb  = embedder.encode(test_df[TEXT_COL].tolist(), show_progress_bar=True, batch_size=64)

    print("Processing tabular features …")
    tab_train = tab_preprocessor.fit_transform(X_train[NUMERIC + ONEHOT + ORDINAL])
    tab_val   = tab_preprocessor.transform(X_val[NUMERIC + ONEHOT + ORDINAL])
    tab_test  = tab_preprocessor.transform(test_df[NUMERIC + ONEHOT + ORDINAL])

    X_train_combined = np.hstack([train_emb, tab_train])
    X_val_combined   = np.hstack([val_emb,   tab_val])
    X_test_combined  = np.hstack([test_emb,  tab_test])

    print(f"\nFitting XGBoost on {X_train_combined.shape[1]} features "
          f"({train_emb.shape[1]} emb + {tab_train.shape[1]} tabular) …")

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train_combined, y_train)

    val_preds = clf.predict(X_val_combined)
    score(y_val, val_preds, label="embeddings+tabular")

    test_preds = clf.predict(X_test_combined)
    save_predictions(test_df, test_preds, "outputs/predictions_emb_tabular.csv")


if __name__ == "__main__":
    main()
