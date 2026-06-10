"""
Ensemble: combine XGBoost + transformer sequence classifier by averaging
class probabilities (soft voting).

Prerequisites — run these first to produce saved models:
    python xgb_pipeline.py          → models/xgb_model.joblib + models/geo_clusterer.joblib
    python transformer_pipeline.py --strategy tiny   (or frozen / full)

Then run:
    python ensemble.py --strategy tiny
    python ensemble.py --strategy frozen --xgb_weight 0.6
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report

from utils import load_data, split, TIER_NAMES
from geo_features import GeoClusterer, add_geo_label_to_df
from transformer_pipeline import build_texts, STRATEGIES, MAX_LEN


# ── XGBoost probabilities (from saved model) ─────────────────────────────────

def get_xgb_proba(X_val_geo: "pd.DataFrame") -> np.ndarray:
    model_path = Path("models/xgb_model.joblib")
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found — run `python xgb_pipeline.py` first."
        )
    pipeline = joblib.load(model_path)
    feature_cols = [
        "minimum_nights", "number_of_reviews", "calculated_host_listings_count",
        "availability_365", "latitude", "longitude",
        "neighbourhood_group", "room_type",
        "neighbourhood", "geo_label",
    ]
    return pipeline.predict_proba(X_val_geo[feature_cols])   # (N, 4)


# ── Transformer probabilities (from saved checkpoint) ────────────────────────

def get_transformer_proba(val_df_geo: "pd.DataFrame", strategy: str) -> np.ndarray:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    cfg        = STRATEGIES[strategy]
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(cfg["output_dir"])

    if not output_dir.exists():
        raise FileNotFoundError(
            f"{output_dir} not found — run `python transformer_pipeline.py "
            f"--strategy {strategy}` first."
        )

    # find the best checkpoint (highest step number that trainer kept)
    checkpoints = sorted(
        [p for p in output_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {output_dir}")

    # use trainer_state to find the best checkpoint if available
    trainer_state = output_dir / "checkpoint-{}/trainer_state.json".format(
        max(int(p.name.split("-")[1]) for p in checkpoints)
    )
    best_ckpt = checkpoints[-1]
    try:
        import json as _json
        state_files = list(output_dir.glob("*/trainer_state.json"))
        if state_files:
            for sf in state_files:
                ts = _json.loads(sf.read_text())
                if "best_model_checkpoint" in ts and ts["best_model_checkpoint"]:
                    candidate = Path(ts["best_model_checkpoint"])
                    if candidate.exists():
                        best_ckpt = candidate
                        break
    except Exception:
        pass

    print(f"  Loading checkpoint: {best_ckpt.name}")
    texts     = build_texts(val_df_geo.reset_index(drop=True))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model     = AutoModelForSequenceClassification.from_pretrained(
        str(best_ckpt), num_labels=4, ignore_mismatched_sizes=True
    ).to(device)
    model.eval()

    all_logits = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i:i + batch_size], truncation=True,
            padding="max_length", max_length=MAX_LEN, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            all_logits.append(model(**enc).logits.cpu())

    logits = torch.cat(all_logits, dim=0)
    return torch.softmax(logits, dim=1).numpy()   # (N, 4)


# ── Ensemble ──────────────────────────────────────────────────────────────────

def ensemble_predict(xgb_proba: np.ndarray, transformer_proba: np.ndarray,
                     xgb_weight: float) -> np.ndarray:
    combined = xgb_weight * xgb_proba + (1 - xgb_weight) * transformer_proba
    return np.argmax(combined, axis=1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(strategy: str, xgb_weight: float):
    train_df, test_df = load_data()
    X_train, X_val, y_train, y_val = split(train_df)

    geo_path = Path("models/geo_clusterer.joblib")
    if geo_path.exists():
        gc = GeoClusterer.load(geo_path)
    else:
        train_rows = train_df.loc[X_train.index]
        gc = GeoClusterer().fit(train_rows)
        gc.save(geo_path)

    X_val_geo = add_geo_label_to_df(X_val, gc)

    print("Loading XGBoost probabilities …")
    xgb_proba = get_xgb_proba(X_val_geo)

    print(f"Loading transformer probabilities (strategy={strategy}) …")
    transformer_proba = get_transformer_proba(X_val_geo, strategy)

    y_val_list = y_val.tolist()

    print(f"\n--- Ensemble (xgb_weight={xgb_weight:.1f}) ---")
    preds = ensemble_predict(xgb_proba, transformer_proba, xgb_weight)
    f1    = f1_score(y_val_list, preds, average="macro")
    print(f"Macro F1 (ensemble):     {f1:.4f}")
    print(f"Macro F1 (xgb alone):   {f1_score(y_val_list, np.argmax(xgb_proba, 1), average='macro'):.4f}")
    print(f"Macro F1 (transformer): {f1_score(y_val_list, np.argmax(transformer_proba, 1), average='macro'):.4f}")
    print()
    print(classification_report(y_val_list, preds, target_names=TIER_NAMES))

    print("\nWeight sweep (xgb_weight from 0.1 → 0.9):")
    print(f"  {'xgb_w':>6}  {'F1':>8}")
    for w in np.arange(0.1, 1.0, 0.1):
        p  = ensemble_predict(xgb_proba, transformer_proba, float(w))
        f1 = f1_score(y_val_list, p, average="macro")
        print(f"  {w:>6.1f}  {f1:>8.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy",   default="tiny",
                        choices=["tiny", "frozen", "full"])
    parser.add_argument("--xgb_weight", type=float, default=0.5,
                        help="Weight for XGBoost probs (0–1). Transformer gets 1−w.")
    args = parser.parse_args()
    main(args.strategy, args.xgb_weight)
