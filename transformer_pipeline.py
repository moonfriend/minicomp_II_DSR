"""
Unified transformer pipeline for price_tier sequence classification.

Choose a strategy via CLI or by setting STRATEGY at the top:
    python transformer_pipeline.py --strategy tiny
    python transformer_pipeline.py --strategy frozen
    python transformer_pipeline.py --strategy full

Strategies
----------
tiny   : google/bert_uncased_L-2_H-128_A-2 (4.4M params), full fine-tune  ~8-12 min CPU
frozen : distilbert-base-uncased (66M), backbone frozen, head only          ~5-8  min CPU
full   : distilbert-base-uncased (66M), full fine-tune                      ~60+  min CPU

Input text is built by row_to_text() which converts every tabular column into
natural language and concatenates it with the listing description, so the model
sees all signals in one unified string.
"""

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from utils import (load_data, split, score, save_predictions,
                   TEXT_COL, RANDOM_STATE)

# ── Strategy registry ─────────────────────────────────────────────────────────

STRATEGIES = {
    "tiny": {
        "model_name":   "google/bert_uncased_L-2_H-128_A-2",
        "epochs":       10,
        "batch_size":   32,
        "lr":           5e-5,   # slightly higher for a small model on limited data
        "freeze":       False,
        "lr_scheduler": "cosine",  # decays slowly — better than linear for fine-tuning
        "warmup_ratio": 0.1,       # 10% of steps used for warmup
        "output_dir":   "outputs/hf_tiny",
        "output_csv":   "outputs/predictions_tiny.csv",
    },
    "frozen": {
        "model_name":   "distilbert-base-uncased",
        "epochs":       15,
        "batch_size":   64,
        "lr":           3e-4,
        "freeze":       True,
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.05,
        "output_dir":   "outputs/hf_frozen",
        "output_csv":   "outputs/predictions_frozen.csv",
    },
    "full": {
        "model_name":   "distilbert-base-uncased",
        "epochs":        6,
        "batch_size":   32,
        "lr":           3e-5,
        "freeze":       False,
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.1,
        "output_dir":   "outputs/hf_full",
        "output_csv":   "outputs/predictions_full.csv",
    },
}

NUM_LABELS   = 4
MAX_LEN      = 128   # fits the combined text+tabular sentence

torch.manual_seed(RANDOM_STATE)


# ── Tabular → natural language ────────────────────────────────────────────────

def _minimum_nights_text(val) -> str:
    try:
        n = int(float(val))
    except (ValueError, TypeError):
        return ""
    if n <= 1:  return "flexible stay (1 night minimum)"
    if n <= 3:  return f"short stay ({n} night minimum)"
    if n <= 7:  return f"weekly stay ({n} night minimum)"
    if n <= 30: return f"extended stay ({n} night minimum)"
    return f"long-term stay ({n} night minimum)"


def _reviews_text(val) -> str:
    try:
        n = int(float(val))
    except (ValueError, TypeError):
        return ""
    if n == 0:   return "new listing, no reviews yet"
    if n <= 5:   return f"very few reviews ({n})"
    if n <= 20:  return f"few reviews ({n})"
    if n <= 50:  return f"several reviews ({n})"
    if n <= 100: return f"many reviews ({n})"
    return f"highly reviewed ({n} reviews)"


def _availability_text(val) -> str:
    try:
        n = int(float(val))
    except (ValueError, TypeError):
        return ""
    if n <= 30:  return "rarely available, almost always booked"
    if n <= 90:  return f"occasionally available ({n} days/year)"
    if n <= 180: return f"often available ({n} days/year)"
    return f"frequently available ({n} days/year)"


def _host_listings_text(val) -> str:
    try:
        n = int(float(val))
    except (ValueError, TypeError):
        return ""
    if n == 1:  return "individual host"
    if n <= 5:  return f"small host ({n} listings)"
    if n <= 20: return f"professional host ({n} listings)"
    return f"large property manager ({n} listings)"


def row_to_text(row: dict) -> str:
    """Convert one listing row to a unified natural-language string."""
    parts = []

    desc = str(row.get("description", "") or "").strip()
    if desc:
        parts.append(desc)

    nb  = str(row.get("neighbourhood", "") or "").strip()
    nbg = str(row.get("neighbourhood_group", "") or "").strip()
    if nb and nbg:
        parts.append(f"located in {nb}, {nbg}")
    elif nbg:
        parts.append(f"located in {nbg}")

    geo = str(row.get("geo_label", "") or "").strip()
    if geo:
        parts.append(geo)

    room_type = str(row.get("room_type", "") or "").strip()
    if room_type:
        parts.append(room_type.lower())

    for fn, key in [
        (_minimum_nights_text, "minimum_nights"),
        (_reviews_text,        "number_of_reviews"),
        (_host_listings_text,  "calculated_host_listings_count"),
        (_availability_text,   "availability_365"),
    ]:
        t = fn(row.get(key))
        if t:
            parts.append(t)

    return ". ".join(parts) + "."


def build_texts(df: pd.DataFrame) -> list[str]:
    return [row_to_text(r) for r in df.to_dict(orient="records")]


# ── Datasets ──────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """For tiny and full strategies: stores token encodings."""
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item


class CachedEmbeddingDataset(Dataset):
    """For frozen strategy: stores pre-computed CLS vectors."""
    def __init__(self, embeddings: np.ndarray, labels=None):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels     = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        item = {"inputs_embeds": self.embeddings[idx]}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item


# ── Linear head for frozen strategy ──────────────────────────────────────────

class LinearHead(torch.nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.pre_classifier = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout        = torch.nn.Dropout(0.1)
        self.classifier     = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, inputs_embeds, labels=None):
        x      = torch.relu(self.pre_classifier(inputs_embeds))
        x      = self.dropout(x)
        logits = self.classifier(x)
        loss   = None
        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


# ── Trainer ───────────────────────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        loss    = torch.nn.CrossEntropyLoss(weight=self.class_weights)(
            outputs.logits, labels
        )
        return (loss, outputs) if return_outputs else loss


# ── CLS vector extraction (frozen only) ──────────────────────────────────────

def extract_cls_vectors(texts: list, tokenizer, backbone, device,
                        batch_size: int = 64) -> np.ndarray:
    backbone.eval()
    vecs = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i:i + batch_size], truncation=True,
            padding="max_length", max_length=MAX_LEN, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            hidden = backbone(**enc).last_hidden_state   # (B, seq, hidden)
        vecs.append(hidden[:, 0, :].cpu().numpy())       # CLS token
    return np.vstack(vecs)


# ── Training dispatch ─────────────────────────────────────────────────────────

def _training_args(cfg: dict) -> TrainingArguments:
    return TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=64,
        learning_rate=cfg["lr"],
        lr_scheduler_type=cfg["lr_scheduler"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=20,
        report_to="none",
        seed=RANDOM_STATE,
    )


def train_standard(cfg, train_texts, val_texts, test_texts,
                   y_train, y_val, device):
    """Shared training loop for 'tiny' and 'full' strategies."""
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    def tok(texts):
        return tokenizer(texts, truncation=True,
                         padding="max_length", max_length=MAX_LEN)

    train_ds = TextDataset(tok(train_texts), y_train)
    val_ds   = TextDataset(tok(val_texts),   y_val)
    test_ds  = TextDataset(tok(test_texts))

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=NUM_LABELS
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg['model_name']}  |  Trainable params: {trainable:,}")

    return model, train_ds, val_ds, test_ds


def train_frozen(cfg, train_texts, val_texts, test_texts,
                 y_train, y_val, device):
    """Pre-compute CLS vectors, then train only the linear head."""
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    backbone  = AutoModel.from_pretrained(cfg["model_name"]).to(device)

    print("Pre-computing CLS vectors (backbone runs once, not during training) …")
    train_vecs = extract_cls_vectors(train_texts, tokenizer, backbone, device)
    val_vecs   = extract_cls_vectors(val_texts,   tokenizer, backbone, device)
    test_vecs  = extract_cls_vectors(test_texts,  tokenizer, backbone, device)
    print(f"CLS shape: {train_vecs.shape}")

    del backbone
    if device == "cuda":
        torch.cuda.empty_cache()

    hidden_size = train_vecs.shape[1]
    model    = LinearHead(hidden_size, NUM_LABELS).to(device)
    train_ds = CachedEmbeddingDataset(train_vecs, y_train)
    val_ds   = CachedEmbeddingDataset(val_vecs,   y_val)
    test_ds  = CachedEmbeddingDataset(test_vecs)

    trainable = sum(p.numel() for p in model.parameters())
    print(f"Linear head trainable params: {trainable:,}")

    return model, train_ds, val_ds, test_ds


# ── Main ──────────────────────────────────────────────────────────────────────

def main(strategy: str):
    from pathlib import Path
    from geo_features import GeoClusterer, add_geo_label_to_df

    cfg = STRATEGIES[strategy]
    print(f"\n{'='*50}\nStrategy: {strategy.upper()}\n{'='*50}")

    train_df, test_df = load_data()
    X_train, X_val, y_train, y_val = split(train_df)

    y_train_list = y_train.tolist()
    y_val_list   = y_val.tolist()

    # fit geo clusterer on training rows (need original df for price_tier)
    geo_model_path = Path("models/geo_clusterer.joblib")
    if geo_model_path.exists():
        gc = GeoClusterer.load(geo_model_path)
        print("Loaded existing GeoClusterer.")
    else:
        train_rows_with_tier = train_df.loc[X_train.index]
        gc = GeoClusterer()
        gc.fit(train_rows_with_tier)
        gc.save(geo_model_path)

    X_train = add_geo_label_to_df(X_train, gc)
    X_val   = add_geo_label_to_df(X_val,   gc)
    test_df = add_geo_label_to_df(test_df, gc)

    print("Building combined text+tabular strings …")
    train_texts = build_texts(X_train.reset_index(drop=True))
    val_texts   = build_texts(X_val.reset_index(drop=True))
    test_texts  = build_texts(test_df.reset_index(drop=True))

    print("Sample:", train_texts[0][:120], "…")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg["freeze"]:
        model, train_ds, val_ds, test_ds = train_frozen(
            cfg, train_texts, val_texts, test_texts,
            y_train_list, y_val_list, device
        )
    else:
        model, train_ds, val_ds, test_ds = train_standard(
            cfg, train_texts, val_texts, test_texts,
            y_train_list, y_val_list, device
        )

    weights       = compute_class_weight("balanced", classes=np.arange(NUM_LABELS),
                                         y=np.array(y_train_list))
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    trainer = WeightedTrainer(
        model=model,
        args=_training_args(cfg),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        class_weights=class_weights,
    )

    trainer.train()

    val_preds  = np.argmax(trainer.predict(val_ds).predictions,  axis=1)
    score(y_val_list, val_preds, label=strategy)

    test_preds = np.argmax(trainer.predict(test_ds).predictions, axis=1)
    save_predictions(test_df, test_preds, cfg["output_csv"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        default="tiny",
        help="tiny | frozen | full  (default: tiny)",
    )
    args = parser.parse_args()
    main(args.strategy)
