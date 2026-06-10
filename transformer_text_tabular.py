"""
Text-tabular fusion: convert every tabular column into natural language,
concatenate with the listing description, and fine-tune a transformer
on the combined text.

Example input row →
  "Cozy Studio in the East Village. Located in East Village, Manhattan.
   Entire home/apt. Flexible stay (1 night minimum). Highly reviewed (120 reviews).
   Individual host. Rarely available (12 days/year)."

This gives the transformer all signals in one unified text — no hstacking needed.
Missing columns are simply omitted from the sentence, making it curveball-resilient.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
)
from utils import (load_data, split, score, save_predictions,
                   TEXT_COL, RANDOM_STATE)

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 4
MAX_LEN    = 128   # longer than before — we're encoding a full sentence now
EPOCHS     = 6
BATCH_SIZE = 32
LR         = 3e-5  # full fine-tune LR (small, backbone is NOT frozen here)

torch.manual_seed(RANDOM_STATE)


# ── Tabular → natural language ────────────────────────────────────────────────

def _minimum_nights_text(val) -> str:
    try:
        n = int(float(val))
    except (ValueError, TypeError):
        return ""
    if n <= 1:   return "flexible stay (1 night minimum)"
    if n <= 3:   return f"short stay ({n} night minimum)"
    if n <= 7:   return f"weekly stay ({n} night minimum)"
    if n <= 30:  return f"extended stay ({n} night minimum)"
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
    if n == 1:   return "individual host"
    if n <= 5:   return f"small host ({n} listings)"
    if n <= 20:  return f"professional host ({n} listings)"
    return f"large property manager ({n} listings)"


def row_to_text(row: dict) -> str:
    """
    Convert one listing row to a single natural-language string.
    Each field is optional — missing values are silently skipped.
    """
    parts = []

    desc = str(row.get("description", "") or "").strip()
    if desc:
        parts.append(desc)

    neighbourhood = str(row.get("neighbourhood", "") or "").strip()
    neighbourhood_group = str(row.get("neighbourhood_group", "") or "").strip()
    if neighbourhood and neighbourhood_group:
        parts.append(f"located in {neighbourhood}, {neighbourhood_group}")
    elif neighbourhood_group:
        parts.append(f"located in {neighbourhood_group}")

    room_type = str(row.get("room_type", "") or "").strip()
    if room_type:
        parts.append(room_type.lower())

    t = _minimum_nights_text(row.get("minimum_nights"))
    if t: parts.append(t)

    t = _reviews_text(row.get("number_of_reviews"))
    if t: parts.append(t)

    t = _host_listings_text(row.get("calculated_host_listings_count"))
    if t: parts.append(t)

    t = _availability_text(row.get("availability_365"))
    if t: parts.append(t)

    # TODO: convert latitude/longitude to area description (backlog)

    return ". ".join(parts) + "."


def build_texts(df: pd.DataFrame) -> list[str]:
    return [row_to_text(row) for row in df.to_dict(orient="records")]


# ── Dataset ───────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    train_df, test_df = load_data()
    X_train, X_val, y_train, y_val = split(train_df)

    y_train_list = y_train.tolist()
    y_val_list   = y_val.tolist()

    print("Converting rows to text …")
    train_texts = build_texts(X_train.reset_index(drop=True))
    val_texts   = build_texts(X_val.reset_index(drop=True))
    test_texts  = build_texts(test_df.reset_index(drop=True))

    print("Sample combined text:")
    print(" ", train_texts[0])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok(texts):
        return tokenizer(texts, truncation=True,
                         padding="max_length", max_length=MAX_LEN)

    train_ds = TextDataset(tok(train_texts), y_train_list)
    val_ds   = TextDataset(tok(val_texts),   y_val_list)
    test_ds  = TextDataset(tok(test_texts))

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    weights = compute_class_weight("balanced", classes=np.arange(NUM_LABELS),
                                   y=np.array(y_train_list))
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    args = TrainingArguments(
        output_dir="outputs/hf_text_tabular",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        learning_rate=LR,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=30,
        report_to="none",
        seed=RANDOM_STATE,
    )

    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        class_weights=class_weights,
    )

    print(f"Fine-tuning {MODEL_NAME} on combined text+tabular …")
    trainer.train()

    val_preds  = np.argmax(trainer.predict(val_ds).predictions,  axis=1)
    score(y_val_list, val_preds, label="text-tabular-distilbert")

    test_preds = np.argmax(trainer.predict(test_ds).predictions, axis=1)
    save_predictions(test_df, test_preds, "outputs/predictions_text_tabular.csv")


if __name__ == "__main__":
    main()
