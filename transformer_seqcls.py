"""
Sequence-classification pipeline: fine-tune a transformer (DistilBERT by default)
on the `description` column to predict price_tier.

Uses the SAME split/scoring/output conventions as utils.py so the result is
directly comparable to the XGBoost pipelines and can be added to compare.py.

Install:
    pip install "transformers>=4.40" "torch" "accelerate"

Run:
    python transformer_seqcls.py
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
)
from utils import (load_data, split, score, save_predictions,
                   TEXT_COL, TARGET_COL, RANDOM_STATE, TIER_NAMES)

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 4
MAX_LEN    = 64          # listing titles are short
EPOCHS     = 6
BATCH_SIZE = 16
LR         = 2e-5

torch.manual_seed(RANDOM_STATE)


class TextDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item


class WeightedTrainer(Trainer):
    """Handle class imbalance (tier 3 is rare) with weighted cross-entropy."""
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    train_df, test_df = load_data()
    X_train, X_val, y_train, y_val = split(train_df)

    y_train = y_train.tolist()
    y_val   = y_val.tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok(texts):
        return tokenizer(list(texts), truncation=True, padding="max_length",
                         max_length=MAX_LEN)

    train_ds = TextDataset(tok(X_train[TEXT_COL]), y_train)
    val_ds   = TextDataset(tok(X_val[TEXT_COL]),   y_val)
    test_ds  = TextDataset(tok(test_df[TEXT_COL]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = compute_class_weight("balanced", classes=np.arange(NUM_LABELS),
                                   y=np.array(y_train))
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    args = TrainingArguments(
        output_dir="outputs/hf_seqcls",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        learning_rate=LR,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
        seed=RANDOM_STATE,
    )

    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        class_weights=class_weights,
    )

    print(f"Fine-tuning {MODEL_NAME} on '{TEXT_COL}' …")
    trainer.train()

    val_logits = trainer.predict(val_ds).predictions
    val_preds  = np.argmax(val_logits, axis=1)
    score(y_val, val_preds, label="transformer-seqcls")

    test_logits = trainer.predict(test_ds).predictions
    test_preds  = np.argmax(test_logits, axis=1)
    save_predictions(test_df, test_preds, "outputs/predictions_transformer.csv")


if __name__ == "__main__":
    main()
