"""
Option 2 — frozen DistilBERT backbone, trainable classification head only.

The transformer weights are locked: no backprop flows through them.
Only the pre_classifier + classifier layers (~590K params) are updated.

Speed: ~20x faster than full fine-tune — effectively training a weighted
linear probe on top of frozen BERT representations.

Extension path to tabular: extract the CLS vector from the frozen backbone
and hstack with tabular features before passing to XGBoost (see bottom).
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
                   TEXT_COL, RANDOM_STATE)

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 4
MAX_LEN    = 64
EPOCHS     = 15   # cheap to run more epochs when backbone is frozen
BATCH_SIZE = 64   # large batch affordable — no gradient through transformer
LR         = 1e-3 # higher LR fine for a linear head

torch.manual_seed(RANDOM_STATE)


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
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.CrossEntropyLoss(weight=self.class_weights)(
            outputs.logits, labels
        )
        return (loss, outputs) if return_outputs else loss


def freeze_backbone(model):
    """Freeze all transformer layers; leave only the classification head trainable."""
    for name, param in model.named_parameters():
        if "classifier" not in name and "pre_classifier" not in name:
            param.requires_grad = False


def main():
    train_df, test_df = load_data()
    X_train, X_val, y_train, y_val = split(train_df)

    y_train_list = y_train.tolist()
    y_val_list   = y_val.tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok(texts):
        return tokenizer(list(texts), truncation=True,
                         padding="max_length", max_length=MAX_LEN)

    train_ds = TextDataset(tok(X_train[TEXT_COL]), y_train_list)
    val_ds   = TextDataset(tok(X_val[TEXT_COL]),   y_val_list)
    test_ds  = TextDataset(tok(test_df[TEXT_COL]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = compute_class_weight("balanced", classes=np.arange(NUM_LABELS),
                                   y=np.array(y_train_list))
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    freeze_backbone(model)

    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {MODEL_NAME}  |  Frozen: {frozen:,}  |  Trainable: {trainable:,}")

    args = TrainingArguments(
        output_dir="outputs/hf_frozen",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=256,
        learning_rate=LR,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=20,
        report_to="none",
        seed=RANDOM_STATE,
    )

    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        class_weights=class_weights,
    )

    print(f"Training classification head only (backbone frozen) …")
    trainer.train()

    val_preds = np.argmax(trainer.predict(val_ds).predictions, axis=1)
    score(y_val_list, val_preds, label="frozen-distilbert")

    test_preds = np.argmax(trainer.predict(test_ds).predictions, axis=1)
    save_predictions(test_df, test_preds, "outputs/predictions_frozen.csv")


# ── Extension: extract CLS vectors for tabular combination ───────────────────
def extract_cls_vectors(texts, model, tokenizer, device, batch_size=64):
    """
    Run the frozen backbone and return the CLS token vector for each text.
    Use this to hstack with tabular features and feed into XGBoost:

        cls_vecs = extract_cls_vectors(descriptions, model, tokenizer, device)
        X_combined = np.hstack([cls_vecs, tabular_features])
        xgb.fit(X_combined, y)
    """
    model.eval()
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i + batch_size])
        enc = tokenizer(batch, truncation=True, padding="max_length",
                        max_length=MAX_LEN, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model.distilbert(**enc).last_hidden_state  # (B, seq, 768)
        all_vecs.append(hidden[:, 0, :].cpu().numpy())           # CLS token
    return np.vstack(all_vecs)


if __name__ == "__main__":
    main()
