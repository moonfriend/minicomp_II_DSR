"""
Option 1 — bert-tiny fine-tune.
prajjwal1/bert-tiny has 4.4M params vs DistilBERT's 66M: ~10x faster on CPU,
comparable quality for short listing titles.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, KFold
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
)
from utils import (load_data, split, score, save_predictions,
                   TEXT_COL, RANDOM_STATE)

MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
NUM_LABELS = 4
MAX_LEN    = 72
EPOCHS     = 10
BATCH_SIZE = 32
LR         = 3e-5

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


def main():
    train_df, test_df = load_data()

    # ── geo_tier via out-of-fold to prevent leakage ───────────────────
    knn_geo = KNeighborsClassifier(n_neighbors=15, weights='distance')
    coords_train = train_df[['latitude', 'longitude']].fillna(0).values
    labels_train = train_df['price_tier'].values
    coords_test  = test_df[['latitude', 'longitude']].fillna(0).values

    # Out-of-fold: each point predicted by KNN that never saw it
    oof_geo = cross_val_predict(
        knn_geo, coords_train, labels_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    )
    train_df['geo_tier'] = oof_geo

    # Fit on full train to predict test
    knn_geo.fit(coords_train, labels_train)
    test_df['geo_tier'] = knn_geo.predict(coords_test)

    print(f'geo_tier train dist: {train_df["geo_tier"].value_counts().sort_index().to_dict()}')

    # Leakage check
    match = (train_df['geo_tier'] == train_df['price_tier']).mean()
    print(f'geo_tier matches price_tier: {match*100:.1f}% (should be ~50-70%, not 98%)')
    # ─────────────────────────────────────────────────────────────────

    # ── split AFTER geo_tier is added ────────────────────────────────
    X_train, X_val, y_train, y_val = split(train_df)

    # ── geo_tier FIRST in text so BERT always sees it ─────────────────
    X_train = X_train.copy()
    X_val   = X_val.copy()

    X_train['combined'] = 'location_tier_' + X_train['geo_tier'].astype(str) + ' ' + X_train[TEXT_COL].fillna('')
    X_val['combined']   = 'location_tier_' + X_val['geo_tier'].astype(str)   + ' ' + X_val[TEXT_COL].fillna('')
    test_df['combined'] = 'location_tier_' + test_df['geo_tier'].astype(str)  + ' ' + test_df[TEXT_COL].fillna('')
    # ─────────────────────────────────────────────────────────────────

    y_train_list = y_train.tolist()
    y_val_list   = y_val.tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok(texts):
        return tokenizer(list(texts), truncation=True,
                         padding="max_length", max_length=MAX_LEN)

    train_ds = TextDataset(tok(X_train['combined']), y_train_list)
    val_ds   = TextDataset(tok(X_val['combined']),   y_val_list)
    test_ds  = TextDataset(tok(test_df['combined']))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    weights = compute_class_weight("balanced", classes=np.arange(NUM_LABELS),
                                   y=np.array(y_train_list))
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {MODEL_NAME}  |  Trainable params: {trainable:,}")

    args = TrainingArguments(
        output_dir="outputs/hf_tiny",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=128,
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

    print(f"Fine-tuning {MODEL_NAME} ...")
    trainer.train()

    val_preds = np.argmax(trainer.predict(val_ds).predictions, axis=1)
    score(y_val_list, val_preds, label="bert-tiny")

    test_preds = np.argmax(trainer.predict(test_ds).predictions, axis=1)
    save_predictions(test_df, test_preds, "outputs/predictions_tiny.csv")


if __name__ == "__main__":
    main()