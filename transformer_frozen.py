"""
Option 2 — frozen DistilBERT backbone, trainable classification head only.

The transformer weights are locked: no backprop flows through them.
Only the pre_classifier + classifier layers (~590K params) are updated.

Key optimisation: CLS vectors are pre-computed ONCE before training so the
frozen backbone is never called again during the training loop — each epoch
trains only on cached numpy arrays, making it nearly instant.

Extension path to tabular: see extract_cls_vectors() at the bottom.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
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
EPOCHS     = 15
BATCH_SIZE = 64
LR         = 3e-4  # AdamW default range; 1e-3 caused unstable early training

torch.manual_seed(RANDOM_STATE)


class CachedEmbeddingDataset(Dataset):
    """Wraps pre-computed CLS vectors so the backbone is never called in training."""
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


class LinearHead(torch.nn.Module):
    """Lightweight replacement: linear probe on top of frozen CLS embeddings."""
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.pre_classifier = torch.nn.Linear(hidden_size, hidden_size)
        self.classifier     = torch.nn.Linear(hidden_size, num_labels)
        self.dropout        = torch.nn.Dropout(0.1)

    def forward(self, inputs_embeds, labels=None):
        x = torch.relu(self.pre_classifier(inputs_embeds))
        x = self.dropout(x)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
        return type("Output", (), {"loss": loss, "logits": logits})()


class WeightedHeadTrainer(Trainer):
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


def extract_cls_vectors(texts, tokenizer, backbone, device, batch_size=64):
    """Run backbone once, return (N, 768) CLS vectors. Never called during training."""
    backbone.eval()
    all_vecs = []
    texts = list(texts)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc   = tokenizer(batch, truncation=True, padding="max_length",
                          max_length=MAX_LEN, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = backbone(**enc).last_hidden_state   # (B, seq, 768)
        all_vecs.append(hidden[:, 0, :].cpu().numpy())   # CLS token
    return np.vstack(all_vecs)


def main():
    train_df, test_df = load_data()
    X_train, X_val, y_train, y_val = split(train_df)

    y_train_list = y_train.tolist()
    y_val_list   = y_val.tolist()

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load backbone to extract CLS vectors — used only here, not in training loop
    from transformers import AutoModel
    print(f"Pre-computing CLS vectors with {MODEL_NAME} …")
    backbone = AutoModel.from_pretrained(MODEL_NAME).to(device)
    backbone.eval()

    train_vecs = extract_cls_vectors(X_train[TEXT_COL], tokenizer, backbone, device)
    val_vecs   = extract_cls_vectors(X_val[TEXT_COL],   tokenizer, backbone, device)
    test_vecs  = extract_cls_vectors(test_df[TEXT_COL], tokenizer, backbone, device)
    print(f"CLS vectors: {train_vecs.shape}  (pre-computation done — backbone no longer needed)")

    # Free backbone from memory
    del backbone
    if device == "cuda":
        torch.cuda.empty_cache()

    hidden_size = train_vecs.shape[1]   # 768 for distilbert-base

    train_ds = CachedEmbeddingDataset(train_vecs, y_train_list)
    val_ds   = CachedEmbeddingDataset(val_vecs,   y_val_list)
    test_ds  = CachedEmbeddingDataset(test_vecs)

    weights       = compute_class_weight("balanced", classes=np.arange(NUM_LABELS),
                                         y=np.array(y_train_list))
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    head = LinearHead(hidden_size, NUM_LABELS).to(device)
    print(f"Training linear head ({sum(p.numel() for p in head.parameters()):,} params) …")

    args = TrainingArguments(
        output_dir="outputs/hf_frozen",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        learning_rate=LR,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        report_to="none",
        seed=RANDOM_STATE,
    )

    trainer = WeightedHeadTrainer(
        model=head, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        class_weights=class_weights,
    )
    trainer.train()

    val_preds  = np.argmax(trainer.predict(val_ds).predictions,  axis=1)
    score(y_val_list, val_preds, label="frozen-distilbert")

    test_preds = np.argmax(trainer.predict(test_ds).predictions, axis=1)
    save_predictions(test_df, test_preds, "outputs/predictions_frozen.csv")


if __name__ == "__main__":
    main()
