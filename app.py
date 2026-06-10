"""
FastAPI wrapper — the Railway endpoint.

POST /predict   : upload 1 or 2 CSV files → predictions + optional evaluation
POST /evaluate  : upload predictions CSV + labels CSV → F1 score
GET  /health    : Railway health check
GET  /          : usage instructions
"""

import io
import json
import pandas as pd
from typing import List, Optional, Tuple
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from agent import run
from tools import check_llm

app = FastAPI(
    title="Airbnb Price Tier Predictor",
    description=(
        "LangGraph agent that predicts NYC Airbnb price tiers (0=Budget, 1=Standard, "
        "2=Premium, 3=Ultra-Luxury). Handles messy CSVs, altered column names, "
        "multilingual descriptions, and missing data."
    ),
)

TARGET_COL = "price_tier"
ID_COL     = "property_id"


# ── File parsing helpers ───────────────────────────────────────────────────────

def _read_csv(upload: UploadFile) -> pd.DataFrame:
    raw = upload.file.read()
    return pd.read_csv(io.BytesIO(raw))


def _detect_inputs(dfs: List[pd.DataFrame]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Given 1 or 2 DataFrames, return (data_df, labels_df_or_None).

    Rules:
      1 file, has price_tier + many cols → combined: split labels off
      1 file, no price_tier             → data only
      2 files: whichever has price_tier (and few cols) is labels;
               the other is data. If both have price_tier, the one
               with more columns is the combined data file.
    """
    if len(dfs) == 1:
        df = dfs[0]
        if TARGET_COL in df.columns:
            labels = df[[ID_COL, TARGET_COL]].copy() if ID_COL in df.columns else df[[TARGET_COL]].copy()
            data   = df.drop(columns=[TARGET_COL])
            return data, labels
        return df, None

    df1, df2 = dfs
    has1 = TARGET_COL in df1.columns
    has2 = TARGET_COL in df2.columns

    if has1 and not has2:
        labels, data = df1, df2
    elif has2 and not has1:
        labels, data = df2, df1
    elif has1 and has2:
        # both have price_tier — richer one is the combined data file
        if len(df1.columns) >= len(df2.columns):
            data   = df1.drop(columns=[TARGET_COL])
            labels = df2
        else:
            data   = df2.drop(columns=[TARGET_COL])
            labels = df1
    else:
        # neither has price_tier — use richer file as data, no labels
        data   = df1 if len(df1.columns) >= len(df2.columns) else df2
        labels = None

    if labels is not None:
        keep = [c for c in [ID_COL, TARGET_COL] if c in labels.columns]
        labels = labels[keep]

    return data, labels


def _compute_f1(predictions_json: str, labels_df: pd.DataFrame) -> dict:
    from sklearn.metrics import f1_score, classification_report
    TIER_NAMES = ["Budget", "Standard", "Premium", "Ultra-Luxury"]

    preds_df = pd.DataFrame(json.loads(predictions_json))
    merged   = preds_df.merge(labels_df.rename(columns={TARGET_COL: "true_tier"}),
                               on=ID_COL, how="inner")
    if merged.empty:
        return {"error": "No matching property_id between predictions and labels."}

    y_true = merged["true_tier"].tolist()
    y_pred = merged[TARGET_COL].tolist()
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    return {
        "macro_f1": round(macro_f1, 4),
        "n_evaluated": len(merged),
        "report": classification_report(
            y_true, y_pred,
            labels=[0, 1, 2, 3], target_names=TIER_NAMES,
            zero_division=0,
        ),
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "endpoints": {
            "POST /predict": (
                "Upload 1 or 2 CSV files. "
                "If the file contains a price_tier column (or you upload a separate labels file), "
                "the response includes accuracy metrics. "
                "If no labels are found, upload them via POST /evaluate."
            ),
            "POST /evaluate": "Upload predictions CSV + labels CSV to get F1 score.",
            "GET  /health":   "Health check.",
            "GET  /docs":     "Interactive Swagger UI.",
        }
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/llm-health")
def llm_health():
    """Check LLM connectivity (OpenRouter or Ollama). Use this to verify the key works."""
    result = check_llm()
    status_code = 200 if result["ok"] else 503
    return JSONResponse(content=result, status_code=status_code)


@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    """
    Accept 1 or 2 CSV files.

    - **1 file without price_tier**: predict only, prompt for labels.
    - **1 file with price_tier**: predict + evaluate automatically.
    - **2 files** (data + labels): predict + evaluate automatically.

    Always returns predictions. Evaluation is included when labels are available.
    """
    if not all(f.filename.endswith(".csv") for f in files):
        raise HTTPException(status_code=400, detail="All uploaded files must be .csv")
    if len(files) > 2:
        raise HTTPException(status_code=400, detail="Upload at most 2 CSV files.")

    try:
        dfs = [_read_csv(f) for f in files]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    data_df, labels_df = _detect_inputs(dfs)
    csv_text = data_df.to_csv(index=False)

    try:
        predictions_json = run(csv_text, labels_csv=labels_df.to_csv(index=False) if labels_df is not None else "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    preds_df = pd.DataFrame(json.loads(predictions_json))
    response = {
        "predictions":     preds_df.to_dict(orient="records"),
        "predictions_csv": preds_df.to_csv(index=False),
        "n_predictions":   len(preds_df),
    }

    if labels_df is not None:
        response["evaluation"] = _compute_f1(predictions_json, labels_df)
        response["labels_provided"] = True
    else:
        response["labels_provided"] = False
        response["message"] = (
            "Labels were not found in the uploaded file(s). "
            "POST to /evaluate with your predictions CSV and a labels CSV to compute accuracy."
        )

    return JSONResponse(content=response)


@app.post("/evaluate")
async def evaluate(
    predictions: UploadFile = File(..., description="CSV with property_id and price_tier (your predictions)"),
    labels:      UploadFile = File(..., description="CSV with property_id and price_tier (ground truth)"),
):
    """Compute F1 score for existing predictions against ground-truth labels."""
    try:
        preds_df  = _read_csv(predictions)
        labels_df = _read_csv(labels)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    preds_json = preds_df.to_json(orient="records")
    result     = _compute_f1(preds_json, labels_df)
    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
