"""
FastAPI wrapper — the Railway endpoint.

POST /predict  : upload a CSV file → returns predictions CSV
GET  /health   : Railway health check
"""

import json
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from agent import run

app = FastAPI(title="Airbnb Price Tier Predictor")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file.")

    csv_bytes = await file.read()
    csv_text  = csv_bytes.decode("utf-8", errors="replace")

    try:
        predictions_json = run(csv_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    records = json.loads(predictions_json)
    df = pd.DataFrame(records)[["property_id", "price_tier"]]

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
