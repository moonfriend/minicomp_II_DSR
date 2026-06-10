"""
FastAPI wrapper — the Railway endpoint.

POST /predict   : upload 1 or 2 CSV files → predictions + optional evaluation
POST /evaluate  : upload predictions CSV + labels CSV → F1 score
GET  /health    : Railway health check
GET  /          : usage instructions
"""

import io
import json
import threading
import pandas as pd
from typing import List, Optional, Tuple
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from contextlib import asynccontextmanager
from agent import run, run_stream
from tools import check_llm

# ── Model preloading ───────────────────────────────────────────────────────────
# Models are loaded once at startup in a background thread.
# /health responds immediately; /ready returns 503 until loading finishes.

_models_ready  = False
_startup_error = ""


def _preload_models():
    global _models_ready, _startup_error
    try:
        from tools import _load_xgb
        _load_xgb()
        print("[startup] XGBoost + GeoClusterer loaded.")
        _models_ready = True
        print("[startup] All models ready.")
    except Exception as e:
        _startup_error = str(e)
        print(f"[startup] Model preload failed: {e}")


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=_preload_models, daemon=True).start()
    yield


app = FastAPI(
    title="Airbnb Price Tier Predictor",
    lifespan=lifespan,
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

@app.get("/", response_class=HTMLResponse)
def root():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Airbnb Price Tier Predictor</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f1117;color:#e2e8f0;min-height:100vh;display:flex;align-items:flex-start;justify-content:center;padding:40px 16px}
  .wrap{width:100%;max-width:780px}
  h1{font-size:1.6rem;font-weight:700;margin-bottom:4px}
  .sub{color:#94a3b8;font-size:.9rem;margin-bottom:28px}
  .card{background:#1e2130;border:1px solid #2d3348;border-radius:12px;padding:24px;margin-bottom:20px}
  .drop{border:2px dashed #3d4464;border-radius:10px;padding:36px;text-align:center;cursor:pointer;transition:border-color .2s,background .2s}
  .drop:hover,.drop.over{border-color:#6366f1;background:#1a1d2e}
  .drop-icon{font-size:2.4rem;margin-bottom:10px}
  .drop p{color:#94a3b8;font-size:.9rem}
  .drop a{color:#818cf8;text-decoration:none}
  .file-list{margin-top:14px;display:flex;flex-wrap:wrap;gap:8px}
  .file-pill{background:#2d3348;border-radius:20px;padding:4px 12px;font-size:.82rem;display:flex;align-items:center;gap:6px}
  .file-pill button{background:none;border:none;color:#64748b;cursor:pointer;font-size:.9rem;padding:0;line-height:1}
  .file-pill button:hover{color:#e2e8f0}
  .btn{width:100%;padding:12px;background:#6366f1;color:#fff;border:none;border-radius:8px;font-size:1rem;font-weight:600;cursor:pointer;transition:background .2s;margin-top:16px}
  .btn:hover:not(:disabled){background:#4f46e5}
  .btn:disabled{background:#3d4464;cursor:default;color:#64748b}
  .spinner{display:inline-block;width:18px;height:18px;border:2px solid #ffffff40;border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:8px}
  @keyframes spin{to{transform:rotate(360deg)}}
  .badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:.78rem;font-weight:600}
  .b0{background:#1e3a5f;color:#60a5fa}
  .b1{background:#14532d;color:#4ade80}
  .b2{background:#4a1d96;color:#c084fc}
  .b3{background:#7f1d1d;color:#f87171}
  .metric-row{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px}
  .metric{background:#2d3348;border-radius:8px;padding:14px 20px;flex:1;min-width:120px;text-align:center}
  .metric .val{font-size:1.8rem;font-weight:700;color:#818cf8}
  .metric .lbl{font-size:.78rem;color:#64748b;margin-top:2px}
  table{width:100%;border-collapse:collapse;font-size:.85rem}
  th{text-align:left;padding:8px 10px;color:#64748b;border-bottom:1px solid #2d3348;font-weight:500}
  td{padding:8px 10px;border-bottom:1px solid #1a1d2e}
  tr:last-child td{border-bottom:none}
  .pre{background:#111420;border-radius:6px;padding:12px;font-family:monospace;font-size:.8rem;white-space:pre;overflow-x:auto;color:#94a3b8;margin-top:10px}
  .dl-btn{display:inline-block;margin-top:12px;padding:7px 16px;background:#1e3a5f;color:#60a5fa;border:1px solid #1e40af;border-radius:6px;cursor:pointer;font-size:.85rem;text-decoration:none}
  .dl-btn:hover{background:#1e40af}
  .err{color:#f87171;background:#2d1515;border:1px solid #7f1d1d;border-radius:8px;padding:12px;font-size:.9rem}
  .section-title{font-size:.85rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px}
  .tier-labels{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}
  input[type=file]{display:none}
  .hint{font-size:.78rem;color:#475569;margin-top:6px}
  .log-pane{font-family:monospace;font-size:.82rem;max-height:240px;overflow-y:auto;background:#111420;border-radius:6px;padding:10px 14px}
  .log-line{padding:3px 0;color:#94a3b8;display:flex;gap:10px;align-items:baseline;line-height:1.5}
  .log-step{color:#818cf8;font-weight:600;min-width:72px;font-size:.75rem;text-transform:uppercase;flex-shrink:0}
  .log-msg{color:#cbd5e1}
</style>
</head>
<body>
<div class="wrap">
  <h1>&#127968; Airbnb Price Tier Predictor</h1>
  <p class="sub">Upload 1 or 2 CSV files. Agent cleans the data, runs the ensemble, and shows its reasoning live.</p>

  <div class="card">
    <div class="drop" id="drop" onclick="document.getElementById('fi').click()" ondragover="ev(event,'over')" ondragleave="ev(event,'out')" ondrop="dropped(event)">
      <div class="drop-icon">&#128196;</div>
      <p>Drop CSV files here or <a href="#" onclick="event.stopPropagation();document.getElementById('fi').click()">browse</a></p>
      <p class="hint">1 file (data only or data + labels combined) &nbsp;&bull;&nbsp; 2 files (data + separate labels)</p>
    </div>
    <input type="file" id="fi" multiple accept=".csv" onchange="picked(this.files)">
    <div class="file-list" id="fl"></div>
    <button class="btn" id="btn" onclick="predict()" disabled>Run Predictions</button>
  </div>

  <div class="card" id="logCard" style="display:none">
    <div class="section-title">Agent Log</div>
    <div class="log-pane" id="logLines"></div>
  </div>

  <div id="out"></div>
</div>

<script>
let files = [];

function ev(e, s) { e.preventDefault(); document.getElementById('drop').classList.toggle('over', s==='over'); }

function dropped(e) {
  e.preventDefault();
  document.getElementById('drop').classList.remove('over');
  addFiles([...e.dataTransfer.files].filter(f => f.name.endsWith('.csv')));
}

function picked(fl) { addFiles([...fl]); }

function addFiles(newFiles) {
  newFiles.forEach(f => { if (!files.find(x => x.name===f.name)) files.push(f); });
  if (files.length > 2) files = files.slice(0,2);
  renderFiles();
}

function removeFile(name) { files = files.filter(f=>f.name!==name); renderFiles(); }

function renderFiles() {
  document.getElementById('fl').innerHTML = files.map(f =>
    `<div class="file-pill">&#128196; ${esc(f.name)} <button onclick="removeFile('${esc(f.name)}')" title="Remove">&#10005;</button></div>`
  ).join('');
  document.getElementById('btn').disabled = files.length === 0;
}

const LABELS = ['Budget','Standard','Premium','Ultra-Luxury'];
const BADGE  = ['b0','b1','b2','b3'];
function badge(t) { return `<span class="badge ${BADGE[t]}">${LABELS[t]}</span>`; }
function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function appendLog(step, msg) {
  const pane = document.getElementById('logLines');
  const line = document.createElement('div');
  line.className = 'log-line';
  line.innerHTML = `<span class="log-step">${esc(step)}</span><span class="log-msg">${esc(msg)}</span>`;
  pane.appendChild(line);
  pane.scrollTop = pane.scrollHeight;
}

function renderResult(data) {
  const out = document.getElementById('out');
  let html = '';

  if (data.evaluation) {
    let ev = data.evaluation;
    try { if (typeof ev==='string') ev = JSON.parse(ev); } catch {}
    if (ev && ev.macro_f1 !== undefined) {
      html += `<div class="card">
        <div class="section-title">Evaluation</div>
        <div class="metric-row">
          <div class="metric"><div class="val">${ev.macro_f1}</div><div class="lbl">Macro F1</div></div>
          <div class="metric"><div class="val">${ev.n_evaluated}</div><div class="lbl">Rows evaluated</div></div>
        </div>`;
      if (ev.report) html += `<div class="section-title" style="margin-top:8px">Classification Report</div><div class="pre">${esc(ev.report)}</div>`;
      html += `</div>`;
    }
  }

  let preds = data.predictions || [];
  try { if (typeof preds==='string') preds = JSON.parse(preds); } catch {}
  const preview = preds.slice(0,100);
  const counts = [0,0,0,0];
  preds.forEach(p => counts[p.price_tier] = (counts[p.price_tier]||0)+1);

  const csv = 'property_id,price_tier\\n' + preds.map(p=>`${p.property_id},${p.price_tier}`).join('\\n');

  html += `<div class="card">
    <div class="section-title">Predictions &mdash; ${preds.length} rows</div>
    <div class="tier-labels">${counts.map((c,i)=>c>0?`${badge(i)} <small style="color:#64748b;font-size:.78rem">${c}</small>`:'').join(' ')}</div>
    <table>
      <thead><tr><th>property_id</th><th>Price Tier</th></tr></thead>
      <tbody>
        ${preview.map(p=>`<tr><td>${p.property_id}</td><td>${badge(p.price_tier)}</td></tr>`).join('')}
        ${preds.length>100?`<tr><td colspan="2" style="color:#64748b;text-align:center">... and ${preds.length-100} more rows</td></tr>`:''}
      </tbody>
    </table>
    <a class="dl-btn" id="dlbtn" href="#">&#11123; Download predictions.csv</a>
  </div>`;

  out.innerHTML = html;
  document.getElementById('dlbtn').onclick = e => {
    e.preventDefault();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
    a.download = 'predictions.csv';
    a.click();
  };
}

async function predict() {
  const btn     = document.getElementById('btn');
  const logCard = document.getElementById('logCard');
  const logLines= document.getElementById('logLines');
  const out     = document.getElementById('out');

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Running&hellip;';
  out.innerHTML = '';
  logLines.innerHTML = '';
  logCard.style.display = 'block';

  const fd = new FormData();
  files.forEach(f => fd.append('files', f));

  try {
    const response = await fetch('/predict-stream', { method:'POST', body:fd });
    if (!response.ok) {
      const err = await response.json().catch(()=>({detail:'Unknown error'}));
      out.innerHTML = `<div class="err">&#9888; ${esc(err.detail||'Request failed')}</div>`;
      return;
    }

    const reader  = response.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\\n\\n');
      buffer = parts.pop();
      for (const part of parts) {
        if (!part.startsWith('data: ')) continue;
        let event;
        try { event = JSON.parse(part.slice(6)); } catch { continue; }
        if (event.type === 'log')    appendLog(event.step, event.msg);
        else if (event.type === 'result') renderResult(event);
        else if (event.type === 'error')
          out.innerHTML = `<div class="err">&#9888; ${esc(event.msg)}</div>`;
      }
    }
  } catch (err) {
    out.innerHTML = `<div class="err">&#9888; Network error: ${esc(err.message)}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Predictions';
  }
}
</script>
</body>
</html>"""


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Returns 200 when models are loaded, 503 while still loading."""
    if _models_ready:
        return {"status": "ready"}
    if _startup_error:
        return JSONResponse({"status": "error", "detail": _startup_error}, status_code=503)
    return JSONResponse({"status": "loading"}, status_code=503)


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
    csv_text   = data_df.to_csv(index=False)
    labels_csv = labels_df.to_csv(index=False) if labels_df is not None else ""

    try:
        result = run(csv_text, labels_csv=labels_csv)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    preds_df = pd.DataFrame(json.loads(result["predictions"]))
    response = {
        "predictions":     preds_df.to_dict(orient="records"),
        "predictions_csv": preds_df.to_csv(index=False),
        "n_predictions":   len(preds_df),
        "logs":            result.get("logs", []),
    }

    if labels_df is not None:
        ev_raw = result.get("evaluation", "")
        response["evaluation"]      = json.loads(ev_raw) if ev_raw else {}
        response["labels_provided"] = True
    else:
        response["labels_provided"] = False

    return JSONResponse(content=response)


@app.post("/predict-stream")
def predict_stream(files: List[UploadFile] = File(...)):
    """Same as /predict but streams agent log lines as Server-Sent Events in real time."""
    if not all(f.filename.endswith(".csv") for f in files):
        raise HTTPException(status_code=400, detail="All uploaded files must be .csv")
    if len(files) > 2:
        raise HTTPException(status_code=400, detail="Upload at most 2 CSV files.")

    try:
        dfs = [_read_csv(f) for f in files]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    data_df, labels_df = _detect_inputs(dfs)
    csv_text   = data_df.to_csv(index=False)
    labels_csv = labels_df.to_csv(index=False) if labels_df is not None else ""

    def generate():
        for event in run_stream(csv_text, labels_csv):
            if event.get("type") == "result":
                preds_raw = event.get("predictions", "")
                ev_raw    = event.get("evaluation", "")
                payload   = {
                    "type":        "result",
                    "predictions": json.loads(preds_raw) if preds_raw else [],
                    "evaluation":  json.loads(ev_raw)    if ev_raw    else None,
                    "logs":        event.get("logs", []),
                }
                yield f"data: {json.dumps(payload)}\n\n"
            else:
                yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
