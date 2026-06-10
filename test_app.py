"""
Local smoke-test suite for the FastAPI endpoint.

Run before every Railway deploy:
    python test_app.py

Uses FastAPI's TestClient — no server needed, runs in-process in ~30 seconds.
Tests are ordered from simplest to most adversarial (mirrors what the instructor
might throw at the endpoint on evaluation day).
"""

import sys
import io
import json
import pandas as pd
from fastapi.testclient import TestClient
from app import app

client = TestClient(app, raise_server_exceptions=False)

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    msg = f"  {status} {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    results.append((name, condition))
    return condition


def _csv_file(df: pd.DataFrame, filename="data.csv"):
    return ("files", (filename, df.to_csv(index=False).encode(), "text/csv"))


def _load_test():
    return pd.read_csv("data/test_01.csv")


def _load_train():
    return pd.read_csv("data/train_01.csv")


# ── 1. Infrastructure ─────────────────────────────────────────────────────────
print("\n=== Infrastructure ===")

r = client.get("/health")
check("GET /health returns 200",   r.status_code == 200)
check("GET /health body is ok",    r.json().get("status") == "ok")

r = client.get("/")
check("GET / returns usage info",  r.status_code == 200)


# ── 2. Happy path: data file only (no labels) ─────────────────────────────────
print("\n=== Predict — data file only ===")

test_df = _load_test()
r = client.post("/predict", files=[_csv_file(test_df, "test.csv")])
check("Status 200",                        r.status_code == 200)
if r.status_code == 200:
    body = r.json()
    check("predictions key present",      "predictions" in body)
    check("correct row count",            body.get("n_predictions") == len(test_df),
          f"got {body.get('n_predictions')}, expected {len(test_df)}")
    check("labels_provided is False",     body.get("labels_provided") is False)
    check("message prompts for labels",   "evaluate" in body.get("message", "").lower())
    preds = body.get("predictions", [])
    check("price_tier values in 0-3",     all(p["price_tier"] in [0,1,2,3] for p in preds))
    check("predictions_csv downloadable", "property_id,price_tier" in body.get("predictions_csv", ""))


# ── 3. Combined file (data + price_tier in one CSV) ───────────────────────────
print("\n=== Predict — combined file (data + labels) ===")

train_df = _load_train()
r = client.post("/predict", files=[_csv_file(train_df, "train.csv")])
check("Status 200",                    r.status_code == 200)
if r.status_code == 200:
    body = r.json()
    check("labels_provided is True",   body.get("labels_provided") is True)
    check("evaluation present",        "evaluation" in body)
    check("macro_f1 present",          "macro_f1" in body.get("evaluation", {}))
    f1 = body.get("evaluation", {}).get("macro_f1", 0)
    check(f"macro_f1 > 0.3 (sanity)", f1 > 0.3, f"got {f1}")
    check("model never saw labels",    len(body.get("predictions", [])) == len(train_df))


# ── 4. Two-file upload (data file + separate labels file) ─────────────────────
print("\n=== Predict — two files (data + labels separate) ===")

data_df   = test_df.copy()
labels_df = test_df[["property_id"]].copy()
labels_df["price_tier"] = 1   # dummy labels

r = client.post("/predict", files=[
    _csv_file(data_df,   "data.csv"),
    _csv_file(labels_df, "labels.csv"),
])
check("Status 200",                  r.status_code == 200)
if r.status_code == 200:
    body = r.json()
    check("labels_provided is True", body.get("labels_provided") is True)
    check("evaluation computed",     "macro_f1" in body.get("evaluation", {}))


# ── 5. Resilience: known column aliases ───────────────────────────────────────
print("\n=== Resilience — known column aliases (no LLM needed) ===")

alias_df = test_df.rename(columns={
    "property_id":        "id",
    "description":        "name",
    "neighbourhood_group":"borough",
    "minimum_nights":     "min_nights",
})
r = client.post("/predict", files=[_csv_file(alias_df, "alias.csv")])
check("Status 200",                  r.status_code == 200)
if r.status_code == 200:
    body = r.json()
    check("correct row count",       body.get("n_predictions") == len(test_df))
    check("valid predictions",       all(p["price_tier"] in [0,1,2,3]
                                         for p in body.get("predictions", [])))


# ── 6. Resilience: missing columns → should use defaults, not crash ───────────
print("\n=== Resilience — missing columns ===")

sparse_df = test_df[["property_id", "description", "latitude", "longitude"]].copy()
r = client.post("/predict", files=[_csv_file(sparse_df, "sparse.csv")])
check("Status 200 (not 500)",        r.status_code == 200,
      f"got {r.status_code}: {r.text[:200]}")


# ── 7. Resilience: empty descriptions ─────────────────────────────────────────
print("\n=== Resilience — empty descriptions ===")

nodesc_df = test_df.copy()
nodesc_df["description"] = ""
r = client.post("/predict", files=[_csv_file(nodesc_df, "nodesc.csv")])
check("Status 200",                  r.status_code == 200)


# ── 8. Resilience: price_tier in data but only labels columns ─────────────────
print("\n=== Resilience — labels-only file + data file ===")

data_only   = test_df.copy()
labels_only = test_df[["property_id"]].copy()
labels_only["price_tier"] = 2

r = client.post("/predict", files=[
    _csv_file(labels_only, "labels.csv"),
    _csv_file(data_only,   "data.csv"),
])
check("Status 200 (files in any order)", r.status_code == 200)
if r.status_code == 200:
    check("correct row count",           r.json().get("n_predictions") == len(test_df))


# ── 9. POST /evaluate endpoint ────────────────────────────────────────────────
print("\n=== POST /evaluate ===")

preds_df          = test_df[["property_id"]].copy()
preds_df["price_tier"] = 1
eval_labels           = test_df[["property_id"]].copy()
eval_labels["price_tier"] = 2

r = client.post("/evaluate", files=[
    ("predictions", ("preds.csv",  preds_df.to_csv(index=False).encode(),  "text/csv")),
    ("labels",      ("labels.csv", eval_labels.to_csv(index=False).encode(), "text/csv")),
])
check("Status 200",              r.status_code == 200)
if r.status_code == 200:
    check("macro_f1 in response", "macro_f1" in r.json())


# ── 10. Error handling ────────────────────────────────────────────────────────
print("\n=== Error handling ===")

r = client.post("/predict", files=[("files", ("bad.txt", b"not a csv", "text/plain"))])
check("Non-CSV rejected (400)",  r.status_code == 400)

r = client.post("/predict", files=[
    _csv_file(test_df, "a.csv"),
    _csv_file(test_df, "b.csv"),
    _csv_file(test_df, "c.csv"),
])
check("3 files rejected (400)",  r.status_code == 400)


# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
total  = len(results)

print(f"\n{'='*45}")
print(f"  {passed}/{total} passed   {failed} failed")
print(f"{'='*45}")

if failed:
    print("\nFailed tests:")
    for name, ok in results:
        if not ok:
            print(f"  - {name}")
    sys.exit(1)
else:
    print("\nAll tests passed. Safe to deploy.")
    sys.exit(0)
