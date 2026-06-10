"""
LangGraph agent: resilient prediction pipeline for the Railway endpoint.

Flow:
  detect_schema → translate → infer_missing → predict → [evaluate] → END

LLM is used for at most 3 calls per request:
  1. Schema mapping   (only if unknown columns found)
  2. Translation      (only if non-English descriptions detected)
  3. Missing inference (only if key tabular cols are absent)

Evaluation node runs only when labels_csv is provided.
"""

import json
import io
import pandas as pd
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from tools import (detect_schema, translate_descriptions,
                   infer_missing_tabular, run_ensemble_predict,
                   KNOWN_ALIASES, CANONICAL_COLUMNS,
                   NUMERIC_DEFAULTS, CATEGORICAL_DEFAULTS,
                   INFERABLE_COLS, _is_likely_non_english)

MAX_RETRIES = 2


class AgentState(TypedDict):
    csv_text:        str
    labels_csv:      str
    records_json:    str
    translated_json: str
    enriched_json:   str
    predictions:     str
    evaluation:      str
    logs:            list
    error:           str
    retries:         int


# ── Nodes ─────────────────────────────────────────────────────────────────────

def node_detect_schema(state: AgentState) -> AgentState:
    logs = list(state.get("logs", []))
    try:
        # analyse input before calling tool
        df_in = pd.read_csv(io.StringIO(state["csv_text"]))
        raw_cols = [c.strip().lower().replace(" ", "_") for c in df_in.columns]
        logs.append(f"Input: {len(df_in)} rows, {len(raw_cols)} columns")

        result = detect_schema.invoke({"csv_text": state["csv_text"]})

        # report renames from known aliases
        renames = [f"'{k}' → '{v}'" for k, v in KNOWN_ALIASES.items() if k in raw_cols]
        if renames:
            logs.append(f"Column aliases resolved: {', '.join(renames)}")

        # report truly unknown cols that needed LLM
        still_unknown = [c for c in raw_cols
                         if c not in CANONICAL_COLUMNS and c not in KNOWN_ALIASES]
        if still_unknown:
            logs.append(f"LLM mapped unknown columns: {', '.join(still_unknown)}")

        # report defaults applied for missing canonical cols
        missing_cols = [c for c in {**NUMERIC_DEFAULTS, **CATEGORICAL_DEFAULTS}
                        if c not in raw_cols and c not in
                        {KNOWN_ALIASES.get(rc, rc) for rc in raw_cols}]
        if missing_cols:
            logs.append(f"Applied defaults for missing columns: {', '.join(missing_cols)}")
        else:
            logs.append("All canonical columns present — no defaults needed")

        return {**state, "records_json": result, "logs": logs, "error": ""}
    except Exception as e:
        logs.append(f"Schema detection failed: {e}")
        return {**state, "logs": logs, "error": f"schema: {e}",
                "retries": state.get("retries", 0) + 1}


def node_translate(state: AgentState) -> AgentState:
    logs = list(state.get("logs", []))
    try:
        records = json.loads(state["records_json"])
        non_en  = [i for i, r in enumerate(records)
                   if _is_likely_non_english(str(r.get("description", "")))]
        if non_en:
            logs.append(f"Detected {len(non_en)} non-English descriptions — translating via LLM")
        else:
            logs.append("All descriptions appear to be English — skipping translation")

        result = translate_descriptions.invoke({"records_json": state["records_json"]})
        return {**state, "translated_json": result, "logs": logs, "error": ""}
    except Exception as e:
        logs.append(f"Translation skipped (error: {e})")
        return {**state, "translated_json": state["records_json"], "logs": logs, "error": ""}


def node_infer_missing(state: AgentState) -> AgentState:
    logs = list(state.get("logs", []))
    try:
        records = json.loads(state["translated_json"])
        missing_rows = {
            i: [c for c in INFERABLE_COLS
                if not rec.get(c) or str(rec.get(c)) in ("Unknown", "nan", "")]
            for i, rec in enumerate(records)
        }
        missing_rows = {i: cols for i, cols in missing_rows.items() if cols}

        if missing_rows:
            all_missing_cols = sorted({c for cols in missing_rows.values() for c in cols})
            logs.append(
                f"Found {len(missing_rows)} rows with missing {', '.join(all_missing_cols)}"
                f" — inferring from description via LLM"
            )
        else:
            logs.append("No missing categorical values — skipping inference step")

        result = infer_missing_tabular.invoke({"records_json": state["translated_json"]})
        return {**state, "enriched_json": result, "logs": logs, "error": ""}
    except Exception as e:
        logs.append(f"Missing-value inference skipped (error: {e})")
        return {**state, "enriched_json": state["translated_json"], "logs": logs, "error": ""}


def node_predict(state: AgentState) -> AgentState:
    logs = list(state.get("logs", []))
    try:
        result = run_ensemble_predict.invoke({"records_json": state["enriched_json"]})
        preds  = json.loads(result)
        counts = [sum(1 for p in preds if p["price_tier"] == t) for t in range(4)]
        names  = ["Budget", "Standard", "Premium", "Ultra-Luxury"]
        dist   = ", ".join(f"{names[t]}: {counts[t]}" for t in range(4) if counts[t] > 0)
        logs.append(f"Ensemble (XGBoost 30% + HF API 70%) → {len(preds)} predictions")
        logs.append(f"Tier distribution: {dist}")
        return {**state, "predictions": result, "logs": logs, "error": ""}
    except Exception as e:
        logs.append(f"Prediction failed: {e}")
        return {**state, "logs": logs, "error": f"predict: {e}",
                "retries": state.get("retries", 0) + 1}


def node_evaluate(state: AgentState) -> AgentState:
    logs = list(state.get("logs", []))
    from sklearn.metrics import f1_score, classification_report
    TIER_NAMES = ["Budget", "Standard", "Premium", "Ultra-Luxury"]
    try:
        labels_df = pd.read_csv(io.StringIO(state["labels_csv"]))
        preds_df  = pd.DataFrame(json.loads(state["predictions"]))
        merged    = preds_df.merge(
            labels_df.rename(columns={"price_tier": "true_tier"}),
            on="property_id", how="inner"
        )
        if merged.empty:
            logs.append("Evaluation skipped — no matching property_id between predictions and labels")
            return {**state, "logs": logs,
                    "evaluation": json.dumps({"error": "No matching property_id."})}

        y_true   = merged["true_tier"].tolist()
        y_pred   = merged["price_tier"].tolist()
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        report   = classification_report(
            y_true, y_pred, labels=[0, 1, 2, 3],
            target_names=TIER_NAMES, zero_division=0,
        )
        logs.append(f"Macro F1: {macro_f1:.4f}  ({len(merged)} rows evaluated)")
        return {**state,
                "logs":       logs,
                "evaluation": json.dumps({
                    "macro_f1":    round(macro_f1, 4),
                    "n_evaluated": len(merged),
                    "report":      report,
                })}
    except Exception as e:
        logs.append(f"Evaluation error: {e}")
        return {**state, "logs": logs, "evaluation": json.dumps({"error": str(e)})}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_schema(state: AgentState) -> str:
    if state.get("error") and state.get("retries", 0) < MAX_RETRIES:
        return "detect_schema"
    if state.get("error"):
        return END
    return "translate"


def route_after_predict(state: AgentState) -> str:
    if state.get("error") and state.get("retries", 0) < MAX_RETRIES:
        return "predict"
    if state.get("error"):
        return END
    return "evaluate" if state.get("labels_csv", "").strip() else END


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_agent():
    g = StateGraph(AgentState)
    g.add_node("detect_schema", node_detect_schema)
    g.add_node("translate",     node_translate)
    g.add_node("infer_missing", node_infer_missing)
    g.add_node("predict",       node_predict)
    g.add_node("evaluate",      node_evaluate)
    g.set_entry_point("detect_schema")
    g.add_conditional_edges("detect_schema", route_after_schema)
    g.add_edge("translate",     "infer_missing")
    g.add_edge("infer_missing", "predict")
    g.add_conditional_edges("predict", route_after_predict)
    g.add_edge("evaluate", END)
    return g.compile()


agent = build_agent()


_INITIAL_STATE = lambda csv_text, labels_csv: {
    "csv_text":        csv_text,
    "labels_csv":      labels_csv,
    "records_json":    "",
    "translated_json": "",
    "enriched_json":   "",
    "predictions":     "",
    "evaluation":      "",
    "logs":            [],
    "error":           "",
    "retries":         0,
}

_STEP_LABELS = {
    "detect_schema": "Schema",
    "translate":     "Translate",
    "infer_missing": "Enrich",
    "predict":       "Predict",
    "evaluate":      "Evaluate",
}


def run(csv_text: str, labels_csv: str = "") -> dict:
    """Synchronous run. Returns {predictions, evaluation, logs}."""
    result = agent.invoke(_INITIAL_STATE(csv_text, labels_csv))
    if result.get("error"):
        raise RuntimeError(result["error"])
    return {
        "predictions": result["predictions"],
        "evaluation":  result.get("evaluation", ""),
        "logs":        result.get("logs", []),
    }


def run_stream(csv_text: str, labels_csv: str = ""):
    """
    Generator: yields SSE-ready dicts as each agent node completes.
    Types: {"type":"log","step":"Schema","msg":"..."} and
           {"type":"result","predictions":...,"evaluation":...,"logs":[...]}
    """
    prev_log_count = 0
    final_state    = None

    for chunk in agent.stream(_INITIAL_STATE(csv_text, labels_csv)):
        for node_name, updates in chunk.items():
            step_label = _STEP_LABELS.get(node_name, node_name)
            new_logs   = updates.get("logs", [])
            for entry in new_logs[prev_log_count:]:
                yield {"type": "log", "step": step_label, "msg": entry}
            prev_log_count = len(new_logs)
            final_state    = updates

    if not final_state:
        yield {"type": "error", "msg": "Agent produced no output"}
    elif final_state.get("error"):
        yield {"type": "error", "msg": final_state["error"]}
    else:
        yield {
            "type":        "result",
            "predictions": final_state.get("predictions", ""),
            "evaluation":  final_state.get("evaluation", ""),
            "logs":        final_state.get("logs", []),
        }
