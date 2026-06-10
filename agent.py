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

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from tools import (detect_schema, translate_descriptions,
                   infer_missing_tabular, run_xgb_predict)

MAX_RETRIES = 2


class AgentState(TypedDict):
    csv_text:        str
    labels_csv:      str        # ground-truth labels CSV, empty string if absent
    records_json:    str
    translated_json: str
    enriched_json:   str
    predictions:     str
    evaluation:      str        # JSON evaluation result, empty if no labels
    error:           str
    retries:         int


# ── Nodes ─────────────────────────────────────────────────────────────────────

def node_detect_schema(state: AgentState) -> AgentState:
    try:
        result = detect_schema.invoke({"csv_text": state["csv_text"]})
        return {**state, "records_json": result, "error": ""}
    except Exception as e:
        return {**state, "error": f"schema: {e}",
                "retries": state.get("retries", 0) + 1}


def node_translate(state: AgentState) -> AgentState:
    try:
        result = translate_descriptions.invoke(
            {"records_json": state["records_json"]}
        )
        return {**state, "translated_json": result, "error": ""}
    except Exception:
        return {**state, "translated_json": state["records_json"], "error": ""}


def node_infer_missing(state: AgentState) -> AgentState:
    try:
        result = infer_missing_tabular.invoke(
            {"records_json": state["translated_json"]}
        )
        return {**state, "enriched_json": result, "error": ""}
    except Exception:
        return {**state, "enriched_json": state["translated_json"], "error": ""}


def node_predict(state: AgentState) -> AgentState:
    try:
        result = run_xgb_predict.invoke(
            {"records_json": state["enriched_json"]}
        )
        return {**state, "predictions": result, "error": ""}
    except Exception as e:
        return {**state, "error": f"predict: {e}",
                "retries": state.get("retries", 0) + 1}


def node_evaluate(state: AgentState) -> AgentState:
    """Compute F1 against ground-truth labels. Only runs when labels_csv is set."""
    import io, json
    import pandas as pd
    from sklearn.metrics import f1_score, classification_report

    TIER_NAMES = ["Budget", "Standard", "Premium", "Ultra-Luxury"]

    try:
        labels_df = pd.read_csv(io.StringIO(state["labels_csv"]))
        preds_df  = pd.DataFrame(json.loads(state["predictions"]))

        # align on property_id
        merged = preds_df.merge(
            labels_df.rename(columns={"price_tier": "true_tier"}),
            on="property_id", how="inner"
        )
        if merged.empty:
            return {**state, "evaluation": json.dumps(
                {"error": "No matching property_id between predictions and labels."}
            )}

        y_true   = merged["true_tier"].tolist()
        y_pred   = merged["price_tier"].tolist()
        macro_f1 = f1_score(y_true, y_pred, average="macro")

        result = {
            "macro_f1":    round(macro_f1, 4),
            "n_evaluated": len(merged),
            "report":      classification_report(y_true, y_pred, target_names=TIER_NAMES),
        }
        print(f"\n[agent] Macro F1: {macro_f1:.4f}  (n={len(merged)})")
        return {**state, "evaluation": json.dumps(result)}

    except Exception as e:
        return {**state, "evaluation": json.dumps({"error": str(e)})}


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
    # only evaluate when labels were provided
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


def run(csv_text: str, labels_csv: str = "") -> str:
    """
    Run the agent. Returns predictions JSON string.
    Pass labels_csv to also trigger evaluation (logged to stdout + stored in state).
    """
    result = agent.invoke({
        "csv_text":        csv_text,
        "labels_csv":      labels_csv,
        "records_json":    "",
        "translated_json": "",
        "enriched_json":   "",
        "predictions":     "",
        "evaluation":      "",
        "error":           "",
        "retries":         0,
    })
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result["predictions"]
