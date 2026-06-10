"""
LangGraph agent: orchestrates the three tools in a linear pipeline
with error-recovery edges so curveball inputs don't crash the app.
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from tools import normalize_columns, extract_text_features, run_xgb_predict


class AgentState(TypedDict):
    csv_text:       str
    records_json:   str
    enriched_json:  str
    predictions:    str
    error:          str
    retries:        int


MAX_RETRIES = 2


# ── Nodes ─────────────────────────────────────────────────────────────────────

def node_normalize(state: AgentState) -> AgentState:
    try:
        result = normalize_columns.invoke({"csv_text": state["csv_text"]})
        return {**state, "records_json": result, "error": ""}
    except Exception as e:
        return {**state, "error": f"normalize: {e}",
                "retries": state.get("retries", 0) + 1}


def node_extract_features(state: AgentState) -> AgentState:
    try:
        result = extract_text_features.invoke({"records_json": state["records_json"]})
        return {**state, "enriched_json": result, "error": ""}
    except Exception as e:
        # feature extraction failure is non-fatal: use records without flags
        return {**state, "enriched_json": state["records_json"], "error": ""}


def node_predict(state: AgentState) -> AgentState:
    try:
        result = run_xgb_predict.invoke({"records_json": state["enriched_json"]})
        return {**state, "predictions": result, "error": ""}
    except Exception as e:
        return {**state, "error": f"predict: {e}",
                "retries": state.get("retries", 0) + 1}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_normalize(state: AgentState) -> str:
    if state.get("error") and state.get("retries", 0) < MAX_RETRIES:
        return "normalize"          # retry
    if state.get("error"):
        return END                  # give up after MAX_RETRIES
    return "extract_features"


def route_after_predict(state: AgentState) -> str:
    if state.get("error") and state.get("retries", 0) < MAX_RETRIES:
        return "predict"            # retry
    return END


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_agent():
    g = StateGraph(AgentState)

    g.add_node("normalize",        node_normalize)
    g.add_node("extract_features", node_extract_features)
    g.add_node("predict",          node_predict)

    g.set_entry_point("normalize")

    g.add_conditional_edges("normalize",        route_after_normalize)
    g.add_edge("extract_features", "predict")
    g.add_conditional_edges("predict",          route_after_predict)

    return g.compile()


agent = build_agent()


def run(csv_text: str) -> str:
    """Run the full pipeline. Returns a predictions JSON string."""
    result = agent.invoke({
        "csv_text":      csv_text,
        "records_json":  "",
        "enriched_json": "",
        "predictions":   "",
        "error":         "",
        "retries":       0,
    })
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result["predictions"]
