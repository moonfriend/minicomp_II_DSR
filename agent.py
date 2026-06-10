"""
LangGraph agent: resilient prediction pipeline for the Railway endpoint.

Flow:
  detect_schema → translate_descriptions → infer_missing_tabular → run_xgb_predict

LLM is used for at most 3 calls per request:
  1. Schema mapping  (only if unknown columns found)
  2. Translation     (only if non-English descriptions detected)
  3. Missing inference (only if key tabular cols are absent)

All nodes have error recovery edges — the agent never crashes on bad input.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from tools import (detect_schema, translate_descriptions,
                   infer_missing_tabular, run_xgb_predict)

MAX_RETRIES = 2


class AgentState(TypedDict):
    csv_text:       str
    records_json:   str
    translated_json: str
    enriched_json:  str
    predictions:    str
    error:          str
    retries:        int


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
    except Exception as e:
        # non-fatal: proceed with original descriptions
        return {**state, "translated_json": state["records_json"], "error": ""}


def node_infer_missing(state: AgentState) -> AgentState:
    try:
        result = infer_missing_tabular.invoke(
            {"records_json": state["translated_json"]}
        )
        return {**state, "enriched_json": result, "error": ""}
    except Exception as e:
        # non-fatal: proceed with defaults
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
    return END


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_agent():
    g = StateGraph(AgentState)

    g.add_node("detect_schema",   node_detect_schema)
    g.add_node("translate",       node_translate)
    g.add_node("infer_missing",   node_infer_missing)
    g.add_node("predict",         node_predict)

    g.set_entry_point("detect_schema")

    g.add_conditional_edges("detect_schema", route_after_schema)
    g.add_edge("translate",     "infer_missing")
    g.add_edge("infer_missing", "predict")
    g.add_conditional_edges("predict", route_after_predict)

    return g.compile()


agent = build_agent()


def run(csv_text: str) -> str:
    result = agent.invoke({
        "csv_text":        csv_text,
        "records_json":    "",
        "translated_json": "",
        "enriched_json":   "",
        "predictions":     "",
        "error":           "",
        "retries":         0,
    })
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result["predictions"]
