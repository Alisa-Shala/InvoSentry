"""
LangGraph orchestrator.

Wires Extraction -> Validation -> Risk-Scoring into an explicit graph.
Even though this pipeline is currently linear, using LangGraph (rather
than plain function calls) is what makes this a genuine "multi-agent
system" for the thesis: state is passed through typed nodes, each node
is independently testable/replaceable, and it's straightforward to
extend later (e.g. add a conditional edge that routes to a "Human
Review" node when risk_score is borderline, or a parallel "Anomaly
Detection Agent" branch).
"""
from __future__ import annotations

from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from src.schemas import Invoice, ExtractionResult, ValidationResult, RiskAssessment, PipelinePrediction
from src.agents import extraction_agent, validation_agent, risk_scoring_agent


class PipelineState(TypedDict):
    invoice: Invoice
    extraction: Optional[ExtractionResult]
    validation: Optional[ValidationResult]
    risk: Optional[RiskAssessment]


def _extraction_node(state: PipelineState) -> PipelineState:
    state["extraction"] = extraction_agent.run(state["invoice"])
    return state


def _validation_node(state: PipelineState) -> PipelineState:
    state["validation"] = validation_agent.run(state["invoice"])
    return state


def _risk_scoring_node(state: PipelineState) -> PipelineState:
    state["risk"] = risk_scoring_agent.run(state["invoice"], state["extraction"], state["validation"])
    return state


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("extraction", _extraction_node)
    graph.add_node("validation", _validation_node)
    graph.add_node("risk_scoring", _risk_scoring_node)

    graph.set_entry_point("extraction")
    graph.add_edge("extraction", "validation")
    graph.add_edge("validation", "risk_scoring")
    graph.add_edge("risk_scoring", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_pipeline(invoice: Invoice) -> PipelinePrediction:
    graph = get_graph()
    result_state = graph.invoke({"invoice": invoice, "extraction": None, "validation": None, "risk": None})

    risk: RiskAssessment = result_state["risk"]
    predicted_label = "fraud" if risk.risk_label in ("medium", "high") else "legit"

    return PipelinePrediction(
        invoice_id=invoice.invoice_id,
        predicted_label=predicted_label,
        risk_score=risk.risk_score,
        risk_label=risk.risk_label,
        reasons=risk.reasons,
        extraction=result_state["extraction"],
        validation=result_state["validation"],
        agent_trace=risk.agent_trace,
    )
