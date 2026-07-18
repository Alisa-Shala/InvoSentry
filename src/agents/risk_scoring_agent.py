"""
Risk-Scoring Agent

The "brain" of the multi-agent pipeline. Takes the outputs of the
Extraction and Validation agents, retrieves relevant policy/fraud-pattern
context via RAG, and asks Claude to reason over all of it to produce a
final, explainable risk assessment.

This is deliberately NOT a black box: the prompt requires the model to
ground its reasoning in the specific validation findings and retrieved
context, and the agent_trace field preserves every intermediate step —
this traceability is exactly what a thesis committee will want to see
in the "Rasti i studimit" chapter (Chapter 5).
"""
from __future__ import annotations

from src.schemas import (
    Invoice,
    ExtractionResult,
    ValidationResult,
    RiskAssessment,
)
from src.rag.retriever import get_retriever
from src.llm_client import call_claude_json

SYSTEM_PROMPT = """Je një agjent i specializuar për vlerësimin e riskut të mashtrimit \
në fatura biznesi. Të jepen rezultatet e validimit të rregullave dhe kontekst \
nga politikat e kompanisë/fraud pattern-et (RAG). Detyra jote është të japësh \
një vlerësim risku final, të mbështetur në dëshmi konkrete.

Përgjigju VETËM me një objekt JSON, pa markdown, pa tekst shtesë, në këtë format:
{
  "risk_score": <numër 0.0-1.0>,
  "risk_label": "low" | "medium" | "high",
  "reasons": ["arsye 1 e shkurtër dhe konkrete", "arsye 2", ...]
}
"""


def _build_user_prompt(
    invoice: Invoice,
    extraction: ExtractionResult,
    validation: ValidationResult,
    context_snippets: list[str],
) -> str:
    findings_text = "\n".join(
        f"- {f.rule}: {'OK' if f.passed else 'DËSHTOI'} — {f.detail}" for f in validation.findings
    )
    context_text = "\n".join(f"- {c}" for c in context_snippets) or "(pa kontekst relevant të gjetur)"

    return f"""FATURA:
ID: {invoice.invoice_id}
Shitësi: {invoice.vendor_name}
Tax ID: {invoice.vendor_tax_id or 'MUNGON'}
Data: {invoice.invoice_date}
Shuma totale: {invoice.total_amount} {invoice.currency}
Besueshmëria e ekstraktimit: {extraction.extraction_confidence:.2f}
Fusha që mungojnë: {extraction.missing_fields or 'asnjë'}

REZULTATET E VALIDIMIT (score {validation.validation_score:.2f}/1.0):
{findings_text}

KONTEKST I RIKTHYER (RAG - politika kompanie dhe fraud patterns):
{context_text}

Bazuar në sa më sipër, jep vlerësimin final të riskut si objekt JSON."""


def run(invoice: Invoice, extraction: ExtractionResult, validation: ValidationResult) -> RiskAssessment:
    trace = [
        f"[Extraction] confidence={extraction.extraction_confidence:.2f}, missing={extraction.missing_fields}",
        f"[Validation] score={validation.validation_score:.2f}, "
        f"failed_rules={[f.rule for f in validation.findings if not f.passed]}",
    ]

    query = invoice.raw_text or f"{invoice.vendor_name} invoice {invoice.total_amount} {invoice.invoice_date}"
    retrieved = get_retriever().retrieve(query, top_k=3)
    trace.append(f"[RAG] retrieved {len(retrieved)} context chunk(s): "
                 f"{[c.source for c in retrieved]}")

    result = call_claude_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=_build_user_prompt(invoice, extraction, validation, [c.snippet for c in retrieved]),
    )
    trace.append(f"[RiskScoring/LLM] risk_score={result['risk_score']}, label={result['risk_label']}")

    return RiskAssessment(
        invoice_id=invoice.invoice_id,
        risk_score=float(result["risk_score"]),
        risk_label=result["risk_label"],
        reasons=result["reasons"],
        retrieved_context=retrieved,
        agent_trace=trace,
    )
