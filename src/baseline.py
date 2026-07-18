"""
Baseline system: a single LLM call classifies the invoice directly, with
no agent pipeline, no rule-based validation, and no RAG context.

This exists purely so the thesis can report a rigorous comparison:
"multi-agent + RAG pipeline vs. naive single-LLM-call baseline", with
precision/recall/F1 for both (see evaluate.py). Uses the SAME model
(src/llm_client.MODEL) as the risk-scoring agent, so any accuracy
difference is attributable to the pipeline architecture, not the
underlying LLM.
"""
from __future__ import annotations

from src.schemas import Invoice, PipelinePrediction, ExtractionResult, ValidationResult
from src.llm_client import call_claude_json

SYSTEM_PROMPT = """Je një sistem për zbulimin e mashtrimit në fatura biznesi. \
Të jepet një faturë e vetme, pa kontekst shtesë. Vendos nëse duket legjitime \
apo e dyshimtë (fraud).

Përgjigju VETËM me JSON, pa markdown:
{
  "risk_score": <numër 0.0-1.0>,
  "risk_label": "low" | "medium" | "high",
  "reasons": ["arsye e shkurtër", ...]
}
"""


def _build_prompt(invoice: Invoice) -> str:
    items = "\n".join(f"- {li.description}: {li.quantity} x {li.unit_price} = {li.amount}"
                       for li in invoice.line_items)
    return f"""FATURA:
ID: {invoice.invoice_id}
Shitësi: {invoice.vendor_name}
Tax ID: {invoice.vendor_tax_id or 'MUNGON'}
Data: {invoice.invoice_date}
Shuma totale: {invoice.total_amount} {invoice.currency}
Artikujt:
{items or '(pa artikuj)'}

Jep vlerësimin si JSON."""


def run_baseline(invoice: Invoice) -> PipelinePrediction:
    result = call_claude_json(SYSTEM_PROMPT, _build_prompt(invoice))
    predicted_label = "fraud" if result["risk_label"] in ("medium", "high") else "legit"

    # Baseline has no extraction/validation agents — fill with empty placeholders
    # so PipelinePrediction's shape stays uniform for evaluate.py.
    dummy_extraction = ExtractionResult(
        invoice_id=invoice.invoice_id, fields_extracted={}, missing_fields=[], extraction_confidence=1.0
    )
    dummy_validation = ValidationResult(invoice_id=invoice.invoice_id, findings=[], validation_score=1.0)

    return PipelinePrediction(
        invoice_id=invoice.invoice_id,
        predicted_label=predicted_label,
        risk_score=float(result["risk_score"]),
        risk_label=result["risk_label"],
        reasons=result["reasons"],
        extraction=dummy_extraction,
        validation=dummy_validation,
        agent_trace=["[Baseline] single LLM call, no agents, no RAG"],
    )
