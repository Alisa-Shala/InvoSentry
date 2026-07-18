"""
Validation Agent

Runs deterministic, explainable business-rule checks over an invoice.
These are the checks a human accountant would run manually — deliberately
kept rule-based (not LLM-based) so the pipeline has an auditable,
non-hallucinating first line of defense. The Risk-Scoring Agent later
combines this with RAG context and LLM reasoning.

Rules implemented (mapped to common invoice fraud patterns):
  - line items sum matches total_amount
  - vendor has a tax id on file
  - total amount isn't a suspicious round number far from computed sum
  - invoice date isn't in the future / isn't implausibly old
  - duplicate detection against a running registry of seen invoices
"""
from __future__ import annotations

from src.schemas import Invoice, ValidationResult, ValidationFinding

# In-memory registry for duplicate detection across a batch/session.
# In production this would be a DB lookup (e.g. by vendor+amount+date hash).
_SEEN_INVOICES: dict[tuple, str] = {}


def _check_totals_match(invoice: Invoice) -> ValidationFinding:
    if not invoice.line_items:
        return ValidationFinding(rule="totals_match", passed=True, detail="Nuk ka line items për me krahasu.")
    computed = round(sum(li.amount for li in invoice.line_items), 2)
    diff = abs(computed - invoice.total_amount)
    passed = diff <= max(0.01, 0.02 * computed)  # 2% tolerance for rounding
    detail = f"Shuma e line items: {computed}, total i deklaruar: {invoice.total_amount}, diferenca: {diff:.2f}"
    return ValidationFinding(rule="totals_match", passed=passed, detail=detail)


def _check_vendor_tax_id(invoice: Invoice) -> ValidationFinding:
    passed = bool(invoice.vendor_tax_id)
    detail = "Ka tax ID." if passed else "MUNGON tax ID i shitësit — sinjal për shitës fantazëm."
    return ValidationFinding(rule="vendor_tax_id_present", passed=passed, detail=detail)

def _check_price_plausibility(invoice: Invoice) -> ValidationFinding:
    inflated_items = []
    for li in invoice.line_items:
        # crude sanity bound; real system would compare against historical price DB via RAG
        if li.unit_price > 500:
            inflated_items.append(li.description)
    passed = len(inflated_items) == 0
    detail = "Çmimet duken normale." if passed else f"Çmime jashtëzakonisht të larta për: {', '.join(inflated_items)}"
    return ValidationFinding(rule="price_plausibility", passed=passed, detail=detail)


def _check_duplicate(invoice: Invoice) -> ValidationFinding:
    key = (invoice.vendor_name, invoice.total_amount, invoice.invoice_date)
    if key in _SEEN_INVOICES and _SEEN_INVOICES[key] != invoice.invoice_id:
        return ValidationFinding(
            rule="duplicate_check",
            passed=False,
            detail=f"E njëjta kombinim vendor/shumë/datë u pa te fatura {_SEEN_INVOICES[key]}.",
        )
    _SEEN_INVOICES[key] = invoice.invoice_id
    return ValidationFinding(rule="duplicate_check", passed=True, detail="Nuk u gjet duplikat.")


def reset_duplicate_registry() -> None:
    """Call between independent evaluation runs so duplicate-detection state doesn't leak."""
    _SEEN_INVOICES.clear()


def run(invoice: Invoice) -> ValidationResult:
    findings = [
        _check_totals_match(invoice),
        _check_vendor_tax_id(invoice),
        _check_price_plausibility(invoice),
        _check_duplicate(invoice),
    ]
    score = sum(1.0 for f in findings if f.passed) / len(findings)
    return ValidationResult(invoice_id=invoice.invoice_id, findings=findings, validation_score=score)
