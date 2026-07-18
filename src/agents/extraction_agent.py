"""
Extraction Agent

Responsible for pulling structured fields out of a raw invoice. In this
pipeline invoices already arrive semi-structured (Invoice schema), so this
agent's real job is: verify completeness, flag missing/suspicious fields,
and produce a confidence score the downstream agents can use.

In a version that ingests scanned PDFs/images, this is where OCR +
LLM-based field extraction would live (e.g. Claude with a document input).
That hook is marked below (`extract_from_raw_text`) so it's a clean place
to plug in real OCR later without touching the rest of the pipeline.
"""
from __future__ import annotations

from src.schemas import Invoice, ExtractionResult

REQUIRED_FIELDS = ["invoice_id", "vendor_name", "invoice_date", "total_amount"]


def run(invoice: Invoice) -> ExtractionResult:
    fields = invoice.model_dump()
    missing = [f for f in REQUIRED_FIELDS if not fields.get(f)]

    confidence = 1.0
    if not invoice.vendor_tax_id:
        confidence -= 0.2  # missing tax id is itself a mild extraction/quality signal
    if not invoice.line_items:
        confidence -= 0.3
    confidence -= 0.15 * len(missing)
    confidence = max(0.0, min(1.0, confidence))

    return ExtractionResult(
        invoice_id=invoice.invoice_id,
        fields_extracted={
            "vendor_name": invoice.vendor_name,
            "vendor_tax_id": invoice.vendor_tax_id,
            "invoice_date": invoice.invoice_date,
            "total_amount": invoice.total_amount,
            "line_item_count": len(invoice.line_items),
        },
        missing_fields=missing,
        extraction_confidence=confidence,
    )


def extract_from_raw_text(raw_text: str) -> dict:
    """Placeholder hook for real OCR/LLM-based extraction from scanned
    invoices. Not used while the dataset is already structured, but keeps
    the extension point explicit for the thesis's 'future work' section.
    """
    raise NotImplementedError(
        "Plug in Claude (vision) or an OCR engine here when working from "
        "scanned invoice images instead of structured records."
    )
