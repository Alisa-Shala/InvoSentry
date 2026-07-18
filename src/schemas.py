"""
Shared data models for the invoice fraud-risk pipeline.

Kept in one place so every agent, the RAG layer, the orchestrator and the
Streamlit UI all speak the same "shape" of data. This also makes it trivial
to swap the dummy/sample dataset for the real Kaggle dataset later: only
`src/data_loader.py` needs to change, nothing downstream.
"""
from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel, Field


class InvoiceLineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    amount: float


class Invoice(BaseModel):
    """Raw invoice as it comes out of the dataset / OCR layer."""

    invoice_id: str
    vendor_name: str
    vendor_tax_id: Optional[str] = None
    invoice_date: str
    due_date: Optional[str] = None
    total_amount: float
    currency: str = "EUR"
    line_items: list[InvoiceLineItem] = Field(default_factory=list)
    raw_text: Optional[str] = None  # OCR / extracted text, used for RAG queries

    # Ground-truth label, only present in labeled train/test data.
    # None for real, unlabeled invoices flowing through the pipeline live.
    label: Optional[Literal["legit", "fraud"]] = None
    fraud_type: Optional[str] = None  # e.g. "duplicate", "phantom_vendor", "price_inflation"


class ExtractionResult(BaseModel):
    invoice_id: str
    fields_extracted: dict
    missing_fields: list[str] = Field(default_factory=list)
    extraction_confidence: float = Field(ge=0.0, le=1.0)


class ValidationFinding(BaseModel):
    rule: str
    passed: bool
    detail: str


class ValidationResult(BaseModel):
    invoice_id: str
    findings: list[ValidationFinding]
    validation_score: float = Field(ge=0.0, le=1.0)  # 1.0 = fully consistent


class RetrievedContext(BaseModel):
    source: str
    snippet: str
    relevance_score: float


class RiskAssessment(BaseModel):
    invoice_id: str
    risk_score: float = Field(ge=0.0, le=1.0)  # 0 = safe, 1 = high risk
    risk_label: Literal["low", "medium", "high"]
    reasons: list[str]
    retrieved_context: list[RetrievedContext] = Field(default_factory=list)
    agent_trace: list[str] = Field(default_factory=list)  # human-readable pipeline log


class PipelinePrediction(BaseModel):
    """Final output for one invoice, used for both live inference and evaluation."""

    invoice_id: str
    predicted_label: Literal["legit", "fraud"]
    risk_score: float
    risk_label: Literal["low", "medium", "high"]
    reasons: list[str]
    extraction: ExtractionResult
    validation: ValidationResult
    agent_trace: list[str]
