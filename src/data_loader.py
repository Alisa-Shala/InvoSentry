"""
Dataset loading layer.

Two modes:

1. SAMPLE MODE (default, works out of the box, no download needed):
   Generates a small synthetic invoice dataset with realistic fraud
   patterns (duplicate invoices, phantom vendors, round-number/price
   inflation, mismatched line-item totals) so you can run and demo the
   whole pipeline immediately.

2. REAL DATASET MODE:
   Once you download the "Procurement Invoice Fraud Dataset" (Kaggle,
   Tokelo) into data/kaggle_raw/, point KAGGLE_CSV_PATH at the metadata
   CSV and this module maps its columns onto our Invoice schema. You
   will need to adjust COLUMN_MAP below once you've inspected the
   actual CSV headers (Kaggle datasets vary in column naming).

Keeping both modes side by side means the rest of the pipeline
(agents, RAG, orchestrator, evaluation) never has to change when you
switch from sample data to the real dataset.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from src.schemas import Invoice, InvoiceLineItem

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KAGGLE_CSV_PATH = DATA_DIR / "kaggle_raw" / "invoices.csv"

# Adjust once you inspect the real Kaggle CSV headers.
COLUMN_MAP = {
    "invoice_id": "invoice_id",
    "vendor_name": "vendor_name",
    "vendor_tax_id": "vendor_tax_id",
    "invoice_date": "invoice_date",
    "due_date": "due_date",
    "total_amount": "total_amount",
    "currency": "currency",
    "label": "label",  # expected values: "legit" / "fraud"
    "fraud_type": "fraud_type",
}

VENDORS_LEGIT = [
    ("Alpha Supplies Ltd", "TX-10023"),
    ("Balkan Office Solutions", "TX-88213"),
    ("Kosova Tech Distribution", "TX-55021"),
    ("Prishtina Paper Co.", "TX-30044"),
    ("NorthStar Logistics", "TX-77102"),
]

VENDORS_PHANTOM = [
    ("QuickPay Consulting LLC", None),
    ("Zenith Global Trading", None),
    ("Rapid Invoice Services", None),
]

ITEM_CATALOG = [
    ("Office chairs", 120.0),
    ("Laptop docking stations", 65.0),
    ("Printer toner cartridges", 35.0),
    ("Consulting hours", 80.0),
    ("Cleaning services (monthly)", 300.0),
    ("Network switches", 210.0),
]


def _make_line_items(n_items: int, inflate: bool = False) -> list[InvoiceLineItem]:
    items = []
    for _ in range(n_items):
        desc, unit_price = random.choice(ITEM_CATALOG)
        qty = random.randint(1, 8)
        if inflate:
            unit_price *= random.uniform(2.5, 4.0)  # price inflation fraud signal
        items.append(
            InvoiceLineItem(
                description=desc,
                quantity=qty,
                unit_price=round(unit_price, 2),
                amount=round(qty * unit_price, 2),
            )
        )
    return items


def generate_sample_dataset(n_legit: int = 60, n_fraud: int = 40, seed: int = 42) -> list[Invoice]:
    """Synthetic dataset with labeled fraud patterns for immediate demoing/testing."""
    random.seed(seed)
    invoices: list[Invoice] = []

    # --- Legit invoices ---
    for i in range(n_legit):
        vendor, tax_id = random.choice(VENDORS_LEGIT)
        items = _make_line_items(random.randint(1, 4))
        total = round(sum(it.amount for it in items), 2)
        invoices.append(
            Invoice(
                invoice_id=f"INV-L-{i:04d}",
                vendor_name=vendor,
                vendor_tax_id=tax_id,
                invoice_date=f"2026-{random.randint(1,6):02d}-{random.randint(1,28):02d}",
                due_date=None,
                total_amount=total,
                line_items=items,
                raw_text=f"Invoice from {vendor} for {len(items)} item(s), total {total} EUR.",
                label="legit",
            )
        )

    # --- Fraudulent invoices, several fraud types ---
    fraud_types = ["duplicate", "phantom_vendor", "price_inflation", "total_mismatch"]
    for i in range(n_fraud):
        fraud_type = random.choice(fraud_types)

        if fraud_type == "phantom_vendor":
            vendor, tax_id = random.choice(VENDORS_PHANTOM)  # no tax id -> red flag
            items = _make_line_items(random.randint(1, 3))
            total = round(sum(it.amount for it in items), 2)

        elif fraud_type == "price_inflation":
            vendor, tax_id = random.choice(VENDORS_LEGIT)
            items = _make_line_items(random.randint(1, 3), inflate=True)
            total = round(sum(it.amount for it in items), 2)

        elif fraud_type == "total_mismatch":
            vendor, tax_id = random.choice(VENDORS_LEGIT)
            items = _make_line_items(random.randint(1, 3))
            real_total = sum(it.amount for it in items)
            total = round(real_total * random.uniform(1.3, 1.8), 2)  # total doesn't match line items

        else:  # duplicate
            vendor, tax_id = random.choice(VENDORS_LEGIT)
            items = _make_line_items(2)
            total = round(sum(it.amount for it in items), 2)

        invoices.append(
            Invoice(
                invoice_id=f"INV-F-{i:04d}",
                vendor_name=vendor,
                vendor_tax_id=tax_id,
                invoice_date=f"2026-{random.randint(1,6):02d}-{random.randint(1,28):02d}",
                due_date=None,
                total_amount=total,
                line_items=items,
                raw_text=f"Invoice from {vendor} for {len(items)} item(s), total {total} EUR.",
                label="fraud",
                fraud_type=fraud_type,
            )
        )

    # A literal duplicate pair, for the "duplicate" fraud type to be detectable
    dup_source = [inv for inv in invoices if inv.fraud_type == "duplicate"]
    if dup_source:
        original = dup_source[0]
        duplicate = original.model_copy(update={"invoice_id": original.invoice_id + "-DUP"})
        invoices.append(duplicate)

    random.shuffle(invoices)
    return invoices


def load_kaggle_dataset(csv_path: Path = KAGGLE_CSV_PATH) -> list[Invoice]:
    """Load the real Procurement Invoice Fraud Dataset once downloaded.

    Download it from Kaggle ("Procurement Invoice Fraud Dataset" by Tokelo),
    place the metadata CSV at data/kaggle_raw/invoices.csv, then run this.
    Adjust COLUMN_MAP above if the real column names differ.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Nuk gjeta {csv_path}. Shkarko dataset-in nga Kaggle dhe vendose CSV-në aty, "
            f"ose thirr generate_sample_dataset() për me testu me data sintetike."
        )
    df = pd.read_csv(csv_path)
    invoices = []
    for _, row in df.iterrows():
        invoices.append(
            Invoice(
                invoice_id=str(row[COLUMN_MAP["invoice_id"]]),
                vendor_name=str(row[COLUMN_MAP["vendor_name"]]),
                vendor_tax_id=row.get(COLUMN_MAP["vendor_tax_id"]),
                invoice_date=str(row[COLUMN_MAP["invoice_date"]]),
                due_date=row.get(COLUMN_MAP["due_date"]),
                total_amount=float(row[COLUMN_MAP["total_amount"]]),
                currency=row.get(COLUMN_MAP["currency"], "EUR"),
                label=row.get(COLUMN_MAP["label"]),
                fraud_type=row.get(COLUMN_MAP["fraud_type"]),
            )
        )
    return invoices


def train_test_split_invoices(invoices: list[Invoice], test_ratio: float = 0.25, seed: int = 42):
    random.seed(seed)
    shuffled = invoices[:]
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


if __name__ == "__main__":
    data = generate_sample_dataset()
    print(f"Generated {len(data)} sample invoices "
          f"({sum(1 for d in data if d.label == 'fraud')} fraud, "
          f"{sum(1 for d in data if d.label == 'legit')} legit)")
