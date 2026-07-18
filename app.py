"""
InvoSentry — Streamlit demo UI.

Run with: streamlit run app.py

Lets you pick a sample invoice (or paste a custom one), run it through
either the multi-agent pipeline or the baseline, and see the full
agent trace, RAG context, and risk assessment side by side. This is
what you'd project during the thesis defense.
"""
import streamlit as st
from dotenv import load_dotenv

from src.data_loader import generate_sample_dataset
from src.orchestrator import run_pipeline
from src.baseline import run_baseline
from src.schemas import Invoice, InvoiceLineItem
from src.agents.validation_agent import reset_duplicate_registry

load_dotenv()

st.set_page_config(page_title="InvoSentry — Invoice Fraud Risk", layout="wide")
st.title("🛡️ InvoSentry")
st.caption("Sistem Multi-Agjentësh për Analizën e Riskut të Mashtrimit në Faturat (RAG)")

if "dataset" not in st.session_state:
    reset_duplicate_registry()
    st.session_state.dataset = generate_sample_dataset(n_legit=15, n_fraud=15)

with st.sidebar:
    st.header("Zgjedh faturën")
    mode = st.radio("Burimi", ["Fatura sample", "Fatura custom"])

    if mode == "Fatura sample":
        options = {f"{inv.invoice_id} — {inv.vendor_name} ({inv.total_amount} EUR)": inv
                   for inv in st.session_state.dataset}
        choice = st.selectbox("Fatura", list(options.keys()))
        selected_invoice = options[choice]
        if selected_invoice.label:
            st.caption(f"Ground truth: **{selected_invoice.label}**"
                       + (f" ({selected_invoice.fraud_type})" if selected_invoice.fraud_type else ""))
    else:
        vendor = st.text_input("Emri i shitësit", "Test Vendor LLC")
        tax_id = st.text_input("Tax ID (bosh nëse mungon)", "")
        date = st.text_input("Data", "2026-07-18")
        total = st.number_input("Shuma totale", value=100.0)
        selected_invoice = Invoice(
            invoice_id="INV-CUSTOM",
            vendor_name=vendor,
            vendor_tax_id=tax_id or None,
            invoice_date=date,
            total_amount=total,
            line_items=[InvoiceLineItem(description="Custom item", quantity=1, unit_price=total, amount=total)],
            raw_text=f"Invoice from {vendor} for {total} EUR.",
        )

    system_choice = st.radio("Sistemi", ["Multi-Agent Pipeline", "Baseline (single LLM call)"])
    run_button = st.button("▶️ Analizo faturën", type="primary")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Fatura")
    st.json(selected_invoice.model_dump())

with col2:
    st.subheader("🔎 Rezultati")
    if run_button:
        with st.spinner("Duke ekzekutuar agjentët..."):
            if system_choice == "Multi-Agent Pipeline":
                prediction = run_pipeline(selected_invoice)
            else:
                prediction = run_baseline(selected_invoice)

        label_color = {"low": "green", "medium": "orange", "high": "red"}[prediction.risk_label]
        st.markdown(f"### Risk: :{label_color}[{prediction.risk_label.upper()}] "
                    f"({prediction.risk_score:.2f})")
        st.markdown(f"**Parashikimi:** `{prediction.predicted_label}`")

        st.markdown("**Arsyet:**")
        for r in prediction.reasons:
            st.markdown(f"- {r}")

        if prediction.validation.findings:
            st.markdown("**Gjetjet e validimit:**")
            for f in prediction.validation.findings:
                icon = "✅" if f.passed else "❌"
                st.markdown(f"{icon} `{f.rule}` — {f.detail}")

        with st.expander("🧠 Agent trace (log i plotë i pipeline-it)"):
            for line in prediction.agent_trace:
                st.text(line)
    else:
        st.info("Zgjedh një faturë dhe kliko 'Analizo faturën'.")

st.divider()
if st.button("📊 Shko te faqja e Evaluation (precision/recall/F1)"):
    st.info("Ekzekuto `python -m src.evaluate` nga terminali për raport të plotë krahasues "
            "pipeline vs. baseline — rezultatet ruhen te results/evaluation_results.json")
