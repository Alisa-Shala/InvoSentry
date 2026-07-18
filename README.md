# InvoSentry

**Sistem Multi-Agjentësh për Analizën e Riskut të Mashtrimit në Faturat përmes RAG**
Punim Diplome BSc — Inxhinieri Kompjuterike dhe Softuerike, FIEK/UP
Mentor: Prof. Avni Rexhepi

## Arkitektura

```
Invoice --> [Extraction Agent] --> [Validation Agent] --> [Risk-Scoring Agent] --> RiskAssessment
                                                                  ^
                                                                  |
                                                        [RAG: FAISS retriever
                                                         mbi politika + fraud patterns]
```

Orkestrimi bëhet me **LangGraph** (`src/orchestrator.py`), ku çdo agjent është
një node i pavarur dhe i testueshëm. Për krahasim akademik ekziston edhe një
**baseline** (`src/baseline.py`): një thirrje e vetme LLM, pa agjentë, pa RAG,
pa rregulla validimi — për të matur sasior vlerën e shtuar të arkitekturës
multi-agjentëshe (shih `src/evaluate.py`).

## Struktura e projektit

```
invosentry/
├── app.py                     # Streamlit demo UI
├── requirements.txt
├── .env.example
├── src/
│   ├── schemas.py             # Pydantic models (Invoice, RiskAssessment, ...)
│   ├── data_loader.py         # dataset sintetik + loader për Kaggle CSV
│   ├── llm_client.py          # wrapper mbi Anthropic API
│   ├── orchestrator.py        # LangGraph pipeline
│   ├── baseline.py            # sistemi baseline (single LLM call)
│   ├── evaluate.py            # precision/recall/F1 + error analysis
│   ├── agents/
│   │   ├── extraction_agent.py
│   │   ├── validation_agent.py
│   │   └── risk_scoring_agent.py
│   └── rag/
│       └── retriever.py       # FAISS + sentence-transformers
├── data/
│   ├── policies/policy_docs.py   # dokumentet e "njohurive" për RAG
│   └── kaggle_raw/               # <-- vendos CSV-në reale të Kaggle këtu
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── results/                   # output i evaluate.py (krijohet automatikisht)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# vendos ANTHROPIC_API_KEY në .env
```

## Përdorimi

**Demo interaktiv (Streamlit):**
```bash
streamlit run app.py
```

**Evaluation — precision/recall/F1, pipeline vs. baseline:**
```bash
python -m src.evaluate --n-legit 30 --n-fraud 20 --system both
```
Rezultatet ruhen te `results/evaluation_results.json`, duke përfshirë listën e
false positives / false negatives — kjo është baza për kapitullin "Error
Analysis and Limitations" të punimit.

**Me Docker:**
```bash
docker compose -f docker/docker-compose.yml up --build
```

## Kalimi te dataset-i real i Kaggle

Aktualisht projekti xhiron me një dataset **sintetik** (`src/data_loader.py:generate_sample_dataset`)
që simulon 4 lloje mashtrimi: duplicate, phantom vendor, price inflation, total mismatch —
kjo mundëson zhvillim dhe demo pa pritur shkarkimin e dataset-it real.

Për të kaluar te **Procurement Invoice Fraud Dataset (Kaggle, Tokelo)**:

1. Shkarko dataset-in nga Kaggle (45,008 files, ~2GB) dhe vendos CSV-në e metadata-ve te
   `data/kaggle_raw/invoices.csv`
2. Hap `src/data_loader.py`, kontrollo `COLUMN_MAP` në krye të file-it dhe përshtate
   emrat e kolonave me ato reale të CSV-së
3. Në `src/evaluate.py` dhe `app.py`, zëvendëso `generate_sample_dataset(...)` me
   `load_kaggle_dataset()`
4. Rrjedha e mbetur (agjentët, RAG, orchestrator, evaluation) **nuk ndryshon fare** —
   kjo është arsyeja pse `Invoice` schema (`src/schemas.py`) qëndron si shtresë abstraksioni
   mes dataset-it dhe pipeline-it

## Komponentët që e bëjnë këtë punë diplome (jo thjesht app)

- ✅ **Pipeline multi-agjentësh** — Extraction / Validation / Risk-Scoring, LangGraph
- ✅ **RAG** — FAISS + sentence-transformers mbi politika kompanie & fraud patterns
- ✅ **Baseline krahasues** — single-LLM-call vs. sistemi i plotë, të njëjtin model
- ✅ **Dataset i etiketuar** — sintetik tani, i zëvendësueshëm me Kaggle
- ✅ **Metrika** — precision/recall/F1/accuracy/confusion matrix (`sklearn`)
- ⏳ **Error analysis** — infrastruktura ekziston (`false_positives`/`false_negatives`
  në `evaluation_results.json`); analiza narrative shkruhet në vetë punimin (Kapitulli 6)

## Shënim mbi përdorimin e IA-së (për deklaratën në punim)

Skeleti fillestar i kodit (struktura e projektit, agjentët, RAG layer, orchestrator,
evaluation harness) u gjenerua me ndihmën e Claude. Kjo duhet deklaruar në formularin
e "Deklaratës mbi origjinalitetin" të punimit (shih template-in FIEK), duke specifikuar
platformën dhe pjesët ku është përdorur.
