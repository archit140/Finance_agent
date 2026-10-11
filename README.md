# CentrAlign AI Finance Worker

CentrAlign is an autonomous finance operations worker built with LangGraph, LangChain, Groq, Streamlit, Chroma, and a FastAPI ERP adapter. The user gives one outcome in natural language. The LLM decides which tools to use, observes every result, retries recoverable failures, asks for clarification when needed, and stops safely when evidence is incomplete.

## What it does

- Answers company questions using approved-document RAG.
- Searches invoices, purchase orders, goods receipts, and the ledger.
- Reads invoice PDFs uploaded through the UI or found as email attachments.
- Performs invoice/PO/GRN reconciliation with line-item, quantity, price, currency, and vendor checks.
- Detects missing POs, missing GRNs, partial receipts, price variance, vendor aliases, and duplicate invoice numbers.
- Requires approval before policy-controlled ledger writes.
- Verifies the ledger after posting and records an audit trace.
- Drafts completion emails; sending requires a separate approval action.
- Stores short-term workflow state in LangGraph checkpoints and verified long-term memory in Chroma/SQLite.

## Architecture

```text
Streamlit UI
    │
    ▼
Unified LangGraph intake
    │
    ├── Knowledge → hybrid company RAG → grounded answer
    │
    └── Action/Hybrid → LLM tool loop
            ├── email and PDF evidence
            ├── invoice/PO/GRN lookup
            ├── deterministic reconciliation
            ├── approval-controlled proposal
            ├── ledger write and read-back verification
            └── audit and long-term memory
```

The LLM chooses the next tool after each observation. Deterministic checks are deliberately limited to financial safety controls and verification.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- A Groq API key

## Setup

```powershell
cd C:\Users\archi\OneDrive\Desktop\centr-align-papeer-worker
uv sync
Copy-Item .env.example .env
```

Edit `.env` and set `GROQ_API_KEY`. Never commit `.env` or place a real key in `.env.example`.

## Run the demo

Use two terminals.

Terminal 1 — mock ERP:

```powershell
uv run uvicorn sandbox_api:app --port 8000 --reload
```

Terminal 2 — Streamlit:

```powershell
uv run streamlit run app.py
```

Open <http://localhost:8501>.

Useful API checks:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/ready
http://127.0.0.1:8000/scenario
http://127.0.0.1:8000/invoices
http://127.0.0.1:8000/ledger
```

## Generate realistic finance scenarios

The generator is deterministic for a seed and keeps generated data separate from the default fixtures:

```powershell
uv run python seed_scenarios.py --seed 42 --count 10 --output sandbox/generated
$env:ERP_DATA_DIR=".\sandbox\generated"
uv run uvicorn sandbox_api:app --port 8000 --reload
```

The generator creates clean matches, missing GRNs, partial receipts, price variance, vendor variants, duplicate invoice numbers, and invoices without POs. Scenario expectations are recorded in `scenario_manifest.json`.

## Example requests

```text
Find invoice INV-1001, inspect its purchase order and goods receipt, reconcile it, and enter it into the ledger.

Process invoice INV-S042-0003 and explain why it cannot be posted if the received quantity is insufficient.

Search my email for the latest invoice from Northwind, inspect the PDF attachment, and reconcile it against the ERP.

What is the company policy for invoices with missing goods receipts?
```

## Uploaded invoice flow

In Streamlit, expand **Upload invoice for processing** and upload a PDF. The worker:

1. Validates the file and computes a checksum.
2. Extracts text and structured invoice fields with the LLM.
3. Saves the original document and extraction evidence.
4. Creates a `pending_validation` invoice record.
5. Lets the autonomous worker investigate matching POs and GRNs.
6. Stops safely on missing or contradictory evidence.

Uploading never posts directly to the ledger.

## Validation commands

```powershell
uv run python -m compileall -q backend app.py sandbox_api.py seed_scenarios.py
git diff --check
uv run pytest -q
```

The deterministic generator check can be run with:

```powershell
uv run python -c "from seed_scenarios import generate; assert generate(42,10) == generate(42,10); print('deterministic generation passed')"
```

## Configuration

```env
GROQ_API_KEY=replace_with_your_groq_key
GROQ_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./data/chroma
CHECKPOINT_DB=./checkpoints.db
MEMORY_DB=./memory.db
ERP_API_URL=http://127.0.0.1:8000
ERP_DATA_DIR=./sandbox/data
HTTP_TIMEOUT_SECONDS=10
MAX_ACTION_RETRIES=2
COMPANY_ID=acme_ops
```

## Project map

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI, uploads, approvals, ledger view |
| `backend/unified_graph.py` | Unified LangGraph orchestration and verification |
| `backend/tool_agent_graph.py` | LLM tool-calling/ReAct loop |
| `backend/company_rag.py` | Hybrid company retrieval and grounded answers |
| `backend/invoice_intake.py` | PDF validation and structured invoice extraction |
| `backend/reconciliation.py` | Finance evidence and three-way matching checks |
| `backend/memory.py` | Short-term/long-term memory integration |
| `sandbox_api.py` | Mock ERP API and pending-invoice endpoints |
| `seed_scenarios.py` | Reproducible finance fixture generator |
| `company_kb/` | Approved SOPs and company documents |

## Production boundary

The mock ERP and mock email provider are local adapters. A production deployment should replace them with authenticated REST/provider adapters while preserving the same tool interfaces, approval controls, idempotency, verification, audit logging, and document-governance rules.
