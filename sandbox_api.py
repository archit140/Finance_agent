from datetime import date
from pathlib import Path
import json
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Acme Ops Mock ERP")
DATA = Path(os.getenv("ERP_DATA_DIR", str(Path(__file__).parent / "sandbox" / "data"))).resolve()
def _load_json(name: str, default):
    path = DATA / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
INVOICES = _load_json("invoices.json", [])
PURCHASE_ORDERS = _load_json("purchase_orders.json", [])
GOODS_RECEIPTS = _load_json("goods_receipts.json", [])
PENDING_PATH = DATA / "pending_invoices.json"
PENDING_INVOICES = _load_json("pending_invoices.json", [])
LEDGER: list[dict] = []
FAIL_NEXT_WRITE = False

@app.get("/health")
def health():
    return {"status": "ok", "service": "mock-erp"}

@app.get("/ready")
def ready():
    missing = [name for name in ("invoices.json", "purchase_orders.json", "goods_receipts.json") if not (DATA / name).exists()]
    if missing:
        raise HTTPException(503, {"status": "not_ready", "data_dir": str(DATA), "missing": missing})
    return {"status": "ready", "data_dir": str(DATA)}

@app.get("/scenario")
def scenario():
    manifest = _load_json("scenario_manifest.json", {})
    return {"data_dir": str(DATA), "seed": manifest.get("seed"), "count": manifest.get("count"), "scenario_counts": manifest.get("scenario_counts", {})}


class LedgerEntry(BaseModel):
    invoice_id: str
    vendor: str
    amount: float
    due_date: date

class PendingInvoice(BaseModel):
    invoice_id: str
    invoice_number: str | None = None
    vendor: str = ""
    invoice_date: date | None = None
    due_date: date | None = None
    amount: float | None = None
    currency: str = "USD"
    po_id: str | None = None
    items: list[dict] = []
    source: str = "upload"
    document_id: str | None = None
    document_hash: str | None = None
    extraction_evidence: list[dict] = []
    confidence: float | None = None
    status: str = "pending_validation"


@app.get("/invoices")
def invoices():
    return INVOICES + PENDING_INVOICES

@app.get("/invoices/pending")
def pending_invoices():
    return PENDING_INVOICES

@app.post("/invoices/pending", status_code=201)
def create_pending_invoice(invoice: PendingInvoice):
    global PENDING_INVOICES
    if any(row.get("document_hash") and row.get("document_hash") == invoice.document_hash for row in PENDING_INVOICES):
        raise HTTPException(409, "Document has already been imported")
    if any(row.get("invoice_id") == invoice.invoice_id for row in PENDING_INVOICES):
        raise HTTPException(409, "Invoice is already pending")
    saved = invoice.model_dump(mode="json")
    PENDING_INVOICES.append(saved)
    DATA.mkdir(parents=True, exist_ok=True)
    PENDING_PATH.write_text(json.dumps(PENDING_INVOICES, indent=2) + "\n", encoding="utf-8")
    return saved

@app.get("/invoices/{invoice_id}")
def invoice(invoice_id: str):
    for item in INVOICES:
        if item["invoice_id"] == invoice_id:
            return item
    for item in PENDING_INVOICES:
        if item["invoice_id"] == invoice_id:
            return item
    raise HTTPException(404, "Invoice not found")

@app.post("/invoices/{invoice_id}/process")
def process_pending_invoice(invoice_id: str):
    for item in PENDING_INVOICES:
        if item["invoice_id"] == invoice_id:
            if item.get("status") == "exception":
                raise HTTPException(422, "Invoice requires review before processing")
            item["status"] = "ready"
            PENDING_PATH.write_text(json.dumps(PENDING_INVOICES, indent=2) + "\n", encoding="utf-8")
            return item
    raise HTTPException(404, "Pending invoice not found")

@app.get("/purchase-orders")
def purchase_orders(vendor: str | None = None, invoice_id: str | None = None):
    rows = PURCHASE_ORDERS
    if vendor:
        rows = [row for row in rows if vendor.lower() in row["vendor"].lower()]
    if invoice_id:
        rows = [row for row in rows if row.get("invoice_id") == invoice_id or row.get("po_id") == invoice_id]
    return rows

@app.get("/purchase-orders/{po_id}")
def purchase_order(po_id: str):
    for item in PURCHASE_ORDERS:
        if item["po_id"] == po_id:
            return item
    raise HTTPException(404, "Purchase order not found")

@app.get("/goods-receipts")
def goods_receipts(po_id: str | None = None, vendor: str | None = None):
    rows = GOODS_RECEIPTS
    if po_id:
        rows = [row for row in rows if row["po_id"] == po_id]
    if vendor:
        rows = [row for row in rows if vendor.lower() in row["vendor"].lower()]
    return rows

@app.get("/goods-receipts/{grn_id}")
def goods_receipt(grn_id: str):
    for item in GOODS_RECEIPTS:
        if item["grn_id"] == grn_id:
            return item
    raise HTTPException(404, "Goods receipt not found")


@app.get("/ledger")
def ledger():
    return LEDGER


@app.post("/ledger", status_code=201)
def add_ledger_entry(entry: LedgerEntry):
    global FAIL_NEXT_WRITE
    if FAIL_NEXT_WRITE:
        FAIL_NEXT_WRITE = False
        raise HTTPException(503, "Injected temporary ledger failure")
    if any(item["invoice_id"] == entry.invoice_id for item in LEDGER):
        raise HTTPException(409, "Invoice is already in the ledger")
    saved = entry.model_dump(mode="json")
    LEDGER.append(saved)
    return saved


@app.post("/reset")
def reset():
    global FAIL_NEXT_WRITE
    LEDGER.clear()
    FAIL_NEXT_WRITE = False
    return {"status": "reset"}


@app.post("/faults/next-ledger-write")
def fail_next_ledger_write():
    global FAIL_NEXT_WRITE
    FAIL_NEXT_WRITE = True
    return {"status": "next ledger write will fail once"}
