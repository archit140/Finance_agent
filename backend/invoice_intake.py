"""Validated PDF invoice intake with structured LLM extraction and evidence."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
from pydantic import BaseModel, Field
from backend.config import settings
from backend.llm_provider import get_llm

MAX_BYTES = 10 * 1024 * 1024

class ExtractedInvoice(BaseModel):
    invoice_number: str | None = None
    vendor: str | None = None
    invoice_date: str | None = None
    due_date: str | None = None
    amount: float | None = None
    currency: str = "USD"
    po_id: str | None = None
    items: list[dict] = Field(default_factory=list)
    confidence: float = 0.0

def extract_invoice_document(filename: str, content: bytes) -> dict:
    if len(content) > MAX_BYTES:
        raise ValueError(f"Invoice file exceeds {MAX_BYTES // (1024 * 1024)} MB limit")
    if Path(filename).suffix.lower() != ".pdf" or not content.startswith(b"%PDF"):
        raise ValueError("Only valid PDF invoice files are accepted")
    from pypdf import PdfReader
    import io
    text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(content)).pages)
    document_hash = sha256(content).hexdigest()
    prompt = """Extract invoice fields from the document text. Do not infer missing values. Return JSON matching the schema. Use null for unknown fields. confidence must be between 0 and 1 and reflect extraction quality.
Required fields: invoice_number, vendor, invoice_date, due_date, amount, currency, po_id, items, confidence.
Document text:\n{text}"""
    extracted = (get_llm().with_structured_output(ExtractedInvoice, method="json_mode").invoke(prompt.format(text=text[:30000])))
    required_ok = bool(extracted.invoice_number and extracted.vendor and extracted.amount is not None)
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name)
    document_id = document_hash[:16]
    document_dir = settings.project_root / "documents" / "invoices" if hasattr(settings, "project_root") else Path(__file__).parents[1] / "documents" / "invoices"
    document_dir.mkdir(parents=True, exist_ok=True)
    (document_dir / f"{document_id}_{safe_name}").write_bytes(content)
    return {
        "invoice_id": f"UPL-{document_id}", "invoice_number": extracted.invoice_number, "vendor": extracted.vendor,
        "invoice_date": extracted.invoice_date, "due_date": extracted.due_date, "amount": extracted.amount,
        "currency": extracted.currency, "po_id": extracted.po_id, "items": extracted.items,
        "source": "upload", "document_id": document_id, "document_hash": document_hash,
        "extraction_evidence": [{"filename": filename, "text_excerpt": text[:2000]}],
        "confidence": extracted.confidence, "status": "pending_validation" if required_ok and extracted.confidence >= 0.7 else "exception",
    }
