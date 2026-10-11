"""Typed finance records shared by the ERP adapters and deterministic checks."""
from __future__ import annotations

from datetime import date
from typing import Literal
from pydantic import BaseModel, Field

class LineItem(BaseModel):
    item_code: str | None = None
    description: str
    quantity: float
    unit_price: float
    total: float | None = None

class InvoiceRecord(BaseModel):
    invoice_id: str
    invoice_number: str | None = None
    vendor: str = ""
    invoice_date: date | None = None
    due_date: date | None = None
    amount: float | None = None
    currency: str = "USD"
    po_id: str | None = None
    items: list[LineItem] = Field(default_factory=list)
    source: Literal["fixture", "upload", "email"] = "fixture"
    document_id: str | None = None
    status: Literal["pending_validation", "ready", "exception", "posted"] = "ready"
    scenario_name: str | None = None
    seed: int | None = None
    document_hash: str | None = None

class PurchaseOrder(BaseModel):
    po_id: str
    vendor: str
    items: list[LineItem] = Field(default_factory=list)
    total: float
    currency: str = "USD"
    status: str = "open"

class GoodsReceipt(BaseModel):
    grn_id: str
    po_id: str
    vendor: str
    items: list[LineItem] = Field(default_factory=list)
    status: str = "received"

class ReconciliationResult(BaseModel):
    status: Literal["matched", "exception", "missing_po", "missing_grn", "insufficient_evidence"]
    invoice_id: str
    po_id: str | None = None
    grn_id: str | None = None
    scenario_type: str | None = None
    checks: dict[str, bool] = Field(default_factory=dict)
    exceptions: list[str] = Field(default_factory=list)
    evidence: list[dict] = Field(default_factory=list)
