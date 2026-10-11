"""Deterministic three-way invoice, PO and goods-receipt reconciliation."""
from __future__ import annotations
import re
from backend.finance_models import ReconciliationResult

VENDOR_ALIASES = {"acmesuppliespvtltd": "acmesupplies", "northwindindustrialparts": "northwindparts"}

def _norm(value: str) -> str:
    raw = re.sub(r"[^a-z0-9]", "", (value or "").lower())
    return VENDOR_ALIASES.get(raw, raw)

def reconcile_invoice(invoice: dict, po: dict | None, grn: dict | None, *, duplicate_count: int = 1, tolerance: float = 0.01) -> ReconciliationResult:
    invoice_id = str(invoice.get("invoice_id", ""))
    if not invoice_id or not invoice.get("vendor") or invoice.get("amount") is None:
        return ReconciliationResult(status="insufficient_evidence", invoice_id=invoice_id, scenario_type=invoice.get("scenario_name"), exceptions=["Invoice lacks id, vendor, or amount."])
    if po is None:
        return ReconciliationResult(status="missing_po", invoice_id=invoice_id, scenario_type=invoice.get("scenario_name"), exceptions=["No purchase order was found for this invoice."], evidence=[{"invoice": invoice}])
    if grn is None:
        return ReconciliationResult(status="missing_grn", invoice_id=invoice_id, po_id=po.get("po_id"), scenario_type=invoice.get("scenario_name"), exceptions=["No goods receipt was found for this purchase order."], evidence=[{"invoice": invoice}, {"purchase_order": po}])
    invoice_items = invoice.get("items") or []
    po_items = po.get("items") or []
    grn_items = grn.get("items") or []
    item_checks = bool(invoice_items and po_items and grn_items)
    if item_checks:
        invoice_item, po_item, grn_item = invoice_items[0], po_items[0], grn_items[0]
        quantity_ok = float(invoice_item.get("quantity", 0)) <= float(grn_item.get("quantity", 0))
        price_ok = abs(float(invoice_item.get("unit_price", 0)) - float(po_item.get("unit_price", 0))) <= tolerance
        item_checks = quantity_ok and price_ok
    else:
        quantity_ok = price_ok = True
    checks = {
        "vendor": _norm(invoice.get("vendor")) == _norm(po.get("vendor")) == _norm(grn.get("vendor")),
        "amount": abs(float(invoice.get("amount", 0)) - float(po.get("total", 0))) <= tolerance,
        "currency": (invoice.get("currency", "USD") == po.get("currency", "USD") == grn.get("currency", "USD")),
        "po_reference": not invoice.get("po_id") or invoice.get("po_id") == po.get("po_id"),
        "receipt_reference": grn.get("po_id") == po.get("po_id"),
        "quantity": quantity_ok,
        "unit_price": price_ok,
        "duplicate": duplicate_count <= 1,
    }
    exceptions = [name for name, passed in checks.items() if not passed]
    if not item_checks and (invoice_items or po_items or grn_items):
        exceptions.append("Line-item evidence is incomplete.")
    return ReconciliationResult(status="matched" if not exceptions else "exception", invoice_id=invoice_id, po_id=po.get("po_id"), grn_id=grn.get("grn_id"), scenario_type=invoice.get("scenario_name"), checks=checks, exceptions=exceptions, evidence=[{"invoice": invoice}, {"purchase_order": po}, {"goods_receipt": grn}])
