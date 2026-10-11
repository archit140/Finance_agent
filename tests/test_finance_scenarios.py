from seed_scenarios import generate
from backend.reconciliation import reconcile_invoice


def test_seed_is_reproducible_and_covers_required_scenarios():
    first = generate(42, 10)
    second = generate(42, 10)
    assert first == second
    required = {"clean_match", "missing_grn", "partial_receipt", "price_variance", "vendor_variant", "duplicate_invoice", "no_po"}
    assert required <= set(first["manifest"]["scenario_counts"])


def test_clean_match():
    result = reconcile_invoice(
        {"invoice_id": "INV-1", "vendor": "Acme Supplies", "amount": 100, "currency": "USD", "items": [{"quantity": 2, "unit_price": 50}]},
        {"po_id": "PO-1", "vendor": "Acme Supplies", "total": 100, "currency": "USD", "items": [{"quantity": 2, "unit_price": 50}]},
        {"grn_id": "GRN-1", "po_id": "PO-1", "vendor": "Acme Supplies", "currency": "USD", "items": [{"quantity": 2, "unit_price": 50}]},
    )
    assert result.status == "matched"


def test_missing_grn_and_partial_receipt_are_blocked():
    invoice = {"invoice_id": "INV-2", "vendor": "Acme Supplies", "amount": 100, "items": [{"quantity": 2, "unit_price": 50}]}
    po = {"po_id": "PO-2", "vendor": "Acme Supplies", "total": 100, "items": [{"quantity": 2, "unit_price": 50}]}
    assert reconcile_invoice(invoice, po, None).status == "missing_grn"
    grn = {"grn_id": "GRN-2", "po_id": "PO-2", "vendor": "Acme Supplies", "items": [{"quantity": 1, "unit_price": 50}]}
    assert reconcile_invoice(invoice, po, grn).status == "exception"


def test_vendor_alias_and_price_variance():
    invoice = {"invoice_id": "INV-3", "vendor": "ACME Supplies Pvt Ltd", "amount": 110, "items": [{"quantity": 1, "unit_price": 110}]}
    po = {"po_id": "PO-3", "vendor": "Acme Supplies", "total": 100, "items": [{"quantity": 1, "unit_price": 100}]}
    grn = {"grn_id": "GRN-3", "po_id": "PO-3", "vendor": "Acme Supplies", "items": [{"quantity": 1, "unit_price": 100}]}
    result = reconcile_invoice(invoice, po, grn)
    assert result.status == "exception"
    assert "amount" in result.exceptions
