"""Generate deterministic, intentionally messy finance fixtures for the mock ERP."""
from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

SCENARIOS = ["clean_match", "missing_grn", "partial_receipt", "price_variance", "vendor_variant", "duplicate_invoice", "no_po"]
VENDORS = [("Acme Supplies", "ACME Supplies Pvt Ltd"), ("Northwind Parts", "Northwind Industrial Parts"), ("Contoso Office", "Contoso Office Ltd")]
ITEMS = [("OPS-001", "Operations supplies"), ("PART-100", "Replacement parts"), ("OFF-010", "Office equipment")]

def _line(rng: random.Random, index: int, quantity: int | None = None, price: float | None = None) -> dict:
    sku, description = ITEMS[index % len(ITEMS)]
    qty = quantity if quantity is not None else rng.randint(1, 8)
    unit_price = price if price is not None else round(rng.uniform(75, 1800), 2)
    return {"item_code": sku, "description": description, "quantity": qty, "unit_price": unit_price, "total": round(qty * unit_price, 2)}

def generate(seed: int = 42, count: int = 10) -> dict[str, list[dict] | dict]:
    rng = random.Random(seed)
    count = max(0, count)
    invoices: list[dict] = []
    purchase_orders: list[dict] = []
    goods_receipts: list[dict] = []
    manifest = {"seed": seed, "count": count, "generated_at": f"seed-{seed}", "scenario_counts": {}, "scenario_ids": {}, "expected_reconciliation_status": {}}
    for i in range(count):
        scenario = SCENARIOS[i] if i < len(SCENARIOS) else "clean_match"
        vendor, variant = VENDORS[i % len(VENDORS)]
        vendor_for_invoice = variant if scenario == "vendor_variant" else vendor
        invoice_id = f"INV-S{seed:03d}-{i + 1:04d}"
        invoice_number = f"V-{seed:03d}-{1000 + i}"
        if scenario == "duplicate_invoice" and invoices:
            invoice_number = invoices[0]["invoice_number"]
            vendor = invoices[0]["vendor"]
            vendor_for_invoice = vendor
        po_id = None if scenario == "no_po" else f"PO-S{seed:03d}-{i + 1:04d}"
        grn_id = None if scenario == "missing_grn" or scenario == "no_po" else f"GRN-S{seed:03d}-{i + 1:04d}"
        line = _line(rng, i)
        po_line = dict(line)
        invoice_line = dict(line)
        received_line = dict(line)
        if scenario == "partial_receipt":
            received_line["quantity"] = max(0, line["quantity"] - 1)
            received_line["total"] = round(received_line["quantity"] * received_line["unit_price"], 2)
        if scenario == "price_variance":
            invoice_line["unit_price"] = round(line["unit_price"] * 1.15, 2)
            invoice_line["total"] = round(invoice_line["quantity"] * invoice_line["unit_price"], 2)
        po_total = po_line["total"]
        invoice_total = invoice_line["total"]
        issued = date(2026, 1, 1) + timedelta(days=i * 3)
        invoice = {
            "invoice_id": invoice_id, "invoice_number": invoice_number, "vendor": vendor_for_invoice,
            "invoice_date": issued.isoformat(), "due_date": (issued + timedelta(days=30)).isoformat(),
            "amount": invoice_total, "currency": "USD", "po_id": po_id, "items": [invoice_line],
            "source": "fixture", "status": "ready", "scenario_name": scenario, "seed": seed,
        }
        invoices.append(invoice)
        if po_id:
            purchase_orders.append({"po_id": po_id, "invoice_id": invoice_id, "vendor": vendor, "items": [po_line], "total": po_total, "currency": "USD", "status": "open", "scenario_name": scenario, "seed": seed})
        if grn_id:
            goods_receipts.append({"grn_id": grn_id, "po_id": po_id, "vendor": vendor, "items": [received_line], "status": "received", "scenario_name": scenario, "seed": seed})
        manifest["scenario_counts"][scenario] = manifest["scenario_counts"].get(scenario, 0) + 1
        manifest["scenario_ids"].setdefault(scenario, []).append(invoice_id)
        manifest["expected_reconciliation_status"][invoice_id] = "matched" if scenario in {"clean_match", "vendor_variant"} else "missing_grn" if scenario == "missing_grn" else "missing_po" if scenario == "no_po" else "exception"
    omitted = SCENARIOS[:min(len(SCENARIOS), count)]
    manifest["omitted_scenarios"] = [scenario for scenario in SCENARIOS if scenario not in omitted]
    return {"invoices": invoices, "purchase_orders": purchase_orders, "goods_receipts": goods_receipts, "manifest": manifest}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("sandbox/generated"))
    args = parser.parse_args()
    data = generate(args.seed, args.count)
    args.output.mkdir(parents=True, exist_ok=True)
    for key, filename in (("invoices", "invoices.json"), ("purchase_orders", "purchase_orders.json"), ("goods_receipts", "goods_receipts.json"), ("manifest", "scenario_manifest.json")):
        (args.output / filename).write_text(json.dumps(data[key], indent=2) + "\n", encoding="utf-8")
    print(f"Generated {args.count} scenarios in {args.output} using seed {args.seed}.")

if __name__ == "__main__":
    main()
