# Acme Ops Invoice Procedure

- Find the requested invoice by invoice ID or vendor alias.
- Record the vendor, amount, and due date exactly as shown by the invoice API.
- Invoices above 5000.00 require human approval before writing to the ledger.
- Never create a duplicate ledger entry for an invoice already recorded.
- After writing, read the ledger and verify invoice ID, amount, and due date.

## Vendor aliases

- Acme means Acme Supplies.
- Northwind means Northwind Parts.
