"""Native LangChain tool-calling loop for action and hybrid requests."""

from __future__ import annotations

import json
import requests
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from backend.config import settings
from backend.llm_provider import get_llm
from backend.vector_store import hybrid_search_company
from backend.reconciliation import reconcile_invoice
from backend.email_provider import MockEmailProvider
from backend.http_client import request as http_request
from pathlib import Path

_email_provider = MockEmailProvider(Path(__file__).parents[1] / "sandbox" / "data" / "emails.json")


@tool
def list_invoices() -> str:
    """Read the current invoice register from the Acme Operations ERP."""
    response = http_request("GET", f"{settings.erp_api_url}/invoices")
    response.raise_for_status()
    return response.text

@tool
def get_invoice(invoice_id: str) -> str:
    """Retrieve one exact invoice by its internal ID. Never substitute a similar invoice."""
    try:
        response = http_request("GET", f"{settings.erp_api_url}/invoices/{invoice_id}")
        if response.status_code == 404:
            return json.dumps({"status": "not_found", "invoice_id": invoice_id, "message": "No invoice with this exact internal ID exists."})
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        return json.dumps({"status": "unavailable", "invoice_id": invoice_id, "error": str(exc), "retryable": True})


@tool
def list_ledger() -> str:
    """Read the current Acme Operations ledger for verification."""
    response = http_request("GET", f"{settings.erp_api_url}/ledger")
    response.raise_for_status()
    return response.text

@tool
def list_purchase_orders(vendor: str = "", invoice_id: str = "") -> str:
    """Find purchase orders by vendor or invoice id before posting an invoice."""
    params = {key: value for key, value in {"vendor": vendor, "invoice_id": invoice_id}.items() if value}
    response = http_request("GET", f"{settings.erp_api_url}/purchase-orders", params=params)
    response.raise_for_status()
    return response.text

@tool
def list_goods_receipts(po_id: str = "", vendor: str = "") -> str:
    """Find goods receipts for a purchase order or vendor."""
    params = {key: value for key, value in {"po_id": po_id, "vendor": vendor}.items() if value}
    response = http_request("GET", f"{settings.erp_api_url}/goods-receipts", params=params)
    response.raise_for_status()
    return response.text

@tool
def reconcile_invoice_records(invoice_id: str, po_id: str = "", grn_id: str = "") -> str:
    """Run deterministic three-way matching. Use this before any ledger proposal."""
    invoice_response = http_request("GET", f"{settings.erp_api_url}/invoices/{invoice_id}")
    invoice_response.raise_for_status()
    invoice = invoice_response.json()
    all_invoices_response = http_request("GET", f"{settings.erp_api_url}/invoices")
    all_invoices_response.raise_for_status()
    all_invoices = all_invoices_response.json()
    duplicate_count = sum(1 for row in all_invoices if row.get("vendor", "").lower() == invoice.get("vendor", "").lower() and row.get("invoice_number") and row.get("invoice_number") == invoice.get("invoice_number"))
    pos_response = http_request("GET", f"{settings.erp_api_url}/purchase-orders/{po_id}") if po_id else http_request("GET", f"{settings.erp_api_url}/purchase-orders", params={"invoice_id": invoice_id})
    pos_response.raise_for_status()
    pos = pos_response.json()
    po = pos[0] if isinstance(pos, list) and pos else pos if isinstance(pos, dict) else None
    grn_response = http_request("GET", f"{settings.erp_api_url}/goods-receipts/{grn_id}") if grn_id else (http_request("GET", f"{settings.erp_api_url}/goods-receipts", params={"po_id": po.get("po_id", "")}) if po else None)
    if grn_response is not None:
        grn_response.raise_for_status()
    grns = grn_response.json() if grn_response is not None else []
    grn = grns[0] if isinstance(grns, list) and grns else grns if isinstance(grns, dict) else None
    return reconcile_invoice(invoice, po, grn, duplicate_count=max(1, duplicate_count)).model_dump_json()

@tool
def search_emails(query: str) -> str:
    """Search the configured mailbox fixture for messages and invoice attachments."""
    return json.dumps(_email_provider.search_emails(query))

@tool
def get_email_attachment(email_id: str, attachment_id: str) -> str:
    """Read attachment metadata from a matching email; pass its path to the existing document ingestion flow."""
    return json.dumps(_email_provider.get_attachment(email_id, attachment_id))

@tool
def read_invoice_attachment(email_id: str, attachment_id: str) -> str:
    """Extract text from a PDF attachment so the agent can compare it with ERP evidence."""
    attachment = _email_provider.get_attachment(email_id, attachment_id)
    path = Path(__file__).parents[1] / attachment.get("path", "")
    if not path.exists():
        return json.dumps({"status": "unavailable", "error": "Attachment file is unavailable."})
    try:
        from pypdf import PdfReader
        text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        return json.dumps({"status": "success", "filename": attachment.get("filename"), "text": text[:30000]})
    except Exception as exc:
        return json.dumps({"status": "unavailable", "error": str(exc)})

@tool
def draft_completion_email(to: str, subject: str, body: str) -> str:
    """Create a completion email draft. Sending always requires a separate human approval."""
    draft = _email_provider.draft_email(to, subject, body)
    return json.dumps({"status": "awaiting_email_approval", "draft": draft})

@tool
def send_completion_email(draft_id: str, approved: bool = False) -> str:
    """Send an existing completion draft only when the application passes explicit approval."""
    if not approved:
        return json.dumps({"status": "awaiting_email_approval", "draft_id": draft_id})
    return json.dumps({"status": "sent", "message": _email_provider.send_email(draft_id)})


@tool
def search_company_knowledge(query: str) -> str:
    """Search approved company SOPs and uploaded documents for policies, aliases, and invoice evidence."""
    documents = hybrid_search_company(query, k=5)
    if not documents:
        return "No approved company document matched the query."
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in documents
    )


@tool
def propose_ledger_entry(invoice_id: str, amount: float, due_date: str, vendor: str = "") -> str:
    """Prepare a ledger-write proposal. This never mutates the ledger; a human approval node must approve it first."""
    return json.dumps({"status": "approval_required", "invoice_id": invoice_id, "amount": amount, "due_date": due_date, "vendor": vendor})


ACTION_TOOLS = [list_invoices, get_invoice, list_ledger, list_purchase_orders, list_goods_receipts, reconcile_invoice_records, search_emails, get_email_attachment, read_invoice_attachment, draft_completion_email, search_company_knowledge, propose_ledger_entry]

SYSTEM_PROMPT = """You are the proactive Acme Operations finance employee.
Own the user's requested outcome. Infer the necessary sequence from the goal and inspect the ERP before acting.
Use the available tools when they provide evidence; do not answer with a proposed procedure when the user asked for work.
After every tool result, reassess what is true and choose the next useful action.
Use search_company_knowledge when the request refers to an uploaded document, a vendor alias, or a company policy.
Use search_emails when the user asks to find an invoice in email; inspect a returned attachment before using it as evidence.
Use read_invoice_attachment to extract invoice facts from a PDF. Use draft_completion_email only after the requested work is verified; it creates a draft and never sends it.
If the user gives an internal invoice ID, call get_invoice with that exact ID first. If it returns not_found, do not select a similar ID or invoice number; explain the mismatch and ask for clarification. Use list_invoices to browse only when no exact ID was provided. Use list_ledger to inspect or verify state, and propose_ledger_entry only when
you have the exact invoice ID, amount, due date, and vendor. The proposal is not a write; the application handles approval.
For invoice processing, inspect the matching purchase order and goods receipt and call reconcile_invoice_records before proposing a ledger entry.
Never propose or post when reconciliation returns exception, missing_po, missing_grn, or insufficient_evidence; report the evidence and blocker.
If an uploaded invoice document and the ERP use different vendor labels, compare invoice ID, amount, and due date and use
the company policy or strongest matching evidence to resolve the alias. Do not reject a task only because the display names differ.
Never invent records, claim completion without verification, or repeat a successful write.
If the target is ambiguous, missing, unsafe, or unsupported, explain the blocker clearly.
"""


def build_tool_agent_graph():
    """Build the standard LLM -> ToolNode -> LLM loop with a safe proposal tool."""
    model = get_llm().bind_tools(ACTION_TOOLS)

    def call_model(state: MessagesState):
        return {"messages": [model.invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(ACTION_TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()
