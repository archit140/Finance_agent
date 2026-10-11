"""Single LangGraph entry point for the CentrAlign autonomous worker."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import json
import requests
from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage

from backend.config import settings
from backend.intake import intake_request
from backend.memory import LongTermMemory
from backend.worker_contracts import WorkerState
from backend.tool_agent_graph import build_tool_agent_graph
from backend.company_rag import answer_company_question
from backend.http_client import request as http_request


def _memory() -> LongTermMemory:
    return LongTermMemory()


def _intake(state: WorkerState) -> dict[str, Any]:
    request = state.get("user_request") or state.get("goal", "")
    # Intake performs LLM classification and creates the normalized state.
    return intake_request(request, company_id=state.get("company_id", settings.company_id), memory=_memory())


def _content_text(content: Any) -> str:
    """Normalize LangChain string or content-block responses for parsing/UI."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content)


def _execute(state: WorkerState) -> dict[str, Any]:
    request = state.get("user_request") or state.get("goal", "")
    context = state.get("final_answer", "")
    goal = state.get("goal_interpretation", {})
    entities = goal.get("entities", {}) if isinstance(goal, dict) else {}
    target_hint = f"\n\nStructured target extracted from the request: {json.dumps(entities)}. Treat these identifiers as exact." if entities else ""
    prompt = (request if not context else f"User request: {request}\n\nGrounded RAG answer already prepared:\n{context}") + target_hint
    if state.get("approved", False):
        prompt += "\n\nHuman approval has been granted. Use the available evidence to call propose_ledger_entry now, then the application will commit and verify the approved write."
    agent_graph = build_tool_agent_graph()
    requires_finance_completion = bool(entities.get("invoice_id")) and any(term in request.lower() for term in ("reconcile", "post", "enter", "ledger", "process"))
    requires_posting = bool(entities.get("invoice_id")) and any(term in request.lower() for term in ("post", "enter", "ledger"))
    input_messages: list[Any] = [HumanMessage(content=prompt)]
    result = {}
    for attempt in range(3):
        result = agent_graph.invoke({"messages": input_messages})
        candidate_messages = result.get("messages", [])
        candidate_tool_text = "\n".join(_content_text(message.content) for message in candidate_messages if type(message).__name__ == "ToolMessage")
        has_proposal = "approval_required" in candidate_tool_text
        has_blocker = any(marker in candidate_tool_text for marker in ('"status":"exception"', '"status":"missing_po"', '"status":"missing_grn"', '"status":"insufficient_evidence"'))
        has_match = '"status":"matched"' in candidate_tool_text
        has_terminal_finance_result = has_proposal or has_blocker or (has_match and not requires_posting)
        has_not_found = '"status":"not_found"' in candidate_tool_text
        if not requires_finance_completion or has_terminal_finance_result or has_not_found or attempt == 2:
            break
        input_messages = candidate_messages + [HumanMessage(content="Continue investigating this exact invoice. You have not reached a terminal finance result yet. Decide the next tool yourself, but inspect the goods receipt and call reconciliation before reporting completion. If evidence is missing, return the appropriate safe blocker.")]
    messages = result.get("messages", [])
    final = _content_text(messages[-1].content) if messages else "No tool-agent response was produced."
    tool_results = [
        {"type": type(message).__name__, "content": _content_text(message.content)}
        for message in messages
        if type(message).__name__ == "ToolMessage"
    ]
    proposal = None
    reconciliation = None
    email_draft = None
    for item in tool_results:
        try:
            payload = json.loads(item["content"])
            candidates = payload if isinstance(payload, list) else [payload]
            for candidate in candidates:
                if isinstance(candidate, dict) and candidate.get("status") == "approval_required":
                    proposal = candidate
                    break
                if isinstance(candidate, dict) and candidate.get("status") == "awaiting_email_approval":
                    email_draft = candidate
                if isinstance(candidate, dict) and candidate.get("status") in {"matched", "exception", "missing_po", "missing_grn", "insufficient_evidence"} and "invoice_id" in candidate:
                    reconciliation = candidate
        except (TypeError, json.JSONDecodeError):
            continue
    if requires_finance_completion and (reconciliation is None or (requires_posting and proposal is None)) and not any('"status":"not_found"' in item["content"] for item in tool_results):
        return {
            "final_answer": final,
            "final_summary": "The worker could not complete invoice reconciliation after bounded follow-up attempts; no ledger write was made.",
            "tool_results": tool_results,
            "status": "failed_safely",
        }
    if reconciliation and reconciliation.get("status") != "matched":
        blocker = "; ".join(reconciliation.get("exceptions", [])) or reconciliation["status"]
        return {
            "final_answer": final,
            "final_summary": f"Invoice was not posted because reconciliation found an exception: {blocker}",
            "tool_results": tool_results,
            "final_evidence": [reconciliation],
            "status": "failed_safely",
        }
    if email_draft:
        return {
            "final_answer": final,
            "final_summary": "The completion email was drafted and is waiting for explicit send approval.",
            "tool_results": tool_results,
            "pending_email_approval": email_draft,
            "approval_state": "awaiting_email_approval",
            "status": "awaiting_approval",
        }
    if proposal is None and any(term in final.lower() for term in ("approval", "authorize", "authorisation", "authorization")):
        return {
            "final_answer": final,
            "final_summary": "Approval is required before the proposed ledger write.",
            "tool_results": tool_results,
            "pending_approval": {"request": request, "reason": final},
            "approval_state": "required",
            "status": "awaiting_approval",
        }
    if proposal and not state.get("approved", False):
        return {
            "final_answer": final,
            "final_summary": "Approval is required before the proposed ledger write.",
            "tool_results": tool_results,
            "pending_approval": proposal,
            "approval_state": "required",
            "status": "awaiting_approval",
        }
    if proposal:
        write_payload = {key: proposal[key] for key in ("invoice_id", "amount", "due_date", "vendor") if key in proposal}
        try:
            response = http_request("POST", f"{settings.erp_api_url}/ledger", json=write_payload)
        except requests.RequestException as exc:
            return {"final_answer": final, "final_summary": f"Ledger service was unavailable after retries: {exc}", "tool_results": tool_results, "status": "failed_safely"}
        if response.status_code not in (201, 409):
            response.raise_for_status()
        ledger_response = http_request("GET", f"{settings.erp_api_url}/ledger")
        ledger_response.raise_for_status()
        ledger = ledger_response.json()
        verified = any(
            item.get("invoice_id") == proposal["invoice_id"]
            and float(item.get("amount", -1)) == float(proposal["amount"])
            and item.get("due_date") == proposal["due_date"]
            for item in ledger
        )
        return {
            "final_answer": final,
            "final_summary": (
                "Ledger entry created and verified."
                if verified and response.status_code == 201
                else "Ledger entry already existed and was verified."
                if verified and response.status_code == 409
                else "Ledger write could not be verified."
            ),
            "tool_results": tool_results,
            "final_evidence": [{"ledger_write_status": response.status_code, "ledger_write": response.json()}, {"verified": verified}],
            "verified": verified,
            "status": "completed" if verified else "failed_safely",
        }
    return {"final_answer": final, "final_summary": final, "tool_results": tool_results, "status": "completed"}


def _general_rag(state: WorkerState) -> dict[str, Any]:
    """Answer general questions using company SOP similarity retrieval."""
    answer, evidence, retrieval = answer_company_question(state.get("user_request", ""))
    return {
        "final_answer": answer,
        "final_summary": answer,
        "retrieved_context": evidence,
        "final_evidence": evidence,
        "observations": [{"retrieval_query": retrieval.query, "attempts": retrieval.attempts, "rewritten": retrieval.rewritten, "relevant": retrieval.relevant}],
        "status": "completed",
    }


def _route_after_intake(state: WorkerState) -> str:
    intent = state.get("intent_type")
    if intent in {"knowledge", "hybrid"}:
        return "general_rag"
    if intent in {"unsafe", "unsupported", "clarification"}:
        return "report"
    return "execute"


def _route_after_rag(state: WorkerState) -> str:
    return "execute" if state.get("intent_type") == "hybrid" else "remember"


def _remember(state: WorkerState) -> dict[str, Any]:
    status = state.get("status", "failed_safely")
    summary = state.get("final_summary") or state.get("summary") or status
    if status in {"completed", "failed_safely"} and summary:
        memory_id = _memory().remember(
            f"Request: {state.get('user_request', state.get('goal', ''))}\nOutcome: {summary}",
            company_id=state.get("company_id", settings.company_id),
            memory_type="task_outcome",
            metadata={"status": status, "request_id": state.get("request_id")},
        )
        return {"final_evidence": state.get("final_evidence", []) + [{"memory_id": memory_id, "type": "task_outcome"}]}
    return {}


def _report(state: WorkerState) -> dict[str, Any]:
    return {
        "final_summary": state.get("final_summary") or state.get("summary") or "Worker finished without a summary.",
        "status": state.get("status", "failed_safely"),
    }


@lru_cache(maxsize=1)
def _checkpointer():
    connection = sqlite3.connect(str(settings.checkpoint_db), check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()
    return saver


def build_unified_graph():
    graph = StateGraph(WorkerState)
    graph.add_node("intake", _intake)
    graph.add_node("general_rag", _general_rag)
    graph.add_node("execute", _execute)
    graph.add_node("remember", _remember)
    graph.add_node("report", _report)
    graph.add_edge(START, "intake")
    graph.add_conditional_edges("intake", _route_after_intake, {"general_rag": "general_rag", "execute": "execute", "report": "report"})
    graph.add_conditional_edges("general_rag", _route_after_rag, {"execute": "execute", "remember": "remember"})
    graph.add_edge("execute", "remember")
    graph.add_edge("remember", "report")
    graph.add_edge("report", END)
    return graph.compile(checkpointer=_checkpointer())


def run_unified_worker(request: str, *, approved: bool = False, company_id: str = settings.company_id) -> WorkerState:
    """Run one complete autonomous request through the unified graph."""
    initial: WorkerState = {
        "user_request": request,
        "company_id": company_id,
        "approved": approved,
        "max_iterations": 8,
    }
    thread_id = f"{company_id}:{request}"
    result = build_unified_graph().invoke(initial, {"configurable": {"thread_id": thread_id}})
    trace_dir = Path(__file__).parents[1] / "data"
    trace_dir.mkdir(exist_ok=True)
    with (trace_dir / "unified_traces.jsonl").open("a", encoding="utf-8") as handle:
        import json
        handle.write(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": result.get("request_id"),
            "request": request,
            "status": result.get("status"),
            "summary": result.get("final_summary"),
            "completed_steps": result.get("completed_steps", []),
            "tool_results": result.get("tool_results", []),
            "evidence": result.get("final_evidence", []),
        }) + "\n")
    return result
