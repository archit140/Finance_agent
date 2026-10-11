"""Unified request understanding and routing for CentrAlign."""

from __future__ import annotations

import re
import uuid
from typing import Any

from backend.config import settings
from backend.memory import LongTermMemory
from backend.worker_contracts import GoalInterpretation, IntentType, WorkerState
from langgraph.graph import END, START, StateGraph
from backend.llm_provider import get_llm
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


class IntentOutput(BaseModel):
    intent: str
    reasoning: str = ""


def classify_request(request: str) -> str:
    """Use the configured LLM classifier; no keyword fallback is used."""
    return llm_classify_request(request)


def llm_classify_request(request: str) -> str:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required for autonomous routing")
    prompt = ChatPromptTemplate.from_template("""Classify this company worker request.
Choose exactly one: knowledge, action, hybrid, clarification, unsafe, unsupported.
knowledge answers from approved context; action changes or reads systems; hybrid does both;
clarification lacks required detail; unsafe requests harmful or unauthorized behavior;
unsupported cannot be handled by the listed company tools.
Return only valid JSON with exactly these fields: {{"intent": "knowledge|action|hybrid|clarification|unsafe|unsupported", "reasoning": "brief explanation"}}.
Request: {request}""")
    result = (prompt | get_llm().with_structured_output(IntentOutput, method="json_mode")).invoke({"request": request})
    if result.intent not in {item.value for item in IntentType}:
        raise ValueError(f"Invalid intent from planner: {result.intent}")
    return result.intent


def interpret_goal(request: str, intent_type: str | None = None) -> GoalInterpretation:
    intent = intent_type or classify_request(request)
    text = request.lower()
    entities: dict[str, Any] = {}
    # Support both legacy IDs (INV-1001) and generated IDs (INV-S042-0001).
    # IDs are carried literally into the action prompt; the LLM must not infer a substitute.
    invoice_match = re.search(r"\b(inv[- ]([a-z0-9]+(?:[- ][a-z0-9]+)*))\b", text, re.IGNORECASE)
    if invoice_match:
        entities["invoice_id"] = "INV-" + re.sub(r"[- ]+", "-", invoice_match.group(2).upper())
    return GoalInterpretation(
        objective=request.strip(),
        success_conditions=(
            ["The requested information is answered from company-approved context"]
            if intent == IntentType.KNOWLEDGE.value
            else ["The requested action is completed and verified"]
        ),
        constraints=["Use approved tools and company data", "Ask for approval before high-risk writes"],
        entities=entities,
        ambiguities=["The requested action or target is not specific enough"] if intent == IntentType.CLARIFICATION.value else [],
        answer_required=intent in {IntentType.KNOWLEDGE.value, IntentType.HYBRID.value},
        action_required=intent in {IntentType.ACTION.value, IntentType.HYBRID.value},
    )


def intake_request(request: str, *, company_id: str = settings.company_id, memory: LongTermMemory | None = None) -> WorkerState:
    """Create the initial unified state and recall relevant long-term memory."""
    intent = llm_classify_request(request)
    goal = interpret_goal(request, intent)
    recalled = []
    if memory is not None:
        recalled = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in memory.recall(request, company_id=company_id, k=4)
        ]
    status = "ready"
    if intent == IntentType.UNSAFE.value:
        status = "unsafe"
    elif intent == IntentType.CLARIFICATION.value:
        status = "awaiting_clarification"
    elif intent == IntentType.UNSUPPORTED.value:
        status = "unsupported"
    return {
        "request_id": str(uuid.uuid4()),
        "company_id": company_id,
        "user_request": request,
        "intent_type": intent,
        "goal_interpretation": goal.model_dump(),
        "success_conditions": goal.success_conditions,
        "constraints": goal.constraints,
        "retrieved_context": recalled,
        "environment_state": {},
        "current_plan": [],
        "completed_steps": [],
        "pending_steps": [],
        "observations": [],
        "tool_calls": [],
        "tool_results": [],
        "retry_counts": {},
        "approval_state": "not_required",
        "clarification_state": "required" if status == "awaiting_clarification" else "not_required",
        "verification_state": {},
        "final_evidence": [],
        "final_answer": "",
        "final_summary": (
            "This request is outside the currently supported company workflows."
            if status == "unsupported"
            else "This request is unsafe and cannot be executed."
            if status == "unsafe"
            else "More information is required before the worker can proceed."
            if status == "awaiting_clarification"
            else ""
        ),
        "status": status,
    }


def build_intake_graph():
    """Build the Phase 3 graph boundary; execution remains side-effect free."""
    graph = StateGraph(WorkerState)

    def understand(state: WorkerState) -> dict[str, Any]:
        request = state.get("user_request") or state.get("goal", "")
        intent = llm_classify_request(request)
        goal = interpret_goal(request, intent)
        return {
            "user_request": request,
            "intent_type": intent,
            "goal_interpretation": goal.model_dump(),
            "success_conditions": goal.success_conditions,
            "constraints": goal.constraints,
            "status": "ready" if intent not in {IntentType.UNSAFE.value, IntentType.CLARIFICATION.value} else ("unsafe" if intent == IntentType.UNSAFE.value else "awaiting_clarification"),
        }

    def route(state: WorkerState) -> dict[str, Any]:
        return {"final_summary": f"Request routed to {state.get('intent_type', 'unsupported')} path."}

    graph.add_node("understand", understand)
    graph.add_node("route", route)
    graph.add_edge(START, "understand")
    graph.add_edge("understand", "route")
    graph.add_edge("route", END)
    return graph.compile()
