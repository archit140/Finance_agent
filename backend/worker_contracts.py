"""Shared contracts for the unified CentrAlign task worker."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class IntentType(StrEnum):
    KNOWLEDGE = "knowledge"
    ACTION = "action"
    HYBRID = "hybrid"
    CLARIFICATION = "clarification"
    UNSAFE = "unsafe"
    UNSUPPORTED = "unsupported"


class TerminalStatus(StrEnum):
    COMPLETED = "completed"
    AWAITING_APPROVAL = "awaiting_approval"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    FAILED_SAFELY = "failed_safely"
    UNSUPPORTED = "unsupported"
    UNSAFE = "unsafe"


class ToolResult(BaseModel):
    """Normalized result returned by every worker tool adapter."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["success", "recoverable_failure", "terminal_failure", "unexpected_state", "no_change"]
    observation: str = ""
    changed_state: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    recoverable: bool = False


class PlanStep(BaseModel):
    """One bounded, observable step in an execution plan."""

    model_config = ConfigDict(extra="forbid")
    step_id: str
    purpose: str
    description: str
    tool_candidate: str | None = None
    arguments_needed: dict[str, Any] = Field(default_factory=dict)
    expected_observation: str = ""
    success_condition: str = ""
    risk_level: Literal["low", "medium", "high"] = "low"
    requires_approval: bool = False


class GoalInterpretation(BaseModel):
    """Structured understanding of the user's end goal."""

    model_config = ConfigDict(extra="forbid")
    objective: str
    success_conditions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    entities: dict[str, Any] = Field(default_factory=dict)
    ambiguities: list[str] = Field(default_factory=list)
    answer_required: bool = False
    action_required: bool = False


class WorkerState(TypedDict, total=False):
    """Unified LangGraph state, with compatibility fields for Papeer."""

    request_id: str
    company_id: str
    user_request: str
    intent_type: str
    goal_interpretation: dict[str, Any]
    success_conditions: list[str]
    constraints: list[str]
    retrieved_context: list[dict[str, Any]]
    environment_state: dict[str, Any]
    current_plan: list[dict[str, Any]]
    current_step: str | None
    completed_steps: list[str]
    pending_steps: list[str]
    scratchpad: list[str]
    observations: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    retry_counts: dict[str, int]
    approval_state: str
    clarification_state: str
    verification_state: dict[str, Any]
    final_evidence: list[dict[str, Any]]
    final_answer: str
    final_summary: str
    status: str

    goal: str
    api_url: str
    approved: bool
    max_retries: int
    invoice: dict[str, Any]
    plan: list[str]
    actions: list[str]
    evidence: list[dict[str, Any]]
    verified: bool
    summary: str
    human_gate: str
    ledger: list[dict[str, Any]]
    sop_context: str
