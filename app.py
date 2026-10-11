import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
from langchain_core.messages import HumanMessage
from backend.llm_provider import get_llm

from backend.btw_handler import handle_btw
from backend.paper_loader import load_arxiv, load_document, load_webpage
from backend.rag_graph import build_graph
from backend.vector_store import add_paper, list_papers, company_documents, ingest_company_document, ingest_company_pdf, search_company
from backend.unified_graph import run_unified_worker
from backend.tool_agent_graph import send_completion_email
from backend.config import settings
from backend.invoice_intake import extract_invoice_document

st.set_page_config(page_title="CentrAlign AI Employee", page_icon="🧭", layout="wide")


@st.cache_resource
def get_graph():
    return build_graph()


SESSIONS_FILE = Path("sessions.json")
_rename_llm = get_llm()


def load_sessions() -> dict:
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_sessions(sessions_meta: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions_meta, indent=2), encoding="utf-8")


def _api_json(path: str, fallback):
    try:
        response = requests.get(f"{settings.erp_api_url}{path}", timeout=settings.http_timeout_seconds)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return fallback


def _status_marker(status: str) -> str:
    return {"completed": "✅ Completed", "awaiting_approval": "⏸️ Approval required", "failed_safely": "🛑 Safely stopped", "awaiting_clarification": "💬 Clarification needed"}.get(status, "🔄 In progress")


def _serialize_state(values: dict) -> dict:
    out = {}
    for k, v in values.items():
        if k == "messages":
            out[k] = [
                {
                    "type": type(m).__name__,
                    "content": (
                        m.content[:300]
                        if isinstance(m.content, str)
                        else repr(m.content)[:300]
                    ),
                }
                for m in (v or [])
            ]
        elif k == "retrieved_docs":
            out[k] = [
                {"content": d.page_content[:300], "metadata": d.metadata}
                for d in (v or [])
            ]
        else:
            out[k] = v
    return out


def generate_session_name(first_message: str) -> str:
    try:
        response = _rename_llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Generate a concise 3-5 word title for a research chat session "
                        "based on the user's first message. Return only the title, "
                        "no punctuation at the end, no quotes."
                    ),
                },
                {"role": "user", "content": first_message[:500]},
            ]
        )
        return response.content.strip()
    except Exception:
        return "New Session"


def maybe_rename_session(session_id: str, first_message: str) -> None:
    if st.session_state.sessions_meta.get(session_id, {}).get("is_named"):
        return
    name = generate_session_name(first_message)
    st.session_state.sessions_meta[session_id]["name"] = name
    st.session_state.sessions_meta[session_id]["is_named"] = True
    save_sessions(st.session_state.sessions_meta)


def create_session() -> str:
    sid = str(uuid.uuid4())
    st.session_state.sessions_meta[sid] = {
        "id": sid,
        "name": "New Session",
        "created_at": datetime.now().isoformat(),
        "is_named": False,
    }
    save_sessions(st.session_state.sessions_meta)
    st.session_state.chats[sid] = []
    st.session_state.turns[sid] = 0
    return sid


def load_session_chats(session_id: str) -> list[dict]:
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        if not state or not state.values:
            return []
        chats = []
        turn = 0
        for msg in state.values.get("messages", []):
            type_name = type(msg).__name__
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if type_name == "HumanMessage":
                chats.append({"role": "user", "content": content})
            elif type_name in ("AIMessage", "AIMessageChunk"):
                turn += 1
                chats.append({"role": "assistant", "content": content, "turn": turn, "graph_state": {}})
        return chats
    except Exception:
        return []


def switch_session(session_id: str) -> None:
    st.session_state.active_session_id = session_id
    if session_id not in st.session_state.chats:
        st.session_state.chats[session_id] = load_session_chats(session_id)
    if session_id not in st.session_state.turns:
        turn_count = sum(1 for m in st.session_state.chats[session_id] if m["role"] == "assistant")
        st.session_state.turns[session_id] = turn_count


graph = get_graph()

# ── Bootstrap ──────────────────────────────────────────────────────────────────
if "sessions_meta" not in st.session_state:
    st.session_state.sessions_meta = load_sessions()
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "turns" not in st.session_state:
    st.session_state.turns = {}
if "active_session_id" not in st.session_state:
    if st.session_state.sessions_meta:
        latest = max(
            st.session_state.sessions_meta.values(),
            key=lambda s: s["created_at"],
        )
        switch_session(latest["id"])
    else:
        sid = create_session()
        st.session_state.active_session_id = sid

active_sid = st.session_state.active_session_id

# ── Unified CentrAlign AI employee interface ──────────────────────────────────
app_mode = "CentrAlign worker"

if app_mode == "CentrAlign worker":
    st.markdown("""
    <style>
    .block-container {padding-top: 2rem; max-width: 1400px;}
    [data-testid="stMetric"] {background: #151923; border: 1px solid #303747; padding: .8rem; border-radius: 12px;}
    .hero {padding: 1.2rem 1.4rem; border-radius: 16px; background: linear-gradient(120deg,#15243b,#1d1834); border: 1px solid #344766; margin-bottom: 1rem;}
    .hero h1 {margin: 0; font-size: 2rem;} .hero p {margin: .35rem 0 0; color: #b8c4d9;}
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>🧭 CentrAlign AI Employee</h1><p>Give the worker an outcome. It investigates, acts with tools, requests approval when needed, and verifies the result.</p></div>', unsafe_allow_html=True)
    health = _api_json("/health", {})
    ledger_snapshot = _api_json("/ledger", [])
    pending_snapshot = _api_json("/invoices/pending", [])
    metric_cols = st.columns(4)
    metric_cols[0].metric("ERP", "Online" if health.get("status") == "ok" else "Offline")
    metric_cols[1].metric("Ledger entries", len(ledger_snapshot) if isinstance(ledger_snapshot, list) else "—")
    metric_cols[2].metric("Pending invoices", len(pending_snapshot) if isinstance(pending_snapshot, list) else "—")
    metric_cols[3].metric("LLM", settings.groq_model)
    st.caption(f"ERP endpoint: `{settings.erp_api_url}` · Company: `{settings.company_id}`")
    st.markdown("#### Quick start")
    quick_cols = st.columns(4)
    quick_prompts = [
        ("📄 Clean invoice", "Find invoice INV-1001, inspect its purchase order and goods receipt, reconcile it, and enter it into the ledger."),
        ("🧪 Exception test", "Process invoice INV-S042-0003 and explain why it cannot be posted if the received quantity is insufficient."),
        ("📬 Email invoice", "Search my email for the latest invoice from Northwind, inspect the attachment, and reconcile it against the ERP."),
        ("📚 Company policy", "What is the company policy for invoices with missing goods receipts?"),
    ]
    for col, (label, prompt) in zip(quick_cols, quick_prompts):
        if col.button(label, use_container_width=True):
            st.session_state.worker_task = prompt
            st.rerun()
    with st.expander("Company knowledge", expanded=False):
        st.caption("Approved SOPs and company documents used by the general-answer RAG route.")
        for name in company_documents():
            st.write(f"• {name}")
        uploaded_company_doc = st.file_uploader("Add company document", type=["md", "markdown", "txt", "pdf"], key="company_doc")
        if uploaded_company_doc is not None and st.button("Index company document", key="index_company_doc"):
            safe_name = Path(uploaded_company_doc.name).name
            file_bytes = uploaded_company_doc.getvalue()
            (Path("company_kb") / safe_name).write_bytes(file_bytes)
            if uploaded_company_doc.name.lower().endswith(".pdf"):
                count = ingest_company_pdf(safe_name, file_bytes)
            else:
                count = ingest_company_document(safe_name, file_bytes.decode("utf-8"))
            st.success(f"Indexed {count} chunks from {uploaded_company_doc.name}.")
        retrieval_query = st.text_input("Test company retrieval", key="company_retrieval_query")
        if retrieval_query and st.button("Search company knowledge", key="search_company_knowledge"):
            st.json([{"source": doc.metadata.get("source"), "content": doc.page_content} for doc in search_company(retrieval_query)])
    with st.expander("Upload invoice for processing", expanded=False):
        st.caption("The invoice is extracted and stored as pending validation. Uploading never posts to the ledger.")
        invoice_file = st.file_uploader("Invoice PDF", type=["pdf"], key="invoice_upload")
        if invoice_file is not None and st.button("Read and create pending invoice", key="create_pending_invoice"):
            try:
                pending = extract_invoice_document(invoice_file.name, invoice_file.getvalue())
                response = requests.post(f"{settings.erp_api_url}/invoices/pending", json=pending, timeout=settings.http_timeout_seconds)
                response.raise_for_status()
                st.session_state.last_pending_invoice = response.json()
                st.success(f"Created {pending['invoice_id']} with status {pending['status']}.")
            except requests.RequestException as exc:
                st.error(f"Could not save the pending invoice: {exc}")
            except Exception as exc:
                st.error(f"Invoice extraction failed: {exc}")
        pending_invoice = st.session_state.get("last_pending_invoice")
        if pending_invoice:
            st.json(pending_invoice)
            if pending_invoice.get("status") == "pending_validation" and st.button("Process uploaded invoice", key="process_pending_invoice"):
                task = f"Process uploaded invoice {pending_invoice['invoice_id']}. Inspect its evidence, find any matching purchase order and goods receipt, reconcile it, and only propose a ledger entry if the evidence supports it."
                with st.spinner("The autonomous worker is investigating the uploaded invoice…"):
                    try:
                        result = run_unified_worker(task)
                        st.session_state.pending_worker_task = task if result.get("status") == "awaiting_approval" else None
                        st.session_state.pending_worker_result = result if result.get("status") == "awaiting_approval" else None
                        st.success(result.get("final_summary", "Invoice processing finished.")) if result.get("status") == "completed" else st.warning(result.get("final_summary", "Invoice requires review."))
                        st.json(result.get("final_evidence", []) or result.get("tool_results", []))
                    except Exception as exc:
                        st.error(f"Uploaded invoice processing failed: {exc}")
    if "worker_task" not in st.session_state:
        st.session_state.worker_task = ""
    task = st.text_area(
        "What would you like the AI employee to do?",
        key="worker_task",
        placeholder="Ask a company question or describe an outcome, for example: Find the latest invoice from Northwind Traders, enter it into the ledger, and verify it.",
        height=90,
    )
    approve = st.checkbox(
        "Approve policy-required ledger writes",
        help="The worker pauses automatically for invoices above the policy threshold.",
    )
    if "pending_worker_task" not in st.session_state:
        st.session_state.pending_worker_task = None
    if "pending_worker_result" not in st.session_state:
        st.session_state.pending_worker_result = None
    col1, col2 = st.columns([1, 1])
    with col1:
        run_worker = st.button("Submit query", type="primary", use_container_width=True)
    with col2:
        if st.button("Reset ledger", use_container_width=True):
            try:
                requests.post(f"{settings.erp_api_url}/reset", timeout=settings.http_timeout_seconds)
                st.success("Ledger reset.")
            except requests.RequestException as exc:
                st.error(f"API unavailable: {exc}")

    with st.expander("Current ledger", expanded=False):
        if st.button("Refresh ledger", key="refresh_ledger"):
            try:
                ledger_response = requests.get(f"{settings.erp_api_url}/ledger", timeout=settings.http_timeout_seconds)
                ledger_response.raise_for_status()
                ledger_rows = ledger_response.json()
                if ledger_rows:
                    st.dataframe(ledger_rows, use_container_width=True, hide_index=True)
                else:
                    st.info("The ledger is empty.")
            except requests.RequestException as exc:
                st.error(f"Could not reach the mock ERP API: {exc}")

    if run_worker and not task.strip():
        st.warning("Enter a question or task first.")
    if run_worker and task.strip():
        with st.spinner("Executing invoice task…"):
            try:
                result = run_unified_worker(task, approved=approve)
                if result.get("status") == "awaiting_approval":
                    st.session_state.pending_worker_task = task
                    st.session_state.pending_worker_result = result
                else:
                    st.session_state.pending_worker_task = None
                    st.session_state.pending_worker_result = None
                st.subheader(_status_marker(result.get("status", "")))
                st.info(result.get("final_summary") or result.get("summary", "Worker finished."))
                result_tabs = st.tabs(["Workflow", "Evidence", "Tool trace"])
                with result_tabs[0]:
                    stage_names = ["Intake", "Agent reasoning", "Evidence retrieval", "Validation / reconciliation", "Approval", "Posting", "Verification", "Audit"]
                    status = result.get("status", "")
                    tool_text = " ".join(str(item.get("content", "")) for item in result.get("tool_results", []))
                    stage_state = {"Intake": True, "Agent reasoning": bool(result.get("tool_results")), "Evidence retrieval": bool(result.get("retrieved_context") or "search_" in tool_text), "Validation / reconciliation": "reconciliation" in tool_text or "matched" in tool_text or "exception" in tool_text, "Approval": status == "awaiting_approval" or bool(result.get("pending_approval")), "Posting": bool(result.get("verified")), "Verification": bool(result.get("verified")), "Audit": True}
                    for stage in stage_names:
                        marker = "⏸️ Waiting" if stage == "Approval" and status == "awaiting_approval" else "✅ Done" if stage_state[stage] else "○ Not reached"
                        st.write(f"**{stage}** · {marker}")
                    if result.get("intent_type"):
                        st.caption(f"LLM route: **{result['intent_type']}**")
                with result_tabs[1]:
                    evidence = result.get("final_evidence") or result.get("evidence", [])
                    st.json(evidence or [{"message": "No structured evidence was returned."}])
                with result_tabs[2]:
                    st.json(result.get("tool_results", []) or [{"message": "No tools were called."}])
                if result.get("status") == "completed" or result.get("verified"):
                    st.success(result.get("final_summary") or result.get("summary", "Verified"))
                elif result.get("status") in {"awaiting_approval", "awaiting_clarification"} or result.get("human_gate"):
                    st.warning(result.get("final_summary") or result.get("human_gate", "User input is required."))
                else:
                    st.error(result.get("final_summary") or result.get("summary", "Verification failed"))
            except requests.RequestException as exc:
                st.error(f"Could not reach the mock ERP API. Start it with uvicorn first.\n\n{exc}")
            except Exception as exc:
                st.exception(exc)
    pending = st.session_state.get("pending_worker_result")
    pending_task = st.session_state.get("pending_worker_task")
    if pending and pending_task:
        st.divider()
        st.subheader("Approval required")
        st.warning(pending.get("final_summary", "A policy-controlled action is waiting for approval."))
        proposal = pending.get("pending_approval")
        if proposal:
            st.json(proposal)
        email_pending = pending.get("pending_email_approval")
        if email_pending:
            st.caption("The agent drafted a completion email. Sending is a separate approval.")
            st.json(email_pending)
        approve_col, reject_col = st.columns(2)
        with approve_col:
            approve_label = "Approve and resume" if proposal else "Approve email draft"
            if st.button(approve_label, type="primary", use_container_width=True):
                with st.spinner("Resuming approved task…"):
                    try:
                        if email_pending and email_pending.get("draft", {}).get("draft_id"):
                            sent = json.loads(send_completion_email.invoke({"draft_id": email_pending["draft"]["draft_id"], "approved": True}))
                            resumed = {"final_summary": "Completion email sent.", "final_evidence": [sent], "status": "completed"}
                        else:
                            resumed = run_unified_worker(pending_task, approved=True)
                        st.session_state.pending_worker_task = None
                        st.session_state.pending_worker_result = None
                        st.success(resumed.get("final_summary", "Approved task completed."))
                        st.json(resumed.get("final_evidence", []) or resumed.get("tool_results", []))
                    except Exception as exc:
                        st.exception(exc)
        with reject_col:
            if st.button("Reject", use_container_width=True):
                st.session_state.pending_worker_task = None
                st.session_state.pending_worker_result = None
                st.info("The proposed action was rejected and no ledger write was performed.")
    trace_path = Path("data/unified_traces.jsonl")
    if trace_path.exists():
        with st.expander("Recent audit traces", expanded=False):
            lines = trace_path.read_text(encoding="utf-8").splitlines()[-5:]
            for line in reversed(lines):
                try:
                    st.json(json.loads(line))
                except json.JSONDecodeError:
                    st.code(line)
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("+ New Chat", use_container_width=True):
        new_sid = create_session()
        st.session_state.active_session_id = new_sid
        active_sid = new_sid
        st.rerun()
    st.divider()
    st.markdown("## 💬 Sessions")

    sorted_sessions = sorted(
        st.session_state.sessions_meta.values(),
        key=lambda s: s["created_at"],
        reverse=True,
    )
    for session in sorted_sessions:
        sid = session["id"]
        is_active = sid == st.session_state.active_session_id
        btn_type = "primary" if is_active else "secondary"
        if st.button(
            session["name"],
            key=f"sess_{sid}",
            use_container_width=True,
            type=btn_type,
        ):
            if not is_active:
                switch_session(sid)
                st.rerun()

    st.divider()
    st.markdown("## 📄 Documents")

    # ── Section 1: File upload ─────────────────────────────────────────────────
    st.markdown("**Upload Files**")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or Markdown",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        key=f"uploader_{active_sid}",
        label_visibility="collapsed",
    )
    if st.button("Add Files", use_container_width=True, key="btn_add_files"):
        if uploaded_files:
            processed_key = f"processed_files_{active_sid}"
            if processed_key not in st.session_state:
                st.session_state[processed_key] = set()
            with st.spinner("Processing files…"):
                for f in uploaded_files:
                    if f.name in st.session_state[processed_key]:
                        st.info(f"Already loaded: {f.name}")
                        continue
                    suffix = Path(f.name).suffix
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(f.read())
                            tmp_path = tmp.name
                        docs = load_document(tmp_path)
                        for doc in docs:
                            doc.metadata["title"] = Path(f.name).stem
                        add_paper(docs, active_sid)
                        st.session_state[processed_key].add(f.name)
                        st.success(f"Added: {f.name}")
                    except Exception as e:
                        st.error(f"Failed: {f.name} — {e}")
                    finally:
                        if tmp_path:
                            Path(tmp_path).unlink(missing_ok=True)
            st.rerun()
        else:
            st.warning("No files selected.")

    # ── Section 2: Web URL loader ──────────────────────────────────────────────
    st.markdown("**Web Pages**")
    url_input = st.text_area(
        "URLs (one per line)",
        key=f"url_area_{active_sid}",
        height=80,
        label_visibility="collapsed",
        placeholder="https://example.com/paper",
    )
    if st.button("Load URLs", use_container_width=True, key="btn_load_urls"):
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]
        if urls:
            with st.spinner("Loading web pages…"):
                for url in urls:
                    try:
                        docs = load_webpage(url)
                        add_paper(docs, active_sid)
                        st.success(f"Loaded: {url[:60]}")
                    except Exception as e:
                        st.error(f"Failed: {url[:60]} — {e}")
            st.rerun()
        else:
            st.warning("Enter at least one URL.")

    # ── Section 3: ArXiv loader ────────────────────────────────────────────────
    st.markdown("**ArXiv Papers**")
    arxiv_title = st.text_input(
        "Paper title or ArXiv ID",
        key=f"arxiv_input_{active_sid}",
        label_visibility="collapsed",
        placeholder="1706.03762  or  Attention Is All You Need",
    )
    if st.button("Load ArXiv Paper", use_container_width=True, key="btn_load_arxiv"):
        if arxiv_title.strip():
            with st.spinner("Loading from ArXiv…"):
                try:
                    docs = load_arxiv(arxiv_title.strip())
                    add_paper(docs, active_sid)
                    loaded_title = docs[0].metadata.get("title") if docs else arxiv_title.strip()
                    st.success(f"Loaded: {loaded_title}")
                except Exception as e:
                    st.error(f"Failed: {e}")
            st.rerun()
        else:
            st.warning("Enter a paper title or ArXiv ID.")

    # ── Loaded Documents list ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### Loaded Documents")
    try:
        doc_titles = list_papers(active_sid)
    except Exception:
        doc_titles = None
    if doc_titles is None:
        st.caption("Could not load document list — try refreshing.")
    elif doc_titles:
        for title in doc_titles:
            st.markdown(f"- {title}")
    else:
        st.caption("No documents loaded yet.")

# ── Page header ────────────────────────────────────────────────────────────────
st.title("📚 Papeer — Research Paper Assistant")
st.markdown(
    "🔍 **Ask questions** from your uploaded papers &nbsp;·&nbsp; "
    "✅ **Verify claims** against recent literature &nbsp;·&nbsp; "
    "🌐 **Search the web** for the latest findings\n\n"
    "> Upload documents in the sidebar and start chatting below."
)
st.divider()

# ── Chat display ───────────────────────────────────────────────────────────────
for msg in st.session_state.chats.get(active_sid, []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            with st.expander(f"📊 Graph state · turn {msg['turn']}", expanded=False):
                st.json(msg["graph_state"])

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about your papers, verify a claim, or search the web…"):
    is_btw = prompt.strip().lower().startswith("/btw")

    if is_btw:
        query = prompt.strip()[4:].strip()

        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption("Side channel — not saved to session history.")

        with st.chat_message("assistant"):
            if not query:
                st.markdown("Please add a question after `/btw`, e.g. `/btw What is attention?`")
            else:
                placeholder = st.empty()
                response_text = ""
                for chunk in handle_btw(query):
                    response_text += chunk
                    placeholder.markdown(response_text + "▌")
                placeholder.markdown(response_text)
            st.caption("Side channel — not saved to session history.")

    else:
        if active_sid not in st.session_state.chats:
            st.session_state.chats[active_sid] = []
        if active_sid not in st.session_state.turns:
            st.session_state.turns[active_sid] = 0

        is_first_message = len(st.session_state.chats[active_sid]) == 0

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chats[active_sid].append({"role": "user", "content": prompt})
        st.session_state.turns[active_sid] += 1
        current_turn = st.session_state.turns[active_sid]

        if is_first_message:
            maybe_rename_session(active_sid, prompt)

        input_state = {
            "messages": [HumanMessage(content=prompt)],
            "session_id": active_sid,
            "query": prompt,
            "route": None,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "claim_verdict": None,
            "claim_source": None,
            "superseding_papers": [],
            "answer": None,
            "is_relevant": None,
            "rewrite_count": 0,
        }
        config = {"configurable": {"thread_id": active_sid}}

        with st.chat_message("assistant"):
            placeholder = st.empty()
            response_text = ""

            for chunk, metadata in graph.stream(input_state, config, stream_mode="messages"):
                if (
                    metadata.get("langgraph_node") == "generate_answer"
                    and hasattr(chunk, "content")
                    and chunk.content
                ):
                    response_text += chunk.content
                    placeholder.markdown(response_text + "▌")

            if not response_text:
                final_values = graph.get_state(config).values
                response_text = final_values.get("answer") or "No response generated."

            placeholder.markdown(response_text)

            final_values = graph.get_state(config).values
            state_snapshot = _serialize_state(final_values)

            with st.expander(f"📊 Graph state · turn {current_turn}", expanded=False):
                st.json(state_snapshot)

        st.session_state.chats[active_sid].append(
            {
                "role": "assistant",
                "content": response_text,
                "graph_state": state_snapshot,
                "turn": current_turn,
            }
        )

        if is_first_message:
            st.rerun()
