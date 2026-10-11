"""Short-term and long-term memory primitives for the task worker.

Short-term memory is the LangGraph checkpoint for an in-flight thread.
Long-term memory is durable semantic memory in Chroma, with a small SQLite
index for filtering and auditability.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver

from backend.config import settings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def checkpoint_store(path: str | Path | None = None) -> Iterator[SqliteSaver]:
    """Yield a durable LangGraph checkpointer and close its connection."""
    db_path = Path(path or settings.checkpoint_db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        saver = SqliteSaver(connection)
        saver.setup()
        yield saver
    finally:
        connection.close()


class LongTermMemory:
    """Semantic memory plus structured metadata for the company worker."""

    def __init__(self, persist_dir: str | Path | None = None, db_path: str | Path | None = None) -> None:
        self.persist_dir = Path(persist_dir or settings.chroma_persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path or settings.memory_db)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = Chroma(
            collection_name="centralign_long_term_memory",
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )
        self._setup_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _setup_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    company_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )"""
            )

    def remember(
        self,
        content: str,
        *,
        company_id: str = "default",
        memory_type: str = "task_outcome",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        memory_id = str(uuid.uuid4())
        metadata = {**(metadata or {}), "company_id": company_id, "memory_type": memory_type}
        self.vectorstore.add_documents(
            [Document(page_content=content, metadata={**metadata, "memory_id": memory_id})],
            ids=[memory_id],
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?)",
                (memory_id, company_id, memory_type, content, json.dumps(metadata), _utc_now()),
            )
        return memory_id

    def recall(self, query: str, *, company_id: str = "default", k: int = 4) -> list[Document]:
        docs = self.vectorstore.similarity_search(query, k=max(k * 2, k))
        return [doc for doc in docs if (doc.metadata or {}).get("company_id") in {company_id, None}][:k]

    def list_recent(self, *, company_id: str = "default", limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT memory_id, memory_type, content, metadata_json, created_at "
                "FROM memories WHERE company_id = ? ORDER BY created_at DESC LIMIT ?",
                (company_id, limit),
            ).fetchall()
        return [
            {"memory_id": row[0], "memory_type": row[1], "content": row[2],
             "metadata": json.loads(row[3]), "created_at": row[4]}
            for row in rows
        ]
