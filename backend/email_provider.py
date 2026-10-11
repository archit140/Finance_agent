"""Email boundary. Replace MockEmailProvider with an MCP/provider adapter later."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Protocol

class EmailProvider(Protocol):
    def search_emails(self, query: str) -> list[dict]: ...
    def get_email(self, email_id: str) -> dict: ...
    def get_attachment(self, email_id: str, attachment_id: str) -> dict: ...
    def draft_email(self, to: str, subject: str, body: str) -> dict: ...
    def send_email(self, draft_id: str) -> dict: ...

@dataclass
class MockEmailProvider:
    fixture_path: Path
    def __post_init__(self):
        self.emails = json.loads(self.fixture_path.read_text(encoding="utf-8")) if self.fixture_path.exists() else []
        self.drafts: dict[str, dict] = {}
    def search_emails(self, query: str) -> list[dict]:
        words = query.lower().split()
        def score(email):
            text = json.dumps(email).lower()
            return sum(word in text for word in words)
        return sorted((email for email in self.emails if score(email)), key=score, reverse=True)
    def get_email(self, email_id: str) -> dict:
        return next((email for email in self.emails if email["email_id"] == email_id), {})
    def get_attachment(self, email_id: str, attachment_id: str) -> dict:
        email = self.get_email(email_id)
        return next((item for item in email.get("attachments", []) if item["attachment_id"] == attachment_id), {})
    def draft_email(self, to: str, subject: str, body: str) -> dict:
        draft_id = f"draft-{len(self.drafts)+1}"
        self.drafts[draft_id] = {"draft_id": draft_id, "to": to, "subject": subject, "body": body, "status": "draft"}
        return self.drafts[draft_id]
    def send_email(self, draft_id: str) -> dict:
        draft = self.drafts.get(draft_id)
        if not draft:
            raise KeyError(f"Unknown draft: {draft_id}")
        draft["status"] = "sent"
        return draft
