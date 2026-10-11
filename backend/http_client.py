"""Small resilient HTTP boundary for external tools and the ERP adapter."""
from __future__ import annotations
import time
import requests
from backend.config import settings

def request(method: str, url: str, **kwargs) -> requests.Response:
    kwargs.setdefault("timeout", settings.http_timeout_seconds)
    attempts = max(1, settings.max_action_retries + 1)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.request(method, url, **kwargs)
            if response.status_code >= 500 and attempt < attempts - 1:
                time.sleep(0.25 * (2 ** attempt))
                continue
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= attempts - 1:
                raise
            time.sleep(0.25 * (2 ** attempt))
    raise last_error or RuntimeError("HTTP request failed")
