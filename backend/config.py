import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    groq_api_key: str = os.getenv("GROQ_API_KEY", "").strip()
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chroma_persist_dir: Path = Path(os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "chroma"))).resolve()
    checkpoint_db: Path = Path(os.getenv("CHECKPOINT_DB", str(PROJECT_ROOT / "checkpoints.db"))).resolve()
    memory_db: Path = Path(os.getenv("MEMORY_DB", str(PROJECT_ROOT / "memory.db"))).resolve()
    erp_api_url: str = os.getenv("ERP_API_URL", "http://127.0.0.1:8000")
    erp_data_dir: Path = Path(os.getenv("ERP_DATA_DIR", str(PROJECT_ROOT / "sandbox" / "data"))).resolve()
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    company_id: str = os.getenv("COMPANY_ID", "acme_ops")
    max_action_retries: int = int(os.getenv("MAX_ACTION_RETRIES", "2"))
    http_timeout_seconds: float = float(os.getenv("HTTP_TIMEOUT_SECONDS", "10"))


settings = Settings()
