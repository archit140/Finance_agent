from pathlib import Path
from io import BytesIO

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader
import re

from backend.config import settings

# ── Config ───────────────────────────────────────────────────────────────────

CHROMA_DIR = str(settings.chroma_persist_dir)
embeddings = HuggingFaceEmbeddings(
    model_name=settings.embedding_model,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


# ── Collection ───────────────────────────────────────────────────────────────

def get_collection_name(session_id: str) -> str:
    return f"papeer_{session_id.replace('-', '_')}"


def get_vectorstore(session_id: str) -> Chroma:
    return Chroma(
        collection_name=get_collection_name(session_id),
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


# ── Public API ───────────────────────────────────────────────────────────────

def add_paper(docs: list[Document], session_id: str) -> None:
    get_vectorstore(session_id).add_documents(docs)


def list_papers(session_id: str) -> list[str]:
    data = get_vectorstore(session_id)._collection.get(include=["metadatas"])
    seen: set[str] = set()
    titles: list[str] = []
    for metadata in data.get("metadatas") or []:
        title = (metadata or {}).get("title")
        if title and title not in seen:
            seen.add(title)
            titles.append(title)
    return titles


def search(query: str, session_id: str, k: int = 4) -> list[Document]:
    return get_vectorstore(session_id).similarity_search(query, k=k)


def ensure_company_kb() -> None:
    """Ingest approved company SOPs once into the shared company collection."""
    kb_dir = Path(__file__).parents[1] / "company_kb"
    store = get_vectorstore("company_kb")
    if store._collection.count() > 0:
        return
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    docs: list[Document] = []
    for path in sorted(kb_dir.glob("*.md")):
        source = path.read_text(encoding="utf-8")
        docs.extend(splitter.create_documents([source], metadatas=[{"source": path.name, "collection": "company_kb"}]))
    if docs:
        store.add_documents(docs)


def search_company(query: str, k: int = 4) -> list[Document]:
    ensure_company_kb()
    return get_vectorstore("company_kb").similarity_search(query, k=k)


def hybrid_search_company(query: str, k: int = 4) -> list[Document]:
    """Fuse dense Chroma retrieval with lexical matching for IDs, names, and dates."""
    ensure_company_kb()
    store = get_vectorstore("company_kb")
    dense = store.similarity_search_with_score(query, k=max(k * 2, 8))
    data = store._collection.get(include=["documents", "metadatas"])
    terms = set(re.findall(r"[a-z0-9-]+", query.lower()))
    lexical: list[tuple[float, Document]] = []
    for content, metadata in zip(data.get("documents") or [], data.get("metadatas") or []):
        words = set(re.findall(r"[a-z0-9-]+", (content or "").lower()))
        overlap = len(terms & words) / max(len(terms), 1)
        if overlap:
            lexical.append((overlap, Document(page_content=content, metadata={**(metadata or {}), "retrieval_method": "lexical"})))
    ranked: dict[str, tuple[float, Document]] = {}
    # Chroma returns (Document, distance), not (distance, Document).
    for rank, (doc, _distance) in enumerate(dense, start=1):
        key = f"{doc.metadata.get('source')}:{doc.page_content[:80]}"
        ranked[key] = (1 / (60 + rank), doc)
    for rank, (_, doc) in enumerate(sorted(lexical, reverse=True, key=lambda item: item[0]), start=1):
        key = f"{doc.metadata.get('source')}:{doc.page_content[:80]}"
        prior = ranked.get(key, (0.0, doc))[0]
        ranked[key] = (prior + 1 / (60 + rank), doc)
    results = []
    for _, doc in sorted(ranked.values(), reverse=True, key=lambda item: item[0])[:k]:
        doc.metadata["retrieval_method"] = "hybrid"
        results.append(doc)
    return results


def ingest_company_document(name: str, content: str) -> int:
    """Chunk and index one approved company text/Markdown document."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = splitter.create_documents([content], metadatas=[{"source": name, "collection": "company_kb"}])
    get_vectorstore("company_kb").add_documents(chunks)
    return len(chunks)


def ingest_company_pdf(name: str, content: bytes) -> int:
    """Extract text from a PDF, then chunk and index it as company knowledge."""
    reader = PdfReader(BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(page for page in pages if page.strip())
    if not text.strip():
        raise ValueError("The PDF does not contain extractable text.")
    return ingest_company_document(name, text)


def company_documents() -> list[str]:
    kb_dir = Path(__file__).parents[1] / "company_kb"
    return sorted(path.name for path in kb_dir.glob("*") if path.is_file())


# Backward-compatible singular alias for older Streamlit processes.
company_document = company_documents
