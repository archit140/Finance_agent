"""Company knowledge RAG with hybrid retrieval, relevance checks, and retry."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from backend.llm_provider import get_llm
from backend.vector_store import hybrid_search_company


class RelevancyOutput(BaseModel):
    is_relevant: bool
    reasoning: str = ""


@dataclass
class CompanyRetrieval:
    query: str
    documents: list[Document]
    attempts: int
    rewritten: bool
    relevant: bool


def _check_relevance(query: str, documents: list[Document]) -> RelevancyOutput:
    if not documents:
        return RelevancyOutput(is_relevant=False, reasoning="No documents were retrieved.")
    snippets = "\n\n---\n\n".join(doc.page_content[:700] for doc in documents[:5])
    prompt = ChatPromptTemplate.from_template("""You evaluate company knowledge retrieval.
Return only JSON: {{"is_relevant": true or false, "reasoning": "brief explanation"}}.
Mark relevant when the supplied chunks contain direct or useful evidence for the question.
Do not use outside knowledge.

Question: {query}
Retrieved chunks:
{snippets}""")
    return (prompt | get_llm().with_structured_output(RelevancyOutput, method="json_mode")).invoke({"query": query, "snippets": snippets})


def _rewrite_query(query: str) -> str:
    response = get_llm().invoke([
        {"role": "system", "content": "Rewrite the company knowledge search query using concrete entities, invoice IDs, vendor names, dates, and policy terms. Return only the rewritten query."},
        {"role": "user", "content": query},
    ])
    return response.content if isinstance(response.content, str) else str(response.content)


def retrieve_company_context(query: str, *, max_attempts: int = 2, k: int = 5) -> CompanyRetrieval:
    current_query = query
    rewritten = False
    documents: list[Document] = []
    for attempt in range(1, max_attempts + 1):
        documents = hybrid_search_company(current_query, k=k)
        relevance = _check_relevance(current_query, documents)
        if relevance.is_relevant:
            return CompanyRetrieval(current_query, documents, attempt, rewritten, True)
        if attempt < max_attempts:
            current_query = _rewrite_query(current_query)
            rewritten = True
    return CompanyRetrieval(current_query, documents, max_attempts, rewritten, False)


def answer_company_question(query: str) -> tuple[str, list[dict], CompanyRetrieval]:
    retrieval = retrieve_company_context(query)
    evidence = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page"),
            "retrieval_method": doc.metadata.get("retrieval_method", "hybrid"),
            "content": doc.page_content,
        }
        for doc in retrieval.documents
    ]
    if not retrieval.relevant:
        return "I could not find enough supported company information to answer that question.", evidence, retrieval
    context = "\n\n---\n\n".join(
        f"Source: {item['source']}\n{item['content']}" for item in evidence
    )
    prompt = ChatPromptTemplate.from_template("""You answer questions for a company operations employee.
Use only the approved context below. Do not invent facts. If the context is incomplete, state what is missing.
Include concise source references using the provided source names.

Question: {query}
Approved context:
{context}""")
    answer = (prompt | get_llm()).invoke({"query": query, "context": context}).content
    return (answer if isinstance(answer, str) else str(answer), evidence, retrieval)
