from langchain_groq import ChatGroq
from backend.config import settings


def get_llm():
    api_key = settings.groq_api_key or "missing-key"
    return ChatGroq(
        api_key=api_key,
        model=settings.groq_model,
        temperature=0,
    )
