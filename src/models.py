from __future__ import annotations

from langchain_ollama import ChatOllama, OllamaEmbeddings

from src.config import Settings


def get_chat_model(settings: Settings) -> ChatOllama:
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )


def get_embedding_model(settings: Settings) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )

