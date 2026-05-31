from __future__ import annotations

from typing import Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import Milvus
from pymilvus import connections, utility

from src.config import Settings


MILVUS_SCHEMA_KWARGS = {
    "text_field": "text",
    "vector_field": "vector",
    "primary_field": "pk",
    "auto_id": True,
}


def _connect(settings: Settings, alias: str = "default") -> None:
    connections.connect(alias=alias, **settings.connection_args)


def has_collection(settings: Settings) -> bool:
    _connect(settings)
    return utility.has_collection(settings.collection_name)


def _build_store(settings: Settings, embeddings: Embeddings) -> Milvus:
    return Milvus(
        embedding_function=embeddings,
        collection_name=settings.collection_name,
        connection_args=settings.connection_args,
        drop_old=False,
        **MILVUS_SCHEMA_KWARGS,
    )


def upsert_documents(settings: Settings, embeddings: Embeddings, docs: Iterable[Document]) -> int:
    doc_list = list(docs)
    if not doc_list:
        return 0

    if has_collection(settings):
        store = _build_store(settings, embeddings)
        store.add_documents(doc_list)
    else:
        Milvus.from_documents(
            doc_list,
            embedding=embeddings,
            collection_name=settings.collection_name,
            connection_args=settings.connection_args,
            drop_old=False,
            **MILVUS_SCHEMA_KWARGS,
        )

    return len(doc_list)


def get_retriever(settings: Settings, embeddings: Embeddings, k: int = 4) -> VectorStoreRetriever:
    if not has_collection(settings):
        raise ValueError(
            f"Collection '{settings.collection_name}' does not exist yet. Index documents first."
        )

    store = _build_store(settings, embeddings)
    return store.as_retriever(search_kwargs={"k": k})

