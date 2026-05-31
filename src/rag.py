from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer the question using only the context below.
If the context is not enough, say you do not have enough information.

Context:
{context}

Question: {question}
"""
    )

    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

