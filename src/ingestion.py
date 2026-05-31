from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings


def _load_text(path: Path) -> list[Document]:
    # Try UTF-8 first, then a permissive fallback for legacy invoice exports.
    try:
        return TextLoader(str(path), encoding="utf-8").load()
    except UnicodeDecodeError:
        return TextLoader(str(path), encoding="latin-1").load()


def _load_pdf(path: Path, settings: Settings) -> list[Document]:
    if settings.llama_cloud_api_key:
        try:
            from llama_parse import LlamaParse

            parser = LlamaParse(
                api_key=settings.llama_cloud_api_key,
                result_type="markdown",
            )
            parsed_docs = parser.load_data(str(path))
            converted: list[Document] = []
            for index, item in enumerate(parsed_docs):
                text = getattr(item, "text", "").strip()
                if text:
                    converted.append(
                        Document(
                            page_content=text,
                            metadata={"source": str(path), "page": index, "parser": "llamaparse"},
                        )
                    )
            if converted:
                return converted
        except Exception:
            # Fall back to standard PDF extraction when LlamaParse fails.
            pass

    return PyPDFLoader(str(path)).load()


def _load_msg(path: Path) -> list[Document]:
    import extract_msg

    message = extract_msg.Message(str(path))
    parts = [message.subject or "", message.body or ""]
    return [
        Document(
            page_content="\n\n".join(part for part in parts if part.strip()),
            metadata={"source": str(path), "parser": "extract-msg"},
        )
    ]


def load_documents(file_paths: Iterable[Path], settings: Settings) -> list[Document]:
    docs: list[Document] = []

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            continue

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            docs.extend(_load_pdf(path, settings))
        elif suffix == ".csv":
            docs.extend(CSVLoader(str(path)).load())
        elif suffix == ".msg":
            docs.extend(_load_msg(path))
        else:
            docs.extend(_load_text(path))

    return [doc for doc in docs if doc.page_content and doc.page_content.strip()]


def split_documents(
    documents: Iterable[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(list(documents))
