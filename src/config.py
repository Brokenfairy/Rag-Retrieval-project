from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    zilliz_uri: str
    zilliz_token: str
    llama_cloud_api_key: str
    ollama_base_url: str
    ollama_model: str
    embedding_model: str
    collection_name: str

    @property
    def connection_args(self) -> dict[str, str]:
        args: dict[str, str] = {"uri": self.zilliz_uri}
        if self.zilliz_token:
            args["token"] = self.zilliz_token
        return args


def get_settings(collection_name: Optional[str] = None) -> Settings:
    # Milvus Lite is used by default when cloud values are not provided.
    default_milvus_path = str(Path("data") / "milvus_lite.db")
    configured_uri = os.getenv("ZILLIZ_CLOUD_URI", "").strip()

    zilliz_uri = configured_uri or default_milvus_path
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3").strip() or "llama3"

    return Settings(
        zilliz_uri=zilliz_uri,
        zilliz_token=os.getenv("ZILLIZ_CLOUD_TOKEN", "").strip(),
        llama_cloud_api_key=os.getenv("LLAMA_CLOUD_API_KEY", "").strip(),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
        ollama_model=ollama_model,
        embedding_model=os.getenv("OLLAMA_EMBED_MODEL", ollama_model).strip() or ollama_model,
        collection_name=collection_name or os.getenv("MILVUS_COLLECTION", "langchain_docs").strip() or "langchain_docs",
    )

