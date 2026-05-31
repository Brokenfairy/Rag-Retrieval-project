import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from pymilvus import MilvusClient

load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MilvusIndexerAgent:
    def __init__(self, collection_name: Optional[str] = None):
        cloud_uri = os.getenv("ZILLIZ_CLOUD_URI", "").strip()
        self.uri = cloud_uri or str(Path("data") / "milvus_lite.db")
        self.token = os.getenv("ZILLIZ_CLOUD_TOKEN", "").strip()
        self.collection_name = (
            collection_name
            or os.getenv("MILVUS_COLLECTION", "invoice_collection").strip()
            or "invoice_collection"
        )

        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.embedding_model = (
            os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text").strip() or "nomic-embed-text"
        )
        self.embeddings = self._build_embeddings(self.embedding_model)

        self.client: Optional[MilvusClient] = None
        self.dimension: Optional[int] = None
        self.init_error: str = ""

        self._initialize_client()

    def _build_embeddings(self, model_name: str) -> OllamaEmbeddings:
        return OllamaEmbeddings(model=model_name, base_url=self.base_url)

    @staticmethod
    def _with_latest_tag(model_name: str) -> str:
        return model_name if ":" in model_name else f"{model_name}:latest"

    def _candidate_embedding_models(self) -> List[str]:
        candidates: List[str] = []
        primary = self.embedding_model.strip()
        if primary:
            candidates.extend([primary, self._with_latest_tag(primary)])

        # Stable fallback for embedding workloads.
        fallback = "nomic-embed-text"
        if primary != fallback:
            candidates.extend([fallback, self._with_latest_tag(fallback)])

        seen = set()
        deduped: List[str] = []
        for model in candidates:
            if model not in seen:
                seen.add(model)
                deduped.append(model)
        return deduped

    def _initialize_client(self) -> None:
        try:
            if self.token:
                self.client = MilvusClient(uri=self.uri, token=self.token)
            else:
                self.client = MilvusClient(uri=self.uri)

            self.dimension = None
            model_errors: List[str] = []
            chosen_model: Optional[str] = None
            for candidate_model in self._candidate_embedding_models():
                self.embeddings = self._build_embeddings(candidate_model)
                try:
                    self.dimension = self._infer_embedding_dimension()
                    chosen_model = candidate_model
                    break
                except Exception as embed_error:
                    model_errors.append(f"{candidate_model}: {embed_error}")

            if self.dimension is None or chosen_model is None:
                tried = ", ".join(self._candidate_embedding_models())
                details = " | ".join(model_errors) if model_errors else "Unknown embedding error"
                raise RuntimeError(
                    f"No usable Ollama embedding model found. Tried [{tried}]. "
                    f"Please run `ollama pull {self.embedding_model}` or `ollama pull nomic-embed-text` to fix this. "
                    f"Details: {details}"
                )

            if chosen_model != self.embedding_model:
                logger.warning(
                    "Embedding model '%s' unavailable. Falling back to '%s'.",
                    self.embedding_model,
                    chosen_model,
                )
                self.embedding_model = chosen_model

            self._ensure_collection()
            self.init_error = ""
        except Exception as e:
            self.client = None
            self.dimension = None
            self.init_error = str(e)
            logger.error(f"Failed to initialize Milvus indexer: {e}")

    def _require_client(self) -> MilvusClient:
        if self.client is None:
            # Retry once in case environment was fixed after startup.
            self._initialize_client()

        if self.client is None:
            raise RuntimeError(
                "Milvus is not available. Configure ZILLIZ_CLOUD_URI/ZILLIZ_CLOUD_TOKEN "
                "or install local support with `pip install pymilvus[milvus_lite]`. "
                f"Details: {self.init_error}"
            )

        return self.client

    def _infer_embedding_dimension(self) -> int:
        sample_vector = self.embeddings.embed_query("dimension probe")
        if not sample_vector:
            raise ValueError("Embedding model returned an empty vector.")
        return len(sample_vector)

    @staticmethod
    def _chunk_text(text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]
        if not chunks and text.strip():
            chunks = [text.strip()]
        return chunks

    def has_collection(self) -> bool:
        try:
            client = self._require_client()
            return client.has_collection(self.collection_name)
        except Exception:
            return False

    def is_ready(self) -> bool:
        return self.client is not None and self.has_collection()

    def get_collection_name(self) -> str:
        return self.collection_name

    def search(self, query: str, limit: int = 5, output_fields: Optional[List[str]] = None) -> List[List[dict]]:
        fields = output_fields or ["text", "filename", "upload_date", "chunk_id"]
        query_vector = self.embeddings.embed_query(query)
        client = self._require_client()
        return client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=fields,
        )

    def _create_collection(self) -> None:
        client = self._require_client()
        if self.dimension is None:
            raise RuntimeError("Embedding dimension was not initialized.")
        logger.info(f"Creating collection: {self.collection_name} (dim={self.dimension})")
        client.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension,
            auto_id=True,
            metric_type="COSINE",
            enable_dynamic_field=True,
        )

    def _ensure_collection(self) -> None:
        """Checks if collection exists, if not creates it."""
        client = self._require_client()
        if not client.has_collection(self.collection_name):
            self._create_collection()
        else:
            logger.info(f"Collection {self.collection_name} already exists.")

    def index_document(self, text: str, metadata: Dict) -> int:
        """
        Chunks the text, creates embeddings, and inserts into Milvus.
        """
        try:
            logger.info(f"Starting indexing for document: {metadata.get('filename', 'Unknown')}")

            if not text or not text.strip():
                raise ValueError("Cannot index empty text.")

            chunks = self._chunk_text(text)
            logger.info(f"Split document into {len(chunks)} chunks.")
            vectors = self.embeddings.embed_documents(chunks)

            data_rows = []
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                data_rows.append({
                    "vector": vector,
                    "text": chunk,
                    "filename": metadata.get("filename", ""),
                    "upload_date": metadata.get("upload_date", ""),
                    "chunk_id": i
                })

            client = self._require_client()
            res = client.insert(
                collection_name=self.collection_name,
                data=data_rows
            )

            logger.info(f"Successfully indexed {len(data_rows)} chunks. Insert Result: {res}")
            return len(data_rows)

        except Exception as e:
            logger.error(f"Failed to index document: {e}")
            raise

if __name__ == "__main__":
    # Test execution
    indexer = MilvusIndexerAgent()
    print("Milvus Indexer initialized.")
