import logging
from typing import Optional

try:
    from crewai.tools import BaseTool
except Exception:  # Backward compatibility for older crewai/crewai-tools combos
    from crewai_tools import BaseTool
from dotenv import load_dotenv

# Use our existing Indexer to get a connection
from agents.indexer import MilvusIndexerAgent

load_dotenv()
logger = logging.getLogger(__name__)

class MilvusSearchTool(BaseTool):
    name: str = "Invoice Search Tool"
    description: str = (
        "Search through processed invoice data stored in Milvus. "
        "Useful for answering questions about line items, totals, dates, and vendors. "
        "Input should be a search query string."
    )
    
    # Lazy-loaded to keep startup fast.
    indexer: Optional[MilvusIndexerAgent] = None

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str) -> str:
        try:
            query = (query or "").strip()
            if not query:
                return "Search query cannot be empty."

            if not self.indexer:
                self.indexer = MilvusIndexerAgent()

            logger.info(f"Searching Milvus for: {query}")

            if self.indexer.client is None:
                return f"Search backend unavailable: {self.indexer.init_error}"

            if not self.indexer.has_collection():
                return (
                    f"No indexed data found in collection '{self.indexer.get_collection_name()}'. "
                    "Upload and process invoices first."
                )

            res = self.indexer.search(query=query, limit=5)
            if not res:
                return "No relevant invoices found."

            # Format results
            results = []
            for hits in res:
                for hit in hits:
                    entity = hit.get("entity", {})
                    text = entity.get("text", "")
                    filename = entity.get("filename", "Unknown")
                    score = hit.get("distance", hit.get("score", 0))
                    snippet = (text[:1200] + "...") if len(text) > 1200 else text

                    results.append(
                        f"Source: {filename}\nScore: {score:.4f}\nContent: {snippet}\n---"
                    )

            return "\n\n".join(results[:10])

        except Exception as e:
            return f"Error searching invoices: {str(e)}"
