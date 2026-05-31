import os
import logging
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LlamaParserAgent:
    def __init__(self):
        self.api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        self._parser: Optional["LlamaParse"] = None
        if not self.api_key:
            logger.warning("LLAMA_CLOUD_API_KEY not found. Falling back to local PDF parsing.")

    def _get_parser(self):
        if self._parser is None:
            from llama_parse import LlamaParse

            self._parser = LlamaParse(
                api_key=self.api_key,
                result_type="markdown",
                verbose=False,
                language="en",
            )
        return self._parser

    @staticmethod
    def _local_pdf_fallback(pdf_path: str) -> str:
        from langchain_community.document_loaders import PyPDFLoader

        docs = PyPDFLoader(pdf_path).load()
        return "\n\n".join(doc.page_content for doc in docs if doc.page_content.strip())

    def parse_pdf(self, pdf_path: str, instruction: str = "Extract all line items, tax, and total amounts accurately into Markdown tables.") -> str:
        """
        Parses a PDF file using LlamaParse into Markdown.
        """
        try:
            logger.info(f"Starting parsing for {pdf_path}")

            if not self.api_key:
                logger.info("Using local PDF fallback parser.")
                return self._local_pdf_fallback(pdf_path)

            parser = self._get_parser()
            documents = parser.load_data(pdf_path)
            full_text = "\n\n".join(doc.text for doc in documents if getattr(doc, "text", "").strip())
            if not full_text.strip():
                raise ValueError("LlamaParse returned empty output.")

            logger.info(f"Successfully parsed {pdf_path}")
            return full_text

        except Exception as e:
            logger.warning(f"LlamaParse failed for {pdf_path}, using local fallback: {e}")
            return self._local_pdf_fallback(pdf_path)

if __name__ == "__main__":
    parser = LlamaParserAgent()
    print("LlamaParser Agent initialized.")
