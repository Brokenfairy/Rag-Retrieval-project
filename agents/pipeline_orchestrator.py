import os
import sys
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.converter import ConverterAgent
from agents.llama_parser import LlamaParserAgent
from agents.indexer import MilvusIndexerAgent
from data.state_manager import StateManager, FileStatus

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()


class InvoicePipeline:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.converted_pdf_dir = self.data_dir / "converted_pdfs"
        self.converted_pdf_dir.mkdir(parents=True, exist_ok=True)
        self.state_manager = StateManager()
        self.converter = ConverterAgent(data_dir=self.data_dir, pdf_dir=self.converted_pdf_dir)
        self.parser = LlamaParserAgent()
        self.indexer = MilvusIndexerAgent()

    def _build_metadata(self, filename: str) -> dict:
        record = self.state_manager.get_file_record(filename)
        return {
            "filename": filename,
            "upload_date": record.get("upload_time", datetime.now(timezone.utc).isoformat()),
            "last_updated": record.get("last_updated", datetime.now(timezone.utc).isoformat()),
        }

    def process_file(self, filename: str):
        """
        Orchestrates the full pipeline for a single file:
        MSG -> PDF -> Validated Markdown -> Milvus Index
        """
        safe_filename = os.path.basename(filename)
        file_path = self.data_dir / safe_filename

        try:
            self.state_manager.add_file(safe_filename)

            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                self.state_manager.update_status(
                    safe_filename,
                    FileStatus.FAILED,
                    {"error": "File not found"},
                    replace_details=True,
                )
                return

            if self.indexer.client is None:
                self.state_manager.update_status(
                    safe_filename,
                    FileStatus.FAILED,
                    {
                        "error": (
                            "Vector index backend unavailable. "
                            f"Details: {self.indexer.init_error}"
                        ),
                        "failed_at": "Indexer Initialization",
                    },
                    replace_details=True,
                )
                return

            ext = file_path.suffix.lower()
            pdf_path = file_path
            markdown_text = None
            stored_pdf_path = None

            if ext == ".msg":
                self.state_manager.update_status(
                    safe_filename,
                    FileStatus.PENDING,
                    {"step": "Converting MSG to PDF"},
                    replace_details=True,
                )
                try:
                    pdf_path = Path(self.converter.convert_msg_to_pdf(str(file_path)))
                    stored_pdf_path = str(pdf_path)
                    self.state_manager.update_status(
                        safe_filename,
                        FileStatus.CONVERTED,
                        {"step": "Converted to PDF", "pdf_path": str(pdf_path)},
                        replace_details=True,
                    )
                except Exception as e:
                    logger.warning(
                        "PDF conversion failed for %s, attempting direct MSG extraction fallback: %s",
                        safe_filename,
                        e,
                    )
                    try:
                        markdown_text = self.converter.extract_msg_as_markdown(str(file_path))
                        if not markdown_text.strip():
                            raise ValueError("MSG fallback produced empty markdown.")
                        self.state_manager.update_status(
                            safe_filename,
                            FileStatus.PARSED,
                            {
                                "step": "Parsed directly from MSG fallback",
                                "markdown_chars": len(markdown_text),
                                "conversion_error": str(e),
                            },
                            replace_details=True,
                        )
                    except Exception as fallback_error:
                        self.state_manager.update_status(
                            safe_filename,
                            FileStatus.FAILED,
                            {
                                "error": str(fallback_error),
                                "failed_at": "Conversion",
                                "conversion_error": str(e),
                            },
                            replace_details=True,
                        )
                        return
            elif ext == ".pdf":
                stored_pdf_path = str(pdf_path)
                self.state_manager.update_status(
                    safe_filename,
                    FileStatus.CONVERTED,
                    {"step": "Source file is already PDF", "pdf_path": str(pdf_path)},
                    replace_details=True,
                )
            else:
                self.state_manager.update_status(
                    safe_filename,
                    FileStatus.FAILED,
                    {"error": f"Unsupported file type: {ext}", "failed_at": "Validation"},
                    replace_details=True,
                )
                return

            if markdown_text is None:
                try:
                    markdown_text = self.parser.parse_pdf(str(pdf_path))
                    if not markdown_text.strip():
                        raise ValueError("Parser returned empty markdown output.")
                    self.state_manager.update_status(
                        safe_filename,
                        FileStatus.PARSED,
                        {"step": "Parsed with LlamaParse", "markdown_chars": len(markdown_text)},
                        replace_details=True,
                    )
                except Exception as e:
                    self.state_manager.update_status(
                        safe_filename,
                        FileStatus.FAILED,
                        {"error": str(e), "failed_at": "Parsing"},
                        replace_details=True,
                    )
                    return

            try:
                metadata = self._build_metadata(safe_filename)
                chunks_indexed = self.indexer.index_document(markdown_text, metadata)
                indexed_details = {"step": "Completed", "chunks_indexed": chunks_indexed}
                if stored_pdf_path:
                    indexed_details["pdf_path"] = stored_pdf_path
                self.state_manager.update_status(
                    safe_filename,
                    FileStatus.INDEXED,
                    indexed_details,
                    replace_details=True,
                )
            except Exception as e:
                self.state_manager.update_status(
                    safe_filename,
                    FileStatus.FAILED,
                    {"error": str(e), "failed_at": "Indexing"},
                    replace_details=True,
                )
                return

            logger.info(f"Successfully processed {safe_filename}")

        except Exception as e:
            logger.exception(f"Critical pipeline error for {safe_filename}: {e}")
            self.state_manager.update_status(
                safe_filename,
                FileStatus.FAILED,
                {"error": f"Critical: {str(e)}"},
                replace_details=True,
            )

    def process_in_background(self, filename: str):
        """Runs the pipeline in a separate thread to not block UI."""
        thread = threading.Thread(
            target=self.process_file,
            args=(filename,),
            daemon=True,
            name=f"pipeline-{os.path.basename(filename)}",
        )
        thread.start()
        return thread

if __name__ == "__main__":
    # Test script usage
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        pipeline = InvoicePipeline()
        pipeline.process_file(filename)
    else:
        print("Usage: python pipeline_orchestrator.py <filename>")
