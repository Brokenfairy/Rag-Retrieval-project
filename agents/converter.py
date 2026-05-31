from html import escape
import contextlib
import importlib
import io
import os
from pathlib import Path
from typing import Dict, Optional, Union

import extract_msg
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConverterAgent:
    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        pdf_dir: Optional[Union[str, Path]] = None,
        prefer_weasyprint: Optional[bool] = None,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir = Path(pdf_dir) if pdf_dir else self.data_dir
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        if prefer_weasyprint is None:
            env_value = os.getenv("USE_WEASYPRINT", "").strip().lower()
            if env_value in {"1", "true", "yes", "on"}:
                self.prefer_weasyprint = True
            elif env_value in {"0", "false", "no", "off"}:
                self.prefer_weasyprint = False
            else:
                # WeasyPrint native dependencies are often missing on Windows.
                self.prefer_weasyprint = os.name != "nt"
        else:
            self.prefer_weasyprint = prefer_weasyprint

    @staticmethod
    def _resolve_msg_opener():
        # extract-msg changed opener names across versions.
        for opener_name in ("open", "openMsg", "open_msg"):
            opener = getattr(extract_msg, opener_name, None)
            if callable(opener):
                return opener

        raise RuntimeError(
            "No supported extract-msg opener found. "
            "Expected one of: open, openMsg, open_msg."
        )

    @staticmethod
    def _stringify(value, default: str = "") -> str:
        if value is None:
            return default
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def _read_msg_fields(self, msg_path: Path) -> Dict[str, str]:
        opener = self._resolve_msg_opener()
        with opener(str(msg_path)) as msg:
            return {
                "subject": self._stringify(msg.subject, "No Subject"),
                "sender": self._stringify(msg.sender, "Unknown Sender"),
                "date": self._stringify(msg.date, "Unknown Date"),
                "body": self._stringify(msg.body, ""),
                "html_body": self._stringify(msg.htmlBody, ""),
            }

    @staticmethod
    def _to_plain_text(body: str, html_body: str) -> str:
        body = (body or "").strip()
        if body:
            return body
        html_body = (html_body or "").strip()
        if not html_body:
            return ""
        return ConverterAgent._strip_html_tags(html_body)

    @staticmethod
    def _strip_html_tags(text: str) -> str:
        # Lightweight fallback when HTML parsing dependencies are unavailable.
        import re

        no_scripts = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", text, flags=re.IGNORECASE | re.DOTALL)
        no_tags = re.sub(r"<[^>]+>", " ", no_scripts)
        normalized = re.sub(r"\s+", " ", no_tags).strip()
        return normalized

    def extract_msg_as_markdown(self, msg_file_path: str) -> str:
        """Extracts MSG metadata/body into Markdown without requiring PDF rendering."""
        msg_path = Path(msg_file_path)
        if not msg_path.exists():
            raise FileNotFoundError(f"MSG file not found: {msg_file_path}")

        fields = self._read_msg_fields(msg_path)
        body = fields["body"].strip()
        if not body and fields["html_body"].strip():
            body = self._strip_html_tags(fields["html_body"])

        if not body:
            raise ValueError("MSG body is empty after extraction.")

        return (
            f"# {fields['subject']}\n\n"
            f"- From: {fields['sender']}\n"
            f"- Date: {fields['date']}\n\n"
            f"{body}\n"
        )

    @staticmethod
    def _write_pdf_with_reportlab(
        pdf_path: Path,
        subject: str,
        sender: str,
        date: str,
        content_text: str,
    ) -> None:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        page_width, page_height = A4
        margin = 40
        line_height = 14
        text_obj = None

        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        y = page_height - margin

        def ensure_text_object(current_y: float):
            text = c.beginText(margin, current_y)
            text.setFont("Helvetica", 10)
            return text

        def draw_header_line(line: str):
            nonlocal y
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, y, line)
            y -= line_height + 2

        draw_header_line(subject or "No Subject")
        c.setFont("Helvetica", 10)
        c.drawString(margin, y, f"From: {sender or 'Unknown Sender'}")
        y -= line_height
        c.drawString(margin, y, f"Date: {date or 'Unknown Date'}")
        y -= line_height + 6

        text_obj = ensure_text_object(y)

        for raw_line in (content_text or "").splitlines():
            line = raw_line.rstrip()
            # Keep lines printable in the PDF text engine.
            safe_line = line.encode("latin-1", errors="replace").decode("latin-1")
            if text_obj.getY() <= margin:
                c.drawText(text_obj)
                c.showPage()
                y = page_height - margin
                text_obj = ensure_text_object(y)
            text_obj.textLine(safe_line)

        c.drawText(text_obj)
        c.save()

    def convert_msg_to_pdf(self, msg_file_path: str) -> str:
        """
        Converts an Outlook .msg file to a PDF.
        Returns the path to the generated PDF.
        """
        try:
            msg_path = Path(msg_file_path)
            if not msg_path.exists():
                raise FileNotFoundError(f"MSG file not found: {msg_file_path}")

            logger.info(f"Starting conversion for {msg_file_path}")

            base_name = msg_path.stem
            pdf_path = self.pdf_dir / f"{base_name}.pdf"
            if pdf_path.exists() and pdf_path.stat().st_mtime >= msg_path.stat().st_mtime:
                logger.info(f"Reusing existing PDF for {msg_file_path}: {pdf_path}")
                return str(pdf_path)

            fields = self._read_msg_fields(msg_path)
            subject = escape(fields["subject"])
            sender = escape(fields["sender"])
            date = escape(fields["date"])
            body = fields["body"]
            html_body = fields["html_body"]
            plain_text = self._to_plain_text(body, html_body)

            content = html_body if html_body else f"<pre>{escape(body)}</pre>"

            html_content = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: sans-serif; padding: 20px; }}
                    .header {{ border-bottom: 1px solid #ccc; padding-bottom: 10px; margin-bottom: 20px; }}
                    .meta {{ color: #555; font-size: 0.9em; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{subject}</h1>
                    <div class="meta">
                        <p><strong>From:</strong> {sender}</p>
                        <p><strong>Date:</strong> {date}</p>
                    </div>
                </div>
                <div class="content">
                    {content}
                </div>
            </body>
            </html>
            """

            stderr_buffer = io.StringIO()
            weasyprint_import_error: Optional[Exception] = None
            HTML = None
            if self.prefer_weasyprint:
                try:
                    with contextlib.redirect_stderr(stderr_buffer):
                        HTML = importlib.import_module("weasyprint").HTML
                except Exception as import_error:
                    weasyprint_import_error = import_error

            if HTML is not None:
                HTML(string=html_content).write_pdf(str(pdf_path))
            else:
                if self.prefer_weasyprint:
                    logger.warning(
                        "WeasyPrint unavailable for %s; using ReportLab fallback. Details: %s",
                        msg_file_path,
                        weasyprint_import_error,
                    )
                else:
                    logger.info("Using ReportLab converter for %s", msg_file_path)
                try:
                    self._write_pdf_with_reportlab(
                        pdf_path=pdf_path,
                        subject=fields["subject"],
                        sender=fields["sender"],
                        date=fields["date"],
                        content_text=plain_text,
                    )
                except Exception as reportlab_error:
                    raise RuntimeError(
                        "PDF conversion failed: WeasyPrint is unavailable and ReportLab fallback failed. "
                        "Install WeasyPrint native libs (GTK/Pango/Cairo) or install reportlab."
                    ) from reportlab_error

            logger.info(f"Successfully converted to {pdf_path}")
            return str(pdf_path)

        except Exception as e:
            logger.error(f"Failed to convert {msg_file_path}: {e}")
            raise

if __name__ == "__main__":
    # Test execution
    # Ensure there is a dummy file or handle the error gracefully for testing
    converter = ConverterAgent()
    print("Converter Agent initialized. Call convert_msg_to_pdf(path) to test.")
