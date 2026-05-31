"""Standalone utility: convert .msg files into PDFs using extract_msg.

Default behavior:
- Input folder:  QuoteAI samples
- Output folder: data/converted_pdfs
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path
from typing import Callable, Iterable

import extract_msg


def resolve_msg_opener() -> Callable[[str], object]:
    """Support multiple extract_msg opener names across versions."""
    for opener_name in ("open", "openMsg", "open_msg"):
        opener = getattr(extract_msg, opener_name, None)
        if callable(opener):
            return opener
    raise RuntimeError(
        "No supported extract_msg opener found. Expected one of: open, openMsg, open_msg."
    )


def safe_print(message: str) -> None:
    text = str(message)
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        sanitized = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(sanitized)


def stringify(value: object, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def strip_html_tags(text: str) -> str:
    no_scripts = re.sub(
        r"<(script|style)\b[^>]*>.*?</\1>",
        "",
        text or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    no_tags = re.sub(r"<[^>]+>", " ", no_scripts)
    return re.sub(r"\s+", " ", no_tags).strip()


def read_msg_fields(msg_path: Path) -> dict[str, str]:
    def safe_attr(msg: object, attr_name: str, default: str = "") -> str:
        try:
            return stringify(getattr(msg, attr_name, None), default)
        except Exception:
            return default

    opener = resolve_msg_opener()
    with opener(str(msg_path)) as msg:
        return {
            "subject": safe_attr(msg, "subject", "No Subject"),
            "sender": safe_attr(msg, "sender", "Unknown Sender"),
            "date": safe_attr(msg, "date", "Unknown Date"),
            "body": safe_attr(msg, "body", ""),
            "html_body": safe_attr(msg, "htmlBody", ""),
        }


def to_plain_text(body: str, html_body: str) -> str:
    if (body or "").strip():
        return body.strip()
    if (html_body or "").strip():
        return strip_html_tags(html_body)
    return ""


def latin1_safe(text: str) -> str:
    return (text or "").encode("latin-1", errors="replace").decode("latin-1")


def iter_wrapped_lines(text: str, width: int = 110) -> Iterable[str]:
    for raw_line in (text or "").splitlines():
        wrapped = textwrap.wrap(raw_line.rstrip(), width=width) or [""]
        for line in wrapped:
            yield line


def write_pdf(pdf_path: Path, subject: str, sender: str, date: str, body_text: str) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    all_lines = [
        subject or "No Subject",
        f"From: {sender or 'Unknown Sender'}",
        f"Date: {date or 'Unknown Date'}",
        "",
    ]
    all_lines.extend(iter_wrapped_lines(body_text, width=100))
    if len(all_lines) == 4:
        all_lines.append("(No message body)")

    pages = paginate_lines(all_lines, lines_per_page=54)
    write_minimal_pdf(pdf_path, pages)


def paginate_lines(lines: list[str], lines_per_page: int) -> list[list[str]]:
    if not lines:
        return [[""]]
    return [lines[i : i + lines_per_page] for i in range(0, len(lines), lines_per_page)]


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_page_stream(lines: list[str]) -> bytes:
    commands: list[str] = ["BT", "/F1 10 Tf", "14 TL", "40 802 Td"]
    first_line = True
    for line in lines:
        safe = pdf_escape(latin1_safe(line))
        if not first_line:
            commands.append("T*")
        commands.append(f"({safe}) Tj")
        first_line = False
    commands.append("ET")
    return "\n".join(commands).encode("latin-1", errors="replace")


def write_minimal_pdf(pdf_path: Path, pages: list[list[str]]) -> None:
    objects: dict[int, bytes] = {}
    font_obj_id = 3
    objects[font_obj_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    next_obj_id = 4
    page_obj_ids: list[int] = []

    for page_lines in pages:
        content_stream = build_page_stream(page_lines)
        content_obj = (
            f"<< /Length {len(content_stream)} >>\nstream\n".encode("ascii")
            + content_stream
            + b"\nendstream"
        )
        content_obj_id = next_obj_id
        next_obj_id += 1
        objects[content_obj_id] = content_obj

        page_obj_id = next_obj_id
        next_obj_id += 1
        objects[page_obj_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 {font_obj_id} 0 R >> >> "
            f"/Contents {content_obj_id} 0 R >>"
        ).encode("ascii")
        page_obj_ids.append(page_obj_id)

    kids = " ".join(f"{obj_id} 0 R" for obj_id in page_obj_ids)
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_obj_ids)} >>".encode("ascii")
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"

    max_obj_id = max(objects)
    with pdf_path.open("wb") as f:
        f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

        offsets: dict[int, int] = {}
        for obj_id in range(1, max_obj_id + 1):
            offsets[obj_id] = f.tell()
            f.write(f"{obj_id} 0 obj\n".encode("ascii"))
            f.write(objects[obj_id])
            f.write(b"\nendobj\n")

        xref_pos = f.tell()
        f.write(f"xref\n0 {max_obj_id + 1}\n".encode("ascii"))
        f.write(b"0000000000 65535 f \n")
        for obj_id in range(1, max_obj_id + 1):
            f.write(f"{offsets[obj_id]:010d} 00000 n \n".encode("ascii"))

        f.write(f"trailer\n<< /Size {max_obj_id + 1} /Root 1 0 R >>\n".encode("ascii"))
        f.write(f"startxref\n{xref_pos}\n%%EOF\n".encode("ascii"))


def convert_folder(source_dir: Path, output_dir: Path, force: bool = False) -> int:
    msg_files = sorted(source_dir.rglob("*.msg"))
    if not msg_files:
        safe_print(f"No .msg files found in: {source_dir}")
        return 0

    converted = 0
    skipped = 0
    failed = 0

    for msg_path in msg_files:
        pdf_path = output_dir / f"{msg_path.stem}.pdf"

        if not force and pdf_path.exists():
            skipped += 1
            safe_print(f"SKIP  {msg_path.name} -> {pdf_path.name} (already exists)")
            continue

        try:
            fields = read_msg_fields(msg_path)
            plain_text = to_plain_text(fields["body"], fields["html_body"])
            write_pdf(
                pdf_path=pdf_path,
                subject=fields["subject"],
                sender=fields["sender"],
                date=fields["date"],
                body_text=plain_text,
            )
            converted += 1
            safe_print(f"OK    {msg_path.name} -> {pdf_path.name}")
        except Exception as exc:
            failed += 1
            safe_print(f"FAIL  {msg_path.name} ({exc})")

    safe_print("\nSummary")
    safe_print(f"Converted: {converted}")
    safe_print(f"Skipped:   {skipped}")
    safe_print(f"Failed:    {failed}")
    safe_print(f"Output:    {output_dir}")

    return 0 if failed == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert .msg files to .pdf using extract_msg.")
    parser.add_argument(
        "--input-dir",
        default="QuoteAI samples",
        help="Folder containing .msg files (default: QuoteAI samples).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("data") / "converted_pdfs"),
        help="Folder to store generated PDFs (default: data/converted_pdfs).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild PDFs even when output is newer than source .msg.",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="replace")

    args = parse_args()
    source_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not source_dir.exists():
        safe_print(f"Input folder not found: {source_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    return convert_folder(source_dir=source_dir, output_dir=output_dir, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
