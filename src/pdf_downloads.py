from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple


def discover_downloadable_pdfs(
    records: Dict[str, dict],
    converted_pdf_dir: Path | str = Path("data") / "converted_pdfs",
) -> List[Tuple[str, Path]]:
    resolved_dir = Path(converted_pdf_dir)
    downloadable: List[Tuple[str, Path]] = []
    seen_paths: set[str] = set()

    for original_name, record in (records or {}).items():
        details = record.get("details", {}) if isinstance(record, dict) else {}
        details = details if isinstance(details, dict) else {}

        pdf_path: Path | None = None
        explicit_path = details.get("pdf_path")
        if isinstance(explicit_path, str) and explicit_path.strip():
            explicit_pdf = Path(explicit_path)
            if explicit_pdf.suffix.lower() == ".pdf" and explicit_pdf.exists():
                pdf_path = explicit_pdf

        if pdf_path is None:
            fallback_pdf = resolved_dir / f"{Path(original_name).stem}.pdf"
            if fallback_pdf.exists():
                pdf_path = fallback_pdf

        if pdf_path is None:
            continue

        normalized_path = str(pdf_path.resolve())
        if normalized_path in seen_paths:
            continue

        seen_paths.add(normalized_path)
        downloadable.append((original_name, pdf_path))

    downloadable.sort(key=lambda item: item[0].lower())
    return downloadable

