import shutil
import tempfile
import unittest
from pathlib import Path

from src.pdf_downloads import discover_downloadable_pdfs


class TestPdfDownloads(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.converted_dir = self.test_dir / "converted_pdfs"
        self.converted_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_discover_downloadable_pdfs_uses_explicit_pdf_path(self):
        explicit_pdf = self.test_dir / "invoice_a.pdf"
        explicit_pdf.write_bytes(b"%PDF-explicit")

        records = {
            "invoice_a.msg": {
                "details": {
                    "pdf_path": str(explicit_pdf),
                }
            }
        }

        results = discover_downloadable_pdfs(records, converted_pdf_dir=self.converted_dir)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "invoice_a.msg")
        self.assertEqual(results[0][1], explicit_pdf)

    def test_discover_downloadable_pdfs_falls_back_to_converted_folder(self):
        fallback_pdf = self.converted_dir / "invoice_b.pdf"
        fallback_pdf.write_bytes(b"%PDF-fallback")

        records = {
            "invoice_b.msg": {
                "details": {},
            }
        }

        results = discover_downloadable_pdfs(records, converted_pdf_dir=self.converted_dir)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "invoice_b.msg")
        self.assertEqual(results[0][1], fallback_pdf)


if __name__ == "__main__":
    unittest.main()

