import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import shutil
import tempfile
from pathlib import Path

# Create a mock for weasyprint before importing agents.converter
# This is necessary because weasyprint fails to import if GTK is not installed
mock_weasyprint = MagicMock()
sys.modules["weasyprint"] = mock_weasyprint

from agents.converter import ConverterAgent

class TestConverterAgent(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.converter = ConverterAgent(data_dir=self.test_dir, prefer_weasyprint=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        # We don't remove weasyprint from sys.modules because it might be needed by other tests
        # and unpatching valid modules is tricky. But here we injected it.

    @patch("agents.converter.ConverterAgent._resolve_msg_opener")
    def test_convert_msg_to_pdf(self, mock_resolve_msg_opener):
        """Test converting an MSG file to PDF."""
        # Setup mock MSG file
        msg_file_path = os.path.join(self.test_dir, "test_email.msg")
        with open(msg_file_path, "w") as f:
            f.write("dummy msg content")

        # Mock extract_msg behavior
        mock_msg_instance = MagicMock()
        mock_msg_instance.subject = "Test Subject"
        mock_msg_instance.sender = "Sender Name"
        mock_msg_instance.date = "2023-10-27"
        mock_msg_instance.body = "Body content"
        mock_msg_instance.htmlBody = None
        mock_msg_opener = MagicMock()
        mock_msg_opener.return_value.__enter__.return_value = mock_msg_instance
        mock_resolve_msg_opener.return_value = mock_msg_opener

        # Mock WeasyPrint HTML behavior
        # We need to configure the mock we put in sys.modules
        mock_html_class = mock_weasyprint.HTML
        mock_html_instance = MagicMock()
        mock_html_class.return_value = mock_html_instance

        # Run conversion
        pdf_path = self.converter.convert_msg_to_pdf(msg_file_path)

        # Assertions
        expected_pdf_path = os.path.join(self.test_dir, "test_email.pdf")
        self.assertEqual(pdf_path, expected_pdf_path)
        
        # Verify WeasyPrint was called
        mock_html_class.assert_called_once()
        # Verify write_pdf was called with correct path
        mock_html_instance.write_pdf.assert_called_once_with(expected_pdf_path)

    def test_convert_nonexistent_file(self):
        """Test that converting a non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.converter.convert_msg_to_pdf("non_existent_file.msg")

    @patch("agents.converter.ConverterAgent._resolve_msg_opener")
    def test_reuse_existing_pdf(self, mock_resolve_msg_opener):
        """Test that existing PDF is reused if it's newer than the MSG file."""
        msg_file_path = os.path.join(self.test_dir, "existing.msg")
        pdf_file_path = os.path.join(self.test_dir, "existing.pdf")
        
        # Create dummy MSG and PDF files
        with open(msg_file_path, "w") as f:
            f.write("msg")
        with open(pdf_file_path, "w") as f:
            f.write("pdf")

        # Set modification times: PDF newer than MSG
        os.utime(msg_file_path, (1000, 1000))
        os.utime(pdf_file_path, (2000, 2000))

        # Run conversion
        result_path = self.converter.convert_msg_to_pdf(msg_file_path)

        # Assertions
        self.assertEqual(result_path, pdf_file_path)
        mock_resolve_msg_opener.assert_not_called()

    @patch("agents.converter.ConverterAgent._resolve_msg_opener")
    def test_convert_msg_to_pdf_in_custom_output_folder(self, mock_resolve_msg_opener):
        """Test that converted PDFs are stored in a dedicated folder when configured."""
        pdf_output_dir = os.path.join(self.test_dir, "converted_pdfs")
        converter = ConverterAgent(
            data_dir=self.test_dir,
            pdf_dir=pdf_output_dir,
            prefer_weasyprint=True,
        )

        msg_file_path = os.path.join(self.test_dir, "folder_target.msg")
        with open(msg_file_path, "w") as f:
            f.write("dummy msg content")

        mock_msg_instance = MagicMock()
        mock_msg_instance.subject = "Folder Test"
        mock_msg_instance.sender = "Sender Name"
        mock_msg_instance.date = "2023-10-27"
        mock_msg_instance.body = "Body content"
        mock_msg_instance.htmlBody = None
        mock_msg_opener = MagicMock()
        mock_msg_opener.return_value.__enter__.return_value = mock_msg_instance
        mock_resolve_msg_opener.return_value = mock_msg_opener

        mock_html_class = mock_weasyprint.HTML
        mock_html_instance = MagicMock()
        mock_html_class.return_value = mock_html_instance

        pdf_path = converter.convert_msg_to_pdf(msg_file_path)

        expected_pdf_path = os.path.join(pdf_output_dir, "folder_target.pdf")
        self.assertEqual(pdf_path, expected_pdf_path)
        self.assertTrue(os.path.isdir(pdf_output_dir))
        mock_html_instance.write_pdf.assert_called_once_with(expected_pdf_path)

    @patch("agents.converter.importlib.import_module")
    @patch("agents.converter.ConverterAgent._resolve_msg_opener")
    def test_convert_msg_to_pdf_falls_back_to_reportlab(
        self,
        mock_resolve_msg_opener,
        mock_import_module,
    ):
        msg_file_path = os.path.join(self.test_dir, "fallback.msg")
        with open(msg_file_path, "w") as f:
            f.write("dummy msg content")

        mock_msg_instance = MagicMock()
        mock_msg_instance.subject = "Fallback Subject"
        mock_msg_instance.sender = "Fallback Sender"
        mock_msg_instance.date = "2023-10-27"
        mock_msg_instance.body = "Fallback body"
        mock_msg_instance.htmlBody = None
        mock_msg_opener = MagicMock()
        mock_msg_opener.return_value.__enter__.return_value = mock_msg_instance
        mock_resolve_msg_opener.return_value = mock_msg_opener

        mock_import_module.side_effect = ModuleNotFoundError("weasyprint unavailable")

        pdf_path = self.converter.convert_msg_to_pdf(msg_file_path)
        expected_pdf_path = os.path.join(self.test_dir, "fallback.pdf")

        self.assertEqual(pdf_path, expected_pdf_path)
        self.assertTrue(os.path.exists(expected_pdf_path))

if __name__ == "__main__":
    unittest.main()
