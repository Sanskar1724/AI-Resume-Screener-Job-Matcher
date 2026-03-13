"""Resume parsing functionality with robust PDF text extraction."""

from __future__ import annotations

from pathlib import Path

import pdfplumber

from utils.helpers import normalize_text


class ResumeParser:
    """Parses PDF resumes into normalized plain text."""

    def extract_text_from_pdf(self, pdf_path: str | Path) -> str:
        """Extract text from a PDF using pdfplumber, fallback to PyMuPDF."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"Resume file not found: {path}")

        text = self._extract_with_pdfplumber(path)
        if text:
            return normalize_text(text)

        text = self._extract_with_pymupdf(path)
        return normalize_text(text)

    def extract_text_from_uploaded_bytes(self, pdf_bytes: bytes) -> str:
        """Extract text from uploaded PDF bytes for dashboard usage."""
        text = ""
        try:
            import io

            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
                text = "\n".join(pages)
        except Exception:
            text = ""

        if not text:
            text = self._extract_bytes_with_pymupdf(pdf_bytes)

        return normalize_text(text)

    @staticmethod
    def _extract_with_pdfplumber(path: Path) -> str:
        try:
            with pdfplumber.open(path) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            return "\n".join(pages)
        except Exception:
            return ""

    @staticmethod
    def _extract_with_pymupdf(path: Path) -> str:
        try:
            import fitz

            doc = fitz.open(path)
            pages = [page.get_text("text") for page in doc]
            doc.close()
            return "\n".join(pages)
        except Exception:
            return ""

    @staticmethod
    def _extract_bytes_with_pymupdf(pdf_bytes: bytes) -> str:
        try:
            import fitz

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages = [page.get_text("text") for page in doc]
            doc.close()
            return "\n".join(pages)
        except Exception:
            return ""
