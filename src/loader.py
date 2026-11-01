from pathlib import Path

from pypdf import PdfReader

SUPPORTED = {".txt", ".md", ".pdf"}


class DocumentError(Exception):
    pass


def load_document(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise DocumentError(f"Unsupported file type: {suffix or 'none'}")

    if suffix == ".pdf":
        try:
            reader = PdfReader(str(path))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as e:
            raise DocumentError(f"Could not read the PDF: {e}")
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")

    if not text.strip():
        raise DocumentError("No readable text in this file. Scanned PDFs are not supported.")
    return {"source": path.name, "text": text}
