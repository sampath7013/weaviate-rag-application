from pathlib import Path
import pymupdf


def load_pdf(file_path: str):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    document = pymupdf.open(file_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()

        if text:
            pages.append(
                {
                    "document_name": path.name,
                    "page_number": page_number,
                    "text": text,
                }
            )

    document.close()

    return pages