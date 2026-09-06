from app.ingestion.loaders import load_pdf
from app.ingestion.chunking import chunk_pages


pages = load_pdf("data/sample.pdf")

print("Pages loaded:", len(pages))

chunks = chunk_pages(pages)

print("Chunks created:", len(chunks))

print("\nFirst chunk:")
print(chunks[0])