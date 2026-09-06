from app.config import settings
from app.generation.llm import create_embedding
from app.ingestion.loaders import load_pdf
from app.ingestion.chunking import chunk_pages
from app.retrieval.weaviate_client import (
    get_weaviate_client,
    create_collection,
)


def ingest_pdf(file_path: str):
    print(f"Loading PDF: {file_path}")

    pages = load_pdf(file_path)

    print(f"Pages loaded: {len(pages)}")

    chunks = chunk_pages(pages)

    print(f"Chunks created: {len(chunks)}")

    client = get_weaviate_client()

    try:
        create_collection(client)

        collection = client.collections.use(
            settings.weaviate_collection
        )

        for index, chunk in enumerate(chunks, start=1):
            print(
                f"Processing chunk {index}/{len(chunks)}..."
            )

            embedding = create_embedding(
                chunk["text"]
            )

            collection.data.insert(
                properties={
                    "text": chunk["text"],
                    "document_name": chunk["document_name"],
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                },
                vector=embedding,
            )

        print("\nPDF ingestion completed successfully!")

        return {
            "filename": chunks[0]["document_name"] if chunks else None,
            "pages": len(pages),
            "chunks": len(chunks),
        }

    finally:
        client.close()