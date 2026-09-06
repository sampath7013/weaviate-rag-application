import uuid

from app.ingestion.ingest import ingest_pdf


if __name__ == "__main__":

    result = ingest_pdf(
        file_path="data/sample.pdf",
        document_id=str(
            uuid.uuid4()
        ),
        document_name="sample.pdf",
    )

    print(result)