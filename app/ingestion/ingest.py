from pathlib import Path
import logging

from weaviate.classes.query import Filter

from app.config import settings

from app.generation.llm import (
    create_embeddings,
)

from app.ingestion.loaders import load_pdf
from app.ingestion.chunking import chunk_pages
from app.ingestion.file_utils import calculate_file_hash

from app.retrieval.weaviate_client import (
    get_weaviate_client,
    create_collection,
    ensure_required_properties,
)


logger = logging.getLogger(__name__)


def find_existing_document(
    collection,
    file_hash: str,
) -> dict | None:

    response = collection.query.fetch_objects(
        filters=Filter.by_property(
            "file_hash"
        ).equal(file_hash),
        limit=1,
    )

    if not response.objects:
        return None

    properties = response.objects[
        0
    ].properties

    return {
        "document_id":
            properties.get(
                "document_id"
            ),

        "filename":
            properties.get(
                "document_name"
            ),
    }


def ingest_pdf(
    file_path: str,
    document_id: str,
    document_name: str | None = None,
) -> dict:

    path = Path(
        file_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    display_name = (
        document_name
        if document_name
        else path.name
    )

    logger.info(
        "ingestion_started | document_id=%s | filename=%s",
        document_id,
        display_name,
    )

    file_hash = calculate_file_hash(
        file_path
    )

    logger.info(
        "file_hash_created | document_id=%s | hash=%s",
        document_id,
        file_hash,
    )

    client = get_weaviate_client()

    try:
        create_collection(
            client
        )

        ensure_required_properties(
            client
        )

        collection = client.collections.use(
            settings.weaviate_collection
        )

        existing_document = (
            find_existing_document(
                collection=collection,
                file_hash=file_hash,
            )
        )

        if existing_document:

            logger.info(
                "duplicate_detected | document_id=%s | existing_document_id=%s",
                document_id,
                existing_document["document_id"],
            )

            return {
                "document_id":
                    existing_document[
                        "document_id"
                    ],

                "filename":
                    existing_document[
                        "filename"
                    ],

                "pages":
                    None,

                "chunks":
                    None,

                "file_hash":
                    file_hash,

                "duplicate":
                    True,
            }

        pages = load_pdf(
            file_path
        )

        if not pages:
            raise ValueError(
                "No readable text was found in the PDF."
            )

        logger.info(
            "pdf_loaded | document_id=%s | pages=%s",
            document_id,
            len(pages),
        )

        for page in pages:
            page[
                "document_name"
            ] = display_name

        chunks = chunk_pages(
            pages
        )

        if not chunks:
            raise ValueError(
                "No text chunks were created from the PDF."
            )

        logger.info(
            "chunks_created | document_id=%s | chunks=%s",
            document_id,
            len(chunks),
        )

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        logger.info(
            "embedding_started | document_id=%s | chunks=%s",
            document_id,
            len(texts),
        )

        embeddings = create_embeddings(
            texts
        )

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                "Embedding count does not match "
                "the number of chunks."
            )

        logger.info(
            "embedding_completed | document_id=%s | embeddings=%s",
            document_id,
            len(embeddings),
        )

        logger.info(
            "weaviate_batch_started | document_id=%s",
            document_id,
        )

        with collection.batch.dynamic() as batch:

            for chunk, embedding in zip(
                chunks,
                embeddings,
            ):

                batch.add_object(
                    properties={
                        "text":
                            chunk["text"],

                        "document_name":
                            display_name,

                        "document_id":
                            document_id,

                        "file_hash":
                            file_hash,

                        "page_number":
                            chunk[
                                "page_number"
                            ],

                        "chunk_index":
                            chunk[
                                "chunk_index"
                            ],
                    },
                    vector=embedding,
                )

        failed_objects = (
            collection.batch.failed_objects
        )

        if failed_objects:

            raise RuntimeError(
                "Weaviate batch ingestion failed "
                f"for {len(failed_objects)} object(s)."
            )

        logger.info(
            "ingestion_completed | document_id=%s | pages=%s | chunks=%s",
            document_id,
            len(pages),
            len(chunks),
        )

        return {
            "document_id":
                document_id,

            "filename":
                display_name,

            "pages":
                len(pages),

            "chunks":
                len(chunks),

            "file_hash":
                file_hash,

            "duplicate":
                False,
        }

    except Exception:

        logger.exception(
            "ingestion_failed | document_id=%s | filename=%s",
            document_id,
            display_name,
        )

        raise

    finally:

        client.close()