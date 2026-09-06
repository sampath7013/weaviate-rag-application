from weaviate.classes.query import Filter

from app.config import settings
from app.generation.llm import create_embedding
from app.retrieval.weaviate_client import (
    get_weaviate_client,
)


def retrieve_documents(
    query: str,
    limit: int = 5,
    document_id: str | None = None,
):
    query_vector = create_embedding(query)

    client = get_weaviate_client()

    try:
        collection = client.collections.use(
            settings.weaviate_collection
        )

        filters = None

        if document_id:
            filters = (
                Filter
                .by_property("document_id")
                .equal(document_id)
            )

        response = collection.query.near_vector(
            near_vector=query_vector,
            filters=filters,
            limit=limit,
        )

        results = []

        for obj in response.objects:
            results.append(
                {
                    "text": obj.properties["text"],
                    "document_name": obj.properties[
                        "document_name"
                    ],
                    "document_id": obj.properties.get(
                        "document_id"
                    ),
                    "page_number": obj.properties[
                        "page_number"
                    ],
                    "chunk_index": obj.properties[
                        "chunk_index"
                    ],
                }
            )

        return results

    finally:
        client.close()