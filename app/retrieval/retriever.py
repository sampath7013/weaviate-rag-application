from weaviate.classes.query import (
    Filter,
    MetadataQuery,
)

from app.config import settings
from app.generation.llm import create_embedding
from app.retrieval.weaviate_client import get_weaviate_client


DEFAULT_MAX_DISTANCE = 0.55


def retrieve_documents(
    query: str,
    limit: int = 5,
    document_id: str | None = None,
    max_distance: float | None = DEFAULT_MAX_DISTANCE,
):
    query_vector = create_embedding(query)

    client = get_weaviate_client()

    try:
        collection = client.collections.use(
            settings.weaviate_collection
        )

        filters = None

        if document_id:
            filters = Filter.by_property(
                "document_id"
            ).equal(document_id)

        response = collection.query.near_vector(
            near_vector=query_vector,
            filters=filters,
            limit=limit,
            return_metadata=MetadataQuery(
                distance=True
            ),
        )

        results = []

        for obj in response.objects:

            distance = obj.metadata.distance

            # Skip weak matches
            if (
                max_distance is not None
                and distance is not None
                and distance > max_distance
            ):
                continue

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
                    "distance": distance,
                }
            )

        return results

    finally:
        client.close()