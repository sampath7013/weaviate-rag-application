from app.config import settings
from app.generation.llm import create_embedding
from app.retrieval.weaviate_client import get_weaviate_client


def retrieve_documents(query: str, limit: int = 5):
    # Convert the user question into an embedding vector
    query_vector = create_embedding(query)

    client = get_weaviate_client()

    try:
        collection = client.collections.use(
            settings.weaviate_collection
        )

        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
        )

        results = []

        for obj in response.objects:
            results.append(
                {
                    "text": obj.properties["text"],
                    "document_name": obj.properties["document_name"],
                    "page_number": obj.properties["page_number"],
                    "chunk_index": obj.properties["chunk_index"],
                }
            )

        return results

    finally:
        client.close()