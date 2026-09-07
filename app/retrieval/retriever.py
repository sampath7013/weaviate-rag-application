from weaviate.classes.query import (
    Filter,
    MetadataQuery,
    HybridFusion,
)

from app.config import settings
from app.generation.llm import create_embedding
from app.retrieval.weaviate_client import get_weaviate_client


DEFAULT_ALPHA = 0.50
DEFAULT_MIN_SCORE = 0.20


def retrieve_documents(
    query: str,
    limit: int = 5,
    document_id: str | None = None,
    alpha: float = DEFAULT_ALPHA,
    min_score: float | None = DEFAULT_MIN_SCORE,
):
    """
    Retrieve relevant document chunks using Weaviate
    hybrid search.

    Hybrid retrieval combines:

    - BM25 keyword search
    - Vector semantic search

    alpha:
        0.0 = keyword only
        1.0 = vector only

    min_score:
        Minimum hybrid relevance score required
        for a result to be returned.
    """

    if not query or not query.strip():
        return []

    # --------------------------------------------------
    # Create OpenAI query embedding
    # --------------------------------------------------

    query_vector = create_embedding(
        query
    )

    # --------------------------------------------------
    # Connect to Weaviate
    # --------------------------------------------------

    client = get_weaviate_client()

    try:

        collection = client.collections.use(
            settings.weaviate_collection
        )

        # --------------------------------------------------
        # Optional document filter
        # --------------------------------------------------

        filters = None

        if document_id:

            filters = Filter.by_property(
                "document_id"
            ).equal(
                document_id
            )

        # --------------------------------------------------
        # Hybrid search
        #
        # query  -> BM25 component
        # vector -> semantic component
        # --------------------------------------------------

        response = collection.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=alpha,
            fusion_type=HybridFusion.RELATIVE_SCORE,
            filters=filters,
            limit=limit,
            query_properties=[
                "text"
            ],
            return_metadata=MetadataQuery(
                score=True,
                explain_score=True,
            ),
        )

        # --------------------------------------------------
        # Build final results
        # --------------------------------------------------

        results = []

        for obj in response.objects:

            score = obj.metadata.score

            # Reject weak hybrid matches
            if (
                min_score is not None
                and score is not None
                and score < min_score
            ):
                continue

            results.append(
                {
                    "text":
                        obj.properties[
                            "text"
                        ],

                    "document_name":
                        obj.properties[
                            "document_name"
                        ],

                    "document_id":
                        obj.properties.get(
                            "document_id"
                        ),

                    "page_number":
                        obj.properties[
                            "page_number"
                        ],

                    "chunk_index":
                        obj.properties[
                            "chunk_index"
                        ],

                    "score":
                        score,

                    "explain_score":
                        obj.metadata.explain_score,
                }
            )

        return results

    finally:

        client.close()