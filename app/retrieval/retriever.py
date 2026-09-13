from __future__ import annotations

import logging
from typing import Any

from weaviate.classes.query import (
    Filter,
    HybridFusion,
    MetadataQuery,
)

from app.config import settings
from app.generation.llm import (
    create_embedding,
    get_embedding_cache_stats,
)
from app.observability.tracing import (
    RAGTrace,
)
from app.retrieval.weaviate_client import (
    get_shared_weaviate_client,
)


logger = logging.getLogger(__name__)


DEFAULT_ALPHA = 0.50
DEFAULT_MIN_SCORE = 0.20


def retrieve_documents(
    query: str,
    limit: int = 5,
    document_id: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    alpha: float = DEFAULT_ALPHA,
    min_score: float | None = DEFAULT_MIN_SCORE,
    trace: RAGTrace | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve relevant chunks from Weaviate.

    Pipeline:

        query
        -> cached query embedding
        -> shared Weaviate connection
        -> hybrid vector + BM25 search
        -> metadata filtering
        -> minimum score filtering

    Supported metadata filters:

        document_id
        page_start
        page_end

    Traced stages:

        embedding
        weaviate_search
    """

    cleaned_query = (
        query.strip()
        if query
        else ""
    )

    if not cleaned_query:
        raise ValueError(
            "Retrieval query cannot be empty."
        )

    limit = max(
        1,
        limit,
    )

    alpha = max(
        0.0,
        min(
            1.0,
            alpha,
        ),
    )

    # ========================================================
    # Validate page filters
    # ========================================================

    if (
        page_start is not None
        and page_start < 1
    ):
        raise ValueError(
            "page_start must be greater than or equal to 1."
        )

    if (
        page_end is not None
        and page_end < 1
    ):
        raise ValueError(
            "page_end must be greater than or equal to 1."
        )

    if (
        page_start is not None
        and page_end is not None
        and page_start > page_end
    ):
        raise ValueError(
            "page_start cannot be greater than page_end."
        )

    logger.info(
        (
            "retrieval_started | "
            "document_id=%s | "
            "page_start=%s | "
            "page_end=%s | "
            "limit=%s | "
            "alpha=%s | "
            "min_score=%s | "
            "query=%s"
        ),
        document_id,
        page_start,
        page_end,
        limit,
        alpha,
        min_score,
        cleaned_query,
    )

    # ========================================================
    # 1. Query embedding
    # ========================================================

    cache_before = (
        get_embedding_cache_stats()
    )

    if trace is not None:
        trace.start_stage(
            "embedding"
        )

    query_vector = (
        create_embedding(
            cleaned_query
        )
    )

    if trace is not None:
        trace.end_stage(
            "embedding",
            vector_dimensions=len(
                query_vector
            ),
        )

    cache_after = (
        get_embedding_cache_stats()
    )

    embedding_cache_hit = (
        cache_after["hits"]
        > cache_before["hits"]
    )

    if trace is not None:
        trace.add_metadata(
            embedding_model=(
                settings.embedding_model
            ),
            embedding_cache_hit=(
                embedding_cache_hit
            ),
            embedding_cache_hits=(
                cache_after["hits"]
            ),
            embedding_cache_misses=(
                cache_after["misses"]
            ),
            embedding_cache_size=(
                cache_after["currsize"]
            ),
            embedding_vector_dimensions=len(
                query_vector
            ),
        )

    logger.info(
        (
            "query_embedding_completed | "
            "dimensions=%s | "
            "cache_hit=%s | "
            "model=%s | "
            "query=%s"
        ),
        len(query_vector),
        embedding_cache_hit,
        settings.embedding_model,
        cleaned_query,
    )

    # ========================================================
    # 2. Build metadata filters
    # ========================================================

    filter_conditions = []

    if document_id:
        filter_conditions.append(
            Filter
            .by_property(
                "document_id"
            )
            .equal(
                document_id
            )
        )

    if page_start is not None:
        filter_conditions.append(
            Filter
            .by_property(
                "page_number"
            )
            .greater_or_equal(
                page_start
            )
        )

    if page_end is not None:
        filter_conditions.append(
            Filter
            .by_property(
                "page_number"
            )
            .less_or_equal(
                page_end
            )
        )

    filters = None

    if len(filter_conditions) == 1:
        filters = (
            filter_conditions[0]
        )

    elif len(filter_conditions) > 1:
        filters = (
            Filter.all_of(
                filter_conditions
            )
        )

    logger.info(
        (
            "retrieval_filters_built | "
            "document_filtered=%s | "
            "page_start=%s | "
            "page_end=%s | "
            "filter_count=%s"
        ),
        bool(document_id),
        page_start,
        page_end,
        len(filter_conditions),
    )

    # ========================================================
    # 3. Weaviate hybrid search
    # ========================================================
    #
    # IMPORTANT:
    #
    # We reuse a single shared Weaviate client across queries.
    # ========================================================

    if trace is not None:
        trace.start_stage(
            "weaviate_search"
        )

    client = (
        get_shared_weaviate_client()
    )

    collection = (
        client.collections.get(
            settings.weaviate_collection
        )
    )

    response = (
        collection.query.hybrid(
            query=cleaned_query,
            vector=query_vector,
            alpha=alpha,
            fusion_type=(
                HybridFusion.RELATIVE_SCORE
            ),
            filters=filters,
            limit=limit,
            query_properties=[
                "text",
            ],
            return_metadata=(
                MetadataQuery(
                    score=True,
                    explain_score=True,
                )
            ),
        )
    )

    raw_objects = list(
        response.objects
    )

    if trace is not None:
        trace.end_stage(
            "weaviate_search",
            returned_objects=len(
                raw_objects
            ),
            limit=limit,
            document_filtered=bool(
                document_id
            ),
            page_start=page_start,
            page_end=page_end,
            page_filtered=(
                page_start is not None
                or page_end is not None
            ),
        )

    logger.info(
        (
            "weaviate_search_completed | "
            "document_id=%s | "
            "page_start=%s | "
            "page_end=%s | "
            "objects=%s | "
            "limit=%s"
        ),
        document_id,
        page_start,
        page_end,
        len(raw_objects),
        limit,
    )

    # ========================================================
    # 4. Convert Weaviate objects
    # ========================================================

    documents: list[
        dict[str, Any]
    ] = []

    for obj in raw_objects:
        properties = (
            obj.properties
            or {}
        )

        metadata = (
            obj.metadata
        )

        score = (
            metadata.score
            if metadata is not None
            else None
        )

        explain_score = (
            metadata.explain_score
            if metadata is not None
            else None
        )

        # ====================================================
        # Minimum relevance filtering
        # ====================================================

        if (
            min_score is not None
            and score is not None
            and score < min_score
        ):
            continue

        document = {
            "text": properties.get(
                "text",
                "",
            ),
            "document_name": (
                properties.get(
                    "document_name"
                )
            ),
            "document_id": (
                properties.get(
                    "document_id"
                )
            ),
            "page_number": (
                properties.get(
                    "page_number"
                )
            ),
            "chunk_index": (
                properties.get(
                    "chunk_index"
                )
            ),
            "score": score,
            "explain_score": (
                explain_score
            ),
        }

        documents.append(
            document
        )

    # ========================================================
    # 5. Add retrieval metadata
    # ========================================================

    if trace is not None:
        trace.add_metadata(
            retrieval_limit=limit,
            retrieval_raw_count=len(
                raw_objects
            ),
            retrieval_filtered_count=len(
                documents
            ),
            retrieval_document_id=(
                document_id
            ),
            retrieval_page_start=(
                page_start
            ),
            retrieval_page_end=(
                page_end
            ),
            retrieval_page_filtered=(
                page_start is not None
                or page_end is not None
            ),
            retrieval_alpha=alpha,
            retrieval_min_score=(
                min_score
            ),
            weaviate_shared_client=True,
        )

    logger.info(
        (
            "retrieval_completed | "
            "document_id=%s | "
            "page_start=%s | "
            "page_end=%s | "
            "raw_chunks=%s | "
            "returned_chunks=%s | "
            "min_score=%s | "
            "embedding_cache_hit=%s | "
            "shared_client=%s"
        ),
        document_id,
        page_start,
        page_end,
        len(raw_objects),
        len(documents),
        min_score,
        embedding_cache_hit,
        True,
    )

    return documents