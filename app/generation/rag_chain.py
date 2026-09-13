from __future__ import annotations

import logging

from app.generation.llm import (
    generate_answer,
    is_context_relevant,
)
from app.observability.tracing import RAGTrace
from app.retrieval.reranker import rerank_documents
from app.retrieval.retriever import retrieve_documents


logger = logging.getLogger(__name__)


FALLBACK_ANSWER = (
    "The available document does not contain enough "
    "relevant information to answer this question."
)


# ============================================================
# Relevance optimization settings
# ============================================================

# Score-based early rejection remains disabled.
#
# Our retrieval diagnostics showed that Weaviate hybrid
# relative scores are not calibrated enough to safely use
# a deterministic threshold for answerability.
#
# Keeping this at 0.0 means normal non-negative hybrid
# scores will not trigger score-based rejection.
EARLY_REJECT_MAX_SCORE = 0.0


def _get_valid_scores(
    documents: list[dict],
) -> list[float]:
    """
    Extract valid numeric retrieval scores.
    """

    scores: list[float] = []

    for document in documents:
        score = document.get(
            "score"
        )

        if isinstance(
            score,
            (int, float),
        ):
            scores.append(
                float(score)
            )

    return scores


def _should_early_reject(
    documents: list[dict],
) -> tuple[
    bool,
    float | None,
]:
    """
    Evaluate the score-based rejection rule.

    The heuristic is currently effectively disabled because
    EARLY_REJECT_MAX_SCORE is 0.0.

    max_score is still calculated for tracing and evaluation.
    """

    scores = _get_valid_scores(
        documents
    )

    if not scores:
        return (
            False,
            None,
        )

    max_score = max(
        scores
    )

    should_reject = (
        max_score
        < EARLY_REJECT_MAX_SCORE
    )

    return (
        should_reject,
        max_score,
    )


def ask_rag(
    question: str,
    top_k: int = 5,
    document_id: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    alpha: float = 0.5,
    min_score: float = 0.2,
    rerank: bool = False,
    candidate_k: int = 10,
) -> dict:
    """
    Execute the RAG pipeline.

    Pipeline:

        question
        -> query embedding
        -> Weaviate hybrid search
        -> document metadata filtering
        -> optional page-range filtering
        -> optional reranking
        -> retrieval score diagnostics
        -> LLM relevance gate
        -> answer generation
        -> citations

    Supported metadata filters:

        document_id
        page_start
        page_end

    Traced stages:

        embedding
        weaviate_search
        reranking       (optional)
        relevance_gate
        generation
        total latency

    Notes:

        Score-based early rejection remains disabled.

        The LLM relevance gate remains the production
        answerability mechanism.

        page_start and page_end are optional. If omitted,
        retrieval searches the entire selected document.
    """

    trace = RAGTrace()

    try:

        # ====================================================
        # Validate input
        # ====================================================

        cleaned_question = (
            question.strip()
            if question
            else ""
        )

        if not cleaned_question:
            raise ValueError(
                "Question cannot be empty."
            )

        top_k = max(
            1,
            top_k,
        )

        candidate_k = max(
            top_k,
            candidate_k,
        )

        # ====================================================
        # Validate page filters
        # ====================================================

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

        page_filtered = (
            page_start is not None
            or page_end is not None
        )

        # ====================================================
        # Initial trace metadata
        # ====================================================

        trace.add_metadata(
            question=cleaned_question,
            document_id=document_id,
            page_start=page_start,
            page_end=page_end,
            page_filtered=page_filtered,
            top_k=top_k,
            candidate_k=candidate_k,
            rerank=rerank,
            alpha=alpha,
            min_score=min_score,
            early_reject_threshold=(
                EARLY_REJECT_MAX_SCORE
            ),
        )

        logger.info(
            (
                "rag_query_started | "
                "request_id=%s | "
                "document_id=%s | "
                "page_start=%s | "
                "page_end=%s | "
                "top_k=%s | "
                "candidate_k=%s | "
                "rerank=%s | "
                "alpha=%s | "
                "min_score=%s | "
                "question=%s"
            ),
            trace.request_id,
            document_id,
            page_start,
            page_end,
            top_k,
            candidate_k,
            rerank,
            alpha,
            min_score,
            cleaned_question,
        )

        # ====================================================
        # Retrieval candidate count
        # ====================================================

        retrieval_limit = (
            candidate_k
            if rerank
            else top_k
        )

        # ====================================================
        # Retrieval
        # ====================================================

        documents = retrieve_documents(
            query=cleaned_question,
            limit=retrieval_limit,
            document_id=document_id,
            page_start=page_start,
            page_end=page_end,
            alpha=alpha,
            min_score=min_score,
            trace=trace,
        )

        logger.info(
            (
                "rag_retrieval_completed | "
                "request_id=%s | "
                "document_id=%s | "
                "page_start=%s | "
                "page_end=%s | "
                "chunks=%s | "
                "rerank=%s"
            ),
            trace.request_id,
            document_id,
            page_start,
            page_end,
            len(documents),
            rerank,
        )

        # ====================================================
        # No usable retrieval results
        # ====================================================

        if not documents:

            trace.add_metadata(
                final_chunk_count=0,
                retrieved_pages=[],
                retrieval_scores=[],
                max_retrieval_score=None,
                early_reject=False,
                relevant=False,
                relevance_gate_called=False,
                fallback_reason=(
                    "no_documents"
                ),
            )

            trace_data = trace.complete(
                success=True
            )

            logger.info(
                (
                    "rag_no_documents | "
                    "request_id=%s | "
                    "document_id=%s | "
                    "page_start=%s | "
                    "page_end=%s"
                ),
                trace.request_id,
                document_id,
                page_start,
                page_end,
            )

            return {
                "answer": FALLBACK_ANSWER,
                "sources": [],
                "reranked": rerank,
                "trace": trace_data,
            }

        # ====================================================
        # Optional reranking
        # ====================================================

        if rerank:

            input_candidate_count = len(
                documents
            )

            trace.start_stage(
                "reranking"
            )

            documents = rerank_documents(
                question=cleaned_question,
                documents=documents,
                top_k=top_k,
            )

            trace.end_stage(
                "reranking",
                input_candidates=(
                    input_candidate_count
                ),
                returned_chunks=len(
                    documents
                ),
            )

        else:

            documents = documents[
                :top_k
            ]

        # ====================================================
        # Retrieval metadata
        # ====================================================

        retrieved_pages = [
            document.get(
                "page_number"
            )
            for document
            in documents
        ]

        retrieval_scores = [
            document.get(
                "score"
            )
            for document
            in documents
        ]

        trace.add_metadata(
            final_chunk_count=len(
                documents
            ),
            retrieved_pages=(
                retrieved_pages
            ),
            retrieval_scores=(
                retrieval_scores
            ),
        )

        if rerank:

            trace.add_metadata(
                rerank_scores=[
                    document.get(
                        "rerank_score"
                    )
                    for document
                    in documents
                ],
                rerank_original_ranks=[
                    document.get(
                        "original_rank"
                    )
                    for document
                    in documents
                ],
            )

        # ====================================================
        # Retrieval score diagnostics
        # ====================================================

        (
            should_early_reject,
            max_retrieval_score,
        ) = _should_early_reject(
            documents
        )

        trace.add_metadata(
            max_retrieval_score=(
                max_retrieval_score
            ),
            early_reject=(
                should_early_reject
            ),
        )

        logger.info(
            (
                "rag_retrieval_score_diagnostics | "
                "request_id=%s | "
                "document_id=%s | "
                "page_start=%s | "
                "page_end=%s | "
                "max_score=%s | "
                "threshold=%s | "
                "early_reject=%s"
            ),
            trace.request_id,
            document_id,
            page_start,
            page_end,
            max_retrieval_score,
            EARLY_REJECT_MAX_SCORE,
            should_early_reject,
        )

        # ====================================================
        # Score-based early rejection
        # ====================================================
        #
        # With EARLY_REJECT_MAX_SCORE = 0.0 this should not
        # normally trigger for Weaviate hybrid scores.
        # ====================================================

        if should_early_reject:

            logger.info(
                (
                    "rag_early_rejected | "
                    "request_id=%s | "
                    "document_id=%s | "
                    "page_start=%s | "
                    "page_end=%s | "
                    "max_score=%s | "
                    "threshold=%s"
                ),
                trace.request_id,
                document_id,
                page_start,
                page_end,
                max_retrieval_score,
                EARLY_REJECT_MAX_SCORE,
            )

            trace.add_metadata(
                relevant=False,
                relevance_gate_called=False,
                fallback_reason=(
                    "weak_retrieval_scores"
                ),
            )

            trace_data = trace.complete(
                success=True
            )

            return {
                "answer": FALLBACK_ANSWER,
                "sources": [],
                "reranked": rerank,
                "trace": trace_data,
            }

        # ====================================================
        # LLM relevance gate
        # ====================================================

        trace.add_metadata(
            relevance_gate_called=True
        )

        trace.start_stage(
            "relevance_gate"
        )

        relevant = is_context_relevant(
            cleaned_question,
            documents,
        )

        trace.end_stage(
            "relevance_gate",
            relevant=relevant,
        )

        trace.add_metadata(
            relevant=relevant
        )

        logger.info(
            (
                "rag_context_relevance | "
                "request_id=%s | "
                "document_id=%s | "
                "page_start=%s | "
                "page_end=%s | "
                "relevant=%s | "
                "max_retrieval_score=%s"
            ),
            trace.request_id,
            document_id,
            page_start,
            page_end,
            relevant,
            max_retrieval_score,
        )

        # ====================================================
        # Reject irrelevant context
        # ====================================================

        if not relevant:

            trace.add_metadata(
                fallback_reason=(
                    "irrelevant_context"
                )
            )

            trace_data = trace.complete(
                success=True
            )

            return {
                "answer": FALLBACK_ANSWER,
                "sources": [],
                "reranked": rerank,
                "trace": trace_data,
            }

        # ====================================================
        # Generation
        # ====================================================

        trace.start_stage(
            "generation"
        )

        answer = generate_answer(
            cleaned_question,
            documents,
        )

        trace.end_stage(
            "generation",
            source_count=len(
                documents
            ),
        )

        # ====================================================
        # Complete trace
        # ====================================================

        trace_data = trace.complete(
            success=True
        )

        logger.info(
            (
                "rag_query_completed | "
                "request_id=%s | "
                "document_id=%s | "
                "page_start=%s | "
                "page_end=%s | "
                "sources=%s | "
                "rerank=%s | "
                "relevance_gate_called=%s | "
                "total_duration_ms=%.2f"
            ),
            trace.request_id,
            document_id,
            page_start,
            page_end,
            len(documents),
            rerank,
            trace_data[
                "metadata"
            ].get(
                "relevance_gate_called"
            ),
            trace_data[
                "total_duration_ms"
            ],
        )

        return {
            "answer": answer,
            "sources": documents,
            "reranked": rerank,
            "trace": trace_data,
        }

    except Exception as error:

        trace.fail(
            error
        )

        raise