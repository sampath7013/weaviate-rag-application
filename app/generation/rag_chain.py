import logging

from app.generation.llm import (
    generate_answer,
    is_context_relevant,
)
from app.retrieval.retriever import retrieve_documents


logger = logging.getLogger(__name__)


FALLBACK_ANSWER = (
    "The available document does not contain enough "
    "relevant information to answer this question."
)


def ask_rag(
    question: str,
    top_k: int = 5,
    document_id: str | None = None,
    alpha: float = 0.5,
    min_score: float = 0.2,
) -> dict:
    """
    Run the complete Retrieval-Augmented Generation pipeline.

    Pipeline:
    1. Validate the question.
    2. Retrieve relevant document chunks from Weaviate.
    3. Apply hybrid search and relevance filtering.
    4. Use the LLM relevance gate.
    5. Generate an answer using only retrieved document context.
    6. Return the answer and source chunks.

    Args:
        question:
            User question.

        top_k:
            Maximum number of chunks to retrieve.

        document_id:
            Optional document identifier used to restrict
            retrieval to the currently selected document.

        alpha:
            Hybrid retrieval weighting.

            0.0 = pure BM25 keyword search
            1.0 = pure semantic vector search
            0.5 = balanced hybrid search

        min_score:
            Minimum hybrid relevance score required for
            a retrieved chunk to be retained.

    Returns:
        dict containing:
        - question
        - answer
        - sources
    """

    # ========================================================
    # Validate question
    # ========================================================

    cleaned_question = question.strip()

    if not cleaned_question:

        logger.warning(
            "rag_empty_question"
        )

        return {
            "question": question,
            "answer": FALLBACK_ANSWER,
            "sources": [],
        }


    # ========================================================
    # Query start logging
    # ========================================================

    logger.info(
        (
            "rag_query_started | "
            "document_id=%s | "
            "top_k=%s | "
            "alpha=%s | "
            "min_score=%s | "
            "question=%s"
        ),
        document_id,
        top_k,
        alpha,
        min_score,
        cleaned_question,
    )


    # ========================================================
    # Retrieve document chunks
    # ========================================================

    documents = retrieve_documents(
        query=cleaned_question,
        limit=top_k,
        document_id=document_id,
        alpha=alpha,
        min_score=min_score,
    )


    logger.info(
        (
            "rag_retrieval_completed | "
            "document_id=%s | "
            "chunks=%s"
        ),
        document_id,
        len(documents),
    )


    # ========================================================
    # No relevant chunks
    # ========================================================

    if not documents:

        logger.info(
            (
                "rag_no_relevant_documents | "
                "document_id=%s | "
                "question=%s"
            ),
            document_id,
            cleaned_question,
        )

        return {
            "question": cleaned_question,
            "answer": FALLBACK_ANSWER,
            "sources": [],
        }


    # ========================================================
    # LLM relevance gate
    # ========================================================

    try:

        relevant = is_context_relevant(
            question=cleaned_question,
            retrieved_documents=documents,
        )

    except Exception:

        logger.exception(
            (
                "rag_relevance_gate_failed | "
                "document_id=%s | "
                "question=%s"
            ),
            document_id,
            cleaned_question,
        )

        raise


    logger.info(
        (
            "rag_context_relevance | "
            "document_id=%s | "
            "relevant=%s"
        ),
        document_id,
        relevant,
    )


    # ========================================================
    # Retrieved chunks are not sufficient
    # ========================================================

    if not relevant:

        logger.info(
            (
                "rag_context_rejected | "
                "document_id=%s | "
                "question=%s"
            ),
            document_id,
            cleaned_question,
        )

        return {
            "question": cleaned_question,
            "answer": FALLBACK_ANSWER,
            "sources": [],
        }


    # ========================================================
    # Generate grounded answer
    # ========================================================

    try:

        answer = generate_answer(
            question=cleaned_question,
            retrieved_documents=documents,
        )

    except Exception:

        logger.exception(
            (
                "rag_generation_failed | "
                "document_id=%s | "
                "question=%s"
            ),
            document_id,
            cleaned_question,
        )

        raise


    # ========================================================
    # Completed
    # ========================================================

    logger.info(
        (
            "rag_query_completed | "
            "document_id=%s | "
            "sources=%s"
        ),
        document_id,
        len(documents),
    )


    return {
        "question": cleaned_question,
        "answer": answer,
        "sources": documents,
    }