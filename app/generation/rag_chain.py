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
    Run the complete RAG pipeline.

    Steps:
    1. Retrieve relevant document chunks from Weaviate.
    2. Apply hybrid relevance filtering.
    3. Use an LLM relevance gate.
    4. Generate an answer grounded only in retrieved context.

    Args:
        question:
            User's question.

        top_k:
            Maximum number of chunks to retrieve.

        document_id:
            Optional document ID used to restrict retrieval
            to a specific uploaded document.

        alpha:
            Hybrid search weighting.

            0.0 = pure BM25 keyword search
            1.0 = pure vector semantic search
            0.5 = balanced hybrid search

        min_score:
            Minimum hybrid relevance score required for
            a retrieved chunk to be kept.

    Returns:
        Dictionary containing:
        - question
        - answer
        - sources
    """

    cleaned_question = question.strip()

    if not cleaned_question:
        return {
            "question": question,
            "answer": FALLBACK_ANSWER,
            "sources": [],
        }


    logger.info(
        (
            "rag_query_started | "
            "document_id=%s | "
            "top_k=%s | "
            "alpha=%s | "
            "min_score=%s"
        ),
        document_id,
        top_k,
        alpha,
        min_score,
    )


    # ========================================================
    # Retrieval
    # ========================================================

    documents = retrieve_documents(
        query=cleaned_question,
        top_k=top_k,
        document_id=document_id,
        alpha=alpha,
        min_score=min_score,
    )


    logger.info(
        "rag_retrieval_completed | chunks=%s",
        len(documents),
    )


    # ========================================================
    # No sufficiently relevant chunks
    # ========================================================

    if not documents:

        logger.info(
            (
                "rag_no_relevant_documents | "
                "document_id=%s"
            ),
            document_id,
        )

        return {
            "question": cleaned_question,
            "answer": FALLBACK_ANSWER,
            "sources": [],
        }


    # ========================================================
    # LLM relevance gate
    # ========================================================

    relevant = is_context_relevant(
        question=cleaned_question,
        retrieved_documents=documents,
    )


    logger.info(
        "rag_context_relevance | relevant=%s",
        relevant,
    )


    if not relevant:

        return {
            "question": cleaned_question,
            "answer": FALLBACK_ANSWER,
            "sources": [],
        }


    # ========================================================
    # Grounded answer generation
    # ========================================================

    answer = generate_answer(
        question=cleaned_question,
        retrieved_documents=documents,
    )


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