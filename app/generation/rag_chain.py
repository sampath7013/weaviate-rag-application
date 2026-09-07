import logging

from app.retrieval.retriever import retrieve_documents
from app.generation.llm import (
    generate_answer,
    is_context_relevant,
)


logger = logging.getLogger(__name__)


FALLBACK_ANSWER = (
    "The available document does not contain enough "
    "relevant information to answer this question."
)


def ask_rag(
    question: str,
    top_k: int = 5,
    document_id: str | None = None,
    alpha: float = 0.50,
    min_score: float | None = 0.20,
):
    """
    Complete RAG pipeline:

    1. Retrieve candidate chunks using hybrid search.
    2. Apply hybrid-score filtering.
    3. Apply LLM relevance gate.
    4. Generate a grounded answer only when the
       retrieved context is relevant.
    """

    cleaned_question = question.strip()

    if not cleaned_question:
        return {
            "question": question,
            "answer": FALLBACK_ANSWER,
            "sources": [],
        }

    # --------------------------------------------------
    # Step 1: Hybrid retrieval
    # --------------------------------------------------

    documents = retrieve_documents(
        query=cleaned_question,
        limit=top_k,
        document_id=document_id,
        alpha=alpha,
        min_score=min_score,
    )

    logger.info(
        "retrieval_completed | document_id=%s | candidates=%s",
        document_id,
        len(documents),
    )

    # --------------------------------------------------
    # Step 2: Nothing survived retrieval filtering
    # --------------------------------------------------

    if not documents:
        logger.info(
            "rag_rejected_no_documents | document_id=%s",
            document_id,
        )

        return {
            "question": cleaned_question,
            "answer": FALLBACK_ANSWER,
            "sources": [],
        }

    # --------------------------------------------------
    # Step 3: LLM relevance gate
    # --------------------------------------------------

    context_is_relevant = is_context_relevant(
        question=cleaned_question,
        retrieved_documents=documents,
    )

    logger.info(
        "relevance_gate | document_id=%s | relevant=%s",
        document_id,
        context_is_relevant,
    )

    # --------------------------------------------------
    # Step 4: Reject unsupported question
    # --------------------------------------------------

    if not context_is_relevant:
        logger.info(
            "rag_rejected_by_relevance_gate | document_id=%s",
            document_id,
        )

        return {
            "question": cleaned_question,
            "answer": FALLBACK_ANSWER,
            "sources": [],
        }

    # --------------------------------------------------
    # Step 5: Grounded generation
    # --------------------------------------------------

    answer = generate_answer(
        question=cleaned_question,
        retrieved_documents=documents,
    )

    logger.info(
        "rag_answer_generated | document_id=%s | sources=%s",
        document_id,
        len(documents),
    )

    return {
        "question": cleaned_question,
        "answer": answer,
        "sources": documents,
    }