from openai import OpenAI

from app.config import settings
from app.generation.prompts import RAG_SYSTEM_PROMPT


# ============================================================
# OpenAI client
# ============================================================

client = OpenAI(
    api_key=settings.openai_api_key
)


# ============================================================
# Single embedding
# Used for user query embeddings during retrieval
# ============================================================

def create_embedding(
    text: str,
) -> list[float]:
    """
    Create an embedding for a single text string.

    Used primarily for user query embeddings.
    """

    if not text or not text.strip():
        raise ValueError(
            "Text cannot be empty."
        )

    response = client.embeddings.create(
        model=settings.embedding_model,
        input=text.strip(),
    )

    return response.data[0].embedding


# ============================================================
# Batch embeddings
# Used during PDF ingestion
# ============================================================

def create_embeddings(
    texts: list[str],
) -> list[list[float]]:
    """
    Create embeddings for multiple text chunks
    in a single OpenAI embeddings request.
    """

    if not texts:
        return []

    cleaned_texts = []

    for text in texts:

        if not text or not text.strip():
            raise ValueError(
                "Embedding input contains empty text."
            )

        cleaned_texts.append(
            text.strip()
        )

    response = client.embeddings.create(
        model=settings.embedding_model,
        input=cleaned_texts,
    )

    embeddings = [
        item.embedding
        for item in response.data
    ]

    if len(embeddings) != len(cleaned_texts):

        raise RuntimeError(
            "Embedding count does not match "
            "the number of input texts."
        )

    return embeddings


# ============================================================
# Build retrieved context
# ============================================================

def build_context(
    retrieved_documents: list[dict],
) -> str:
    """
    Convert retrieved document chunks into
    a formatted context block.
    """

    context_parts = []

    for document in retrieved_documents:

        document_name = document.get(
            "document_name",
            "Unknown document",
        )

        page_number = document.get(
            "page_number",
            "Unknown",
        )

        text = document.get(
            "text",
            "",
        )

        context_parts.append(
            f"""
Document: {document_name}
Page: {page_number}

{text}
""".strip()
        )

    return "\n\n---\n\n".join(
        context_parts
    )


# ============================================================
# Relevance gate
# ============================================================

def is_context_relevant(
    question: str,
    retrieved_documents: list[dict],
) -> bool:
    """
    Determine whether the retrieved document context
    contains enough information to answer the question.

    Returns:
        True  -> context is relevant
        False -> context is insufficient or unrelated
    """

    if not question or not question.strip():
        return False

    if not retrieved_documents:
        return False

    context = build_context(
        retrieved_documents
    )

    relevance_prompt = f"""
You are a relevance classifier for a
Retrieval-Augmented Generation system.

Your job is NOT to answer the user's question.

Determine whether the provided DOCUMENT CONTEXT
contains enough information to answer the QUESTION.

Rules:

1. Return only YES or NO.
2. Return YES only when the answer is directly supported
   by the supplied document context.
3. Return NO if the context is unrelated.
4. Return NO if the context only partially relates to the
   question but does not contain enough information to
   answer it.
5. Do not use outside knowledge.
6. Do not guess.
7. Do not explain your decision.

DOCUMENT CONTEXT:

{context}

QUESTION:

{question}
""".strip()

    response = client.responses.create(
        model=settings.llm_model,
        input=relevance_prompt,
    )

    decision = (
        response.output_text
        .strip()
        .upper()
    )

    return decision.startswith(
        "YES"
    )


# ============================================================
# Grounded answer generation
# ============================================================

def generate_answer(
    question: str,
    retrieved_documents: list[dict],
) -> str:
    """
    Generate an answer using only retrieved
    document context.
    """

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    if not retrieved_documents:

        return (
            "The available document does not contain "
            "enough relevant information to answer "
            "this question."
        )

    context = build_context(
        retrieved_documents
    )

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}

QUESTION:

{question}
""".strip()

    response = client.responses.create(
        model=settings.llm_model,
        instructions=RAG_SYSTEM_PROMPT,
        input=user_prompt,
    )

    answer = (
        response.output_text
        .strip()
    )

    if not answer:

        return (
            "The available document does not contain "
            "enough relevant information to answer "
            "this question."
        )

    return answer