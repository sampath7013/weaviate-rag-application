from openai import OpenAI

from app.config import settings
from app.generation.prompts import RAG_SYSTEM_PROMPT


client = OpenAI(
    api_key=settings.openai_api_key
)


def create_embedding(
    text: str,
) -> list[float]:
    """
    Create an embedding for a single text string.

    Used for query embeddings during retrieval.
    """

    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    response = client.embeddings.create(
        model=settings.embedding_model,
        input=text,
    )

    return response.data[0].embedding


def create_embeddings(
    texts: list[str],
) -> list[list[float]]:
    """
    Create embeddings for multiple text chunks
    in one OpenAI embeddings request.
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
            "Embedding count does not match input text count."
        )

    return embeddings


def generate_answer(
    question: str,
    retrieved_documents: list[dict],
) -> str:
    """
    Generate a grounded answer using
    retrieved document chunks.
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

    context = "\n\n---\n\n".join(
        context_parts
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

    return response.output_text