from openai import OpenAI

from app.config import settings
from app.generation.prompts import RAG_SYSTEM_PROMPT


client = OpenAI(
    api_key=settings.openai_api_key
)


def create_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=text,
    )

    return response.data[0].embedding


def generate_answer(
    question: str,
    retrieved_documents: list[dict],
) -> str:

    context_parts = []

    for document in retrieved_documents:
        context_parts.append(
            f"""
Document: {document["document_name"]}
Page: {document["page_number"]}

{document["text"]}
"""
        )

    context = "\n\n---\n\n".join(context_parts)

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}

QUESTION:

{question}
"""

    response = client.responses.create(
        model=settings.llm_model,
        instructions=RAG_SYSTEM_PROMPT,
        input=user_prompt,
    )

    return response.output_text