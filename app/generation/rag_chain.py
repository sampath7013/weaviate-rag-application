from app.retrieval.retriever import retrieve_documents
from app.generation.llm import generate_answer


def ask_rag(
    question: str,
    top_k: int = 5,
    document_id: str | None = None,
    max_distance: float | None = 0.55,
):
    documents = retrieve_documents(
        query=question,
        limit=top_k,
        document_id=document_id,
        max_distance=max_distance,
    )

    if not documents:
        return {
            "question": question,
            "answer": (
                "The available document does not contain "
                "enough relevant information to answer "
                "this question."
            ),
            "sources": [],
        }

    answer = generate_answer(
        question=question,
        retrieved_documents=documents,
    )

    return {
        "question": question,
        "answer": answer,
        "sources": documents,
    }