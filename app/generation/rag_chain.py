from app.retrieval.retriever import retrieve_documents
from app.generation.llm import generate_answer


def ask_rag(
    question: str,
    top_k: int = 5,
    document_id: str | None = None,
):
    documents = retrieve_documents(
        query=question,
        limit=top_k,
        document_id=document_id,
    )

    if not documents:
        return {
            "question": question,
            "answer": (
                "No relevant document chunks were "
                "found for the selected document."
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