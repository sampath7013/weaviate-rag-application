from app.retrieval.retriever import retrieve_documents
from app.generation.llm import generate_answer


def ask_rag(
    question: str,
    top_k: int = 5,
):
    documents = retrieve_documents(
        query=question,
        limit=top_k,
    )

    answer = generate_answer(
        question=question,
        retrieved_documents=documents,
    )

    return {
        "question": question,
        "answer": answer,
        "sources": documents,
    }