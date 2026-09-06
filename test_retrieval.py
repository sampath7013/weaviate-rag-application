from app.retrieval.retriever import retrieve_documents


query = "What experience does Sampath have with RAG?"

results = retrieve_documents(
    query=query,
    limit=5,
)

print("\nQUESTION:")
print(query)

print("\nRETRIEVED CHUNKS:")

for index, result in enumerate(results, start=1):
    print("\n" + "=" * 80)

    print(f"Result {index}")
    print(f"Document: {result['document_name']}")
    print(f"Page: {result['page_number']}")
    print(f"Chunk: {result['chunk_index']}")

    print("\nText:")
    print(result["text"])