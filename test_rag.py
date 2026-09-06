from app.generation.rag_chain import ask_rag


question = input("Ask a question: ")

result = ask_rag(question)

print("\n" + "=" * 80)
print("ANSWER")
print("=" * 80)

print(result["answer"])

print("\n" + "=" * 80)
print("SOURCES")
print("=" * 80)

for source in result["sources"]:
    print(
        f"{source['document_name']} "
        f"- Page {source['page_number']} "
        f"- Chunk {source['chunk_index']}"
    )