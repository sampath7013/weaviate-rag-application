from app.generation.llm import create_embedding


text = "Retrieval-Augmented Generation combines retrieval with an LLM."

embedding = create_embedding(text)

print("Embedding type:", type(embedding))
print("Embedding length:", len(embedding))
print("First 5 values:", embedding[:5])