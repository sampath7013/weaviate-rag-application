def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
):
    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - chunk_overlap

    return chunks


def chunk_pages(
    pages,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
):
    all_chunks = []

    for page in pages:
        chunks = split_text(
            page["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for index, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "text": chunk,
                    "document_name": page["document_name"],
                    "page_number": page["page_number"],
                    "chunk_index": index,
                }
            )

    return all_chunks