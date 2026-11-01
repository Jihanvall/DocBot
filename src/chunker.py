def split_text(text, size=500, overlap=100):
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be between 0 and size - 1")

    words = text.split()
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        if piece:
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def chunk_document(document, size=500, overlap=100):
    pieces = split_text(document["text"], size, overlap)
    return [{"source": document["source"], "position": i, "text": piece} for i, piece in enumerate(pieces)]
