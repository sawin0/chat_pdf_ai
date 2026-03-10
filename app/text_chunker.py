def text_chunker(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Splits text into overlapping chunks.
    :param text: full text to split
    :param chunk_size: number of words per chunk
    :param overlap: number of words to overlap between chunks
    :return: list of text chunks
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap # move with overlap

    return chunks
