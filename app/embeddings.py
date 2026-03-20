from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-large")  # 1024-d embeddings


def get_embeddings(text_chunks: list):
    """
    Returns a list of embeddings for each chunk
    """
    embeddings = model.encode(text_chunks, show_progress_bar=True)
    return embeddings
