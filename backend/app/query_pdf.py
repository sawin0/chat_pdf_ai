from app.vector_store import ensure_collection, search
from app.embeddings import model


def search_pdf(question: str, pdf_id: str, top_k: int = 3):
    ensure_collection()
    query_vector = model.encode([question])[0]
    query_list = (
        query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
    )
    texts = search(query_list, pdf_id=pdf_id, top_k=top_k)
    return "\n\n".join(texts)
