from app.qdrant_client import qdrant, COLLECTION_NAME
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from app.embeddings import model


def search_pdf(question: str, pdf_id: str, top_k: int = 5):
    # Encode as a list to always get a 2D result, then take first vector.
    query_vector = model.encode([question])[0]

    # MUST convert to Python list (not numpy)
    query_list = query_vector.tolist()

    # 2. Build filter if user wants to restrict to a specific PDF
    q_filter = None
    if pdf_id:
        q_filter = Filter(
            must=[FieldCondition(key="pdf_id", match=MatchValue(value=pdf_id))]
        )

    # 3) Perform nearest neighbor search
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_list,
        query_filter=q_filter,
        limit=top_k,
        with_payload=True,  # include stored payload in results
    )

    # 4) Combine retrieved text into a single context string
    hits = results.points if hasattr(results, "points") else results
    context = "\n".join([hit.payload.get("text", "") for hit in hits])
    return context
