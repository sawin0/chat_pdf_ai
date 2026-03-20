from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue
import uuid


# Connect to local Qdrant
qdrant = QdrantClient(url="http://qdrant:6333")

COLLECTION_NAME = "pdf_embeddings"


def ensure_collection(vector_size: int = 384):
    collections = qdrant.get_collections().collections
    if COLLECTION_NAME not in [c.name for c in collections]:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        )


def pdf_exists(pdf_id: str) -> bool:
    result = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="pdf_id", match=MatchValue(value=pdf_id))]
        ),
        limit=1,
    )

    points, _ = result
    return len(points) > 0


def store_embeddings(chunks: list, embeddings: list, pdf_id: str):
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{pdf_id}:{i}")),
            vector=embeddings[i].tolist() if hasattr(embeddings[i], "tolist") else list(embeddings[i]),
            payload={"text": chunks[i], "pdf_id": pdf_id, "chunk_index": i},
        )
        for i in range(len(chunks))
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
