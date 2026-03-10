from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct

# Connect to local Qdrant
qdrant = QdrantClient(url="https://qdrant:6333")

COLLECTION_NAME = "pdf_embeddings"


def create_collection(vector_size: int = 384):
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size, distance=models.Distance.COSINE
        ),
    )


def store_embeddings(chunks: list, embeddings: list):
    points = [
        PointStruct(id=i, vector=embeddings[i], payload={"text": chunks[i]})
        for i in range(len(chunks))
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
