import os
import time
import uuid

from pinecone import Pinecone, ServerlessSpec

_INDEX_NAME = "pdf-embeddings"
_SCORE_THRESHOLD = 0.3


def _client() -> Pinecone:
    return Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def _get_index():
    return _client().Index(_INDEX_NAME)


def ensure_collection(vector_size: int = 1024) -> None:
    pc = _client()
    existing = [idx.name for idx in pc.list_indexes()]
    if _INDEX_NAME not in existing:
        pc.create_index(
            name=_INDEX_NAME,
            dimension=vector_size,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(_INDEX_NAME).status["ready"]:
            time.sleep(1)


def pdf_exists(pdf_id: str) -> bool:
    first_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{pdf_id}:0"))
    result = _get_index().fetch(ids=[first_id])
    return first_id in result.vectors


def store_embeddings(chunks: list, embeddings: list, pdf_id: str) -> None:
    index = _get_index()
    vectors = [
        {
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{pdf_id}:{i}")),
            "values": emb.tolist() if hasattr(emb, "tolist") else list(emb),
            "metadata": {"pdf_id": pdf_id, "chunk_index": i, "text": chunks[i]},
        }
        for i, emb in enumerate(embeddings)
    ]
    batch_size = 100
    for start in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[start : start + batch_size])


def search(query_vector: list, pdf_id: str | None = None, top_k: int = 3) -> list[str]:
    filter_dict = {"pdf_id": {"$eq": pdf_id}} if pdf_id else None
    results = _get_index().query(
        vector=query_vector,
        top_k=top_k,
        filter=filter_dict,
        include_metadata=True,
    )

    total_matches = len(results.matches)
    top_scores = [
        round(match.score, 4) for match in results.matches[: min(5, total_matches)]
    ]
    print(
        "\n--- RETRIEVAL ---\n"
        f"pdf_id filter: {pdf_id}\n"
        f"top_k: {top_k}\n"
        f"raw matches: {total_matches}\n"
        f"top scores: {top_scores}\n"
    )

    filtered = [
        match.metadata["text"]
        for match in results.matches
        if match.score >= _SCORE_THRESHOLD and match.metadata.get("text", "").strip()
    ]

    if not filtered and total_matches > 0:
        # Fallback so context is not empty when scores are just below threshold.
        filtered = [
            match.metadata.get("text", "")
            for match in results.matches
            if match.metadata.get("text", "").strip()
        ]
        print(
            "\n⚠️ No matches passed score threshold; using raw top matches as fallback.\n"
        )

    print(f"Returned chunks: {len(filtered)}\n")
    return filtered
