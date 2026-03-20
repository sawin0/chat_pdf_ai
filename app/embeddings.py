from sentence_transformers import SentenceTransformer

# Load once
# | Model            | Dim | Quality | Speed | Best For           |
# | ---------------- | --- | ------- | ----- | ------------------ |
# | `MiniLM‑L6`      | 384 | ⭐⭐⭐     | ⭐⭐⭐⭐  | fastest small      |
# | **`MiniLM‑L12`** | 384 | ⭐⭐⭐⭐    | ⭐⭐⭐⭐  | best free balance  |
# | `mpnet‑base`     | 768 | ⭐⭐⭐⭐⭐   | ⭐⭐⭐   | most accurate free |
# | intfloat/multilingual-e5-large | 1024 | ⭐⭐⭐⭐⭐   | ⭐⭐    | best multilingual  |

# all-MiniLM-L12-v2 and all-MiniLM-16-v2
model = SentenceTransformer("all-MiniLM-L12-v2") # 384-d embeddings

def get_embeddings(text_chunks: list):
    """
    Returns a list of embeddings for each chunk
    """
    embeddings = model.encode(text_chunks, show_progress_bar=True)
    return embeddings
