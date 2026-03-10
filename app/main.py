from fastapi import FastAPI
from pydantic import BaseModel

from app.embeddings import get_embeddings
from app.pdf_downloader import download_pdf
from app.pdf_text_extractor import extract_text_from_pdf
from app.pdf_utils import generate_pdf_hash
from app.qdrant_client import (
    ensure_collection,
    pdf_exists,
    store_embeddings,
)
from app.text_chunker import text_chunker
from app.routers import query_router

app = FastAPI()

# Include the query PDF router
app.include_router(query_router.router)


class PDFReuqest(BaseModel):
    url: str


@app.post("/process-pdf")
def process_pdf(data: PDFReuqest):

    pdf_path = download_pdf(data.url)

    pdf_id = generate_pdf_hash(pdf_path)

    ensure_collection()

    print("pdf_id " + pdf_id)

    # Prevent duplicate embeddings
    if pdf_exists(pdf_id):
        return {"status": "already_processed", "pdf_id": pdf_id}

    text = extract_text_from_pdf(pdf_path)

    # Chunk text
    chunks = text_chunker(text, chunk_size=500, overlap=50)

    # Create embeddings
    embeddings = get_embeddings(chunks)

    # Store in Qdrant
    store_embeddings(chunks, embeddings, pdf_id)

    return {
        "status": "success",
        "pdf_id": pdf_id,
        "chunks_created": len(chunks),
    }
