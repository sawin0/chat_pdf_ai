# Chat PDF AI

A local-first RAG API that ingests PDF files, creates embeddings, stores them in Qdrant, and answers questions using a local Ollama model.

This project is designed to run fully self-hosted with open-source components.

## What We Have Built

- PDF ingestion endpoint (`/process-pdf`) that:
  - downloads a PDF from URL
  - computes a stable SHA-256 PDF id
  - prevents duplicate processing for the same PDF
  - extracts text using PDFMiner
  - falls back to OCR (Tesseract Nepali language pack) for scanned PDFs
  - splits text into overlapping chunks
  - generates embeddings using Sentence Transformers (`all-MiniLM-L12-v2`)
  - stores vectors and payload in Qdrant
- Query endpoint (`/query-pdf`) that:
  - embeds the user question
  - performs semantic vector search in Qdrant
  - optionally filters results by `pdf_id`
  - builds context from top-k retrieved chunks
  - asks a local Ollama model (`phi3:mini`) with guardrails to answer only from context
- Dockerized multi-service setup with:
  - FastAPI app
  - Qdrant vector database
  - Ollama model server

## Tech Stack

- Backend: FastAPI, Uvicorn
- Vector DB: Qdrant
- Embeddings: sentence-transformers (`all-MiniLM-L12-v2`, 384 dimensions)
- LLM Inference: Ollama (`phi3:mini` by default)
- PDF Text Extraction: pdfminer.six
- OCR Fallback: pytesseract + pdf2image + Poppler
- Containerization: Docker, Docker Compose

## Architecture

1. Client sends PDF URL to `/process-pdf`
2. API downloads PDF and calculates `pdf_id`
3. API extracts text (PDFMiner, then OCR fallback if needed)
4. API chunks text and creates embeddings
5. API upserts chunks + vectors into Qdrant
6. Client asks a question at `/query-pdf`
7. API retrieves top-k semantically similar chunks
8. API sends context + question to Ollama
9. API returns generated answer

## Project Structure

```
chat_pdf_ai/
  app/
    main.py                # FastAPI app + PDF processing endpoint
    routers/query_router.py# Query endpoint
    query_pdf.py           # Vector search and context building
    llm.py                 # Ollama chat integration
    embeddings.py          # SentenceTransformer model
    qdrant_client.py       # Qdrant collection + storage helpers
    pdf_downloader.py      # PDF download utility
    pdf_text_extractor.py  # PDFMiner + OCR fallback
    text_chunker.py        # Overlapping chunking logic
    pdf_utils.py           # PDF hash generation
  Dockerfile
  docker-compose.yml
  requirements.txt
```

## Prerequisites

- Docker and Docker Compose
- At least 8 GB RAM recommended for smooth local model and embedding workflow

## Run With Docker (Recommended)

1. Build and start services:

```bash
docker compose up --build -d
```

2. Pull the Ollama model inside the Ollama container:

```bash
docker compose exec ollama ollama pull phi3:mini
```

3. Verify API is up:

```bash
curl http://localhost:8000/docs
```

Swagger UI will be available at:

- http://localhost:8000/docs

## API Usage

### 1) Process a PDF

```bash
curl -X POST 'http://localhost:8000/process-pdf' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "url": "https://example.com/sample.pdf"
}'
```

Sample response:

```json
{
  "status": "success",
  "pdf_id": "<sha256>",
  "chunks_created": 42
}
```

If already processed:

```json
{
  "status": "already_processed",
  "pdf_id": "<sha256>"
}
```

### 2) Query a Processed PDF

```bash
curl -X POST 'http://localhost:8000/query-pdf' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "What is this document about?",
  "pdf_id": "<sha256>",
  "top_k": 5
}'
```

Sample response:

```json
{
  "answer": "..."
}
```

Notes:

- `pdf_id` can be omitted to query across all indexed PDFs.
- `top_k` controls the number of retrieved chunks used as context.

## Environment and Defaults

Current key defaults in code:

- Ollama host: `http://ollama:11434`
- Ollama model: `phi3:mini`
- Max generation tokens: `256`
- Prompt length cap: `3000` characters
- Chunk size: `500` words
- Chunk overlap: `50` words
- Qdrant collection: `pdf_embeddings`
- Vector size: `384`

## Known Limitations

- PDF downloader currently does not validate URL, timeout, or HTTP status robustly.
- No authentication/authorization on API endpoints.
- OCR language is currently set to Nepali (`nep`) only.
- No automated test suite yet.

## Suggested Next Improvements

- Add request validation and proper error handling for download/extraction failures.
- Add health/readiness endpoints for API, Qdrant, and Ollama.
- Add unit/integration tests for ingestion and query pipelines.
- Add optional multilingual OCR and language auto-detection.
- Add metadata filters (page number, source URL, tags) in retrieval.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
