# Chat PDF AI

A FastAPI-based RAG API that ingests PDF files, creates multilingual embeddings, stores them in Pinecone, and answers questions with a grounded Groq-backed response pipeline.

## What We Have Built

- PDF ingestion endpoint (`/process-pdf`) that:
  - downloads a PDF from URL
  - computes a stable SHA-256 PDF id
  - prevents duplicate processing for the same PDF
  - extracts text using SarvamAI Document Intelligence (PDF split into chunks)
  - includes utility paths for PDFMiner and OCR (Nepali) extraction
  - splits text into overlapping chunks
  - generates embeddings using Sentence Transformers (`intfloat/multilingual-e5-large`)
  - stores vectors and metadata in Pinecone
- Query endpoint (`/query-pdf`) that:
  - embeds the user question
  - performs semantic vector search in Pinecone
  - optionally filters results by `pdf_id`
  - builds context from top-k retrieved chunks
  - sends grounded prompt to Groq via LangChain
  - returns Nepali response with a strict fallback message when context is insufficient
- Dockerized setup with:
  - FastAPI app service
  - external managed services via API keys (Pinecone, Groq, SarvamAI)

## Tech Stack

- Backend: FastAPI, Uvicorn
- Vector DB: Pinecone (serverless index)
- Embeddings: sentence-transformers (`intfloat/multilingual-e5-large`, 1024 dimensions)
- LLM Orchestration: LangChain (`langchain-core` + `langchain-groq`)
- LLM Provider: Groq
- PDF Text Extraction (primary): SarvamAI Document Intelligence (markdown output)
- PDF Text Extraction (available utilities): pdfminer.six
- OCR Utility (available path): pytesseract + pdf2image + Poppler
- Containerization: Docker, Docker Compose

## Architecture

1. Client sends PDF URL to `/process-pdf`
2. API downloads PDF and calculates `pdf_id`
3. API ensures Pinecone index exists (`pdf-embeddings`, cosine, 1024 dims)
4. API checks duplicate by deterministic first chunk id (`uuid5(pdf_id:0)`)
5. API extracts text from PDF using SarvamAI Document Intelligence
6. API cleans extracted markdown text
7. API chunks text (word-based overlap)
8. API creates embeddings
9. API upserts vectors + metadata into Pinecone
10. Client asks a question at `/query-pdf`
11. API embeds question and retrieves semantic matches from Pinecone
12. API applies score threshold fallback logic and builds context
13. API prompts Groq with strict grounded rules
14. API returns Nepali answer or fallback message

## Project Structure

```
chat_pdf_ai/
  app/
    main.py                 # FastAPI app + /process-pdf endpoint
    routers/query_router.py # /query-pdf endpoint
    query_pdf.py            # Query embedding + retrieval context build
    llm.py                  # Prompting, intent checks, Groq call, fallback
    embeddings.py           # SentenceTransformer model loader
    vector_store.py         # Pinecone index management + upsert/search
    pdf_downloader.py       # URL download to /tmp
    pdf_text_extractor.py   # Sarvam extraction + helper PDF/OCR paths
    split_pdf.py            # Splits large PDF into smaller chunks
    extract_zip.py          # Extracts Sarvam output markdown from ZIP
    clean_data.py           # Markdown cleanup utility
    text_chunker.py         # Overlapping word chunk logic
    pdf_utils.py            # SHA-256 hash generation for PDF id
    remove_tmp.py           # Cleans local ./tmp directory
  Dockerfile
  docker-compose.yml
  requirements.txt
```

## Prerequisites

- Docker and Docker Compose
- At least 4 GB RAM recommended for embedding and OCR/image conversion workloads
- External service credentials:
  - Pinecone API key
  - Groq API key
  - Sarvam API key

## Environment Variables

Create a `.env` file at project root.

Required for current flow:

- `PINECONE_API_KEY`
- `GROQ_API_KEY`
- `SARVAM_API_KEY`

Optional:

- `GROQ_MODEL` (default: `llama-3.3-70b-versatile`)

## Run With Docker (Recommended)

1. Build and start service:

```bash
docker compose up --build -d
```

2. Verify API is up:

```bash
curl http://localhost:8000/docs
```

Swagger UI:

- http://localhost:8000/docs

## Run Locally (Without Docker)

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

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

Sample success response:

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
- `top_k` default is `3`.



## License

This project is licensed under the MIT License. See the LICENSE file for details.
