from fastapi import APIRouter
from pydantic import BaseModel
from app.query_pdf import search_pdf
from app.llm import ask_rag_llm

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    pdf_id: str | None = None
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str


@router.post("/query-pdf", response_model=QueryResponse)
def query_pdf_endpoint(data: QueryRequest):
    context = search_pdf(data.question, data.pdf_id, data.top_k)
    answer = ask_rag_llm(data.question, context)
    return {"answer": answer}
