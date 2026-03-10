from fastapi import APIRouter
from pydantic import BaseModel
from app.query_pdf import search_pdf

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    pdf_id: str
    top_k: int = 5


class QueryResponse(BaseModel):
    context: str


@router.post("/query-pdf", response_model=QueryResponse)
def query_pdf(data: QueryRequest):
    context = search_pdf(data.question, data.pdf_id, data.top_k)
    return {"context": context}
