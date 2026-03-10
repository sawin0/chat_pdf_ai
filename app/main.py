from fastapi import FastAPI
from pydantic import BaseModel

from app.pdf_downloader import download_pdf
from app.pdf_text_extractor import extract_text_from_pdf

app = FastAPI()


class PDFReuqest(BaseModel):
    url: str


@app.post("/process-pdf")
def process_pdf(data: PDFReuqest):

    pdf_path = download_pdf(data.url)

    text = extract_text_from_pdf(pdf_path)

    return {"status": "Success", "characters_extracterd(length)": len(text)}
