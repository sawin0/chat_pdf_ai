from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
from sarvamai import SarvamAI
from PyPDF2 import PdfReader, PdfWriter
from app.clean_data import clean_extracted_text
from app.extract_zip import extract_document_from_zip
from app.split_pdf import split_pdf

import os
import pytesseract
import time
import zipfile
import re


def extract_text_pdfmine(pdf_path: str) -> str:
    return extract_text(pdf_path)


def extract_text_ocr(pdf_path: str) -> str:
    pages = convert_from_path(pdf_path)

    full_text = ""

    for page in pages:
        text = pytesseract.image_to_string(page, lang="nep")
        full_text += text + "\n"

    return full_text


def extract_text_from_pdf(pdf_path: str) -> str:

    text = extract_text_pdfmine(pdf_path)

    if len(text.strip()) < 100:
        text = extract_text_ocr(pdf_path)

    return text


def extract_text_from_pdf_sarvamai(pdf_path: str) -> str:
    client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])

    # Split PDF into chunks
    chunks = split_pdf(pdf_path, "./tmp")

    result = ""

    # Upload document
    for i, chunk in enumerate(chunks):
        # Create a document intelligence job
        job = client.document_intelligence.create_job(
            language="ne-IN", output_format="md"
        )

        job.upload_file(chunk)

        # Start processing
        job.start()

        # Wait for completion
        status = job.wait_until_complete()

        # Get processing metrics
        metrics = job.get_page_metrics()

        # Download output (ZIP file containing the processed document)
        job.download_output("./tmp/outputs/output.zip")
        os.rename("./tmp/outputs/output.zip", f"./tmp/outputs/output_{i}.zip")

        extracted_markdown_path = extract_document_from_zip(
            f"./tmp/outputs/output_{i}.zip"
        )

        content = ""

        # Read the Markdown file
        with open(extracted_markdown_path, "r", encoding="utf-8") as f:
            content = f.read()

        clean_text = clean_extracted_text(content)
        result += clean_text + "\n"
        os.remove(extracted_markdown_path)

    return result.strip()
