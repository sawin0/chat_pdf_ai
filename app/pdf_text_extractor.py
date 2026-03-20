from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
import pytesseract

from sarvamai import SarvamAI

from PyPDF2 import PdfReader, PdfWriter

import os

from app.split_pdf import split_pdf
import time


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

    # Upload document
    for i, chunk in enumerate(chunks):
        # Create a document intelligence job
        job = client.document_intelligence.create_job(
            language="ne-IN", output_format="md"
        )
        print(f"Job created: {job.job_id}")

        job.upload_file(chunk)
        print(f"File uploaded: {chunk}")

        # Start processing
        job.start()
        print("Job started")

        # Wait for completion
        status = job.wait_until_complete()
        print(f"Job completed with state: {status.job_state}")

        # Get processing metrics
        metrics = job.get_page_metrics()
        print(f"Page metrics: {metrics}")

        # Download output (ZIP file containing the processed document)
        job.download_output("./output.zip")
        os.rename("./output.zip", f"./output_{i}.zip")
        print(f"Output saved to ./output_{i}.zip")

        time.sleep(5)  # Sleep to avoid hitting rate limits
