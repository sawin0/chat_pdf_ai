from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
import pytesseract


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
