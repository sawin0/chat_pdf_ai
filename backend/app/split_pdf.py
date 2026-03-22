from PyPDF2 import PdfReader, PdfWriter
import os

def split_pdf(input_path, output_dir, chunk_size=10):
    reader = PdfReader(input_path)
    total_pages = len(reader.pages)

    os.makedirs(output_dir, exist_ok=True)

    file_paths = []

    for start in range(0, total_pages, chunk_size):
        writter = PdfWriter()
        end = min(start + chunk_size, total_pages)

        for i in range(start, end):
            writter.add_page(reader.pages[i])

        output_path = os.path.join(output_dir, f"chunk_{start+1}_to_{end}.pdf")

        with open(output_path, "wb") as f:
            writter.write(f)

        file_paths.append(output_path)

    return file_paths
