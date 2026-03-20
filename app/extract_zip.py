import os
import zipfile


def extract_document_from_zip(path: str) -> str:
    files_to_extract = ["document.md"]
    output_dir = "./tmp/outputs"

    with zipfile.ZipFile(path, "r") as zip_ref:
        for file in files_to_extract:
            if file in zip_ref.namelist():
                extracted_path = zip_ref.extract(file, output_dir)
                return extracted_path
            else:
                print(f"{file} not found in the ZIP archive.")

    raise FileNotFoundError("document.md not found in the ZIP archive")
