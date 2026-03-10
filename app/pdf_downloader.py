import requests
import os
import uuid

TEMP_DIR = "temp"

def download_pdf(url: str) -> str:
    os.makedirs(TEMP_DIR, exist_ok=True)

    file_name = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(TEMP_DIR, file_name)

    response = requests.get(url)

    with open(file_path, "wb") as f:
        f.write(response.content)

    return file_name
