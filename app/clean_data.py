import re


def clean_extracted_text(content: str) -> str:
    # Remove image links (including base64)
    content = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", content)

    # Remove horizontal rules (--- or *** or ___)
    content = re.sub(r"^\s*([-*_]){3,}\s*$", "", content, flags=re.MULTILINE)

    # Remove Markdown links but keep text: [text](url) -> text
    content = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", content)

    # Remove headings (#, ##, ###, etc.) but keep text
    content = re.sub(r"^\s*#{1,6}\s+", "", content, flags=re.MULTILINE)

    # Remove extra blank lines
    content = re.sub(r"\n\s*\n+", "\n\n", content)

    # Strip leading/trailing whitespace
    clean_text = content.strip()
    return clean_text
