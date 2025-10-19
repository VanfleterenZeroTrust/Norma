from pypdf import PdfReader

def pdf_to_chunks(path):
    # Simple page-based chunking; improve later if needed
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        yield {"id": f"{path}-p{i}", "text": text.strip()}
