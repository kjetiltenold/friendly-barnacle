"""Decode base64 attachments and prepare content for Claude."""

import base64
import csv
import io
import logging

import fitz  # pymupdf

from app.models import FileAttachment

logger = logging.getLogger(__name__)


def process_attachments(files: list[FileAttachment]) -> list[dict]:
    """Convert file attachments into Claude content blocks.

    Returns a list of content blocks:
    - {"type": "text", "text": "..."} for extracted PDF text
    - {"type": "image", "source": {...}} for images
    """
    blocks = []
    for f in files:
        raw = base64.b64decode(f.content_base64)

        if f.mime_type == "application/pdf":
            blocks.extend(_process_pdf(raw, f.filename))
        elif f.mime_type == "text/csv":
            blocks.append({
                "type": "text",
                "text": _process_csv(raw, f.filename),
            })
        elif f.mime_type.startswith("text/"):
            blocks.append({
                "type": "text",
                "text": _process_text(raw, f.filename),
            })
        elif f.mime_type.startswith("image/"):
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": f.mime_type,
                    "data": f.content_base64,
                },
            })
        else:
            logger.warning(f"Unsupported attachment type: {f.mime_type}")

    return blocks


def _process_pdf(raw: bytes, filename: str) -> list[dict]:
    blocks = []
    doc = fitz.open(stream=raw, filetype="pdf")
    text_parts = []
    page_count = len(doc)
    prefer_image_first = False

    for page in doc:
        text = page.get_text()
        weak_text = _should_render_pdf_page_as_image(text)
        if text.strip():
            text_parts.append(text)
        if page_count == 1 or weak_text:
            # Single-page contest documents are often visually structured even
            # when text extraction succeeds, so include the image too. For
            # multi-page PDFs, keep the existing weak-text heuristic.
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(img_bytes).decode(),
                },
            })
            if page_count == 1 and weak_text:
                prefer_image_first = True

    if text_parts:
        header = f"[Content from {filename}]"
        if prefer_image_first:
            header += (
                "\n\n[Extracted text note]\n"
                "This OCR text may miss separators or layout details on short receipts. "
                "Verify exact amounts, dates, and merchant details against the attached page image."
            )
        text_block = {
            "type": "text",
            "text": header + "\n\n" + "\n---\n".join(text_parts),
        }
        if prefer_image_first and blocks:
            blocks.append(text_block)
        else:
            blocks.insert(0, text_block)

    doc.close()
    return blocks


def _process_text(raw: bytes, filename: str) -> str:
    text = raw.decode("utf-8-sig", errors="replace")
    return f"[Content from {filename}]\n\n{text}"


def _process_csv(raw: bytes, filename: str) -> str:
    text = raw.decode("utf-8-sig", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return f"[Content from {filename}]\n\n<empty csv>"

    sample = "\n".join(lines[:5])
    delimiter = ";"
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        pass

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = [[cell.strip() for cell in row] for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return f"[Content from {filename}]\n\n{text}"

    max_rows = 80
    max_cols = 12
    rendered_rows = rows[:max_rows]
    normalized_rows = [row[:max_cols] for row in rendered_rows]
    col_count = max(len(row) for row in normalized_rows)
    padded_rows = [row + [""] * (col_count - len(row)) for row in normalized_rows]

    header = padded_rows[0]
    body = padded_rows[1:]
    table_lines = ["\t".join(header)]
    for row in body:
        table_lines.append("\t".join(row))

    summary = f"[Content from {filename}]\n\nParsed CSV with {len(rows)} row(s). Columns are tab-separated below.\n\n"
    return summary + "\n".join(table_lines)


def _should_render_pdf_page_as_image(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if "\ufffd" in stripped:
        return True
    if len(stripped) < 500:
        return True
    non_empty_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    return len(non_empty_lines) < 4
