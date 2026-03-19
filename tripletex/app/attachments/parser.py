"""Decode base64 attachments and prepare content for Claude."""

import base64
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

    for page in doc:
        text = page.get_text()
        if text.strip():
            text_parts.append(text)
        else:
            # Scanned page — render as image for Claude vision
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

    if text_parts:
        blocks.insert(0, {
            "type": "text",
            "text": f"[Content from {filename}]\n\n" + "\n---\n".join(text_parts),
        })

    doc.close()
    return blocks
