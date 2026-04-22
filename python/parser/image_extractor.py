"""PDF image extraction and enrichment utilities."""

from __future__ import annotations

import logging
import re
from typing import Optional

import fitz

from .cloudinary_uploader import upload_image
from .pix2tex_runner import run_pix2tex

logger = logging.getLogger(__name__)

_VISION_CLIENT = None
_VISION_INIT_ATTEMPTED = False


def _get_vision_client():
    """
    Lazily initialize and return Google Vision client.

    Returns:
        ImageAnnotatorClient instance or None when unavailable.
    """
    global _VISION_CLIENT, _VISION_INIT_ATTEMPTED
    if _VISION_INIT_ATTEMPTED:
        return _VISION_CLIENT

    _VISION_INIT_ATTEMPTED = True
    try:
        from google.cloud import vision

        _VISION_CLIENT = vision.ImageAnnotatorClient()
    except Exception:
        logger.exception('Google Vision client initialization failed')
        _VISION_CLIENT = None
    return _VISION_CLIENT


def _clean_ocr_text(text: Optional[str]) -> Optional[str]:
    """
    Clean OCR output by stripping whitespace and removing single-character lines.
    """
    if not text:
        return None

    cleaned_lines = []
    for line in text.splitlines():
        normalized = re.sub(r'\s+', ' ', line).strip()
        if not normalized:
            continue
        if len(normalized) <= 1:
            continue
        cleaned_lines.append(normalized)

    cleaned = '\n'.join(cleaned_lines).strip()
    return cleaned or None


def _run_google_ocr(image_bytes: bytes) -> Optional[str]:
    """
    Run Google Vision OCR and return cleaned text.
    """
    client = _get_vision_client()
    if client is None:
        return None

    try:
        from google.cloud import vision

        image = vision.Image(content=image_bytes)
        response = client.text_detection(image=image)
    except Exception:
        logger.exception('Google Vision OCR call failed')
        return None

    if getattr(response, 'error', None) and response.error.message:
        logger.error('Google Vision OCR error: %s', response.error.message)
        return None

    text = None
    if getattr(response, 'full_text_annotation', None):
        text = response.full_text_annotation.text
    elif getattr(response, 'text_annotations', None):
        if response.text_annotations:
            text = response.text_annotations[0].description

    return _clean_ocr_text(text)


def _sanitize_for_public_id(value: str) -> str:
    """
    Convert subject/chapter fragments to Cloudinary-safe path components.
    """
    lowered = value.strip().lower().replace(' ', '_')
    cleaned = re.sub(r'[^a-z0-9_\-]', '', lowered)
    return cleaned or 'unknown'


def _determine_position(page_rect: fitz.Rect, bbox: tuple[float, float, float, float]) -> str:
    """
    Determine image position relative to page center.
    """
    page_center_x = (page_rect.x0 + page_rect.x1) / 2.0
    page_center_y = (page_rect.y0 + page_rect.y1) / 2.0
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0

    x_threshold = (page_rect.width * 0.15)
    y_threshold = (page_rect.height * 0.15)

    if abs(center_x - page_center_x) <= x_threshold and abs(center_y - page_center_y) <= y_threshold:
        return 'center'
    if center_x < page_center_x and center_y < page_center_y:
        return 'top-left'
    if center_x >= page_center_x and center_y < page_center_y:
        return 'top-right'
    if center_x < page_center_x and center_y >= page_center_y:
        return 'bottom-left'
    return 'bottom-right'


def _extract_image_blob(doc: fitz.Document, block: dict) -> tuple[Optional[bytes], int, int]:
    """
    Extract raw image bytes and dimensions from an image block.
    """
    xref = int(block.get('xref', 0) or 0)
    if xref > 0:
        image_data = doc.extract_image(xref)
        image_bytes = image_data.get('image')
        width = int(image_data.get('width') or 0)
        height = int(image_data.get('height') or 0)
        return image_bytes, width, height

    image_bytes = block.get('image')
    width = int(block.get('width') or 0)
    height = int(block.get('height') or 0)
    return image_bytes, width, height


def extract_images_per_page(
    pdf_path: str,
    subject: str = 'unknown',
    chapter_num: int | str = 'unknown',
) -> list[dict]:
    """
    Extract and process images from each PDF page.

    Returns:
        [
          {
            "page_number": int,
            "is_image_only_page": bool,
            "images": [...]
          }
        ]
    """
    pages_output: list[dict] = []
    safe_subject = _sanitize_for_public_id(str(subject))
    safe_chapter = _sanitize_for_public_id(str(chapter_num))

    doc = fitz.open(pdf_path)
    try:
        for page_number, page in enumerate(doc, start=1):
            try:
                page_dict = page.get_text('dict') or {}
                blocks = page_dict.get('blocks', [])
                image_blocks = [block for block in blocks if block.get('type') == 1]
                images: list[dict] = []

                for block in image_blocks:
                    try:
                        image_bytes, width, height = _extract_image_blob(doc, block)
                        if not image_bytes:
                            continue
                        if width < 50 or height < 50:
                            continue

                        bbox = block.get('bbox') or (0.0, 0.0, 0.0, 0.0)
                        if len(bbox) != 4:
                            bbox = (0.0, 0.0, 0.0, 0.0)
                        position = _determine_position(page.rect, bbox)

                        image_area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                        page_area = max(1.0, page.rect.width * page.rect.height)
                        is_full_page = (image_area / page_area) > 0.6

                        image_index = len(images)
                        public_id = f'ncert/{safe_subject}/{safe_chapter}/img_{page_number}_{image_index}'

                        latex = run_pix2tex(image_bytes)
                        is_mathematical = latex is not None
                        ocr_text = None if is_mathematical else _run_google_ocr(image_bytes)
                        cloudinary_url = upload_image(image_bytes, public_id)

                        images.append(
                            {
                                'image_index': image_index,
                                'position': position,
                                'cloudinary_url': cloudinary_url,
                                'pix2tex_latex': latex,
                                'ocr_text': ocr_text,
                                'is_mathematical': is_mathematical,
                                'width': int(width),
                                'height': int(height),
                                'is_full_page': is_full_page,
                            }
                        )
                    except Exception:
                        logger.exception(
                            'Failed to process image on page %s in %s',
                            page_number,
                            pdf_path,
                        )
                        continue

                page_text = page.get_text('text') or ''
                text_chars = len(re.sub(r'\s+', '', page_text))
                is_image_only_page = text_chars < 50 and len(images) > 0

                pages_output.append(
                    {
                        'page_number': page_number,
                        'is_image_only_page': is_image_only_page,
                        'images': images,
                    }
                )
            except Exception:
                logger.exception('Failed to process page %s in %s', page_number, pdf_path)
                continue
    finally:
        doc.close()

    return pages_output
