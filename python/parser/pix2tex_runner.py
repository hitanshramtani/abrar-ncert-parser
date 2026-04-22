"""Utilities for running pix2tex OCR with a lazy singleton model."""

from __future__ import annotations

import logging
from io import BytesIO
from threading import Lock
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

_MODEL = None
_MODEL_LOCK = Lock()


def _get_model():
    """
    Lazily initialize and return a singleton LatexOCR model.

    Returns:
        LatexOCR model instance, or None if model loading fails.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            from pix2tex.cli import LatexOCR

            _MODEL = LatexOCR()
            logger.info('pix2tex model initialized')
        except Exception:
            logger.exception('Failed to initialize pix2tex model')
            _MODEL = None

    return _MODEL


def run_pix2tex(image_bytes: bytes) -> Optional[str]:
    """
    Run pix2tex against image bytes and return detected LaTeX.

    Returns:
        LaTeX string if detection succeeds; otherwise None.
    """
    if not image_bytes:
        return None

    model = _get_model()
    if model is None:
        return None

    try:
        with BytesIO(image_bytes) as image_buffer:
            image = Image.open(image_buffer).convert('RGB')
            latex = model(image)
    except Exception:
        logger.debug('pix2tex inference failed for image', exc_info=True)
        return None

    if latex is None:
        return None

    latex_str = str(latex).strip()
    return latex_str or None
