"""Cloudinary upload helpers."""

from __future__ import annotations

import logging
import os
from io import BytesIO
from typing import Optional

import cloudinary
from cloudinary import uploader

logger = logging.getLogger(__name__)

_CLOUDINARY_CONFIGURED = False


def _configure_cloudinary() -> bool:
    """
    Configure Cloudinary SDK from environment variables.

    Returns:
        True if configuration is present and applied; otherwise False.
    """
    global _CLOUDINARY_CONFIGURED
    if _CLOUDINARY_CONFIGURED:
        return True

    cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME')
    api_key = os.getenv('CLOUDINARY_API_KEY')
    api_secret = os.getenv('CLOUDINARY_API_SECRET')

    if not cloud_name or not api_key or not api_secret:
        logger.warning('Cloudinary credentials are missing in environment variables')
        return False

    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        secure=True,
    )
    _CLOUDINARY_CONFIGURED = True
    return True


def upload_image(image_bytes: bytes, public_id: str) -> Optional[str]:
    """
    Upload image bytes to Cloudinary.

    Args:
        image_bytes: Raw image bytes.
        public_id: Cloudinary public ID.

    Returns:
        Secure Cloudinary URL on success; otherwise None.
        This function never raises.
    """
    if not image_bytes:
        return None
    if not public_id:
        return None
    if not _configure_cloudinary():
        return None

    try:
        with BytesIO(image_bytes) as image_stream:
            result = uploader.upload(
                image_stream,
                public_id=public_id,
                resource_type='image',
                overwrite=True,
            )
        secure_url = result.get('secure_url') if isinstance(result, dict) else None
        return str(secure_url) if secure_url else None
    except Exception:
        logger.exception('Cloudinary upload failed for public_id=%s', public_id)
        return None
