"""Headless-safe display helpers shared across wrappers and tracking modules."""
from __future__ import annotations

import os

import cv2


def safe_imshow(window_name: str, image) -> None:
    """Show *image* in a named OpenCV window, silently no-op when headless."""
    if image is None:
        return
    if os.name != "nt" and os.environ.get("DISPLAY") is None:
        return
    try:
        cv2.imshow(window_name, image)
        cv2.waitKey(1)
    except cv2.error:
        pass
