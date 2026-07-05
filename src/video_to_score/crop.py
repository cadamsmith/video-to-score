"""[5] crop - isolate the notation region.

Sheet-music videos letterbox the page against a darker background. Threshold to
find the bright page rectangle and crop to its bounding box, trimming the
letterboxing and background. Overlay/cursor/hand removal is deliberately *not*
part of the MVP.
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import Page


def find_page_bbox(
    image: np.ndarray,
    margin: int = 4,
    min_area_ratio: float = 0.2,
) -> tuple[int, int, int, int] | None:
    """Find the bright page rectangle. Returns ``(x, y, w, h)`` or ``None``.

    ``None`` means no confident page region was found (the caller should keep the
    frame uncropped rather than guess).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Otsu picks the page/background split automatically across lighting.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(mask)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    if w * h < min_area_ratio * image.shape[0] * image.shape[1]:
        return None

    # Grow slightly so we don't shave the outermost staff lines.
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(image.shape[1], x + w + margin)
    y1 = min(image.shape[0], y + h + margin)
    return x0, y0, x1 - x0, y1 - y0


def crop_page(page: Page) -> Page:
    """Crop a page to its detected notation region (no-op if none is found)."""
    bbox = find_page_bbox(page.image)
    if bbox is None:
        return page
    x, y, w, h = bbox
    return Page(image=page.image[y : y + h, x : x + w], timestamp=page.timestamp)


def crop_pages(pages: list[Page]) -> list[Page]:
    """Crop every page to its notation region."""
    return [crop_page(p) for p in pages]
