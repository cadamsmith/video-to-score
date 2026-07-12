"""[5] crop - isolate the notation region.

Sheet-music videos show the page as a bright, near-full-width horizontal *band*.
Anything else bright in the frame -- a piano keyboard filmed below the page, a
title card, a hand -- is what we want to trim. A plain bounding box of every
bright pixel fails here: it swallows the keyboard because the white keys are just
as bright as the page.

Instead we project the bright mask onto the vertical axis and keep the tallest
run of rows that are *mostly* white across their width. The page fits that
description (a solid strip); a keyboard does not -- its black keys carve every row
down to roughly half-white, and the dark piano body separates it from the page.
Overlay/cursor/hand removal is deliberately *not* part of the MVP.
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import Page


def _dominant_band(solid: np.ndarray, max_gap: int) -> tuple[int, int] | None:
    """Longest run of ``True`` rows in ``solid``, merging gaps up to ``max_gap``.

    Returns ``(y0, y1)`` half-open, or ``None`` if no row qualifies. Merging short
    gaps keeps a single dense chord row (which momentarily dips below the fill
    threshold) from splitting the page into two shorter bands.
    """
    idx = np.flatnonzero(solid)
    if idx.size == 0:
        return None
    breaks = np.flatnonzero(np.diff(idx) > max_gap + 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    best = np.argmax(ends - starts)
    return int(starts[best]), int(ends[best]) + 1


def find_page_bbox(
    image: np.ndarray,
    margin: int = 4,
    min_area_ratio: float = 0.2,
    band_frac: float = 0.6,
) -> tuple[int, int, int, int] | None:
    """Find the bright page band. Returns ``(x, y, w, h)`` or ``None``.

    ``band_frac`` is the fraction of a row that must be white for the row to count
    as part of the page band -- high enough to reject a keyboard (~half-white per
    row once black keys are subtracted), low enough to survive a dense page.

    ``None`` means no confident page region was found (the caller should keep the
    frame uncropped rather than guess).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Otsu picks the page/background split automatically across lighting.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = mask.shape

    # Vertical projection: keep the tallest run of near-full-width bright rows.
    solid = mask.mean(axis=1) / 255.0 >= band_frac
    band = _dominant_band(solid, max_gap=int(0.02 * h))
    if band is None:
        return None
    y0, y1 = band

    # Within that band, trim left/right to the bright content (side letterboxing).
    coords = cv2.findNonZero(mask[y0:y1])
    if coords is None:
        return None
    bx, _, bw, _ = cv2.boundingRect(coords)

    # Grow slightly so we don't shave the outermost staff lines.
    x0 = max(0, bx - margin)
    y0 = max(0, y0 - margin)
    x1 = min(w, bx + bw + margin)
    y1 = min(h, y1 + margin)
    if (x1 - x0) * (y1 - y0) < min_area_ratio * h * w:
        return None
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
