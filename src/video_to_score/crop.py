"""[5] crop - isolate the notation region, then split it into systems.

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

A captured strip is a video *viewport*, so it typically holds a couple of systems
plus their surrounding whitespace. If ``assemble`` packed those whole strips it
could only fit two per portrait page (three overflow), leaving a wide gap down the
middle and forcing the reader to flip more than necessary. So after cropping we
also *split* each strip into its individual systems along the horizontal
whitespace between them, using the same vertical-projection band logic. Packing
the shorter system units fills each page tightly (four to six systems) and reads
like a real engraved score.
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import Page


def _row_runs(flags: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """Half-open ``(start, end)`` runs of ``True`` in ``flags``, merging gaps up to ``max_gap``.

    Merging short gaps keeps a single dense chord row (which momentarily dips below
    the flag threshold) from splitting one band into two.
    """
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > max_gap + 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]])) + 1
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def _dominant_band(solid: np.ndarray, max_gap: int) -> tuple[int, int] | None:
    """Longest run of ``True`` rows in ``solid``, merging gaps up to ``max_gap``.

    Returns ``(y0, y1)`` half-open, or ``None`` if no row qualifies.
    """
    runs = _row_runs(solid, max_gap)
    if not runs:
        return None
    return max(runs, key=lambda r: r[1] - r[0])


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


def _system_bands(
    image: np.ndarray,
    gap_frac: float,
    min_system_frac: float,
) -> list[tuple[int, int]]:
    """Vertical ``(y0, y1)`` bands, one per system, top to bottom.

    Rows carrying ink are grouped into runs (gaps up to ``gap_frac`` of the height
    are bridged so a grand staff's treble/bass halves stay together). A run shorter
    than ``min_system_frac`` of the height is a label, not a system -- a chord-symbol
    row, a title, a watermark -- so it is folded into the system just *below* it
    (its owner), or into the last system if nothing follows. Nothing is dropped.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h = mask.shape[0]

    ink = 1.0 - mask.mean(axis=1) / 255.0
    runs = _row_runs(ink > 0.005, max_gap=int(gap_frac * h))
    if not runs:
        return []

    min_system = int(min_system_frac * h)
    bands: list[list[int]] = []
    carry: int | None = None  # top of pending short run(s) awaiting the system below
    for y0, y1 in runs:
        if y1 - y0 < min_system:
            carry = y0 if carry is None else carry
        else:
            bands.append([carry if carry is not None else y0, y1])
            carry = None
    if carry is not None:  # trailing label(s) with no system below: keep with the last
        if bands:
            bands[-1][1] = runs[-1][1]
        else:
            bands.append([carry, runs[-1][1]])
    return [(y0, y1) for y0, y1 in bands]


def split_systems(
    page: Page,
    gap_frac: float = 0.035,
    min_system_frac: float = 0.12,
    pad_frac: float = 0.02,
) -> list[Page]:
    """Split one cropped strip into per-system pages (top to bottom, order preserved).

    Each returned page keeps the strip's full width and inherits its timestamp, so
    downstream ordering is unchanged. Bands are padded by ``pad_frac`` of the height
    so ledger lines and the outermost staff lines are not shaved. A strip with a
    single band (or none) is returned tightly cropped but not split.
    """
    h = page.image.shape[0]
    bands = _system_bands(page.image, gap_frac, min_system_frac)
    if not bands:
        return [page]
    pad = int(pad_frac * h)
    return [
        Page(image=page.image[max(0, y0 - pad) : min(h, y1 + pad)], timestamp=page.timestamp)
        for y0, y1 in bands
    ]


def split_pages(pages: list[Page]) -> list[Page]:
    """Flatten every strip into its systems, preserving timestamp order."""
    return [system for page in pages for system in split_systems(page)]
