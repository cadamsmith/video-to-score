"""[6] assemble - pages -> PDF.

The captured pages are wide landscape strips (a couple of systems each), but real
sheet music reads as a vertical page. So we stack several strips per portrait
page. Rather than a fixed strips-per-page count, we pack by resolution: scale each
strip to the page width, then greedily fill a portrait page with as many strips
(in timestamp order) as fit its height before starting the next page. A short,
wide strip takes little vertical room, so more fit; a tall strip takes more, so
fewer do. Every strip keeps its natural proportions, and the packing count falls
out of the strip aspect ratios instead of being hard-coded. ``img2pdf`` embeds
the result losslessly.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import img2pdf
import numpy as np

from .types import Page

# Portrait page shape as width / height (US Letter, 8.5 x 11).
LETTER_ASPECT = 8.5 / 11.0


def _resize_to_page(image: np.ndarray, page_w: int, page_h: int) -> np.ndarray:
    """Scale ``image`` to fill the page width without exceeding the page height."""
    h, w = image.shape[:2]
    # Fill the width; if that would make the strip taller than a whole page
    # (a rare, very tall strip), fall back to fitting the height instead.
    scale = min(page_w / w, page_h / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def stack_pages(
    pages: list[Page],
    max_rows_per_page: int | None = None,
    page_aspect: float = LETTER_ASPECT,
) -> list[np.ndarray]:
    """Pack strips onto portrait canvases, fitting as many per page as resolution allows.

    Each strip is scaled to the page width, then strips are placed onto portrait
    pages in order, filling a page until the next strip would overflow its height.

    Args:
        pages: Strips in timestamp order.
        max_rows_per_page: Optional cap on strips per page. ``None`` fits as many
            as the strip heights allow; an int is an upper bound on top of that.
        page_aspect: Portrait page width / height (< 1 is taller than wide).

    Returns:
        One white BGR canvas per output page. Every strip keeps its natural
        proportions; leftover vertical space is shared as equal gaps.
    """
    if not pages:
        return []
    if max_rows_per_page is not None and max_rows_per_page < 1:
        raise ValueError(f"max_rows_per_page must be >= 1, got {max_rows_per_page}")

    # Page dimensions are derived from the widest strip so nothing is upscaled
    # past its capture resolution; height follows from the portrait aspect.
    page_w = max(p.image.shape[1] for p in pages)
    page_h = max(1, round(page_w / page_aspect))

    resized = [_resize_to_page(p.image, page_w, page_h) for p in pages]

    # First-fit in timestamp order: never reorder (strips are systems in reading
    # order), just decide where each page break falls.
    groups: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []
    current_h = 0
    for strip in resized:
        strip_h = strip.shape[0]
        capped = max_rows_per_page is not None and len(current) >= max_rows_per_page
        if current and (current_h + strip_h > page_h or capped):
            groups.append(current)
            current, current_h = [], 0
        current.append(strip)
        current_h += strip_h
    if current:
        groups.append(current)

    canvases: list[np.ndarray] = []
    for group in groups:
        canvas = np.full((page_h, page_w, 3), 255, dtype=np.uint8)
        content_h = sum(strip.shape[0] for strip in group)
        # Share the leftover space as equal gaps above, between, and below strips.
        gap = max(0, page_h - content_h) / (len(group) + 1)
        y = gap
        for strip in group:
            new_h, new_w = strip.shape[:2]
            x0 = (page_w - new_w) // 2
            y0 = min(round(y), page_h - new_h)  # clamp against rounding overflow
            canvas[y0 : y0 + new_h, x0 : x0 + new_w] = strip
            y += new_h + gap
        canvases.append(canvas)
    return canvases


def assemble_pdf(
    pages: list[Page],
    output_path: str | Path,
    max_rows_per_page: int | None = None,
) -> Path:
    """Order ``pages`` by timestamp, pack them, and write a portrait PDF."""
    if not pages:
        raise ValueError("no pages to assemble; nothing to write")

    ordered = sorted(pages, key=lambda p: p.timestamp)
    canvases = stack_pages(ordered, max_rows_per_page=max_rows_per_page)

    png_bytes = []
    for canvas in canvases:
        ok, buf = cv2.imencode(".png", canvas)
        if not ok:
            raise RuntimeError("failed to encode page as PNG")
        png_bytes.append(buf.tobytes())

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(img2pdf.convert(png_bytes))
    return out
