"""[6] assemble - pages -> PDF.

The captured pages are wide landscape strips (a couple of systems each), but real
sheet music reads as a vertical page. So we stack several strips per portrait
page: order the strips by timestamp, group them ``rows_per_page`` at a time, and
place each group onto a portrait canvas, scaling every strip down to fit its slot
(zoom out to fit). ``img2pdf`` embeds the result losslessly.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import img2pdf
import numpy as np

from .types import Page

# Portrait page shape as width / height (US Letter, 8.5 x 11).
LETTER_ASPECT = 8.5 / 11.0


def stack_pages(
    pages: list[Page],
    rows_per_page: int = 3,
    page_aspect: float = LETTER_ASPECT,
) -> list[np.ndarray]:
    """Stack strips ``rows_per_page`` at a time onto portrait canvases.

    Args:
        pages: Strips in timestamp order.
        rows_per_page: How many strips to stack on each portrait page.
        page_aspect: Portrait page width / height (< 1 is taller than wide).

    Returns:
        One white BGR canvas per output page. Each strip is scaled to fit its
        equal-height slot without distortion; the last page may have empty slots.
    """
    if not pages:
        return []
    if rows_per_page < 1:
        raise ValueError(f"rows_per_page must be >= 1, got {rows_per_page}")

    # Page dimensions are derived from the widest strip so nothing is upscaled
    # past its capture resolution; height follows from the portrait aspect.
    page_w = max(p.image.shape[1] for p in pages)
    page_h = max(1, round(page_w / page_aspect))
    slot_h = page_h // rows_per_page

    canvases: list[np.ndarray] = []
    for start in range(0, len(pages), rows_per_page):
        chunk = pages[start : start + rows_per_page]
        canvas = np.full((page_h, page_w, 3), 255, dtype=np.uint8)
        for row, page in enumerate(chunk):
            h, w = page.image.shape[:2]
            scale = min(page_w / w, slot_h / h)  # zoom out to fit the slot
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(page.image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            slot_top = row * slot_h
            x0 = (page_w - new_w) // 2
            y0 = slot_top + (slot_h - new_h) // 2
            canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
        canvases.append(canvas)
    return canvases


def assemble_pdf(
    pages: list[Page],
    output_path: str | Path,
    rows_per_page: int = 3,
) -> Path:
    """Order ``pages`` by timestamp, stack them, and write a portrait PDF."""
    if not pages:
        raise ValueError("no pages to assemble; nothing to write")

    ordered = sorted(pages, key=lambda p: p.timestamp)
    canvases = stack_pages(ordered, rows_per_page=rows_per_page)

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
