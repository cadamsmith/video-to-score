"""[3] select - segment -> one clean frame.

Within a stable segment, pick the sharpest frame so we dodge motion blur and
half-turned pages that bracket a flip. Sharpness is measured by the variance of
the Laplacian: a crisp page has strong high-frequency edges (high variance); a
blurred or mid-turn frame is smooth (low variance).
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import Frame, Page, PageSegment


def focus_measure(image: np.ndarray) -> float:
    """Variance of the Laplacian - higher is sharper."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_page(segment: PageSegment, frames: list[Frame]) -> Page:
    """Pick the sharpest frame in ``segment`` and return it as a :class:`Page`."""
    best_idx = max(segment.frame_indices, key=lambda i: focus_measure(frames[i].image))
    best = frames[best_idx]
    return Page(image=best.image, timestamp=best.timestamp)


def select_pages(segments: list[PageSegment], frames: list[Frame]) -> list[Page]:
    """Select one clean page per segment, preserving order."""
    return [select_page(seg, frames) for seg in segments]
