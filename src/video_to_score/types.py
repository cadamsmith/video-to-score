"""Data contracts passed between pipeline stages.

These small structures are the interfaces that keep each stage independently
tunable and swappable:

    Frame[] -> [segment] -> PageSegment[] -> [select] -> Page[]
            -> [dedup] -> [crop] -> Page[] -> [assemble] -> PDF
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    """A single sampled frame from the source video.

    ``image`` is a BGR ``uint8`` array (OpenCV convention). ``timestamp`` is the
    absolute position in the *original* video in seconds, independent of any
    ``--start`` trim, so page ordering survives across stages.
    """

    index: int
    timestamp: float
    image: np.ndarray


@dataclass
class PageSegment:
    """A contiguous "stable" stretch of frames showing one page.

    Produced by ``segment``: the run of frames between two page transitions.
    ``frame_indices`` refer to positions in the sampled ``Frame[]`` list.
    """

    start_ts: float
    end_ts: float
    frame_indices: list[int]


@dataclass
class Page:
    """One clean, representative image for a distinct page of the score."""

    image: np.ndarray
    timestamp: float
