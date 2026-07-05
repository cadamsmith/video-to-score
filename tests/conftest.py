"""Shared helpers for building synthetic frames/pages."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from video_to_score.types import Frame, Page

FPS = 2.0
H, W = 90, 120


def solid(value: int) -> np.ndarray:
    """A uniform BGR image of the given intensity."""
    return np.full((H, W, 3), value, dtype=np.uint8)


def page_image(seed: int) -> np.ndarray:
    """A distinct, texture-rich page image (so focus/hash/SSIM have signal)."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    for _ in range(40):
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        cv2.line(img, (x, y), (x + 15, y + 3), (0, 0, 0), 1)
    return img


def blur(img: np.ndarray, k: int = 9) -> np.ndarray:
    return cv2.GaussianBlur(img, (k, k), 0)


def make_frames(images: list[np.ndarray], fps: float = FPS) -> list[Frame]:
    """Wrap images as Frames with 1/fps-spaced timestamps."""
    return [Frame(index=i, timestamp=i / fps, image=img) for i, img in enumerate(images)]


@pytest.fixture
def two_pages() -> tuple[Page, Page]:
    return Page(page_image(1), 0.0), Page(page_image(2), 10.0)
