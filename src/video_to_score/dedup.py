"""[4] dedup - drop repeated pages.

Videos revisit page 1 (intro/outro) or loop, so the same page can appear as more
than one segment. Collapse near-duplicate pages via a perceptual hash (fast, robust
to small rendering differences) with a structural-similarity (SSIM) confirmation to
avoid collapsing two genuinely different pages that happen to hash alike. This stage
*removes* duplicates but never reorders what remains.
"""

from __future__ import annotations

import cv2
import imagehash
import numpy as np
from PIL import Image

from .types import Page


def _phash(image: np.ndarray) -> imagehash.ImageHash:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb))


def ssim(a: np.ndarray, b: np.ndarray, size: int = 256) -> float:
    """Structural similarity of two images in ``[-1, 1]`` (1.0 == identical).

    Both images are converted to grayscale and resized to ``size x size`` so pages
    captured at slightly different scales still compare cleanly.
    """
    ga = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (size, size)).astype(np.float64)
    gb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (size, size)).astype(np.float64)

    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (11, 11)
    mu_a = cv2.GaussianBlur(ga, k, 1.5)
    mu_b = cv2.GaussianBlur(gb, k, 1.5)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = cv2.GaussianBlur(ga * ga, k, 1.5) - mu_a2
    var_b = cv2.GaussianBlur(gb * gb, k, 1.5) - mu_b2
    cov = cv2.GaussianBlur(ga * gb, k, 1.5) - mu_ab

    num = (2 * mu_ab + c1) * (2 * cov + c2)
    den = (mu_a2 + mu_b2 + c1) * (var_a + var_b + c2)
    return float((num / den).mean())


def dedup_pages(
    pages: list[Page],
    max_hash_distance: int = 8,
    min_ssim: float = 0.85,
) -> list[Page]:
    """Return ``pages`` with near-duplicates removed, order preserved.

    A page is dropped if it matches any already-kept page on *both* signals: a
    perceptual-hash Hamming distance ``<= max_hash_distance`` and an SSIM
    ``>= min_ssim``.
    """
    kept: list[Page] = []
    kept_hashes: list[imagehash.ImageHash] = []

    for page in pages:
        page_hash = _phash(page.image)
        is_dup = False
        for kept_page, kept_hash in zip(kept, kept_hashes, strict=True):
            if (page_hash - kept_hash) <= max_hash_distance and ssim(
                page.image, kept_page.image
            ) >= min_ssim:
                is_dup = True
                break
        if not is_dup:
            kept.append(page)
            kept_hashes.append(page_hash)

    return kept
