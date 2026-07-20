"""[4] filter - drop frames that aren't sheet music.

The whole pipeline assumes a page is bright paper (see ``crop``). But a video often
opens or closes on a non-paper card -- a black "thank you" outro, a title slate --
that segments and selects into a page like any other, then survives ``crop`` as a
full-frame "system" and lands in the score.

The distinguishing trait is that a card is near-*black*, while any real page shows a
large bright paper band -- even a half-scrolled page, where the paper occupies only
part of the frame and black letterbox fills the rest, still has plenty of bright
pixels. So we gate on the *fraction* of bright pixels, not overall brightness: a
median-gray gate would wrongly reject a letterboxed page (its dark bars drag the
median down), whereas real pages hold ~0.4+ of the frame bright and a dark card
~0.0 -- a wide, safe gap.

The default threshold sits low, near the observed junk ceiling (~0.02) rather than
midway to the real-page floor, because the two errors are asymmetric. A false *drop*
loses a real page silently from the user's PDF; a false *keep* leaves a visible junk
page the user can trim with ``--end``. So we bias toward keeping -- a sparse
single-system scroll view is still a real page with little paper on screen.

This runs before ``dedup`` so the distinct-page count already excludes non-pages
(and so ``dedup`` never wastes a comparison on one).
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import Page

# Gray at/above which a pixel counts as "paper-bright" (paper sits ~200+).
_BRIGHT_GRAY = 180


def paper_fraction(image: np.ndarray) -> float:
    """Fraction of the frame that is paper-bright, in ``[0, 1]``."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float((gray >= _BRIGHT_GRAY).mean())


def is_paper_page(image: np.ndarray, min_paper_frac: float = 0.10) -> bool:
    """True if the frame shows enough paper to be a page, not a dark title/outro card."""
    return paper_fraction(image) >= min_paper_frac


def drop_non_pages(pages: list[Page], min_paper_frac: float = 0.10) -> list[Page]:
    """Return ``pages`` with dark non-sheet-music frames removed, order preserved."""
    return [p for p in pages if is_paper_page(p.image, min_paper_frac)]
