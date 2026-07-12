"""select / dedup / crop / assemble stage checks."""

from __future__ import annotations

import numpy as np
import pikepdf
from conftest import blur, make_frames, page_image

from video_to_score.assemble import assemble_pdf
from video_to_score.crop import crop_page, find_page_bbox
from video_to_score.dedup import dedup_pages
from video_to_score.segment import PageSegment
from video_to_score.select import focus_measure, select_page
from video_to_score.types import Page


def test_select_picks_sharpest_frame():
    sharp = page_image(3)
    frames = make_frames([blur(sharp, 11), sharp, blur(sharp, 7)])
    seg = PageSegment(0.0, 1.0, [0, 1, 2])
    page = select_page(seg, frames)
    # The middle (unblurred) frame is sharpest.
    assert focus_measure(page.image) == max(focus_measure(f.image) for f in frames)
    assert np.array_equal(page.image, sharp)


def test_dedup_collapses_identical_pages():
    p = page_image(5)
    pages = [Page(p, 0.0), Page(p.copy(), 5.0), Page(page_image(6), 10.0)]
    kept = dedup_pages(pages)
    assert len(kept) == 2


def test_dedup_keeps_distinct_pages():
    pages = [Page(page_image(i), float(i)) for i in range(4)]
    assert len(dedup_pages(pages)) == 4


def test_dedup_preserves_order():
    pages = [Page(page_image(1), 0.0), Page(page_image(2), 1.0), Page(page_image(1), 2.0)]
    kept = dedup_pages(pages)
    assert [p.timestamp for p in kept] == [0.0, 1.0]


def test_crop_isolates_bright_band():
    # Dark canvas with a bright, near-full-width page band offset vertically.
    canvas = np.full((200, 200, 3), 20, dtype=np.uint8)
    canvas[40:160, 20:180] = 240
    x, y, w, h = find_page_bbox(canvas, margin=0)
    assert 15 <= x <= 25 and 35 <= y <= 45
    assert 155 <= w <= 165 and 115 <= h <= 125


def test_crop_rejects_keyboard_below_page():
    # A page band on top and a bright-but-striped keyboard below it (white keys
    # broken up by black keys). The crop must keep the page and drop the keyboard.
    canvas = np.full((300, 200, 3), 20, dtype=np.uint8)
    canvas[20:140, 20:180] = 240  # page band: solid across its width
    keys = np.full((100, 160, 3), 240, dtype=np.uint8)
    keys[:, ::4] = 20  # black keys carve every row down to ~75% white
    keys[:, 1::4] = 20  # ~50% white -> below band_frac
    canvas[180:280, 20:180] = keys
    x, y, w, h = find_page_bbox(canvas, margin=0)
    assert 15 <= y <= 25 and 115 <= h <= 125  # bounded to the page band
    assert y + h <= 180  # keyboard excluded


def test_crop_page_shrinks_image():
    canvas = np.full((200, 200, 3), 20, dtype=np.uint8)
    canvas[40:160, 20:180] = 240
    cropped = crop_page(Page(canvas, 0.0))
    assert cropped.image.shape[0] < 200 and cropped.image.shape[1] < 200


def test_assemble_packs_by_resolution(tmp_path):
    from video_to_score.assemble import stack_pages

    # A portrait page (aspect 8.5/11) is ~1.29x as tall as wide. Short, wide
    # strips (width 200, height 50 -> scaled height 50 at full width) pack several
    # per page; the count falls out of the aspect ratio, not a hard-coded number.
    strips = [Page(np.full((50, 200, 3), 255, np.uint8), float(i)) for i in range(10)]
    canvases = stack_pages(strips)
    # page_h = round(200 * 11 / 8.5) = 259 -> floor(259 / 50) = 5 strips per page.
    assert len(canvases) == 2  # 10 strips, 5 per page


def test_assemble_taller_strips_pack_fewer(tmp_path):
    from video_to_score.assemble import stack_pages

    # Same width but twice as tall -> half as many fit per page.
    short = stack_pages([Page(np.full((50, 200, 3), 255, np.uint8), float(i)) for i in range(6)])
    tall = stack_pages([Page(np.full((100, 200, 3), 255, np.uint8), float(i)) for i in range(6)])
    assert len(tall) > len(short)


def test_assemble_max_rows_per_page_caps(tmp_path):
    from video_to_score.assemble import stack_pages

    # Short strips would auto-fit several per page; the cap forces at most 2.
    strips = [Page(np.full((50, 200, 3), 255, np.uint8), float(i)) for i in range(6)]
    canvases = stack_pages(strips, max_rows_per_page=2)
    assert len(canvases) == 3  # 6 strips, capped at 2 per page


def test_assemble_writes_pdf(tmp_path):
    pages = [Page(page_image(i), float(i)) for i in range(7)]
    out = assemble_pdf(pages, tmp_path / "out.pdf")
    assert out.exists()
    assert out.read_bytes()[:4] == b"%PDF"
    with pikepdf.open(out) as pdf:
        assert len(pdf.pages) >= 1


def test_assemble_pages_are_portrait(tmp_path):
    from video_to_score.assemble import stack_pages

    # Wide landscape strips should be stacked onto taller-than-wide pages.
    strips = [Page(np.full((60, 200, 3), 255, np.uint8), float(i)) for i in range(3)]
    canvases = stack_pages(strips)
    h, w = canvases[0].shape[:2]
    assert h > w


def test_assemble_orders_by_timestamp(tmp_path):
    # Out-of-order input, one strip per page, still yields time-ordered pages.
    a, b, c = page_image(0), page_image(1), page_image(2)
    pages = [Page(c, 20.0), Page(a, 0.0), Page(b, 10.0)]
    out = assemble_pdf(pages, tmp_path / "ordered.pdf", max_rows_per_page=1)
    with pikepdf.open(out) as pdf:
        assert len(pdf.pages) == 3


def test_assemble_empty_raises(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        assemble_pdf([], tmp_path / "empty.pdf")
