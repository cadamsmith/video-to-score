"""Segment is the crux: verify it finds the right page boundaries."""

from __future__ import annotations

import numpy as np
from conftest import make_frames, page_image, solid

from video_to_score.segment import classify_gaps, compute_signal, segment_frames


def test_clean_cut_two_pages():
    # 6 frames of page A, hard cut, 6 frames of page B.
    imgs = [page_image(1)] * 6 + [page_image(2)] * 6
    result = segment_frames(make_frames(imgs))
    assert len(result.segments) == 2
    assert result.segments[0].frame_indices[0] == 0
    assert result.segments[1].frame_indices[-1] == 11


def test_stable_signal_near_zero():
    imgs = [page_image(7)] * 8
    signal = compute_signal(make_frames(imgs))
    assert float(signal.max()) < 0.01


def test_fade_is_one_transition_and_fragments_dropped():
    # page A (6) -> 3-frame cross-fade -> page B (6). The fade frames are short
    # fragments and must be dropped, leaving exactly two pages. Use high-contrast
    # pages so each fade step is genuinely elevated (as a real page flip is).
    a = np.zeros((90, 120, 3), dtype=float)
    a[:, :60] = 255  # left half white
    b = np.zeros((90, 120, 3), dtype=float)
    b[:, 60:] = 255  # right half white
    fade = [(a * (1 - t) + b * t).astype("uint8") for t in (0.25, 0.5, 0.75)]
    imgs = [a.astype("uint8")] * 6 + fade + [b.astype("uint8")] * 6
    result = segment_frames(make_frames(imgs), min_stable_sec=1.0)
    assert len(result.segments) == 2


def test_classify_gaps_hysteresis():
    # Once "in transition" above enter, stays until it drops below exit.
    signal = np.array([0.0, 0.08, 0.04, 0.02, 0.0])
    states = classify_gaps(signal, enter_threshold=0.06, exit_threshold=0.03)
    assert list(states) == [False, True, True, False, False]


def test_no_transitions_single_segment():
    imgs = [solid(200)] * 5
    result = segment_frames(make_frames(imgs))
    assert len(result.segments) == 1


def test_clean_segmentation_reports_no_missed_flips():
    # A hard cut is well above threshold; nothing internal should be flagged.
    imgs = [page_image(1)] * 6 + [page_image(2)] * 6
    result = segment_frames(make_frames(imgs))
    assert result.suspected_missed == []


def test_sub_threshold_flip_is_flagged_as_missed():
    # Build a flip whose per-gap signal lands *between* warn_level and enter, so
    # it is never classified as a transition but is still a suspicious internal
    # spike. warn_level = 0.75 * enter = 0.03; enter = 0.04.
    base = np.zeros((100, 100, 3), dtype=np.uint8)
    changed = base.copy()
    changed[:4, :] = 255  # ~4% of pixels flip white -> MAD ~0.04
    imgs = [base] * 6 + [changed] * 6
    result = segment_frames(make_frames(imgs), enter_threshold=0.05, exit_threshold=0.03)
    # The flip peak (~0.04) is below enter (0.05) -> collapses into one segment...
    assert len(result.segments) == 1
    # ...but it is loud enough to be reported as a suspected missed flip.
    assert len(result.suspected_missed) == 1
