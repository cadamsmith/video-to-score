"""[2] segment - frames -> page segments  (the crux).

A page-flip video is a sequence of long *stable* stretches (one page held on
screen) separated by short *transition* events (a flip or cross-fade). This stage
recovers that structure:

1. Compute a per-gap dissimilarity signal between consecutive frames on
   downscaled grayscale. A stable page -> signal near zero; a flip or cross-fade
   -> a sustained spike.
2. Classify each gap stable/transition with a two-threshold **hysteresis** rule,
   so a slow fade (signal elevated for a *run* of gaps) is treated as one
   transition rather than fragmenting into many.
3. Split the frame sequence at transition gaps and keep the runs that stay stable
   long enough to be a real page (``min_stable_sec``), dropping the short
   fragments produced *during* a multi-gap fade.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .types import Frame, PageSegment

# Similarity signal methods.
METHOD_MAD = "mad"  # mean absolute difference of normalized grayscale
METHOD_HIST = "hist"  # 1 - histogram correlation


@dataclass
class SegmentResult:
    """Segmentation output plus the raw signals that produced it (for ``--debug``)."""

    segments: list[PageSegment]
    signal: np.ndarray  # per-gap dissimilarity, length len(frames) - 1
    gap_is_transition: np.ndarray  # bool per gap, length len(frames) - 1
    timestamps: np.ndarray  # per-frame timestamp, length len(frames)


def _prep(image: np.ndarray, downscale: int) -> np.ndarray:
    """Downscale to grayscale float in [0, 1] for cheap, robust comparison."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if max(h, w) > downscale:
        scale = downscale / float(max(h, w))
        gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))))
    return gray.astype(np.float32) / 255.0


def compute_signal(
    frames: list[Frame],
    method: str = METHOD_MAD,
    downscale: int = 128,
) -> np.ndarray:
    """Per-gap dissimilarity between consecutive frames, in ``[0, 1]``.

    Returns an array of length ``len(frames) - 1`` (empty if fewer than 2 frames).
    """
    if len(frames) < 2:
        return np.empty(0, dtype=np.float32)

    prepped = [_prep(f.image, downscale) for f in frames]
    signal = np.empty(len(frames) - 1, dtype=np.float32)

    for i in range(len(prepped) - 1):
        a, b = prepped[i], prepped[i + 1]
        if method == METHOD_MAD:
            signal[i] = float(np.mean(np.abs(a - b)))
        elif method == METHOD_HIST:
            ha = cv2.calcHist([a], [0], None, [64], [0.0, 1.0])
            hb = cv2.calcHist([b], [0], None, [64], [0.0, 1.0])
            corr = cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)
            signal[i] = float(np.clip(1.0 - corr, 0.0, 1.0))
        else:
            raise ValueError(f"unknown method: {method!r}")

    return signal


def classify_gaps(
    signal: np.ndarray,
    enter_threshold: float,
    exit_threshold: float,
) -> np.ndarray:
    """Hysteresis threshold: enter transition above ``enter``, stay until below ``exit``.

    ``enter_threshold`` should be >= ``exit_threshold``. Returns a bool array the
    same length as ``signal`` (True = transition gap).
    """
    states = np.zeros(len(signal), dtype=bool)
    in_transition = False
    for i, value in enumerate(signal):
        if not in_transition and value >= enter_threshold:
            in_transition = True
        elif in_transition and value <= exit_threshold:
            in_transition = False
        states[i] = in_transition
    return states


def segment_frames(
    frames: list[Frame],
    method: str = METHOD_MAD,
    enter_threshold: float = 0.06,
    exit_threshold: float = 0.03,
    min_stable_sec: float = 1.0,
    downscale: int = 128,
) -> SegmentResult:
    """Group ``frames`` into stable page segments.

    Args:
        frames: Sampled frames in timestamp order.
        method: Dissimilarity signal (``"mad"`` or ``"hist"``).
        enter_threshold: Signal at/above which a gap starts a transition.
        exit_threshold: Signal at/below which a transition ends.
        min_stable_sec: Minimum wall-clock duration for a run to count as a page.
            Filters out the short fragments produced during a multi-gap fade.
        downscale: Longest-edge size used when comparing frames.

    Returns:
        A :class:`SegmentResult` with the page segments and the raw signals.
    """
    timestamps = np.array([f.timestamp for f in frames], dtype=np.float64)
    signal = compute_signal(frames, method=method, downscale=downscale)

    if len(frames) == 0:
        return SegmentResult([], signal, np.empty(0, dtype=bool), timestamps)
    if len(frames) == 1:
        seg = PageSegment(timestamps[0], timestamps[0], [0])
        return SegmentResult([seg], signal, np.empty(0, dtype=bool), timestamps)

    gap_is_transition = classify_gaps(signal, enter_threshold, exit_threshold)

    # Split the frame sequence at transition gaps into candidate runs.
    runs: list[list[int]] = []
    current = [0]
    for i, is_transition in enumerate(gap_is_transition):
        if is_transition:
            runs.append(current)
            current = [i + 1]
        else:
            current.append(i + 1)
    runs.append(current)

    # Keep runs that stay stable long enough to be a real page. A single run
    # spanning the whole clip (no transitions found) is always kept.
    segments: list[PageSegment] = []
    for run in runs:
        start_ts = float(timestamps[run[0]])
        end_ts = float(timestamps[run[-1]])
        if len(runs) == 1 or end_ts - start_ts >= min_stable_sec:
            segments.append(PageSegment(start_ts, end_ts, run))

    return SegmentResult(segments, signal, gap_is_transition, timestamps)
