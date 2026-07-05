"""``--debug`` signal dump - the instrument used to tune ``segment``.

Writes, to a debug directory, everything needed to *see* why the video segmented
the way it did:

- ``signal.csv``  - per-gap timestamp, dissimilarity value, and stable/transition
  classification.
- ``signal.png``  - a plot of that signal with the enter/exit thresholds drawn in
  and transition regions shaded (rendered with OpenCV, no plotting dependency).
- ``segments/``   - one representative frame per detected segment, named by order
  and timestamp, so you can eyeball whether the page boundaries are right.

Deliberately has no matplotlib dependency: the plot is drawn with OpenCV so the
debug path stays lightweight.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .segment import SegmentResult


def dump_debug(
    result: SegmentResult,
    frames,
    out_dir: str | Path,
    enter_threshold: float,
    exit_threshold: float,
) -> Path:
    """Write the signal CSV, signal plot, and per-segment frames to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    _write_csv(result, out / "signal.csv")
    _write_plot(result, out / "signal.png", enter_threshold, exit_threshold)
    _write_segment_frames(result, frames, out / "segments")
    return out


def _write_csv(result: SegmentResult, path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gap_index", "from_ts", "to_ts", "signal", "is_transition"])
        for i, value in enumerate(result.signal):
            writer.writerow(
                [
                    i,
                    f"{result.timestamps[i]:.3f}",
                    f"{result.timestamps[i + 1]:.3f}",
                    f"{value:.6f}",
                    int(bool(result.gap_is_transition[i])),
                ]
            )


def _write_plot(
    result: SegmentResult,
    path: Path,
    enter_threshold: float,
    exit_threshold: float,
) -> None:
    signal = result.signal
    w, h = 1200, 400
    pad = 40
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    if len(signal) == 0:
        cv2.imwrite(str(path), canvas)
        return

    y_max = max(float(signal.max()), enter_threshold) * 1.15 or 1.0
    plot_w, plot_h = w - 2 * pad, h - 2 * pad

    def to_xy(i: int, value: float) -> tuple[int, int]:
        x = pad + int(i / max(1, len(signal) - 1) * plot_w)
        y = pad + plot_h - int(value / y_max * plot_h)
        return x, y

    # Shade transition regions.
    for i, is_transition in enumerate(result.gap_is_transition):
        if is_transition:
            x0 = pad + int(i / max(1, len(signal) - 1) * plot_w)
            x1 = pad + int((i + 1) / max(1, len(signal) - 1) * plot_w)
            cv2.rectangle(canvas, (x0, pad), (max(x0 + 1, x1), pad + plot_h), (230, 230, 255), -1)

    # Threshold lines: enter (red), exit (orange).
    for thr, color in ((enter_threshold, (0, 0, 220)), (exit_threshold, (0, 140, 255))):
        y = pad + plot_h - int(thr / y_max * plot_h)
        cv2.line(canvas, (pad, y), (pad + plot_w, y), color, 1, cv2.LINE_AA)

    # Signal polyline.
    pts = [to_xy(i, float(v)) for i, v in enumerate(signal)]
    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (200, 0, 0), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (pad, pad), (pad + plot_w, pad + plot_h), (0, 0, 0), 1)
    cv2.putText(
        canvas,
        f"gaps={len(signal)}  segments={len(result.segments)}  y_max={y_max:.3f}",
        (pad, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), canvas)


def _write_segment_frames(result: SegmentResult, frames, seg_dir: Path) -> None:
    seg_dir.mkdir(parents=True, exist_ok=True)
    for order, segment in enumerate(result.segments):
        mid = segment.frame_indices[len(segment.frame_indices) // 2]
        name = f"seg_{order:03d}_t{segment.start_ts:07.2f}.png"
        cv2.imwrite(str(seg_dir / name), frames[mid].image)
