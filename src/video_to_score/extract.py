"""[1] extract - video -> sampled frames.

Decode the MP4 and sample at a low fixed rate (~2 fps by default). Page flips are
slow, human-scale events, so there is no need to touch every frame. The
``--start``/``--end`` window is applied *here* by seeking the decoder, so no
downstream stage ever sees the intro/outro.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2

from .types import Frame


def extract_frames(
    video_path: str | Path,
    fps: float = 2.0,
    start: float | None = None,
    end: float | None = None,
) -> list[Frame]:
    """Sample frames from ``video_path`` at ``fps`` frames per second.

    Args:
        video_path: Path to a local MP4.
        fps: Target sampling rate. Must be > 0.
        start: Optional start of the window, in seconds of the original video.
        end: Optional end of the window, in seconds of the original video.

    Returns:
        A list of :class:`Frame`, in ascending timestamp order, with timestamps
        expressed in seconds of the *original* video.
    """
    return list(iter_frames(video_path, fps=fps, start=start, end=end))


def iter_frames(
    video_path: str | Path,
    fps: float = 2.0,
    start: float | None = None,
    end: float | None = None,
) -> Iterator[Frame]:
    """Streaming variant of :func:`extract_frames` that yields one frame at a time."""
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")

    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        native_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # Fall back to a frame-count/duration estimate if FPS metadata is missing.
        if native_fps <= 0:
            native_fps = 30.0

        window_start = start if start is not None else 0.0
        # Seek by timestamp so the decoder skips the intro entirely.
        if window_start > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, window_start * 1000.0)

        sample_interval = 1.0 / fps
        next_sample_ts = window_start
        emitted = 0

        while True:
            ok, image = cap.read()
            if not ok:
                break

            # Prefer the decoder's own timestamp; fall back to frame-index math
            # for containers that don't report POS_MSEC reliably.
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if ts <= 0:
                pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                ts = pos / native_fps

            if end is not None and ts > end:
                break

            if ts + 1e-9 >= next_sample_ts:
                yield Frame(index=emitted, timestamp=ts, image=image)
                emitted += 1
                # Advance to the next slot at or after the current timestamp so we
                # stay on grid even if the decoder overshoots.
                next_sample_ts += sample_interval
                while next_sample_ts <= ts:
                    next_sample_ts += sample_interval

        _ = native_frames  # reserved for future progress reporting
    finally:
        cap.release()
