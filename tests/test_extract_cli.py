"""extract sampling/windowing and CLI helpers."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from video_to_score.cli import parse_timecode
from video_to_score.extract import extract_frames


def _write_video(path, seconds=5, native_fps=30, w=160, h=120):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, native_fps, (w, h))
    for i in range(seconds * native_fps):
        # Encode the elapsed second into intensity so we can check timestamps.
        writer.write(np.full((h, w, 3), (i // native_fps) * 40 % 256, dtype=np.uint8))
    writer.release()
    return path


def test_extract_samples_near_target_fps(tmp_path):
    video = _write_video(tmp_path / "v.mp4", seconds=5, native_fps=30)
    frames = extract_frames(video, fps=2.0)
    # ~2 fps over ~5 s -> ~10 frames; allow slack for encoder timing.
    assert 8 <= len(frames) <= 12
    # Timestamps must be ascending.
    ts = [f.timestamp for f in frames]
    assert ts == sorted(ts)


def test_extract_start_end_window(tmp_path):
    video = _write_video(tmp_path / "v.mp4", seconds=6, native_fps=30)
    frames = extract_frames(video, fps=2.0, start=2.0, end=4.0)
    assert frames, "expected frames inside the window"
    assert all(1.9 <= f.timestamp <= 4.1 for f in frames)


def test_extract_bad_fps_raises(tmp_path):
    video = _write_video(tmp_path / "v.mp4", seconds=1)
    with pytest.raises(ValueError):
        extract_frames(video, fps=0)


def test_extract_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_frames(tmp_path / "nope.mp4")


@pytest.mark.parametrize(
    "value,expected",
    [(None, None), ("30", 30.0), ("1:30", 90.0), ("2:05", 125.0), ("1:00:00", 3600.0)],
)
def test_parse_timecode(value, expected):
    assert parse_timecode(value) == expected


def test_parse_timecode_invalid():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        parse_timecode("aa:bb")
