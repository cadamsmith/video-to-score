"""CLI - orchestration.

    video-to-score input.mp4 -o output.pdf [--start mm:ss] [--end mm:ss]

Runs the linear pipeline end to end:

    extract -> segment -> select -> dedup -> crop -> assemble

``--start``/``--end`` let the user manually skip non-notation intros/outros (title
cards, talking-head intros, bumpers) in place of auto-detecting the first page of
music. ``--debug`` dumps the per-frame similarity signal and the selected frames so
you can *see* why it segmented the way it did.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .assemble import assemble_pdf
from .crop import crop_pages
from .dedup import dedup_pages
from .extract import extract_frames
from .segment import segment_frames
from .select import select_pages


def parse_timecode(value: str | None) -> float | None:
    """Parse ``ss``, ``mm:ss``, or ``hh:mm:ss`` into seconds. ``None`` stays ``None``."""
    if value is None:
        return None
    parts = value.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid timecode: {value!r}") from exc
    if len(nums) > 3:
        raise argparse.ArgumentTypeError(f"invalid timecode: {value!r}")
    seconds = 0.0
    for n in nums:
        seconds = seconds * 60 + n
    return seconds


def run(
    input_path: str | Path,
    output_path: str | Path,
    *,
    fps: float = 2.0,
    start: float | None = None,
    end: float | None = None,
    method: str = "mad",
    enter_threshold: float = 0.045,
    exit_threshold: float = 0.03,
    min_stable_sec: float = 1.0,
    no_crop: bool = False,
    rows_per_page: int = 3,
    debug_dir: str | Path | None = None,
    verbose: bool = True,
) -> Path:
    """Run the full pipeline and return the written PDF path."""

    def log(msg: str) -> None:
        if verbose:
            print(msg, file=sys.stderr)

    log(f"[1/6] extract: decoding {input_path} at {fps} fps")
    frames = extract_frames(input_path, fps=fps, start=start, end=end)
    log(f"      {len(frames)} frames sampled")
    if not frames:
        raise SystemExit("no frames extracted; check the input and --start/--end window")

    log("[2/6] segment: detecting page transitions")
    result = segment_frames(
        frames,
        method=method,
        enter_threshold=enter_threshold,
        exit_threshold=exit_threshold,
        min_stable_sec=min_stable_sec,
    )
    log(f"      {len(result.segments)} page segments")

    if result.suspected_missed:
        # Not gated by --quiet: a missed flip means a page is silently absent from
        # the output, which is worth flagging even in a quiet run.
        at = ", ".join(f"{t:.1f}s" for t in result.suspected_missed)
        print(
            f"      warning: {len(result.suspected_missed)} possible missed page "
            f"flip(s) at {at} -- the signal spiked just below --threshold "
            f"({enter_threshold}); a page may be missing. Lower --threshold or raise "
            f"--fps and re-run, or use --debug to inspect debug/signal.png.",
            file=sys.stderr,
        )

    if debug_dir is not None:
        from .debug import dump_debug

        out = dump_debug(result, frames, debug_dir, enter_threshold, exit_threshold)
        log(f"      debug dump written to {out}")

    log("[3/6] select: choosing the sharpest frame per page")
    pages = select_pages(result.segments, frames)

    log("[4/6] dedup: dropping repeated pages")
    pages = dedup_pages(pages)
    log(f"      {len(pages)} distinct pages")

    if no_crop:
        log("[5/6] crop: skipped (--no-crop)")
    else:
        log("[5/6] crop: isolating the notation region")
        pages = crop_pages(pages)

    log(f"[6/6] assemble: writing {output_path} ({rows_per_page} strips/page, portrait)")
    return assemble_pdf(pages, output_path, rows_per_page=rows_per_page)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-to-score",
        description="Turn an MP4 of page-flip sheet music into a clean, printable PDF score.",
    )
    parser.add_argument("input", type=Path, help="local MP4 file of sheet music")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output PDF path")
    parser.add_argument(
        "--start",
        type=parse_timecode,
        default=None,
        help="skip everything before this point (ss, mm:ss, or hh:mm:ss)",
    )
    parser.add_argument(
        "--end",
        type=parse_timecode,
        default=None,
        help="skip everything after this point (ss, mm:ss, or hh:mm:ss)",
    )
    parser.add_argument("--fps", type=float, default=2.0, help="frame sampling rate (default: 2.0)")
    parser.add_argument(
        "--method",
        choices=["mad", "hist"],
        default="mad",
        help="frame-similarity signal (default: mad)",
    )
    parser.add_argument(
        "--threshold",
        dest="enter_threshold",
        type=float,
        default=0.045,
        help="dissimilarity at/above which a page transition begins (default: 0.045)",
    )
    parser.add_argument(
        "--exit-threshold",
        type=float,
        default=0.03,
        help="dissimilarity at/below which a transition ends (hysteresis; default: 0.03)",
    )
    parser.add_argument(
        "--min-stable-sec",
        type=float,
        default=1.0,
        help="minimum seconds a page must hold to count as a page (default: 1.0)",
    )
    parser.add_argument("--no-crop", action="store_true", help="skip the crop stage")
    parser.add_argument(
        "--rows-per-page",
        type=int,
        default=3,
        help="number of captured strips stacked per portrait PDF page (default: 3)",
    )
    parser.add_argument(
        "--debug",
        nargs="?",
        const="debug",
        default=None,
        metavar="DIR",
        help="dump similarity signal and per-segment frames (default dir: ./debug)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress progress logs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    try:
        out = run(
            args.input,
            args.output,
            fps=args.fps,
            start=args.start,
            end=args.end,
            method=args.method,
            enter_threshold=args.enter_threshold,
            exit_threshold=args.exit_threshold,
            min_stable_sec=args.min_stable_sec,
            no_crop=args.no_crop,
            rows_per_page=args.rows_per_page,
            debug_dir=args.debug,
            verbose=not args.quiet,
        )
    except (SystemExit, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
