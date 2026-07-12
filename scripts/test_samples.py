#!/usr/bin/env python3
"""Run the pipeline against your own sample videos and check page counts.

`samples/` is bring-your-own (gitignored). Drop MP4s in there, describe each one
in `samples/manifest.toml`, then:

    uv run python scripts/test_samples.py

For every sample it runs the real pipeline in-process (extract -> segment ->
select -> dedup -> crop -> assemble), writes the assembled score to
`samples/<name>.pdf` next to the MP4, then compares the number of distinct pages
against the manifest's `expected_pages`. It also reports the raw segment count and
any suspected missed page flips, so a FAIL doubles as a tuning signal:

  * distinct == segments : dedup found nothing to drop
  * distinct <  segments : dedup collapsed repeats (revisited pages, loops)
  * suspected_missed set  : a flip spiked just below --threshold; a page may be
                            silently missing (lower --threshold or raise --fps)

Exit status is non-zero if any sample fails, so this works in CI too.
"""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

# Run against the source tree without needing an editable install.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from video_to_score.assemble import assemble_pdf  # noqa: E402
from video_to_score.cli import parse_timecode  # noqa: E402
from video_to_score.crop import crop_pages  # noqa: E402
from video_to_score.dedup import dedup_pages  # noqa: E402
from video_to_score.extract import extract_frames  # noqa: E402
from video_to_score.segment import segment_frames  # noqa: E402
from video_to_score.select import select_pages  # noqa: E402

SAMPLES_DIR = ROOT / "samples"
MANIFEST = SAMPLES_DIR / "manifest.toml"

# Manifest keys that are pipeline flags, mapped to segment_frames / extract args.
# Kept explicit so a typo'd key in the manifest is caught rather than ignored.
_PIPELINE_KEYS = {"fps", "method", "threshold", "exit_threshold", "min_stable_sec"}
_TIMECODE_KEYS = {"start", "end"}


@dataclass
class Result:
    name: str
    expected: int
    distinct: int
    segments: int
    suspected_missed: list[float]
    error: str | None = None
    pdf_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.distinct == self.expected


def run_sample(spec: dict) -> Result:
    """Run the pipeline for one manifest entry and collect its counts."""
    name = spec.get("file")
    if not name:
        return Result("<missing file>", 0, 0, 0, [], error="entry is missing `file`")
    if "expected_pages" not in spec:
        return Result(name, 0, 0, 0, [], error="entry is missing `expected_pages`")

    unknown = set(spec) - {"file", "expected_pages"} - _PIPELINE_KEYS - _TIMECODE_KEYS
    if unknown:
        return Result(
            name,
            spec["expected_pages"],
            0,
            0,
            [],
            error=f"unknown manifest key(s): {', '.join(sorted(unknown))}",
        )

    path = SAMPLES_DIR / name
    if not path.exists():
        return Result(name, spec["expected_pages"], 0, 0, [], error=f"not found: {path}")

    try:
        frames = extract_frames(
            path,
            fps=spec.get("fps", 2.0),
            start=parse_timecode(spec.get("start")),
            end=parse_timecode(spec.get("end")),
        )
        if not frames:
            raise ValueError("no frames extracted (check start/end window)")

        result = segment_frames(
            frames,
            method=spec.get("method", "mad"),
            enter_threshold=spec.get("threshold", 0.045),
            exit_threshold=spec.get("exit_threshold", 0.03),
            min_stable_sec=spec.get("min_stable_sec", 1.0),
        )
        pages = dedup_pages(select_pages(result.segments, frames))
    except Exception as exc:  # noqa: BLE001 - report any failure per-sample, keep going
        return Result(name, spec["expected_pages"], 0, 0, [], error=str(exc))

    # Write the assembled PDF next to the MP4 (same name, .pdf). Kept off the
    # page-count path above so a PDF-write hiccup can't flip a correct-count
    # sample to FAIL -- it's reported as a note instead.
    pdf_error = None
    try:
        assemble_pdf(crop_pages(pages), path.with_suffix(".pdf"))
    except Exception as exc:  # noqa: BLE001 - artifact failure shouldn't sink the count check
        pdf_error = f"pdf write failed: {exc}"

    return Result(
        name=name,
        expected=spec["expected_pages"],
        distinct=len(pages),
        segments=len(result.segments),
        suspected_missed=list(result.suspected_missed),
        pdf_error=pdf_error,
    )


def main() -> int:
    if not MANIFEST.exists():
        print(f"no manifest at {MANIFEST}", file=sys.stderr)
        print(
            "Create it and add a [[sample]] block per video. See scripts/test_samples.py.",
            file=sys.stderr,
        )
        return 2

    samples = tomllib.loads(MANIFEST.read_text()).get("sample", [])
    if not samples:
        print(f"{MANIFEST} has no [[sample]] entries.", file=sys.stderr)
        return 2

    results = [run_sample(s) for s in samples]

    width = max(len(r.name) for r in results)
    print(f"{'':4} {'sample':<{width}}  {'expected':>8} {'got':>4} {'segs':>4}  notes")
    print("-" * (width + 32))
    for r in results:
        tag = "PASS" if r.ok else "FAIL"
        if r.error:
            print(f"{tag:<4} {r.name:<{width}}  {r.expected:>8} {'--':>4} {'--':>4}  {r.error}")
            continue
        notes = []
        if r.distinct < r.segments:
            notes.append(f"dedup dropped {r.segments - r.distinct}")
        if r.suspected_missed:
            at = ", ".join(f"{t:.1f}s" for t in r.suspected_missed)
            notes.append(f"suspected missed flip(s) at {at}")
        if r.pdf_error:
            notes.append(r.pdf_error)
        print(
            f"{tag:<4} {r.name:<{width}}  {r.expected:>8} {r.distinct:>4} "
            f"{r.segments:>4}  {'; '.join(notes)}"
        )

    passed = sum(r.ok for r in results)
    print("-" * (width + 32))
    print(f"{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
