---
name: test-samples
description: Run the video-to-score pipeline against the user's own sample videos in samples/ and check that the number of distinct pages detected matches the expected count in samples/manifest.toml. Use when asked to test samples, check sample accuracy, or add/tune a sample.
---

# test-samples

Regression harness for real videos. `samples/` is gitignored (bring-your-own MP4);
each video's expected page count and per-sample flags live in `samples/manifest.toml`.

## Run it

```bash
uv run python scripts/test_samples.py
```

Output is one row per sample: `PASS`/`FAIL`, `expected` vs `got` distinct pages, the
raw `segs` (segment) count, and notes (dedup drops, suspected missed flips). Exit
status is non-zero if any sample fails.

Each sample's assembled score is also written to `samples/<name>.pdf` (same name as
the MP4, `.pdf` extension) so you can eyeball the actual output. PDF assembly runs
off the page-count path, so a PDF-write failure is reported as a note rather than
failing the sample.

## Add a sample

1. Copy the MP4 into `samples/`.
2. Add a block to `samples/manifest.toml`. `file` and `expected_pages` are required;
   the rest are optional CLI-matching flags (`start`, `end`, `fps`, `method`,
   `threshold`, `exit_threshold`, `min_stable_sec`):

   ```toml
   [[sample]]
   file = "my_video.mp4"
   expected_pages = 14
   start = "0:08"
   ```

3. Determine `expected_pages` by counting the real pages in the video (that's the
   ground truth), not by copying whatever the tool currently outputs.

## Reading a FAIL

`dedup` is the last stage that changes the page count (`crop`/`assemble` are 1:1 /
layout-only), so the `got` count reflects the pipeline through `dedup`. Use the
extra columns to tune:

- **got < expected** — pages are being merged. A missed flip (see the "suspected
  missed flip" note) means two pages collapsed into one; lower `threshold` or raise
  `fps` for that sample. If `segs` already equals `expected` but `got` is lower,
  `dedup` over-collapsed distinct pages.
- **got > expected** — pages are over-split; raise `threshold`/`min_stable_sec`.
- Tune by editing that sample's flags in the manifest, not the source defaults.
