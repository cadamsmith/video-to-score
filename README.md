# 🎹 video-to-score

Turn an MP4 video of **page-flip** sheet music into a clean, printable PDF score.

A lot of great piano sheet music only exists as videos where the notation pages through on
screen. Playing from it means constantly pausing and scrubbing. `video-to-score` captures each
distinct page once and assembles them into a normal PDF you can print or read straight through.

> **Scope:** the MVP handles **page-flip** videos (notation advances one static page at a time).
> Continuous-scroll videos, URL downloading, and full notation recognition (OMR) are out of scope.

## install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## usage

```bash
uv run video-to-score input.mp4 -o output.pdf [--start mm:ss] [--end mm:ss]
```

`--start`/`--end` skip non-notation intros/outros (title cards, talking-head intros, bumpers) —
a quick manual step per video. Timecodes accept `ss`, `mm:ss`, or `hh:mm:ss`.

### options

| flag | default | description |
| --- | --- | --- |
| `-o, --output` | *(required)* | Output PDF path. |
| `--start` / `--end` | none | Trim the notation window. |
| `--fps` | `2.0` | Frame sampling rate. Page flips are slow, so a low rate is plenty. |
| `--method` | `mad` | Frame-similarity signal: `mad` (mean abs difference) or `hist`. |
| `--threshold` | `0.045` | Dissimilarity at/above which a page transition begins. |
| `--exit-threshold` | `0.03` | Dissimilarity at/below which a transition ends (hysteresis). |
| `--min-stable-sec` | `1.0` | Minimum seconds a page must hold to count as a page. |
| `--rows-per-page` | `3` | Number of captured strips stacked per portrait PDF page. |
| `--no-crop` | off | Skip cropping to the notation region. |
| `--debug [DIR]` | off | Dump the similarity signal (`signal.csv`, `signal.png`) and one frame per detected segment. |

### tuning with `--debug`

`segment` is the heart of the tool. If pages are missed or split, run with `--debug` and open
`debug/signal.png`: stable pages sit near zero, page flips are spikes, and the enter/exit thresholds
are drawn in. Nudge `--threshold`/`--exit-threshold` so every real flip clears the line and nothing
else does. The `debug/segments/` frames let you eyeball the detected page boundaries.

A missed flip is silent — the two pages on either side collapse into one and only one survives, so a
page just disappears from the PDF. To catch this, `segment` watches for a spike that lands *just
below* `--threshold` inside an otherwise stable page and prints a `warning: possible missed page
flip(s) at ...` (even under `--quiet`). If you see it, lower `--threshold` or raise `--fps` and re-run.

## how it works

A linear pipeline; each stage does one job and hands a simple data structure to the next:

```
MP4 -> [1] extract -> [2] segment -> [3] select -> [4] dedup -> [5] crop -> [6] assemble -> PDF
```

1. **extract** — decode and sample frames (~2 fps) within the `--start`/`--end` window.
2. **segment** — compute a per-frame similarity signal; classify stable vs. transition with
   threshold + hysteresis; group stable runs into page segments.
3. **select** — pick the sharpest frame per segment (variance of the Laplacian).
4. **dedup** — drop repeated pages (perceptual hash + SSIM confirmation).
5. **crop** — isolate the bright notation rectangle.
6. **assemble** — order by timestamp and stack `--rows-per-page` strips onto each portrait PDF page.

## development

```bash
uv run pytest              # unit tests (synthetic frames per stage)
uv run ruff check          # lint
uv run pre-commit install  # lint + format on every commit (one-time setup)
```
