# jev-mnist

A program that shows MNIST handwritten digit images to TypeSafe AI's **Jev 1.14** (which supports image input), asks it which digit from 0 to 9 each one is, and measures its accuracy.

- Each 28x28 grayscale image is converted to PNG (upscaled 10x to 280x280 by default) and passed in `state` as a `typesafe_sdk.Image`.
- The question is a `Choice` (options `"0"` to `"9"`), and Jev returns the digit with the highest probability.
- The default model is `jev-preview` (an alias pointing to Jev 1.14.0). It can be changed with `--model`.

> Image input in Jev 1.14 is positioned by TypeSafe as an "experimental preview" feature.

## Results (Jev 1.14)

The full MNIST test set (10,000 images), default settings (10x upscale, 8 concurrent requests):

```sh
uv run jev-mnist --limit 10000 --output results.csv
```

| Metric | Value |
|---|---|
| Model | `jev-1.14.0` |
| Samples | 10,000 (answered: 10,000, errors: 0) |
| Accuracy | **92.45%** (9245/10000) |
| Input tokens | 5,942,745 |
| Latency per image | mean 0.17s, median 0.15s, p95 0.30s, min 0.13s, max 1.38s |
| Total time | 216.5s (46.18 images/s) |

Per-digit accuracy:

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 91.43% | 98.68% | 90.12% | 96.34% | 90.94% | 93.61% | 85.49% | 98.25% | 85.73% | 92.57% |
| Correct / total | 896/980 | 1120/1135 | 930/1032 | 973/1010 | 893/982 | 835/892 | 819/958 | 1010/1028 | 835/974 | 934/1009 |

Confusion matrix (rows = label, columns = prediction):

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| **0** | **896** | 1 | 2 | 0 | 0 | 0 | 19 | 30 | 1 | 31 |
| **1** | 0 | **1120** | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 |
| **2** | 1 | 3 | **930** | 3 | 1 | 0 | 7 | 56 | 0 | 31 |
| **3** | 0 | 0 | 8 | **973** | 0 | 17 | 0 | 10 | 1 | 1 |
| **4** | 1 | 0 | 0 | 0 | **893** | 0 | 0 | 86 | 0 | 2 |
| **5** | 0 | 1 | 11 | 1 | 0 | **835** | 16 | 25 | 0 | 3 |
| **6** | 0 | 5 | 8 | 0 | 8 | 0 | **819** | 88 | 0 | 30 |
| **7** | 0 | 0 | 7 | 1 | 6 | 0 | 0 | **1010** | 0 | 4 |
| **8** | 0 | 0 | 10 | 2 | 2 | 3 | 19 | 34 | **835** | 69 |
| **9** | 1 | 3 | 0 | 1 | 44 | 1 | 3 | 22 | 0 | **934** |

Observations:

- 1 (98.68%) and 7 (98.25%) are recognized best; 6 (85.49%) and 8 (85.73%) are the weakest.
- Jev is strongly biased toward answering "7": 366 of the 755 errors (48%) are wrong "7"s, most often for 6 (88), 4 (86), and 2 (56).
- "9" is the second most common wrong answer (171 errors, led by 8→9 at 69), and 9 is often mistaken for 4 (44).

## Setup

Use [mise](https://mise.jdx.dev/) to provide Python and [uv](https://docs.astral.sh/uv/), then install the dependencies with uv (versions are in `mise.toml`, and the dependency lock is in `uv.lock`).

```sh
mise install                  # Install Python and uv as specified in mise.toml
uv sync --extra test          # Install into .venv exactly as locked in uv.lock
cp .env.example .env          # Put your TypeSafe AI API key in .env
```

If you put your API key in `TYPESAFE_API_KEY` in `.env`, it is loaded at startup (`.env` is not tracked by Git).
If the `TYPESAFE_API_KEY` environment variable is set, it takes precedence.

## Usage

```sh
# Evaluate on the first 100 images of the test set (default)
uv run jev-mnist

# 500 random images, 16 in parallel, save the results to CSV
uv run jev-mnist --limit 500 --seed 0 --concurrency 16 --output results.csv

# Don't call the API; save the images that would be sent to data/samples/ for inspection
uv run jev-mnist --dry-run --limit 10
```

The MNIST test set (10,000 images) is downloaded to `data/` on the first run.
The output shows the accuracy (answered / total), per-digit accuracy, the confusion matrix, input token counts, the name of the model that actually responded, and classification speed (mean, median, p95, min, and max response time per image, plus total elapsed time and throughput).
In the `--output` CSV, the response time for each image is recorded in the `latency_ms` column (milliseconds).
Images that hit an API error are counted as "errors", and processing does not stop.

| Option | Default | Description |
|---|---|---|
| `--limit` | 100 | Number of images to evaluate |
| `--offset` | 0 | Starting position when evaluating from the beginning |
| `--seed` | none | If set, randomly sample `--limit` images |
| `--model` | `jev-preview` | Model to use |
| `--concurrency` | 8 | Number of concurrent requests |
| `--upscale` | 10 | Image upscale factor (10–56, since the API only accepts 280–1568px per side) |
| `--timeout` | 60 | Per-request timeout (seconds) |
| `--output` | none | CSV file to write per-image results to |
| `--dry-run` | - | Save images only, without calling the API |

## Viewing misclassified images

From a CSV saved with `--output`, collect only the images Jev got wrong into an HTML gallery (does not call the API).

```sh
.venv/bin/jev-mnist --limit 500 --seed 0 --output results.csv
.venv/bin/jev-mnist-errors results.csv                         # All misclassified images → errors.html
.venv/bin/jev-mnist-errors results.csv --label 4 --prediction 9  # Only true 4s answered as 9
.venv/bin/jev-mnist-errors results.csv --min-confidence 0.9      # Only confident mistakes
```

By default, images are sorted by confidence, highest first. Main options: `--label`, `--prediction`, `--min-confidence`, `--max-confidence`,
`--include-errors` (also include images that hit an API error), `--sort index`, `--limit`, `--output` (default `errors.html`).

## Tests

```sh
uv run pytest
```

Offline tests (MNIST parser, PNG conversion, aggregation) that run without an API key or network access.
