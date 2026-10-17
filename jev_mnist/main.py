"""Command-line entry point: run Jev on MNIST test images and report accuracy."""

import argparse
import asyncio
import csv
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from jev_mnist.classify import DEFAULT_UPSCALE, IMAGES_OPT_IN_HEADERS, MAX_IMAGE_SIDE, MIN_IMAGE_SIDE, Prediction, classify, to_png
from jev_mnist.mnist import load_test_set

DEFAULT_MODEL = "jev-preview"  # Points to jev-1.14.0.

Classifier = Callable[[bytes], Awaitable[Prediction]]


@dataclass
class Result:
    index: int
    label: int
    prediction: Prediction | None
    error: str | None = None
    latency: float | None = None  # Seconds the classifier call took.

    @property
    def correct(self) -> bool:
        return self.prediction is not None and self.prediction.digit == self.label


async def evaluate(
    images: np.ndarray,
    labels: np.ndarray,
    indices: Sequence[int],
    classifier: Classifier,
    *,
    concurrency: int,
    upscale: int,
    progress: bool = True,
) -> list[Result]:
    semaphore = asyncio.Semaphore(concurrency)
    done = 0

    async def run(index: int) -> Result:
        nonlocal done
        label = int(labels[index])
        async with semaphore:
            png = to_png(images[index], upscale)
            start = time.perf_counter()
            try:
                result = Result(index, label, await classifier(png))
            except Exception as error:  # noqa: BLE001 - Count API failures instead of aborting the run.
                result = Result(index, label, None, f"{type(error).__name__}: {error}")
            result.latency = time.perf_counter() - start
        done += 1
        if progress:
            correct = "o" if result.correct else ("!" if result.error else "x")
            print(f"\r[{done}/{len(indices)}] #{index} label={label} {correct}", end="", file=sys.stderr, flush=True)
        return result

    results = await asyncio.gather(*(run(i) for i in indices))
    if progress:
        print(file=sys.stderr)
    return list(results)


def confusion_matrix(results: Sequence[Result]) -> np.ndarray:
    matrix = np.zeros((10, 10), dtype=int)
    for r in results:
        if r.prediction is not None:
            matrix[r.label, r.prediction.digit] += 1
    return matrix


def report(results: Sequence[Result], elapsed: float | None = None) -> str:
    answered = [r for r in results if r.prediction is not None]
    correct = sum(r.correct for r in results)
    lines = []
    models = sorted({r.prediction.model for r in answered})
    if models:
        lines.append(f"Model: {', '.join(models)}")
    lines.append(f"Samples: {len(results)}  answered: {len(answered)}  errors: {len(results) - len(answered)}")
    if answered:
        lines.append(f"Accuracy (answered): {correct / len(answered):.2%} ({correct}/{len(answered)})")
    if results:
        lines.append(f"Accuracy (all):      {correct / len(results):.2%} ({correct}/{len(results)})")
    lines.append(f"Input tokens: {sum(r.prediction.input_tokens for r in answered)}")
    latencies = [r.latency for r in answered if r.latency is not None]
    if latencies:
        lines.append(
            f"Latency per image (answered): mean {np.mean(latencies):.2f}s  median {np.median(latencies):.2f}s  "
            f"p95 {np.percentile(latencies, 95):.2f}s  min {min(latencies):.2f}s  max {max(latencies):.2f}s"
        )
    if elapsed is not None and elapsed > 0:
        lines.append(f"Total time: {elapsed:.1f}s  throughput: {len(results) / elapsed:.2f} images/s")

    matrix = confusion_matrix(results)
    lines.append("")
    lines.append("Per-digit accuracy:")
    for digit in range(10):
        total = matrix[digit].sum()
        if total:
            lines.append(f"  {digit}: {matrix[digit, digit] / total:.2%} ({matrix[digit, digit]}/{total})")
    lines.append("")
    lines.append("Confusion matrix (rows = label, columns = prediction):")
    lines.append("     " + "".join(f"{d:>5}" for d in range(10)))
    for digit in range(10):
        lines.append(f"{digit:>5}" + "".join(f"{n:>5}" for n in matrix[digit]))

    errors = [r for r in results if r.error]
    if errors:
        lines.append("")
        lines.append(f"First error: #{errors[0].index}: {errors[0].error}")
    return "\n".join(lines)


def write_csv(results: Sequence[Result], path: Path) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "label", "prediction", "confidence", "correct", "error", "latency_ms"])
        for r in results:
            p = r.prediction
            latency = f"{r.latency * 1000:.0f}" if r.latency is not None else ""
            writer.writerow([r.index, r.label, p.digit if p else "", f"{p.confidence:.4f}" if p else "", r.correct, r.error or "", latency])


def select_indices(total: int, limit: int, offset: int, seed: int | None) -> list[int]:
    if seed is not None:
        return sorted(np.random.default_rng(seed).choice(total, size=min(limit, total), replace=False).tolist())
    return list(range(offset, min(offset + limit, total)))


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recognize MNIST test digits with Jev 1.14 and report accuracy.")
    parser.add_argument("--limit", type=int, default=100, help="number of test images to evaluate (default: 100)")
    parser.add_argument("--offset", type=int, default=0, help="index of the first test image (default: 0)")
    parser.add_argument("--seed", type=int, help="sample --limit images at random with this seed instead of using --offset")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--concurrency", type=int, default=8, help="parallel requests (default: 8)")
    parser.add_argument(
        "--upscale", type=int, default=DEFAULT_UPSCALE, help=f"enlarge the 28x28 images by this factor (default: {DEFAULT_UPSCALE})"
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="per-request timeout in seconds (default: 60)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="MNIST cache directory (default: data)")
    parser.add_argument("--output", type=Path, help="write per-image results to this CSV file")
    parser.add_argument("--dry-run", action="store_true", help="save the selected images as PNG without calling the API")
    args = parser.parse_args(argv)
    side = 28 * args.upscale
    if not args.dry_run and not MIN_IMAGE_SIDE <= side <= MAX_IMAGE_SIDE:
        parser.error(f"--upscale {args.upscale} makes {side}x{side} images; the API accepts {MIN_IMAGE_SIDE}-{MAX_IMAGE_SIDE} pixels per side")
    return args


async def run(args: argparse.Namespace) -> int:
    images, labels = load_test_set(args.data_dir)
    indices = select_indices(len(images), args.limit, args.offset, args.seed)

    if args.dry_run:
        out = args.data_dir / "samples"
        out.mkdir(parents=True, exist_ok=True)
        for i in indices:
            (out / f"{i:05d}_label{labels[i]}.png").write_bytes(to_png(images[i], args.upscale))
        print(f"Saved {len(indices)} images to {out}")
        return 0

    from typesafe_sdk import AsyncTypeSafeClient

    async with AsyncTypeSafeClient(timeout=args.timeout, headers=IMAGES_OPT_IN_HEADERS) as client:
        start = time.perf_counter()
        results = await evaluate(
            images,
            labels,
            indices,
            lambda png: classify(client, png, args.model),
            concurrency=args.concurrency,
            upscale=args.upscale,
        )
        elapsed = time.perf_counter() - start
    print(report(results, elapsed))
    if args.output:
        write_csv(results, args.output)
        print(f"\nWrote {args.output}")
    return 0 if any(r.prediction for r in results) else 1


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()  # Read TYPESAFE_API_KEY etc. from .env; real environment variables take precedence.
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
