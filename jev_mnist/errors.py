"""Build an HTML gallery of the images Jev got wrong, from the CSV written by `jev-mnist --output`."""

import argparse
import base64
import csv
import html
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from jev_mnist.classify import to_png
from jev_mnist.mnist import load_test_set


@dataclass(frozen=True)
class Row:
    index: int
    label: int
    prediction: int | None
    confidence: float | None
    error: str


@dataclass(frozen=True)
class Filters:
    label: int | None = None
    prediction: int | None = None
    min_confidence: float | None = None
    max_confidence: float | None = None
    include_errors: bool = False
    sort: str = "confidence"
    limit: int | None = None

    def describe(self) -> str:
        parts = []
        if self.label is not None:
            parts.append(f"label={self.label}")
        if self.prediction is not None:
            parts.append(f"prediction={self.prediction}")
        if self.min_confidence is not None:
            parts.append(f"confidence>={self.min_confidence}")
        if self.max_confidence is not None:
            parts.append(f"confidence<={self.max_confidence}")
        if self.include_errors:
            parts.append("including API errors")
        return ", ".join(parts) or "all misclassified"


def read_rows(path: Path) -> list[Row]:
    with path.open(newline="") as file:
        return [
            Row(
                index=int(r["index"]),
                label=int(r["label"]),
                prediction=int(r["prediction"]) if r["prediction"] else None,
                confidence=float(r["confidence"]) if r["confidence"] else None,
                error=r.get("error", ""),
            )
            for r in csv.DictReader(file)
        ]


def select_rows(rows: Iterable[Row], filters: Filters) -> list[Row]:
    selected = []
    for r in rows:
        if r.prediction is None:
            if not filters.include_errors:
                continue
        elif r.prediction == r.label:
            continue
        if filters.label is not None and r.label != filters.label:
            continue
        if filters.prediction is not None and r.prediction != filters.prediction:
            continue
        # API-error rows have no confidence, so confidence filters exclude them.
        if filters.min_confidence is not None and (r.confidence is None or r.confidence < filters.min_confidence):
            continue
        if filters.max_confidence is not None and (r.confidence is None or r.confidence > filters.max_confidence):
            continue
        selected.append(r)
    if filters.sort == "confidence":
        selected.sort(key=lambda r: (-(r.confidence if r.confidence is not None else -1.0), r.index))
    else:
        selected.sort(key=lambda r: r.index)
    return selected[: filters.limit] if filters.limit is not None else selected


STYLE = """
:root { color-scheme: light dark; --bg: #fafafa; --fg: #1a1a1a; --card: #fff; --muted: #666; --bad: #c62828; --border: #ddd; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #161616; --fg: #eee; --card: #222; --muted: #aaa; --bad: #ef6c6c; --border: #333; }
}
body { margin: 0; padding: 16px; background: var(--bg); color: var(--fg); font-family: system-ui, sans-serif; }
h1 { font-size: 1.3rem; margin: 0 0 4px; }
p.meta { color: var(--muted); margin: 0 0 16px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 12px; }
figure { margin: 0; padding: 8px; background: var(--card); border: 1px solid var(--border); border-radius: 8px; text-align: center; }
img { width: 112px; height: 112px; image-rendering: pixelated; }
figcaption { font-size: 0.85rem; line-height: 1.4; }
.pred { color: var(--bad); font-weight: 600; }
.muted { color: var(--muted); }
"""


def render_html(items: Sequence[tuple[Row, bytes]], title: str, description: str) -> str:
    cards = []
    for row, png in items:
        src = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        if row.prediction is None:
            result = '<span class="pred">API error</span>'
            detail = f'<span class="muted" title="{html.escape(row.error)}">no answer</span>'
        else:
            result = f'{row.label} → <span class="pred">{row.prediction}</span>'
            detail = f'<span class="muted">confidence {row.confidence:.2f}</span>' if row.confidence is not None else ""
        cards.append(
            f'<figure><img src="{src}" alt="MNIST #{row.index}, label {row.label}">'
            f'<figcaption>#{row.index}<br>{result}<br>{detail}</figcaption></figure>'
        )
    return (
        "<!doctype html>\n<html lang=\"ja\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f'<p class="meta">{html.escape(description)} — {len(items)} images (label → prediction)</p>'
        f'<div class="grid">{"".join(cards)}</div></body></html>\n'
    )


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show the MNIST images Jev misclassified, from a jev-mnist --output CSV, as an HTML gallery.")
    parser.add_argument("csv", type=Path, help="results CSV written by jev-mnist --output")
    parser.add_argument("--label", type=int, help="only images whose correct digit is this")
    parser.add_argument("--prediction", type=int, help="only images Jev answered as this digit")
    parser.add_argument("--min-confidence", type=float, help="only answers with at least this confidence")
    parser.add_argument("--max-confidence", type=float, help="only answers with at most this confidence")
    parser.add_argument("--include-errors", action="store_true", help="also show images whose API request failed")
    parser.add_argument("--sort", choices=["confidence", "index"], default="confidence", help="order (default: confidence, highest first)")
    parser.add_argument("--limit", type=int, help="show at most this many images")
    parser.add_argument("--output", type=Path, default=Path("errors.html"), help="HTML file to write (default: errors.html)")
    parser.add_argument("--upscale", type=int, default=4, help="enlarge the 28x28 images by this factor (default: 4)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="MNIST cache directory (default: data)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    filters = Filters(
        label=args.label,
        prediction=args.prediction,
        min_confidence=args.min_confidence,
        max_confidence=args.max_confidence,
        include_errors=args.include_errors,
        sort=args.sort,
        limit=args.limit,
    )
    rows = select_rows(read_rows(args.csv), filters)
    images, _ = load_test_set(args.data_dir)
    items = [(r, to_png(images[r.index], args.upscale)) for r in rows]
    args.output.write_text(render_html(items, "Jev MNIST misclassifications", filters.describe()), encoding="utf-8")
    print(f"Wrote {len(items)} images to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
