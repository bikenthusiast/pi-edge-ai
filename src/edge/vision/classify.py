"""CLI: classify a single image.

Run on either machine:
    python -m edge.vision.classify photo.jpg --fit crop --runs 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from edge.vision.model import (
    DEFAULT_LABELS,
    DEFAULT_MODEL,
    ModelError,
    TFLiteModel,
    load_labels,
    preprocess,
)


def classify(model: TFLiteModel, labels: list[str], image: Image.Image,
             fit: str, runs: int) -> tuple[np.ndarray, np.ndarray]:
    """Run inference `runs` times and return (scores, timings in ms)."""
    prepared = preprocess(image, model.width, model.height, fit=fit)
    tensor = model.make_tensor(prepared)

    if runs > 1:
        model.invoke(tensor)  # warm-up: the first pass is always slower

    timings = np.array([model.invoke(tensor) for _ in range(runs)])
    scores = model.output(0)

    if scores.ndim != 1:
        raise ModelError(
            f"Expected a 1D output vector, got shape {list(scores.shape)}. "
            "This looks like a detection model, not a classifier."
        )
    if len(labels) != scores.shape[0]:
        raise ModelError(
            f"Label/output mismatch: {len(labels)} labels but "
            f"{scores.shape[0]} classes. ImageNet models usually need 1001 "
            "labels — index 0 is the 'background' pseudo-class."
        )
    return scores, timings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify an image with a TFLite/LiteRT model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", type=Path, help="path to the image")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--fit", choices=("crop", "pad", "stretch"), default="crop")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--min-score", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.top < 1 or args.threads < 1 or args.runs < 1:
        print("--top, --threads and --runs must be >= 1", file=sys.stderr)
        return 2

    try:
        labels = load_labels(args.labels)

        if not args.image.is_file():
            raise ModelError(f"Image not found: {args.image}")
        try:
            image = Image.open(args.image)
            image.load()
        except UnidentifiedImageError as exc:
            raise ModelError(f"Not a readable image file: {args.image}") from exc

        model = TFLiteModel(args.model, num_threads=args.threads)
        scores, timings = classify(model, labels, image, args.fit, args.runs)
    except ModelError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130

    print(f"Model    : {Path(args.model).name}")
    print(f"Input    : {model.width}x{model.height}, {model.dtype.__name__}, "
          f"fit={args.fit}")
    if args.runs > 1:
        print(f"Inference: {timings.mean():.1f} ms avg over {args.runs} runs "
              f"(min {timings.min():.1f} / max {timings.max():.1f})")
    else:
        print(f"Inference: {timings[0]:.1f} ms (cold, no warm-up)")
    print()

    shown = 0
    for rank, idx in enumerate(np.argsort(scores)[::-1][: args.top], start=1):
        if scores[idx] < args.min_score:
            break
        print(f"{rank}. {labels[idx][:45]:<45s} {scores[idx] * 100:6.2f} %")
        shown += 1

    if shown == 0:
        print(f"No class above --min-score {args.min_score:.2f} "
              f"(best was {scores.max() * 100:.2f} %)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
