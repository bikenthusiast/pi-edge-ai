"""Compare two or more models on latency and predictions.

    python -m edge.vision.benchmark models/*.tflite --images samples

Two things are measured, and they are not the same thing:

* **Latency** — objective. Warm-up first, then percentiles, because the mean
  hides the stutter that thermal throttling introduces.
* **Agreement** — how often a model's top-1 matches the reference model (the
  first one listed). This is NOT accuracy: it says how much behaviour changes
  when you swap models, which is the question you actually face when deciding
  whether a faster model is good enough. Real accuracy needs a labelled set.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from edge.vision.model import (
    DEFAULT_LABELS,
    ModelError,
    TFLiteModel,
    load_labels,
    preprocess,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.iterdir()
                                if p.suffix.lower() in IMAGE_SUFFIXES))
        elif path.is_file():
            files.append(path)
    if not files:
        raise ModelError(f"No images found in {[str(p) for p in paths]}")
    return files


def load_images(files: list[Path]) -> list[tuple[str, Image.Image]]:
    images = []
    for path in files:
        try:
            image = Image.open(path)
            image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise ModelError(f"Cannot read {path}: {exc}") from exc
        images.append((path.name, image.convert("RGB")))
    return images


def measure(model: TFLiteModel, images: list[tuple[str, Image.Image]],
            labels: list[str], runs: int, fit: str) -> dict:
    """Return timings and top-1 predictions for one model."""
    timings: list[float] = []
    predictions: dict[str, tuple[str, float]] = {}

    # Warm-up on the first image: XNNPACK repacks weights on the first call,
    # so including it would inflate every result by tens of milliseconds.
    warm = model.make_tensor(
        preprocess(images[0][1], model.width, model.height, fit=fit))
    for _ in range(3):
        model.invoke(warm)

    for name, image in images:
        tensor = model.make_tensor(
            preprocess(image, model.width, model.height, fit=fit))
        for _ in range(runs):
            timings.append(model.invoke(tensor))
        scores = model.output(0)
        best = int(np.argmax(scores))
        predictions[name] = (labels[best], float(scores[best]))

    ordered = sorted(timings)
    return {
        "mean": statistics.fmean(timings),
        "median": statistics.median(timings),
        "p95": ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)],
        "min": ordered[0],
        "max": ordered[-1],
        "predictions": predictions,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare models on latency and top-1 predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("models", nargs="+", type=Path,
                        help="model files to compare")
    parser.add_argument("--reference",
                        help="model filename to compare the others against; "
                             "defaults to the slowest, which is normally the "
                             "most accurate one")
    parser.add_argument("--images", nargs="+", type=Path,
                        default=[Path("samples")],
                        help="image files or directories")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--runs", type=int, default=20,
                        help="inference passes per image")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--fit", choices=("crop", "pad", "stretch"),
                        default="crop")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        labels = load_labels(args.labels)
        files = collect_images(args.images)
        images = load_images(files)
        results = {}
        for path in args.models:
            model = TFLiteModel(path, num_threads=args.threads)
            results[path.name] = {
                "shape": f"{model.width}x{model.height}",
                "size_kb": path.stat().st_size / 1024,
                **measure(model, images, labels, args.runs, args.fit),
            }
    except ModelError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Shell globs expand alphabetically, so "first argument" is a bad default
    # for the reference — mobilenet_v1_* would silently become the baseline and
    # every ratio would read inverted. Pick the slowest model instead, which is
    # the accuracy baseline in practice.
    if args.reference and args.reference not in results:
        print(f"Error: --reference {args.reference} is not among the models "
              f"given: {', '.join(results)}", file=sys.stderr)
        return 1
    reference = args.reference or max(results, key=lambda n: results[n]["median"])
    ordered = dict(sorted(results.items(),
                          key=lambda kv: (kv[0] != reference, kv[0])))
    results = ordered

    total = len(images) * args.runs
    print(f"{len(images)} images x {args.runs} runs = {total} inferences per "
          f"model, {args.threads} threads, fit={args.fit}\n")

    print(f"{'model':<36} {'input':>9} {'size':>8} {'median':>8} {'p95':>8} "
          f"{'FPS':>6}")
    print("-" * 80)
    for name, row in results.items():
        print(f"{name[:36]:<36} {row['shape']:>9} "
              f"{row['size_kb']:>7.0f}K {row['median']:>7.1f}m "
              f"{row['p95']:>7.1f}m {1000 / row['median']:>6.1f}")

    if len(results) > 1:
        base = results[reference]
        print(f"\nSpeed relative to {reference} (reference):")
        for name, row in results.items():
            factor = base["median"] / row["median"]
            print(f"  {name[:44]:<44} {factor:>5.2f}x")

    print(f"\nTop-1 per image (reference: {reference})")
    header = f"{'image':<18}" + "".join(f"{n[:26]:<28}" for n in results)
    print(header)
    print("-" * len(header))
    for name, _ in images:
        row = f"{name[:18]:<18}"
        for model_name in results:
            label, score = results[model_name]["predictions"][name]
            marker = ""
            if model_name != reference:
                ref_label = results[reference]["predictions"][name][0]
                marker = "  " if label == ref_label else " *"
            row += f"{label[:20]:<20}{score * 100:4.0f}%{marker}".ljust(28)
        print(row)
    if len(results) > 1:
        print("  * top-1 differs from the reference")

    if len(results) > 1:
        print("\nAgreement with the reference (not accuracy — see module docstring):")
        for name in list(results)[1:]:
            same = sum(
                results[name]["predictions"][img][0]
                == results[reference]["predictions"][img][0]
                for img, _ in images
            )
            print(f"  {name[:44]:<44} {same}/{len(images)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())