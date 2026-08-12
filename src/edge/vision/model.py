"""Model loading, preprocessing and dequantisation.

Device-independent: no camera, no GPIO, no argparse. That is what makes this
module testable on the Mac and reusable for Project 5 (object detection).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# LiteRT is imported lazily inside TFLiteModel. Reason: macOS x86_64 has no
# wheels past ai-edge-litert 1.0.1, so on an Intel Mac this module must still
# import — preprocess() and load_labels() are pure Python and stay testable.

# Model weights live outside the package: same relative path on Mac and Pi.
# EDGE_MODEL_DIR overrides it (useful in tests and on the Pi).
# repo root = three levels up from src/edge/vision/model.py
REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = Path(os.environ.get("EDGE_MODEL_DIR", REPO_ROOT / "models"))
DEFAULT_MODEL = MODEL_DIR / "mobilenet_v2_1.0_224_quant.tflite"
DEFAULT_LABELS = MODEL_DIR / "imagenet_labels.txt"


class ModelError(RuntimeError):
    """Raised for problems the user can act on (missing files, shape mismatch)."""


# --------------------------------------------------------------------------- #
# Reusable helpers
# --------------------------------------------------------------------------- #

def load_labels(path: Path | str) -> list[str]:
    """Read a plain-text label file, one class per line.

    The list index IS the class id, so blank lines must be dropped — a stray
    empty line in the middle of the file would shift every label after it.
    """
    path = Path(path)
    if not path.is_file():
        raise ModelError(f"Label file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        labels = [line.strip() for line in handle if line.strip()]

    if not labels:
        raise ModelError(f"Label file is empty: {path}")
    return labels


def preprocess(
    image: Image.Image,
    width: int,
    height: int,
    fit: str = "crop",
    pad_value: int = 114,
) -> Image.Image:
    """Resize an image to the model input size using one of three strategies.

    crop    Scale the short edge, then centre-crop. Matches how classification
            models such as MobileNet were trained. Loses the image borders.
    pad     Letterbox: scale the long edge, pad the rest. Keeps the whole frame
            and the aspect ratio — the correct choice for object detectors,
            because bounding boxes stay geometrically valid.
    stretch Plain resize. Fastest, but distorts the aspect ratio.
    """
    image = image.convert("RGB")

    if fit == "stretch":
        return image.resize((width, height), Image.BILINEAR)

    if fit == "crop":
        scale = max(width / image.width, height / image.height)
        new_size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(new_size, Image.BILINEAR)
        left = (image.width - width) // 2
        top = (image.height - height) // 2
        return image.crop((left, top, left + width, top + height))

    if fit == "pad":
        scale = min(width / image.width, height / image.height)
        new_size = (round(image.width * scale), round(image.height * scale))
        resized = image.resize(new_size, Image.BILINEAR)
        canvas = Image.new("RGB", (width, height), (pad_value,) * 3)
        canvas.paste(resized, ((width - new_size[0]) // 2,
                               (height - new_size[1]) // 2))
        return canvas

    raise ValueError(f"Unknown fit mode: {fit!r}")


class TFLiteModel:
    """Thin wrapper around the LiteRT interpreter.

    Handles the two things every TFLite script needs and most get wrong:
    building the input tensor with the dtype the model actually wants, and
    dequantising the output using the model's own scale and zero point.
    """

    def __init__(self, model_path: Path | str, num_threads: int = 4):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise ModelError(f"Model file not found: {model_path}")

        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as exc:  # pragma: no cover - platform dependent
            raise ModelError(
                "ai-edge-litert is not installed for this interpreter "
                f"({sys.executable}). On macOS it requires arm64; there are no "
                "x86_64 wheels past 1.0.1. Run inference on the Pi instead."
            ) from exc

        try:
            self.interpreter = Interpreter(
                model_path=str(model_path), num_threads=num_threads
            )
            self.interpreter.allocate_tensors()
        except ValueError as exc:
            raise ModelError(f"Could not load model {model_path.name}: {exc}") from exc

        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()

        shape = self.input_detail["shape"]
        if len(shape) != 4:
            raise ModelError(
                f"Expected a 4D input tensor (NHWC), got shape {list(shape)}."
            )
        _, self.height, self.width, self.channels = (int(v) for v in shape)
        self.dtype = self.input_detail["dtype"]

    def __repr__(self) -> str:
        return (f"<TFLiteModel {self.width}x{self.height}x{self.channels} "
                f"{self.dtype.__name__}>")

    def make_tensor(self, image: Image.Image) -> np.ndarray:
        """Turn a correctly sized PIL image into a batched input tensor."""
        array = np.asarray(image, dtype=np.float32)

        if self.dtype == np.float32:
            # MobileNet-style normalisation to [-1, 1]. Other families (ResNet,
            # EfficientNet-B*) use per-channel mean/std — check before swapping.
            array = (array - 127.5) / 127.5
        else:
            array = array.astype(self.dtype)

        return np.expand_dims(array, axis=0)

    def invoke(self, tensor: np.ndarray) -> float:
        """Run one inference pass. Returns the elapsed time in milliseconds."""
        self.interpreter.set_tensor(self.input_detail["index"], tensor)
        start = time.perf_counter()
        self.interpreter.invoke()
        return (time.perf_counter() - start) * 1000

    def output(self, index: int = 0) -> np.ndarray:
        """Fetch an output tensor, dequantised and with the batch axis removed."""
        detail = self.output_details[index]
        values = self.interpreter.get_tensor(detail["index"])[0]

        scale, zero_point = detail["quantization"]
        if scale:
            # The float32 cast must come first: in uint8 arithmetic
            # 100 - 128 wraps around to 228 instead of giving -28.
            values = scale * (values.astype(np.float32) - zero_point)
        return values
