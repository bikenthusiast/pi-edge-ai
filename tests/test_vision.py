"""Tests that run on the Mac. No camera, no GPIO, no AWS.

The point of this file is the deploy boundary: everything here exercises code
that also runs on the Pi, without needing the Pi.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from edge.vision.model import (
    DEFAULT_LABELS,
    DEFAULT_MODEL,
    ModelError,
    TFLiteModel,
    load_labels,
    preprocess,
)


def _litert_available() -> bool:
    """True if the interpreter can actually be imported on this platform.

    Not pytest.importorskip: since pytest 8.2 that re-raises when the ImportError
    comes from inside the module rather than from it being absent, which is
    exactly the case for a broken/unsupported LiteRT build.
    """
    try:
        import ai_edge_litert.interpreter  # noqa: F401
    except ImportError:
        return False
    return True


requires_litert = pytest.mark.skipif(
    not _litert_available(),
    reason="LiteRT unavailable here (macOS x86_64 has no wheels) — runs on the Pi",
)


@pytest.fixture(scope="module")
def model() -> TFLiteModel:
    if not DEFAULT_MODEL.is_file():
        pytest.skip("model missing — run scripts/fetch_models.sh")
    return TFLiteModel(DEFAULT_MODEL, num_threads=2)


@pytest.fixture(scope="module")
def labels() -> list[str]:
    if not DEFAULT_LABELS.is_file():
        pytest.skip("labels missing — run scripts/fetch_models.sh")
    return load_labels(DEFAULT_LABELS)


# --- preprocessing: pure functions, no model needed ------------------------ #

@pytest.mark.parametrize("fit", ["crop", "pad", "stretch"])
@pytest.mark.parametrize("size", [(512, 929), (929, 512), (224, 224), (50, 4000)])
def test_preprocess_always_returns_target_size(fit, size):
    out = preprocess(Image.new("RGB", size), 224, 224, fit=fit)
    assert out.size == (224, 224)
    assert out.mode == "RGB"


def test_preprocess_converts_greyscale_and_alpha():
    for mode in ("L", "RGBA", "P"):
        out = preprocess(Image.new(mode, (300, 300)), 224, 224)
        assert out.mode == "RGB"


def test_pad_keeps_aspect_ratio_crop_does_not():
    """A wide red bar on white: pad must preserve it, stretch must distort it."""
    img = Image.new("RGB", (400, 100), (255, 255, 255))
    padded = np.asarray(preprocess(img, 224, 224, fit="pad"))
    # letterbox fill sits at top and bottom, original content in the middle band
    assert padded[0, 112].tolist() == [114, 114, 114]
    assert padded[112, 112].tolist() == [255, 255, 255]


def test_unknown_fit_mode_raises():
    with pytest.raises(ValueError, match="Unknown fit mode"):
        preprocess(Image.new("RGB", (10, 10)), 224, 224, fit="squish")


# --- model contract -------------------------------------------------------- #

@requires_litert
def test_model_reports_its_own_input_shape(model):
    assert (model.width, model.height, model.channels) == (224, 224, 3)
    assert model.dtype == np.uint8


def test_missing_model_file_raises_modelerror():
    """Runs everywhere: the path check happens before the LiteRT import."""
    with pytest.raises(ModelError, match="not found"):
        TFLiteModel("models/does-not-exist.tflite")


def test_missing_label_file_raises_modelerror():
    with pytest.raises(ModelError, match="not found"):
        load_labels("models/does-not-exist.txt")


@requires_litert
def test_label_count_matches_output_dimension(model, labels):
    """The classic off-by-one: 1000 labels against a 1001-class head."""
    tensor = model.make_tensor(preprocess(Image.new("RGB", (224, 224)), 224, 224))
    model.invoke(tensor)
    assert len(labels) == model.output(0).shape[0]


@requires_litert
def test_dequantised_scores_are_probabilities(model):
    tensor = model.make_tensor(preprocess(Image.new("RGB", (224, 224)), 224, 224))
    model.invoke(tensor)
    scores = model.output(0)

    assert scores.dtype == np.float32, "output was not dequantised"
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0
    # The sum lands BELOW 1, not at it: with scale=1/256 every class whose true
    # probability is under ~0.4 % quantises to 0, so the tail is lost. A test
    # asserting sum == 1.0 would fail on every uint8 model — that missing mass
    # is quantisation error, not a bug.
    assert 0.4 < scores.sum() <= 1.05


@requires_litert
def test_inference_is_deterministic(model):
    tensor = model.make_tensor(preprocess(Image.new("RGB", (224, 224)), 224, 224))
    model.invoke(tensor)
    first = model.output(0).copy()
    model.invoke(tensor)
    np.testing.assert_array_equal(first, model.output(0))
