"""Offline tests: no network or API key needed."""

import asyncio
import struct
from io import BytesIO

import numpy as np
import PIL.Image

from jev_mnist.classify import Prediction, to_png
from jev_mnist.main import Result, confusion_matrix, evaluate, report, select_indices
from jev_mnist.mnist import parse_idx


def test_parse_idx_images():
    pixels = np.arange(2 * 28 * 28, dtype=np.uint32).astype(np.uint8)
    raw = struct.pack(">HBB3I", 0, 0x08, 3, 2, 28, 28) + pixels.tobytes()
    assert (parse_idx(raw) == pixels.reshape(2, 28, 28)).all()


def test_parse_idx_labels():
    raw = struct.pack(">HBBI", 0, 0x08, 1, 3) + bytes([7, 2, 1])
    assert parse_idx(raw).tolist() == [7, 2, 1]


def test_to_png_upscales():
    pixels = np.zeros((28, 28), dtype=np.uint8)
    pixels[10, 10] = 255
    image = PIL.Image.open(BytesIO(to_png(pixels, upscale=4)))
    assert image.size == (112, 112)
    assert image.getpixel((41, 41)) == 255


def test_evaluate_and_report():
    images = np.zeros((4, 28, 28), dtype=np.uint8)
    labels = np.array([0, 1, 2, 3])
    answers = iter([0, 7, 2, None])

    async def fake_classifier(png: bytes) -> Prediction:
        digit = next(answers)
        if digit is None:
            raise RuntimeError("boom")
        return Prediction(digit=digit, confidence=0.9, model="jev-1.14.0", input_tokens=10)

    results = asyncio.run(evaluate(images, labels, range(4), fake_classifier, concurrency=1, upscale=1, progress=False))
    assert [r.correct for r in results] == [True, False, True, False]
    matrix = confusion_matrix(results)
    assert matrix[1, 7] == 1 and matrix.sum() == 3
    text = report(results)
    assert "Accuracy (answered): 66.67% (2/3)" in text
    assert "Accuracy (all):      50.00% (2/4)" in text
    assert "errors: 1" in text


def test_select_indices():
    assert select_indices(10, 3, 5, None) == [5, 6, 7]
    sampled = select_indices(10000, 5, 0, 42)
    assert len(set(sampled)) == 5 and sampled == select_indices(10000, 5, 0, 42)


def test_result_without_prediction_is_wrong():
    assert not Result(0, 1, None, "err").correct
