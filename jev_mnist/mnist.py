"""Download and parse the MNIST test set (IDX format)."""

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
TEST_IMAGES = "t10k-images-idx3-ubyte.gz"
TEST_LABELS = "t10k-labels-idx1-ubyte.gz"


def _download(name: str, data_dir: Path) -> Path:
    path = data_dir / name
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(BASE_URL + name, tmp)
        tmp.rename(path)
    return path


def parse_idx(raw: bytes) -> np.ndarray:
    """Parse an uncompressed IDX file containing unsigned bytes."""
    zero, dtype, ndim = struct.unpack(">HBB", raw[:4])
    if zero != 0 or dtype != 0x08:
        raise ValueError("not an unsigned-byte IDX file")
    shape = struct.unpack(f">{ndim}I", raw[4 : 4 + 4 * ndim])
    data = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim)
    return data.reshape(shape)


def load_test_set(data_dir: Path = Path("data")) -> tuple[np.ndarray, np.ndarray]:
    """Return the 10,000 test images (N, 28, 28) and their labels (N,)."""
    images = parse_idx(gzip.decompress(_download(TEST_IMAGES, data_dir).read_bytes()))
    labels = parse_idx(gzip.decompress(_download(TEST_LABELS, data_dir).read_bytes()))
    if len(images) != len(labels):
        raise ValueError("image and label counts differ")
    return images, labels
