"""Dataset loading for the training experiments.

Primary dataset: MNIST (downloaded once, parsed from the IDX format).
If the download fails (offline machine), a deterministic synthetic
fallback with the same API keeps the whole pipeline runnable.
"""

import gzip
import os
import struct
import urllib.request

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
IMAGE_MAGIC = 2051  # 0x00000803
LABEL_MAGIC = 2049  # 0x00000801


def _download(name):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, FILES[name])
    if not os.path.exists(path):
        urllib.request.urlretrieve(MIRROR + FILES[name], path)
    return path


def _read_images(path):
    with gzip.open(path, "rb") as fh:
        magic, n, rows, cols = struct.unpack(">IIII", fh.read(16))
        if magic != IMAGE_MAGIC:
            raise ValueError(f"Bad MNIST image magic {magic} in {path}")
        pixels = np.frombuffer(fh.read(), dtype=np.uint8)
    expected = n * rows * cols
    if pixels.size != expected:
        raise ValueError(
            f"Truncated image file {path}: got {pixels.size}, expected {expected}"
        )
    return pixels.reshape(n, rows * cols).astype(np.float32) / 255.0


def _read_labels(path):
    with gzip.open(path, "rb") as fh:
        magic, n = struct.unpack(">II", fh.read(8))
        if magic != LABEL_MAGIC:
            raise ValueError(f"Bad MNIST label magic {magic} in {path}")
        labels = np.frombuffer(fh.read(), dtype=np.uint8)
    if labels.size != n:
        raise ValueError(
            f"Truncated label file {path}: got {labels.size}, expected {n}"
        )
    return labels.astype(np.int64)


def _synthetic(n_train, n_test, dim=784, classes=10, seed=0):
    """Deterministic class-structured blobs (offline fallback)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 1.0, (classes, dim)).astype(np.float32)

    def make(n):
        y = rng.integers(0, classes, size=n)
        x = centers[y] + rng.normal(0.0, 0.6, (n, dim)).astype(np.float32)
        return x.astype(np.float32), y.astype(np.int64)

    x_train, y_train = make(n_train)
    x_test, y_test = make(n_test)
    return {
        "name": "synthetic-blobs (MNIST unavailable)",
        "x_train": x_train, "y_train": y_train,
        "x_test": x_test, "y_test": y_test,
        "n_classes": classes, "input_dim": dim,
    }


def load_dataset(name="mnist", n_train=12000, n_test=2000):
    """Return dict with x/y train+test splits (images flattened to 784)."""
    if name == "mnist":
        try:
            x_train = _read_images(_download("train_images"))
            y_train = _read_labels(_download("train_labels"))
            x_test = _read_images(_download("test_images"))
            y_test = _read_labels(_download("test_labels"))
            rng = np.random.default_rng(0)
            tr = rng.permutation(len(x_train))[:n_train]
            te = rng.permutation(len(x_test))[:n_test]
            return {
                "name": "MNIST",
                "x_train": x_train[tr], "y_train": y_train[tr],
                "x_test": x_test[te], "y_test": y_test[te],
                "n_classes": 10, "input_dim": 784,
            }
        except Exception as exc:  # offline → fallback
            print(f"MNIST download failed ({exc}); using synthetic fallback.")
    return _synthetic(n_train, n_test)
