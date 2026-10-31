#!/usr/bin/env python
"""Download and preprocess MNIST for the Sudoku project (reproducible, fixed seed).

Steps
  1. Obtain the four official MNIST files (LeCun's page, then the torchvision S3 mirror),
     or - only if those hosts are unreachable - a GitHub mirror of the same 60k/10k data.
     A local copy is kept in data/raw/  (re-used on later runs).
  2. Take the 60,000 training images, split 50,000 train / 10,000 validation with a
     seeded permutation (SPLIT_SEED). The 10,000 test images are untouched.
  3. Drop digit 0 everywhere (the Sudoku uses digits 1-9; blank cells are detected by a
     brightness rule, not by the network).
  4. Save data/processed/mnist19.npz  (uint8 pixels 0-255; scaled to 0-1 on loading).

Usage:  python make_data.py [--source auto|official|mirror]
"""
import argparse
import gzip
import hashlib
import pickle
import struct
import sys
import urllib.request
import warnings

import numpy as np

from sudoku_rescue.config import DATA_PROCESSED, DATA_RAW, MNIST_FILE, SPLIT_SEED

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
OFFICIAL_BASES = [
    "http://yann.lecun.com/exdb/mnist/",          # original page
    "https://ossci-datasets.s3.amazonaws.com/mnist/",  # mirror used by torchvision
]
# Fallback: same data in the classic mnist.pkl.gz layout (50k train / 10k val / 10k test).
# Pixels there are stored as k/256, so we multiply back by 256 to recover the exact uint8 values.
MIRROR_URL = ("https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/"
              "master/data/mnist.pkl.gz")


def fetch(url, dest, timeout=60):
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=timeout) as r, open(dest, "wb") as f:
        f.write(r.read())


def read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, "not an IDX image file"
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)


def read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, "not an IDX label file"
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_official():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        dest = DATA_RAW / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        for base in OFFICIAL_BASES:
            try:
                fetch(base + name, dest)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  failed ({e.__class__.__name__}: {e})")
                dest.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"could not download {name} from any official host")
    xtr = read_idx_images(DATA_RAW / FILES["train_images"])
    ytr = read_idx_labels(DATA_RAW / FILES["train_labels"])
    xte = read_idx_images(DATA_RAW / FILES["test_images"])
    yte = read_idx_labels(DATA_RAW / FILES["test_labels"])
    return xtr, ytr, xte, yte, "official MNIST (LeCun / torchvision mirror)"


def load_mirror():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    dest = DATA_RAW / "mnist.pkl.gz"
    if not dest.exists() or dest.stat().st_size == 0:
        fetch(MIRROR_URL, dest)
    # NOTE: unpickling executes code from the file; only use this fallback with the known mirror.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with gzip.open(dest, "rb") as f:
            tr, va, te = pickle.load(f, encoding="latin1")
    to_u8 = lambda x: np.rint(np.asarray(x, dtype=np.float64) * 256).astype(np.uint8)
    xtr = np.concatenate([to_u8(tr[0]), to_u8(va[0])])   # 50k + 10k = original 60k training set
    ytr = np.concatenate([tr[1], va[1]]).astype(np.uint8)
    return xtr, ytr, to_u8(te[0]), te[1].astype(np.uint8), "GitHub mirror of MNIST (mnist.pkl.gz)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["auto", "official", "mirror"], default="auto")
    args = ap.parse_args()

    loaders = {"official": [load_official], "mirror": [load_mirror], "auto": [load_official, load_mirror]}
    for loader in loaders[args.source]:
        try:
            xtr, ytr, xte, yte, source = loader()
            break
        except Exception as e:  # noqa: BLE001
            print(f"{loader.__name__} failed: {e}")
    else:
        sys.exit("No data source worked. Download the four MNIST files manually into data/raw/.")

    assert xtr.shape == (60000, 784) and xte.shape == (10000, 784), "unexpected MNIST size"
    print(f"source: {source}")

    perm = np.random.default_rng(SPLIT_SEED).permutation(60000)
    tr_idx, va_idx = perm[:50000], perm[50000:]
    splits = {
        "train": (xtr[tr_idx], ytr[tr_idx]),
        "val": (xtr[va_idx], ytr[va_idx]),
        "test": (xte, yte),
    }
    out = {}
    for name, (x, y) in splits.items():
        keep = y > 0                      # drop digit 0
        out[f"X_{name}"], out[f"y_{name}"] = x[keep], y[keep]
        print(f"{name:5s}: {len(y):6d} images -> {keep.sum():6d} after dropping digit 0")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MNIST_FILE, source=source, split_seed=SPLIT_SEED, **out)
    h = hashlib.sha256()
    for k in sorted(out):
        h.update(out[k].tobytes())
    print(f"saved {MNIST_FILE}  (sha256 of arrays: {h.hexdigest()[:16]}...)")


if __name__ == "__main__":
    main()
