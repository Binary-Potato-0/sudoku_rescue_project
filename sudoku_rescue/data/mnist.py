"""Load the preprocessed MNIST (digits 1-9) produced by make_data.py."""
import numpy as np

from sudoku_rescue.config import MNIST_FILE


def load_mnist19():
    """Return dict with X_train/X_val/X_test as float32 in [0,1] (N,784) and y_* in 1..9."""
    if not MNIST_FILE.exists():
        raise FileNotFoundError(f"{MNIST_FILE} not found - run `python make_data.py` first")
    z = np.load(MNIST_FILE, allow_pickle=False)
    out = {}
    for s in ("train", "val", "test"):
        out[f"X_{s}"] = z[f"X_{s}"].astype(np.float32) / 255.0   # scale pixels to 0-1
        out[f"y_{s}"] = z[f"y_{s}"].astype(np.int64)              # digits 1..9
    return out
