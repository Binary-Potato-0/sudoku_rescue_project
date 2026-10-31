"""Accuracy, confusion matrix, calibration error (ECE), NLL and temperature scaling."""
import numpy as np


def accuracy(probs, y):
    return float((probs.argmax(1) == y).mean())


def confusion_matrix(pred, y, k=9):
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (y, pred), 1)          # rows = true, cols = predicted
    return cm


def nll(probs, y):
    return float(-np.log(np.maximum(probs[np.arange(len(y)), y], 1e-12)).mean())


def reliability_bins(probs, y, n_bins=15):
    """Per confidence-bin: (count, mean confidence, accuracy)."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = ids == b
        rows.append((int(m.sum()), float(conf[m].mean()) if m.any() else 0.0,
                     float(correct[m].mean()) if m.any() else 0.0))
    return rows


def ece(probs, y, n_bins=15):
    """Expected calibration error: weighted mean |accuracy - confidence| over bins."""
    rows = reliability_bins(probs, y, n_bins)
    n = len(y)
    return float(sum(c / n * abs(a - m) for c, m, a in rows if c))


def softmax_T(logits, T):
    z = logits.astype(np.float64) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits, y, lo=0.25, hi=8.0, iters=60):
    """Temperature scaling: the single T>0 minimising validation NLL (golden-section on log T)."""
    f = lambda lt: nll(softmax_T(logits, np.exp(lt)), y)
    a, b = np.log(lo), np.log(hi)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(iters):
        if f(c) < f(d):
            b = d
        else:
            a = c
        c, d = b - g * (b - a), a + g * (b - a)
    return float(np.exp((a + b) / 2))
