"""Build 9x9 mosaics of MNIST digit images from a Sudoku puzzle, inject hard digits, read them back.

Documented injection rule ("hard digit"):
  A cell whose true digit is d normally shows a RANDOM test image of d that the network reads
  CORRECTLY. To inject a misread we instead show a test image of d that the network reads WRONGLY
  but for which d is still among its top-3 digits (so a correction within 2 alternatives can fix it).
  The `m` injected cells are chosen uniformly at random among the givens. Every team uses its
  own seed for the choice of puzzle, cells and images.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from sudoku_rescue.config import BLANK_MEAN_THRESHOLD


@dataclass
class DigitBank:
    """Test digits grouped by digit (1..9) into 'correct' and 'hard' pools for one network."""
    X: np.ndarray                      # (N,784) test images in [0,1]
    y: np.ndarray                      # (N,) true digits 1..9
    correct: List[np.ndarray]          # index 0..8 -> indices of correctly read images
    hard: List[np.ndarray]             # index 0..8 -> indices of misread images (true digit in top-3)

    @classmethod
    def build(cls, model, X, y, T=1.0):
        P = model.predict_proba(X, T)
        pred = P.argmax(1) + 1
        rank = (P > P[np.arange(len(y)), y - 1][:, None]).sum(1) + 1    # rank of the true digit
        correct, hard = [], []
        for d in range(1, 10):
            correct.append(np.where((y == d) & (pred == d))[0])
            hard.append(np.where((y == d) & (pred != d) & (rank <= 3))[0])
        return cls(X, y, correct, hard)


@dataclass
class Mosaic:
    images: np.ndarray                 # (81,28,28) float32; blank cells = faint noise
    img_index: np.ndarray              # (81,) test-set index shown in each cell, -1 = blank
    puzzle: List[int]                  # true puzzle (0 = blank)
    solution: List[int]                # true full solution
    injected: List[int] = field(default_factory=list)   # cells showing an injected hard digit

    def as_image(self):
        """The 252x252 picture (what a user would photograph)."""
        return self.images.reshape(9, 9, 28, 28).transpose(0, 2, 1, 3).reshape(252, 252)


def build_mosaic(puzzle, solution, n_misreads, bank: DigitBank, rng: np.random.Generator) -> Mosaic:
    givens = [i for i, d in enumerate(puzzle) if d]
    ok = [i for i in givens if len(bank.hard[puzzle[i] - 1]) > 0]
    if n_misreads > len(ok):
        raise ValueError("not enough cells with a hard digit available for this many misreads")
    injected = sorted(int(c) for c in rng.choice(ok, size=n_misreads, replace=False)) if n_misreads else []
    images = np.zeros((81, 28, 28), dtype=np.float32)
    idx = -np.ones(81, dtype=int)
    for i in range(81):
        if puzzle[i] == 0:
            images[i] = np.clip(rng.normal(0.0, 0.02, (28, 28)), 0, 1)       # blank cell (faint noise)
            continue
        pool = bank.hard[puzzle[i] - 1] if i in injected else bank.correct[puzzle[i] - 1]
        j = int(rng.choice(pool))
        idx[i] = j
        images[i] = bank.X[j].reshape(28, 28)
    return Mosaic(images, idx, list(puzzle), list(solution), injected)


@dataclass
class Reading:
    grid: List[int]                    # network's first reading (0 = blank)
    probs: np.ndarray                  # (81,9) probabilities per digit 1..9 (zeros for blank cells)
    blank: np.ndarray                  # (81,) bool


def read_mosaic(model, images: np.ndarray, T: float = 1.0) -> Reading:
    """Blank cells by a brightness rule; the network classifies every non-blank cell."""
    flat = images.reshape(81, -1)
    blank = flat.mean(axis=1) < BLANK_MEAN_THRESHOLD
    probs = np.zeros((81, 9))
    if (~blank).any():
        probs[~blank] = model.predict_proba(flat[~blank], T)
    grid = [0 if blank[i] else int(probs[i].argmax()) + 1 for i in range(81)]
    return Reading(grid, probs, blank)
