"""Shared fixtures: the classic Wikipedia Sudoku (known solution) and fake network outputs."""
import math

import numpy as np

PUZZLE = [int(c) for c in
          "530070000" "600195000" "098000060" "800060003" "400803001" "700020006" "060000280" "000419005" "000080079"]
SOLUTION = [int(c) for c in
            "534678912" "672195348" "198342567" "859761423" "426853791" "713924856" "961537284" "287419635" "345286179"]


def fake_probs(grid, uncertain=None, floor=0.001):
    """Probabilities for a read grid: 0.99 on the read digit, or custom {cell: {digit: p}} (rest share the remainder)."""
    uncertain = uncertain or {}
    P = np.zeros((81, 9))
    for i, d in enumerate(grid):
        if d == 0:
            continue
        if i in uncertain:
            row = np.full(9, floor)
            for dig, p in uncertain[i].items():
                row[dig - 1] = p
            row[row == floor] = (1 - sum(uncertain[i].values())) / (9 - len(uncertain[i]))
            P[i] = row / row.sum()
        else:
            row = np.full(9, 0.01 / 8)
            row[d - 1] = 0.99
            P[i] = row
    return P
