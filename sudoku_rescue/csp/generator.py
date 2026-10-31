"""Sudoku generator: full grid by randomised backtracking, then remove cells while unique."""
import random

from sudoku_rescue.csp.sudoku import SudokuSolver


def generate_full_grid(rng: random.Random):
    """Random complete valid grid (randomised value order, backtracking + forward checking)."""
    res = SudokuSolver(use_fc=True, ordering="mrv", rng=rng).solve([0] * 81)
    return res.solution


def make_puzzle(rng: random.Random, n_givens: int, max_tries: int = 40):
    """Return (puzzle, solution) with exactly `n_givens` givens and a UNIQUE solution.

    Cells are removed in random order; a removal is kept only if the puzzle still has exactly one
    solution. If a greedy pass gets stuck above n_givens we restart with a new grid/order.
    """
    checker = SudokuSolver(use_fc=True, ordering="mrv")
    for _ in range(max_tries):
        sol = generate_full_grid(rng)
        puzzle = list(sol)
        order = list(range(81))
        rng.shuffle(order)
        n = 81
        for c in order:
            if n <= n_givens:
                break
            keep = puzzle[c]
            puzzle[c] = 0
            if checker.count_solutions(puzzle, 2) == 1:
                n -= 1
            else:
                puzzle[c] = keep
        if n == n_givens:
            return puzzle, sol
    raise RuntimeError(f"could not generate a unique puzzle with {n_givens} givens")
