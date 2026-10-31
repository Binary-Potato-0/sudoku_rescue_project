import random

import pytest

from sudoku_rescue.csp.generator import make_puzzle
from sudoku_rescue.csp.sudoku import SudokuSolver, is_valid_solution
from tests.common import PUZZLE, SOLUTION

MODES = {"BT": {}, "BT+FC": dict(use_fc=True), "BT+AC3": dict(ac3="root"),
         "BT+FC+AC3": dict(use_fc=True, ac3="root"), "BT+FC+MAC": dict(ac3="mac")}


# ----------------------------------------------------------------------- normal cases
@pytest.mark.parametrize("name", MODES)
@pytest.mark.parametrize("ordering", ["static", "mrv"])
def test_normal_every_mode_solves_known_puzzle(name, ordering):
    res = SudokuSolver(ordering=ordering, **MODES[name]).solve(PUZZLE)
    assert res.solved and res.solution == SOLUTION


def test_normal_forward_checking_needs_fewer_assignments_than_plain_bt():
    p, sol = make_puzzle(random.Random(11), 25)
    bt = SudokuSolver(ordering="static").solve(p)
    fc = SudokuSolver(use_fc=True, ordering="static").solve(p)
    assert bt.solution == fc.solution == sol
    assert fc.stats.assignments < bt.stats.assignments
    assert fc.stats.fc_prunes > 0 and bt.stats.fc_prunes == 0


def test_normal_ac3_prunes_domains_and_counts_work():
    res = SudokuSolver(use_fc=True, ac3="root").solve(PUZZLE)
    assert res.stats.ac3_prunes > 0 and res.stats.ac3_revisions >= 81 * 20


@pytest.mark.parametrize("n", [25, 30, 35])
def test_normal_generator_gives_unique_puzzles_with_requested_givens(n):
    p, sol = make_puzzle(random.Random(n), n)
    assert sum(1 for v in p if v) == n and is_valid_solution(sol)
    assert all(a in (0, b) for a, b in zip(p, sol))
    assert len(SudokuSolver(ordering="mrv").solve(p, max_solutions=3).solutions) == 1


def test_normal_generator_is_reproducible():
    assert make_puzzle(random.Random(5), 30) == make_puzzle(random.Random(5), 30)


# ----------------------------------------------------------------------- edge cases
def test_edge_empty_grid_is_solvable():
    res = SudokuSolver(use_fc=True, ordering="mrv").solve([0] * 81)
    assert res.solved and is_valid_solution(res.solution)


def test_edge_complete_grid_needs_no_search():
    res = SudokuSolver().solve(SOLUTION)
    assert res.solved and res.solution == SOLUTION and res.stats.assignments == 0


def test_edge_assignment_limit_stops_search():
    p, _ = make_puzzle(random.Random(3), 25)
    res = SudokuSolver(node_limit=5).solve(p)
    assert res.status == "limit" and "limit" in res.message


def test_edge_multiple_solutions_counted():
    g = list(PUZZLE)
    g[2] = 0; g[3] = 0                                   # keep it solvable
    assert SudokuSolver(use_fc=True).count_solutions([0] * 81, 3) == 3


def test_edge_bad_input_rejected():
    with pytest.raises(ValueError):
        SudokuSolver().solve([0] * 80)


# ----------------------------------------------------------------------- impossible cases
@pytest.mark.parametrize("name", MODES)
def test_impossible_duplicate_givens_report_no_solution(name):
    g = [0] * 81
    g[0] = g[1] = 5
    res = SudokuSolver(**MODES[name]).solve(g)
    assert res.status == "no_solution" and res.solution is None
    assert res.message.startswith("NO SOLUTION")


@pytest.mark.parametrize("name", MODES)
def test_impossible_contradiction_without_any_duplicate(name):
    g = [0] * 81
    for c in range(8):
        g[c] = c + 1                                     # row 1 holds 1..8, so (r1,c9) must be 9 ...
    g[17] = 9                                            # ... but (r2,c9)=9 forbids 9 in column 9
    res = SudokuSolver(**MODES[name]).solve(g)
    assert res.status == "no_solution" and "NO SOLUTION" in res.message
