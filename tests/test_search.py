import math

import pytest

from sudoku_rescue.integration.pipeline import GoalChecker, PipelineConfig
from sudoku_rescue.search.correction import build_problem, change_cost, correction_search, lower_bound
from tests.common import PUZZLE, SOLUTION, fake_probs

# Toy example with a known answer (hand-checkable):
#  - cell 0 truly holds 5 but is read as 3 (p=0.52 vs 0.45)  -> duplicate 3 in row 1 and box 1
#  - cell 79 truly holds 7 but is read as 9 (p=0.55 vs 0.40) -> duplicate 9 in row 9
READ1 = list(PUZZLE); READ1[0] = 3
READ2 = list(READ1);  READ2[79] = 9
P1 = fake_probs(READ1, {0: {3: 0.52, 5: 0.45}})
P2 = fake_probs(READ2, {0: {3: 0.52, 5: 0.45}, 79: {9: 0.55, 7: 0.40}})
COST0, COST79 = math.log(0.52 / 0.45), math.log(0.55 / 0.40)


def run(read, probs, algo, goal="solvable", **kw):
    pr = build_problem(read, probs, kw.pop("n_candidates", 10), kw.pop("k_alts", 2), kw.pop("max_corrections", 3))
    chk = GoalChecker(PipelineConfig(goal=goal))
    return correction_search(pr, chk, algo, **kw), chk


def test_normal_cost_is_never_negative_and_matches_formula():
    assert change_cost(0.52, 0.45) == pytest.approx(COST0)
    assert change_cost(0.5, 0.5) == 0.0 and change_cost(0.3, 0.6) == 0.0     # clipped at 0
    pr = build_problem(READ2, P2, 10, 2, 3)
    assert all(c >= 0 for a in pr.alts.values() for _, c in a)


def test_normal_ucs_finds_known_single_correction_and_counters():
    res, chk = run(READ1, P1, "ucs")
    assert res.status == "found" and res.solution == SOLUTION
    assert [(c, o, n) for c, o, n, _ in res.corrections] == [(0, 3, 5)]
    assert res.total_cost == pytest.approx(COST0)
    # root (1) + 10 candidate cells x 2 alternatives = 21 generated; root expanded once; 2 goal tests
    assert (res.stats.generated, res.stats.expanded, res.stats.goal_tests) == (21, 1, 2)
    assert chk.calls == 2


def test_normal_ucs_finds_known_two_corrections():
    res, _ = run(READ2, P2, "ucs")
    assert sorted((c, n) for c, _, n, _ in res.corrections) == [(0, 5), (79, 7)]
    assert res.total_cost == pytest.approx(COST0 + COST79)
    assert res.solution == SOLUTION


def test_normal_astar_is_optimal_and_expands_no_more_than_ucs():
    ucs, _ = run(READ2, P2, "ucs")
    astar, _ = run(READ2, P2, "astar")
    assert astar.total_cost == pytest.approx(ucs.total_cost)
    assert astar.stats.expanded <= ucs.stats.expanded


def test_normal_heuristic_is_admissible_on_toy_example():
    pr = build_problem(READ2, P2, 10, 2, 3)
    h = lower_bound(READ2, pr, ())
    assert h <= COST0 + COST79 + 1e-12 and h > 0
    # after fixing cell 0 only the second conflict remains
    from sudoku_rescue.search.correction import apply_corrections
    state = ((0, 5),)
    assert lower_bound(apply_corrections(READ2, state), pr, state) <= COST79 + 1e-12


def test_normal_greedy_also_solves_toy_example():
    res, _ = run(READ2, P2, "greedy")
    assert res.status == "found" and res.solution == SOLUTION


def test_normal_unique_goal_rejects_corrections_that_leave_many_solutions():
    # only the first correction is needed; with the 'unique' goal the answer must still be the true puzzle
    res, _ = run(READ1, P1, "ucs", goal="unique")
    assert res.solution == SOLUTION


def test_edge_no_misread_means_zero_corrections():
    res, _ = run(list(PUZZLE), fake_probs(PUZZLE), "ucs")
    assert res.status == "found" and res.corrections == [] and res.stats.expanded == 0


def test_edge_zero_correction_budget():
    res, _ = run(READ1, P1, "ucs", max_corrections=0)
    assert res.status == "exhausted"


def test_edge_node_limit_returns_limit_status():
    res, _ = run(READ2, P2, "ucs", node_limit=1)
    assert res.status == "limit"


def test_impossible_four_disjoint_conflicts_need_more_than_three_corrections():
    g = [0] * 81
    for r in range(4):
        g[r * 9] = g[r * 9 + 1] = r + 1                  # four rows, each with a duplicated digit
    P = fake_probs(g, {i: {g[i]: 0.6, (g[i] % 9) + 1: 0.3} for i in (0, 1, 9, 10, 18, 19, 27, 28)})
    for algo in ("ucs", "astar", "greedy"):
        res, _ = run(g, P, algo)
        assert res.status == "exhausted" and res.solution is None
