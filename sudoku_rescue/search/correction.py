"""Search over sets of corrections to the network's first reading of the grid.

State   : a set of corrections {(cell, new_digit)}, at most `max_corrections`, one per cell.
Action  : change ONE more uncertain cell to one of its top alternative digits.
Cost    : change_cost(p_read, p_new) = ln(p_read / p_new)  >= 0  because the read digit is the
          network's top choice. Costs add up, so the cost of a state = -ln(P(new digits)/P(read digits))
          = how much less likely the whole corrected reading is than the first reading.
Goal    : goal_fn(corrected_grid) returns a solution (CSP says "solvable", or "uniquely solvable").

Algorithms
  ucs    : priority g              -> optimal (cheapest) correction set
  greedy : priority h              -> fast, not optimal
  astar  : priority g + h          -> optimal, h admissible (see lower_bound)

Counters: generated (states put on the frontier), expanded (states whose successors were
generated), goal_tests (CSP calls), max_frontier.
"""
from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sudoku_rescue.csp.sudoku import UNITS

EPS = 1e-12
INF = math.inf


def change_cost(p_read: float, p_new: float) -> float:
    """ln(p_read / p_new), clipped at 0 (never negative, as UCS requires)."""
    return max(0.0, math.log(max(p_read, EPS) / max(p_new, EPS)))


@dataclass
class CorrectionProblem:
    read_grid: List[int]                                   # first reading, 0 = blank
    alts: Dict[int, List[Tuple[int, float]]]               # cell -> [(new_digit, cost)] cheapest first
    max_corrections: int = 3
    min_cost: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        self.min_cost = {c: a[0][1] for c, a in self.alts.items() if a}

    @property
    def candidates(self):
        return sorted(self.alts)


def build_problem(read_grid: Sequence[int], probs: np.ndarray, n_candidates=10, k_alts=2,
                  max_corrections=3) -> CorrectionProblem:
    """Pick the n_candidates least-confident given cells and, for each, its k_alts most likely
    alternative digits. probs[i] is the 9-vector of probabilities for digits 1..9."""
    givens = [i for i, d in enumerate(read_grid) if d]
    cands = sorted(givens, key=lambda i: probs[i, read_grid[i] - 1])[:n_candidates]
    alts = {}
    for i in cands:
        p_read = probs[i, read_grid[i] - 1]
        order = [d for d in np.argsort(-probs[i]) + 1 if d != read_grid[i]][:k_alts]
        alts[i] = [(int(d), change_cost(p_read, probs[i, d - 1])) for d in order]
    return CorrectionProblem(list(read_grid), alts, max_corrections)


def apply_corrections(read_grid, state):
    g = list(read_grid)
    for cell, d in state:
        g[cell] = d
    return g


def conflict_pairs(grid):
    """All pairs of given cells that hold the same digit inside one row/column/box."""
    pairs = set()
    for unit in UNITS:
        by = {}
        for c in unit:
            if grid[c]:
                by.setdefault(grid[c], []).append(c)
        for cells in by.values():
            for a, b in itertools.combinations(cells, 2):
                pairs.add((a, b))
    return pairs


def lower_bound(grid, problem: CorrectionProblem, state) -> float:
    """Admissible A* heuristic: a COST lower bound for fixing the remaining duplicate conflicts.

    Every duplicate pair (a, b) needs at least one of its two cells changed, and changing cell c
    costs at least min_cost[c]. Take a set of vertex-disjoint conflicting pairs: each needs its own
    changed cell, so the costs add. Hence  h = sum over the disjoint pairs of min(min_cost[a], min_cost[b]).
    It never exceeds the true remaining cost, so A* stays optimal. Returns INF (dead end) if some
    conflict can no longer be repaired (no allowed cell left in the pair, or too few corrections
    left in the budget).
    """
    pairs = conflict_pairs(grid)
    if not pairs:
        return 0.0
    corrected = {c for c, _ in state}

    def cheapest(c):
        return INF if (c in corrected or c not in problem.min_cost) else problem.min_cost[c]

    scored = []
    for a, b in pairs:
        pc = min(cheapest(a), cheapest(b))
        if pc == INF:
            return INF
        scored.append((pc, a, b))
    scored.sort(reverse=True)
    used, total, n = set(), 0.0, 0
    for pc, a, b in scored:
        if a not in used and b not in used:
            used.update((a, b)); total += pc; n += 1
    if n > problem.max_corrections - len(state):
        return INF
    return total


@dataclass
class SearchStats:
    generated: int = 0
    expanded: int = 0
    goal_tests: int = 0
    max_frontier: int = 0
    pruned_by_h: int = 0


@dataclass
class SearchResult:
    status: str                       # "found" | "exhausted" | "limit"
    algorithm: str
    corrections: List[Tuple[int, int, int, float]]   # (cell, read_digit, new_digit, cost)
    total_cost: float
    solution: Optional[List[int]]
    stats: SearchStats
    trace: List[dict]


def correction_search(problem: CorrectionProblem, goal_fn: Callable[[List[int]], Optional[List[int]]],
                      algorithm: str = "ucs", node_limit: int = 5000, record_trace: bool = True) -> SearchResult:
    if algorithm not in ("ucs", "greedy", "astar"):
        raise ValueError("algorithm must be ucs, greedy or astar")
    stats, trace = SearchStats(), []
    counter = itertools.count()

    def h_of(state):
        if algorithm == "ucs":
            return 0.0
        return lower_bound(apply_corrections(problem.read_grid, state), problem, state)

    def priority(g, h):
        return {"ucs": g, "greedy": h, "astar": g + h}[algorithm]

    root = ()
    h0 = h_of(root)
    frontier = []
    if h0 == INF:
        stats.pruned_by_h += 1
        return SearchResult("exhausted", algorithm, [], 0.0, None, stats, trace)
    heapq.heappush(frontier, (priority(0.0, h0), 0.0, next(counter), root, h0))
    reached = {root: 0.0}
    stats.generated = 1

    while frontier:
        stats.max_frontier = max(stats.max_frontier, len(frontier))
        f, g, _, state, h = heapq.heappop(frontier)
        grid = apply_corrections(problem.read_grid, state)
        stats.goal_tests += 1
        sol = goal_fn(grid)
        if record_trace:
            trace.append({"order": len(trace) + 1, "corrections": state, "g": g, "h": h,
                          "f": f, "goal": sol is not None})
        if sol is not None:
            log = []
            for cell, new in state:
                old = problem.read_grid[cell]
                cost = dict(problem.alts[cell])[new]
                log.append((cell, old, new, cost))
            return SearchResult("found", algorithm, log, g, sol, stats, trace)
        if stats.expanded >= node_limit:
            return SearchResult("limit", algorithm, [], math.nan, None, stats, trace)
        stats.expanded += 1
        if len(state) >= problem.max_corrections:
            continue
        done = {c for c, _ in state}
        for cell in problem.candidates:
            if cell in done:
                continue
            for digit, cost in problem.alts[cell]:
                child = tuple(sorted(state + ((cell, digit),)))
                if child in reached:
                    continue
                gc = g + cost
                hc = h_of(child)
                if hc == INF:
                    stats.pruned_by_h += 1
                    reached[child] = gc          # remember so we do not recompute
                    continue
                reached[child] = gc
                stats.generated += 1
                heapq.heappush(frontier, (priority(gc, hc), gc, next(counter), child, hc))
    return SearchResult("exhausted", algorithm, [], math.nan, None, stats, trace)
