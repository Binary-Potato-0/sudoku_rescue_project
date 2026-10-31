"""NN reading -> AC-3 conflict check -> correction search -> CSP solve (or "unreadable")."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from sudoku_rescue.csp.sudoku import CSPStats, SudokuSolver, cell_name
from sudoku_rescue.data.mosaic import Reading
from sudoku_rescue.search.correction import SearchResult, build_problem, correction_search


@dataclass
class PipelineConfig:
    algorithm: str = "ucs"            # ucs | greedy | astar
    max_corrections: int = 3
    n_candidates: int = 10            # how many least-confident cells may be changed
    k_alts: int = 2                   # alternatives per cell
    goal: str = "solvable"            # "solvable" | "unique" (extension: exactly one solution)
    use_fc: bool = True
    ac3: str = "root"
    ordering: str = "mrv"
    node_limit: int = 5000            # max expanded search states


class GoalChecker:
    """The CSP as the search's goal test. Accumulates CSP counters over all calls."""

    def __init__(self, cfg: PipelineConfig):
        self.solver = SudokuSolver(use_fc=cfg.use_fc, ac3=cfg.ac3, ordering=cfg.ordering)
        self.need = 2 if cfg.goal == "unique" else 1
        self.calls = 0
        self.ac3_rejects = 0
        self.stats = CSPStats()

    def __call__(self, grid):
        self.calls += 1
        res = self.solver.solve(grid, max_solutions=self.need)
        self.stats.add(res.stats)
        if res.stats.ac3_conflict:
            self.ac3_rejects += 1
        if res.status != "solved":
            return None
        if self.need == 2 and len(res.solutions) != 1:
            return None
        return res.solution


@dataclass
class PipelineResult:
    status: str                       # "solved" | "unreadable"
    message: str
    read_grid: List[int]
    ac3_flagged: bool                 # did AC-3 find a contradiction in the first reading?
    ac3_message: str
    corrections: list                 # (cell, read_digit, new_digit, cost)
    total_cost: float
    solution: Optional[List[int]]
    search: Optional[SearchResult]
    csp_stats: CSPStats
    csp_calls: int
    least_confident: list = field(default_factory=list)   # [(cell, p_top1)] for the "unreadable" report


def least_confident_cells(reading: Reading, n=5):
    rows = [(i, float(reading.probs[i].max())) for i in range(81) if not reading.blank[i]]
    return sorted(rows, key=lambda r: r[1])[:n]


def solve_reading(reading: Reading, cfg: PipelineConfig = PipelineConfig()) -> PipelineResult:
    """Stages 2-5 of the pipeline, given the network's reading."""
    # Stage 2: AC-3 on the first reading
    probe = SudokuSolver(use_fc=False, ac3="root").solve(reading.grid)
    flagged = probe.stats.ac3_conflict
    ac3_msg = probe.message if flagged else "AC-3: no contradiction found in the first reading"

    # Stage 3+4: search proposes corrections, CSP tests each corrected puzzle
    problem = build_problem(reading.grid, reading.probs, cfg.n_candidates, cfg.k_alts, cfg.max_corrections)
    checker = GoalChecker(cfg)
    res = correction_search(problem, checker, cfg.algorithm, cfg.node_limit)

    if res.status == "found":
        n = len(res.corrections)
        msg = "SOLVED" if n == 0 else f"SOLVED after correcting {n} cell(s)"
        return PipelineResult("solved", msg, reading.grid, flagged, ac3_msg, res.corrections,
                              res.total_cost, res.solution, res, checker.stats, checker.calls)
    # Stage 5: nothing worked
    lc = least_confident_cells(reading)
    why = ("search budget exhausted" if res.status == "limit"
           else f"no solvable puzzle within {cfg.max_corrections} corrections of the {cfg.n_candidates} "
                "least confident cells")
    cells = ", ".join(f"{cell_name(c)} (p={p:.2f})" for c, p in lc)
    return PipelineResult("unreadable", f"UNREADABLE: {why}. Least confident cells: {cells}",
                          reading.grid, flagged, ac3_msg, [], float("nan"), None, res,
                          checker.stats, checker.calls, lc)


def solve_without_correction(reading: Reading, cfg: PipelineConfig = PipelineConfig()):
    """'NN off' baseline: trust the first reading and just run the CSP."""
    res = SudokuSolver(use_fc=cfg.use_fc, ac3=cfg.ac3, ordering=cfg.ordering).solve(reading.grid)
    return res
