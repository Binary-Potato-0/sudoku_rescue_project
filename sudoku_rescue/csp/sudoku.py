"""Sudoku as a constraint satisfaction problem.

Variables : 81 cells (index = 9*row + col)
Domains   : digits 1-9 (givens have a one-value domain); stored as bit masks, bit d = digit d
Constraints: the 9 cells of every row, column and 3x3 box must be pairwise different.

Three separately switchable techniques (all report counters):
  * Backtracking (BT)         - always on; assigns one cell at a time and undoes failed choices.
  * Forward checking (FC)     - after each assignment remove that digit from the peers' domains;
                                fail as soon as some domain becomes empty.
  * Arc consistency (AC-3)    - ac3="root": run once before search (detects contradictory givens,
                                shrinks domains);  ac3="mac": also re-run after every assignment.

Counters (CSPStats): assignments, backtracks, fc_prunes, ac3_revisions, ac3_prunes.
  assignments = number of (cell := value) tries; backtracks = number of tries that were undone
  because they led to failure.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence

ALL = 0b1111111110          # digits 1..9


def _build_structure():
    rows = [[r * 9 + c for c in range(9)] for r in range(9)]
    cols = [[r * 9 + c for r in range(9)] for c in range(9)]
    boxes = [[(br * 3 + r) * 9 + (bc * 3 + c) for r in range(3) for c in range(3)]
             for br in range(3) for bc in range(3)]
    units = rows + cols + boxes
    peers = [set() for _ in range(81)]
    for u in units:
        for i in u:
            peers[i].update(j for j in u if j != i)
    return units, [tuple(sorted(p)) for p in peers]


UNITS, PEERS = _build_structure()
UNIT_NAME = [f"row {i + 1}" for i in range(9)] + [f"column {i + 1}" for i in range(9)] + \
            [f"box {i + 1}" for i in range(9)]
ROW = [i // 9 for i in range(81)]
COL = [i % 9 for i in range(81)]
BOX = [(i // 27) * 3 + (i % 9) // 3 for i in range(81)]
DIGITS = [tuple(d for d in range(1, 10) if (m >> d) & 1) for m in range(1024)]   # mask -> digits


def cell_name(i: int) -> str:
    return f"(r{ROW[i] + 1},c{COL[i] + 1})"


def find_given_conflict(grid: Sequence[int]):
    """First pair of equal digits sharing a unit, as (cell_a, cell_b, digit, unit_name), or None."""
    for u, unit in enumerate(UNITS):
        seen = {}
        for c in unit:
            d = grid[c]
            if d:
                if d in seen:
                    return seen[d], c, d, UNIT_NAME[u]
                seen[d] = c
    return None


def is_valid_solution(grid: Sequence[int]) -> bool:
    return all(sorted(grid[c] for c in u) == list(range(1, 10)) for u in UNITS)


@dataclass
class CSPStats:
    assignments: int = 0
    backtracks: int = 0
    fc_prunes: int = 0
    ac3_revisions: int = 0
    ac3_prunes: int = 0
    ac3_conflict: bool = False      # AC-3 / FC emptied a domain before any search
    wipeout_cell: int = -1
    limit_hit: bool = False
    seconds: float = 0.0

    def add(self, o: "CSPStats"):
        for k in ("assignments", "backtracks", "fc_prunes", "ac3_revisions", "ac3_prunes"):
            setattr(self, k, getattr(self, k) + getattr(o, k))
        self.seconds += o.seconds

    def as_dict(self):
        return asdict(self)


@dataclass
class SolveResult:
    status: str                              # "solved" | "no_solution" | "limit"
    solution: Optional[List[int]]
    message: str
    stats: CSPStats
    solutions: Optional[List[List[int]]] = None

    @property
    def solved(self):
        return self.status == "solved"


class _Limit(Exception):
    pass


class SudokuSolver:
    """Backtracking solver with switchable forward checking and AC-3.

    use_fc : forward checking on/off
    ac3    : "off" | "root" | "mac"   ("mac" re-runs AC-3 after every assignment and implies FC)
    ordering: "static" (first empty cell) | "mrv" (fewest candidate digits first)
    node_limit: stop after this many assignments (status "limit")
    rng    : optional random.Random used to shuffle value order (puzzle generation only)
    """

    def __init__(self, use_fc=False, ac3="off", ordering="static", node_limit=None, rng=None):
        ac3 = {True: "root", False: "off"}.get(ac3, ac3)
        if ac3 not in ("off", "root", "mac"):
            raise ValueError("ac3 must be 'off', 'root' or 'mac'")
        if ordering not in ("static", "mrv"):
            raise ValueError("ordering must be 'static' or 'mrv'")
        self.use_fc = use_fc or ac3 == "mac"
        self.ac3 = ac3
        self.ordering = ordering
        self.node_limit = node_limit
        self.rng = rng

    # ------------------------------------------------------------------ public API
    def solve(self, grid: Sequence[int], max_solutions: int = 1) -> SolveResult:
        if len(grid) != 81 or any((not isinstance(v, (int,))) and v != int(v) or not 0 <= int(v) <= 9 for v in grid):
            raise ValueError("grid must be 81 integers in 0..9 (0 = blank)")
        grid = [int(v) for v in grid]
        stats = CSPStats()
        t0 = time.perf_counter()
        try:
            res = self._run(grid, max_solutions, stats)
        finally:
            stats.seconds = time.perf_counter() - t0
        return res

    def count_solutions(self, grid, limit=2) -> int:
        res = self.solve(grid, max_solutions=limit)
        return len(res.solutions or [])

    # ------------------------------------------------------------------ internals
    def _digits(self, mask):
        ds = DIGITS[mask]
        if self.rng is not None:
            ds = list(ds)
            self.rng.shuffle(ds)
        return ds

    def _no_solution(self, msg, stats):
        return SolveResult("no_solution", None, "NO SOLUTION: " + msg, stats, [])

    def _run(self, grid, limit, stats) -> SolveResult:
        sols: List[List[int]] = []
        dom = None

        # ---- stage 1: consistency of the givens (each technique does it its own way)
        if self.ac3 != "off":
            dom = [(1 << d) if d else ALL for d in grid]
            if not self._ac3(dom, stats):
                stats.ac3_conflict = True
                return self._no_solution(
                    f"AC-3 emptied the domain of cell {cell_name(stats.wipeout_cell)}; "
                    "the given digits contradict each other.", stats)
        elif self.use_fc:
            dom = [(1 << d) if d else ALL for d in grid]
            for i, d in enumerate(grid):
                if d:
                    for p in PEERS[i]:
                        if dom[p] & (1 << d):
                            dom[p] &= ~(1 << d)
                            stats.fc_prunes += 1
                            if dom[p] == 0:
                                stats.ac3_conflict = True
                                stats.wipeout_cell = p
                                return self._no_solution(
                                    f"forward checking emptied the domain of cell {cell_name(p)}; "
                                    "the given digits contradict each other.", stats)
        else:
            c = find_given_conflict(grid)
            if c:
                a, b, d, u = c
                return self._no_solution(
                    f"cells {cell_name(a)} and {cell_name(b)} both contain {d} in {u}.", stats)

        # ---- stage 2: search
        try:
            if self.use_fc:
                self._search_fc(grid, dom, limit, sols, stats)
            else:
                self._search_bt(grid, dom, limit, sols, stats)
        except _Limit:
            stats.limit_hit = True
            status = "solved" if sols else "limit"
            return SolveResult(status, sols[0] if sols else None,
                               "SOLVED" if sols else f"STOPPED: assignment limit {self.node_limit} reached",
                               stats, sols)
        if sols:
            return SolveResult("solved", sols[0], "SOLVED", stats, sols)
        return self._no_solution(
            f"search exhausted after {stats.assignments} assignments (a cell ran out of legal digits); no placement "
            "satisfies every row, column and box constraint.", stats)

    # -- AC-3 -----------------------------------------------------------------------
    def _ac3(self, dom, stats, seeds=None) -> bool:
        """Enforce arc consistency in place. Returns False if some domain becomes empty.

        Arc (xi, xj): every value of xi needs a supporting value in xj. For a 'different'
        constraint a value v of xi loses support only when xj's domain is exactly {v}.
        """
        q = deque()
        if seeds is None:
            for i in range(81):
                for j in PEERS[i]:
                    q.append((i, j))
        else:
            for c in seeds:
                for k in PEERS[c]:
                    q.append((k, c))
        while q:
            xi, xj = q.popleft()
            stats.ac3_revisions += 1
            dj = dom[xj]
            if dj and (dj & (dj - 1)) == 0 and dom[xi] & dj:    # xj fixed to one value v: remove v from xi
                dom[xi] &= ~dj
                stats.ac3_prunes += 1
                if dom[xi] == 0:
                    stats.wipeout_cell = xi
                    return False
                for xk in PEERS[xi]:
                    if xk != xj:
                        q.append((xk, xi))
        return True

    # -- plain backtracking (optionally restricted by AC-3 domains) ------------------------
    def _search_bt(self, grid, dom0, limit, sols, stats):
        g = list(grid)
        rowm, colm, boxm = [0] * 9, [0] * 9, [0] * 9
        for i, d in enumerate(g):
            if d:
                b = 1 << d
                rowm[ROW[i]] |= b; colm[COL[i]] |= b; boxm[BOX[i]] |= b
        node_limit, mrv = self.node_limit, self.ordering == "mrv"

        def cand(i):
            m = ALL & ~(rowm[ROW[i]] | colm[COL[i]] | boxm[BOX[i]])
            return m & dom0[i] if dom0 is not None else m

        def rec(remaining):
            if not remaining:
                sols.append(list(g))
                return len(sols) >= limit
            if mrv:
                i, m, best = -1, 0, 10
                for c in remaining:
                    mc = cand(c)
                    n = mc.bit_count()
                    if n < best:
                        i, m, best = c, mc, n
                        if n <= 1:
                            break
                rest = [c for c in remaining if c != i]
            else:
                i, rest = remaining[0], remaining[1:]
                m = cand(i)
            r, c_, b_ = ROW[i], COL[i], BOX[i]
            for d in self._digits(m):
                stats.assignments += 1
                if node_limit is not None and stats.assignments > node_limit:
                    raise _Limit
                bit = 1 << d
                g[i] = d; rowm[r] |= bit; colm[c_] |= bit; boxm[b_] |= bit
                stop = rec(rest)
                rowm[r] &= ~bit; colm[c_] &= ~bit; boxm[b_] &= ~bit; g[i] = 0
                if stop:
                    return True
                stats.backtracks += 1
            return False

        rec([i for i, d in enumerate(g) if d == 0])

    # -- backtracking + forward checking (+ MAC) -----------------------------------------
    def _search_fc(self, grid, dom0, limit, sols, stats):
        assigned = [d != 0 for d in grid]
        node_limit, mrv, mac = self.node_limit, self.ordering == "mrv", self.ac3 == "mac"

        def rec(dom):
            i, best = -1, 10
            for c in range(81):
                if not assigned[c]:
                    if not mrv:
                        i = c
                        break
                    n = dom[c].bit_count()
                    if n < best:
                        i, best = c, n
                        if n <= 1:
                            break
            if i < 0:
                sols.append([d.bit_length() - 1 for d in dom])
                return len(sols) >= limit
            for d in self._digits(dom[i]):
                stats.assignments += 1
                if node_limit is not None and stats.assignments > node_limit:
                    raise _Limit
                bit = 1 << d
                nd = dom[:]
                nd[i] = bit
                ok, changed = True, [i]
                for p in PEERS[i]:
                    v = nd[p]
                    if v & bit:
                        v &= ~bit
                        nd[p] = v
                        stats.fc_prunes += 1
                        changed.append(p)
                        if v == 0:
                            ok = False
                            break
                if ok and mac:
                    ok = self._ac3(nd, stats, seeds=changed)
                if ok:
                    assigned[i] = True
                    stop = rec(nd)
                    assigned[i] = False
                    if stop:
                        return True
                stats.backtracks += 1
            return False

        rec(dom0)
