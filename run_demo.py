#!/usr/bin/env python
"""Integrated demo: handwritten Sudoku -> NN reading -> AC-3 -> correction search -> CSP solve.

Usage:  python run_demo.py [--seed 2026] [--misreads 2] [--givens 30] [--algorithm ucs|greedy|astar]
                           [--unique] [--show]
Prints every stage and saves plots to outputs/demo/  (demo_overview.png + one PNG per panel).
"""
import argparse
import random

import matplotlib
import numpy as np

from sudoku_rescue.config import DEMO_OUT, TEAM_SEED
from sudoku_rescue.csp.generator import make_puzzle
from sudoku_rescue.csp.sudoku import SudokuSolver, cell_name
from sudoku_rescue.data.mosaic import DigitBank, build_mosaic, read_mosaic
from sudoku_rescue.integration.pipeline import PipelineConfig, solve_reading, solve_without_correction
from sudoku_rescue.nn.mlp import MLP
from sudoku_rescue.data.mnist import load_mnist19

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=TEAM_SEED)
ap.add_argument("--misreads", type=int, default=2)
ap.add_argument("--givens", type=int, default=30)
ap.add_argument("--algorithm", default="ucs", choices=["ucs", "greedy", "astar"])
ap.add_argument("--unique", action="store_true", help="extension: accept only a UNIQUELY solvable correction")
ap.add_argument("--show", action="store_true", help="open the plot window (default: just save PNGs)")
args = ap.parse_args()
if not args.show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def show_grid(grid, mark=None, title=""):
    mark = mark or {}
    print(title)
    for r in range(9):
        if r % 3 == 0:
            print("  +-------+-------+-------+")
        row = "  "
        for c in range(9):
            if c % 3 == 0:
                row += "| "
            d = grid[r * 9 + c]
            row += (mark.get(r * 9 + c, "") + (str(d) if d else ".")) + " "
        print(row + "|")
    print("  +-------+-------+-------+")


# ---------------------------------------------------------------- 0. build the input
data = load_mnist19()
model, T = MLP.load(__import__("sudoku_rescue.config", fromlist=["MODELS"]).MODELS / "mlp.npz")
bank = DigitBank.build(model, data["X_test"], data["y_test"], T)
puzzle, solution = make_puzzle(random.Random(f"{args.seed}-demo"), args.givens)
mosaic = build_mosaic(puzzle, solution, args.misreads, bank, np.random.default_rng([args.seed, 99]))
reading = read_mosaic(model, mosaic.images, T)
cfg = PipelineConfig(algorithm=args.algorithm, goal="unique" if args.unique else "solvable")

print("=" * 72)
print(f"INPUT: a 252x252 image of a handwritten Sudoku ({args.givens} givens), seed {args.seed}")
print(f"       ({args.misreads} hard handwritten digit(s) were injected at cells "
      f"{', '.join(cell_name(c) for c in mosaic.injected) or 'none'})")
print("=" * 72)
show_grid(puzzle, title="True puzzle (what the writer meant; unknown to the system):")
show_grid(reading.grid, mark={c: "*" for c in range(81) if reading.grid[c] != puzzle[c]},
          title="\nStage 1 - NN reading (* = misread; unknown to the system):")
low = sorted(((reading.probs[i].max(), i) for i in range(81) if not reading.blank[i]))[:4]
print("  least confident cells: " + ", ".join(f"{cell_name(i)}={reading.grid[i]} (p={p:.2f})" for p, i in low))

# ---------------------------------------------------------------- 2. AC-3
res = solve_reading(reading, cfg)
print("\nStage 2 - AC-3 on the first reading:\n  " + res.ac3_message)

# ---------------------------------------------------------------- 3/4. search + CSP
print(f"\nStage 3/4 - {args.algorithm.upper()} proposes correction sets; CSP (BT+FC+AC-3) tests each one")
print("   order  corrections (cell:read->new)             cost g    CSP verdict")
for t in res.search.trace[:14]:
    desc = ", ".join(f"{cell_name(c)}:{reading.grid[c]}->{n}" for c, n in t["corrections"]) or "(none: first reading)"
    print(f"   {t['order']:5d}  {desc:42s} {t['g']:6.3f}    {'SOLVABLE -> goal' if t['goal'] else 'no solution'}")
if len(res.search.trace) > 14:
    print(f"   ... {len(res.search.trace) - 14} more")
s = res.search.stats
print(f"  counters: generated={s.generated} expanded={s.expanded} goal tests (CSP calls)={s.goal_tests}")

print("\nResult: " + res.message)
if res.status == "solved":
    print("  correction log (cell, read digit, new digit, cost):")
    for c, o, n, k in res.corrections:
        truth = "correct" if puzzle[c] == n else "WRONG"
        print(f"    {cell_name(c)}  {o} -> {n}   cost {k:.3f}   ({truth} vs. the true digit {puzzle[c]})")
    show_grid(res.solution, title="\nSolution:")
    print(f"  matches the true solution: {res.solution == solution}")

# ---------------------------------------------------------------- compare search algorithms
print("\nSearch algorithms on this input:")
algo_rows = {}
for a in ("ucs", "greedy", "astar"):
    r = solve_reading(reading, PipelineConfig(algorithm=a, goal=cfg.goal))
    algo_rows[a] = r
    print(f"  {a:6s} expanded={r.search.stats.expanded:4d} generated={r.search.stats.generated:4d} "
          f"cost={r.total_cost:.3f} exact={r.solution == solution}")

# ---------------------------------------------------------------- CSP pruning with / without FC and AC-3
final_grid = list(reading.grid)
for c, o, n, k in res.corrections:
    final_grid[c] = n
hard, _ = make_puzzle(random.Random(f"{args.seed}-hard"), 25)
variants = [("BT", {}), ("BT+FC", dict(use_fc=True)), ("BT+FC+AC3", dict(use_fc=True, ac3="root")),
            ("BT+FC+MAC", dict(ac3="mac"))]
print("\nCSP pruning (static variable order):")
csp_rows = {}
for label, g in (("this puzzle", final_grid), ("a 25-given puzzle", hard)):
    print(f"  {label}:")
    for name, kw in variants:
        r = SudokuSolver(ordering="static", **kw).solve(g)
        csp_rows[(label, name)] = r.stats.assignments
        print(f"    {name:10s} assignments={r.stats.assignments:6d} backtracks={r.stats.backtracks:6d} "
              f"FC prunes={r.stats.fc_prunes:6d} AC-3 prunes={r.stats.ac3_prunes:4d}")

# ---------------------------------------------------------------- NN on vs off
off = solve_without_correction(reading, cfg)
print("\nSystem with NN OFF (trust the first reading, no correction):")
print("  " + (off.message if off.status != "solved" else f"solved, matches truth: {off.solution == solution}"))
print("System with NN ON  (probabilities guide the corrections):")
print("  " + res.message + (f", matches truth: {res.solution == solution}" if res.solution else ""))

# ---------------------------------------------------------------- plots
DEMO_OUT.mkdir(parents=True, exist_ok=True)
fig = plt.figure(figsize=(16, 9.5))
gs = fig.add_gridspec(2, 3)

ax = fig.add_subplot(gs[0, 0])
ax.imshow(mosaic.as_image(), cmap="gray")
for k in range(10):
    ax.axhline(k * 28 - 0.5, color="#4a90d9", lw=1.6 if k % 3 == 0 else 0.4)
    ax.axvline(k * 28 - 0.5, color="#4a90d9", lw=1.6 if k % 3 == 0 else 0.4)
from matplotlib.patches import Rectangle
for c in mosaic.injected:
    ax.add_patch(Rectangle(((c % 9) * 28 - 0.5, (c // 9) * 28 - 0.5), 28, 28, fill=False, ec="red", lw=2.5))
for c, o, n, k in res.corrections:
    ax.add_patch(Rectangle(((c % 9) * 28 + 3 - 0.5, (c // 9) * 28 + 3 - 0.5), 22, 22, fill=False, ec="lime", lw=2, ls="--"))
ax.set_title("1. Input mosaic\nred = injected hard digit, green dashed = corrected by the system", fontsize=10)
ax.axis("off")

ax = fig.add_subplot(gs[0, 1])
tr = res.search.trace
gs_ = [t["g"] for t in tr]
cols = ["#2ca02c" if t["goal"] else "#bbbbbb" for t in tr]
ax.bar(range(1, len(tr) + 1), gs_, color=cols)
ax.set_xlabel(f"order in which {args.algorithm.upper()} tests states (CSP calls)")
ax.set_ylabel("cost g = -ln(P(corrected)/P(read))")
ax.set_title(f"2. Search trace ({args.algorithm.upper()}): grey = CSP says no solution, green = solvable", fontsize=10)

ax = fig.add_subplot(gs[0, 2])
names = ["ucs", "greedy", "astar"]
ax.bar(names, [algo_rows[a].search.stats.expanded for a in names], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
for i, a in enumerate(names):
    tc = algo_rows[a].total_cost
    ax.text(i, algo_rows[a].search.stats.expanded, "no solution" if tc != tc else f"cost {tc:.2f}",
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("nodes expanded"); ax.set_title("3. Search algorithms on this input", fontsize=10)

ax = fig.add_subplot(gs[1, 0])
w = 0.38
shown = [lab for lab in ("this puzzle", "a 25-given puzzle") if lab != "this puzzle" or res.status == "solved"]
for j, label in enumerate(shown):
    off_x = (j - (len(shown) - 1) / 2) * w
    ax.bar(np.arange(len(variants)) + off_x, [max(csp_rows[(label, n)], 0.5) for n, _ in variants], w, label=label)
ax.set_xticks(range(len(variants)), [n for n, _ in variants], fontsize=8); ax.set_yscale("log")
ax.set_ylabel("assignments (log)"); ax.legend(fontsize=8)
ax.set_title("4. CSP pruning: BT vs +FC vs +AC-3 (static order)", fontsize=10)

ax = fig.add_subplot(gs[1, 1])
labels = ["NN off\n(no correction)", "NN on\n(guided correction)"]
vals = [int(off.status == "solved" and off.solution == solution), int(res.solution == solution)]
ax.bar(labels, vals, color=["#d62728" if v == 0 else "#2ca02c" for v in vals])
ax.set_ylim(0, 1.25); ax.set_yticks([0, 1], ["fail", "exact\nsolution"])
def _outcome(status, sol):
    if status != "solved":
        return "no solution\n(UNREADABLE)" if status == "unreadable" else "no solution"
    return "exact solution" if sol == solution else "solvable but WRONG"
notes = [_outcome("solved" if off.status == "solved" else "unreadable", off.solution if off.status == "solved" else None),
         _outcome(res.status, res.solution)]
for i, (v, t) in enumerate(zip(vals, notes)):
    ax.text(i, v + 0.04, t, ha="center", va="bottom", fontsize=9)
ax.set_title("5. System with the network on vs off", fontsize=10)

ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
lines = ["Correction log (cell, read -> new, cost)"] + \
        [f"{cell_name(c)}: {o} -> {n}   cost {k:.3f}" for c, o, n, k in res.corrections] + \
        ["", f"status: {res.status.upper()}", f"search: {s.expanded} expanded, {s.generated} generated",
         f"CSP calls: {res.csp_calls}", f"AC-3 flagged first reading: {res.ac3_flagged}",
         f"exact solution recovered: {res.solution == solution}"]
if res.status == "unreadable":
    lines += ["", "Least confident cells (check these by hand):"] + \
             [f"  {cell_name(c)}: read {reading.grid[c]} (p={p_:.2f})" for c, p_ in res.least_confident]
ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=10)
fig.suptitle("Handwritten Sudoku reader & solver: NN + AC-3 + search + CSP", fontsize=14)
fig.tight_layout()
fig.savefig(DEMO_OUT / "demo_overview.png", dpi=130)
print(f"\nSaved plots to {DEMO_OUT}/demo_overview.png")
if args.show:
    plt.show()
