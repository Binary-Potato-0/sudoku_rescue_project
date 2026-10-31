"""The three experiments. Each runs over several seeds; we report mean and std ACROSS SEEDS
(each seed contributes one number: a rate or an average over its puzzles).

Exp 1  UCS vs Greedy vs A*          : nodes + exact recovery for 0-4 injected misreads
Exp 2  BT vs BT+FC vs BT+FC+AC-3    : counters on puzzles with 25 / 30 / 35 givens
Exp 3  NN-guided vs none vs random  : exact recovery, solvable-but-wrong, unreadable
"""
from __future__ import annotations

import csv
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sudoku_rescue.config import FIGURES, RESULTS, TEAM_SEED
from sudoku_rescue.csp.generator import make_puzzle
from sudoku_rescue.csp.sudoku import SudokuSolver
from sudoku_rescue.data.mnist import load_mnist19
from sudoku_rescue.data.mosaic import DigitBank, build_mosaic, read_mosaic
from sudoku_rescue.integration.pipeline import GoalChecker, PipelineConfig, solve_reading, solve_without_correction
from sudoku_rescue.nn.mlp import MLP

COLORS = {"ucs": "#1f77b4", "greedy": "#ff7f0e", "astar": "#2ca02c"}


# ------------------------------------------------------------------------------ helpers
def load_context():
    d = load_mnist19()
    model, T = MLP.load(Path(__file__).resolve().parents[2] / "models" / "mlp.npz")
    bank = DigitBank.build(model, d["X_test"], d["y_test"], T)
    return model, T, bank


def puzzle_set(seed, n_givens, count):
    rng = random.Random(f"{TEAM_SEED}-puzzles-{seed}-{n_givens}")
    return [make_puzzle(rng, n_givens) for _ in range(count)]


def write_csv(path, rows):
    RESULTS.mkdir(exist_ok=True, parents=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def mean_std_over_seeds(rows, keys, value):
    """Group rows by `keys`; per seed average `value`; return {group: (mean, std, n_seeds)}."""
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per[tuple(r[k] for k in keys)][r["seed"]].append(float(r[value]))
    out = {}
    for g, seeds in per.items():
        vals = np.array([np.mean(v) for v in seeds.values()])
        out[g] = (float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, len(vals))
    return out


def fmt(ms, pct=False):
    m, s, _ = ms
    return f"{100 * m:.1f} ± {100 * s:.1f}" if pct else f"{m:.1f} ± {s:.1f}"


def corr_str(corrections):
    return ";".join(f"{c}:{o}->{n}({k:.2f})" for c, o, n, k in corrections)


# ------------------------------------------------------------------------------ experiment 1
def experiment1(model, T, bank, seeds=5, per_seed=20, givens=30, levels=range(5), node_limit=5000):
    rows = []
    for s in range(seeds):
        puzzles = puzzle_set(s, givens, per_seed)
        for m in levels:
            nrng = np.random.default_rng([TEAM_SEED, 1, s, m])
            for pid, (p, sol) in enumerate(puzzles):
                mo = build_mosaic(p, sol, m, bank, nrng)
                rd = read_mosaic(model, mo.images, T)
                for algo in ("ucs", "greedy", "astar"):
                    t0 = time.perf_counter()
                    r = solve_reading(rd, PipelineConfig(algorithm=algo, node_limit=node_limit))
                    dt = time.perf_counter() - t0
                    st = r.search.stats
                    rows.append(dict(seed=s, puzzle=pid, misreads=m, algorithm=algo, status=r.status,
                                     exact=int(r.solution == sol), solvable_wrong=int(r.status == "solved" and r.solution != sol),
                                     expanded=st.expanded, generated=st.generated, goal_tests=st.goal_tests,
                                     pruned_by_h=st.pruned_by_h, n_corrections=len(r.corrections),
                                     total_cost=r.total_cost, csp_assignments=r.csp_stats.assignments,
                                     seconds=round(dt, 4), injected=";".join(map(str, mo.injected)),
                                     corrections=corr_str(r.corrections)))
    write_csv(RESULTS / "exp1_raw.csv", rows)

    # A* must be optimal: same cost as UCS whenever both succeed
    by = defaultdict(dict)
    for r in rows:
        by[(r["seed"], r["puzzle"], r["misreads"])][r["algorithm"]] = r
    mism = sum(1 for d in by.values() if d["ucs"]["status"] == "solved" and d["astar"]["status"] == "solved"
               and abs(float(d["ucs"]["total_cost"]) - float(d["astar"]["total_cost"])) > 1e-9)
    greedy_worse = sum(1 for d in by.values() if d["ucs"]["status"] == "solved" and d["greedy"]["status"] == "solved"
                       and float(d["greedy"]["total_cost"]) > float(d["ucs"]["total_cost"]) + 1e-9)
    total = len(by)

    algos = ("ucs", "greedy", "astar")
    summ = []
    lines = ["| misreads | algorithm | exact recovery % | solvable-but-wrong % | unreadable % | nodes expanded | nodes generated | CSP calls (goal tests) |",
             "|---|---|---|---|---|---|---|---|"]
    stats = {k: mean_std_over_seeds(rows, ("misreads", "algorithm"), k)
             for k in ("exact", "solvable_wrong", "expanded", "generated", "goal_tests")}
    for r in rows:
        r["unreadable"] = int(r["status"] != "solved")
    stats["unreadable"] = mean_std_over_seeds(rows, ("misreads", "algorithm"), "unreadable")
    for m in levels:
        for a in algos:
            k = (m, a)
            lines.append(f"| {m} | {a} | {fmt(stats['exact'][k], True)} | {fmt(stats['solvable_wrong'][k], True)} | "
                         f"{fmt(stats['unreadable'][k], True)} | {fmt(stats['expanded'][k])} | {fmt(stats['generated'][k])} | "
                         f"{fmt(stats['goal_tests'][k])} |")
            summ.append(dict(misreads=m, algorithm=a, **{f"{n}_mean": stats[n][k][0] for n in stats},
                             **{f"{n}_std": stats[n][k][1] for n in stats}))
    write_csv(RESULTS / "exp1_summary.csv", summ)
    note = (f"\nA* vs UCS cost mismatches (both solved): {mism} of {total} instances (must be 0 - A* is optimal).\n"
            f"Greedy returned a costlier correction set than UCS in {greedy_worse} instances.\n")
    (RESULTS / "exp1_table.md").write_text("\n".join(lines) + "\n" + note)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    lv = list(levels)
    for a in algos:
        for axi, key, ylab in ((0, "exact", "exact solution recovery (%)"), (1, "expanded", "nodes expanded")):
            mu = np.array([stats[key][(m, a)][0] for m in lv]) * (100 if key == "exact" else 1)
            sd = np.array([stats[key][(m, a)][1] for m in lv]) * (100 if key == "exact" else 1)
            off = {"ucs": -0.04, "greedy": 0, "astar": 0.04}[a]
            ax[axi].errorbar(np.array(lv) + off, mu, yerr=[np.minimum(sd, mu), sd], marker="o", capsize=3,
                             label=a, color=COLORS[a])
            ax[axi].set_xlabel("injected misreads"); ax[axi].set_ylabel(ylab); ax[axi].set_xticks(lv)
    ax[1].set_yscale("symlog", linthresh=1); ax[1].set_ylim(bottom=0); ax[0].legend(); ax[0].set_ylim(-3, 103)
    ax[0].set_title("Exp 1: recovery rate (mean ± std over seeds)"); ax[1].set_title("Exp 1: search effort")
    fig.tight_layout(); fig.savefig(FIGURES / "exp1_search_comparison.png", dpi=140); plt.close(fig)
    print("\n".join(lines) + note)


# ------------------------------------------------------------------------------ experiment 2
def experiment2(seeds=5, per_seed=10, givens_levels=(25, 30, 35), node_limit=200_000):
    variants = [("BT", dict()), ("BT+FC", dict(use_fc=True)), ("BT+FC+AC3", dict(use_fc=True, ac3="root")),
                ("BT+FC+MAC", dict(ac3="mac"))]
    rows = []
    for ordering in ("static", "mrv"):
        for s in range(seeds):
            for g in givens_levels:
                for pid, (p, sol) in enumerate(puzzle_set(s, g, per_seed)):
                    for name, kw in variants:
                        r = SudokuSolver(ordering=ordering, node_limit=node_limit, **kw).solve(p)
                        assert r.status != "solved" or r.solution == sol
                        st = r.stats
                        rows.append(dict(seed=s, puzzle=pid, givens=g, ordering=ordering, method=name,
                                         solved=int(r.status == "solved"), assignments=st.assignments,
                                         backtracks=st.backtracks, fc_prunes=st.fc_prunes,
                                         ac3_revisions=st.ac3_revisions, ac3_prunes=st.ac3_prunes,
                                         seconds=round(st.seconds, 5)))
    write_csv(RESULTS / "exp2_raw.csv", rows)
    md = []
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    for oi, ordering in enumerate(("static", "mrv")):
        sub = [r for r in rows if r["ordering"] == ordering]
        md.append(f"\n**Variable ordering: {ordering}** (mean ± std over {seeds} seeds, {per_seed} puzzles per seed)\n")
        md += ["| givens | method | assignments | backtracks | FC prunes | AC-3 revisions | AC-3 prunes | time (ms) | solved % |",
               "|---|---|---|---|---|---|---|---|---|"]
        st = {k: mean_std_over_seeds(sub, ("givens", "method"), k)
              for k in ("assignments", "backtracks", "fc_prunes", "ac3_revisions", "ac3_prunes", "seconds", "solved")}
        for g in givens_levels:
            for name, _ in variants:
                k = (g, name)
                t = (st["seconds"][k][0] * 1000, st["seconds"][k][1] * 1000, 0)
                md.append(f"| {g} | {name} | {fmt(st['assignments'][k])} | {fmt(st['backtracks'][k])} | {fmt(st['fc_prunes'][k])} | "
                          f"{fmt(st['ac3_revisions'][k])} | {fmt(st['ac3_prunes'][k])} | {fmt(t)} | {fmt(st['solved'][k], True)} |")
        for ci, key in enumerate(("assignments", "backtracks")):
            a = ax[oi][ci]
            w = 0.2
            for vi, (name, _) in enumerate(variants):
                mu = [st[key][(g, name)][0] for g in givens_levels]
                sd = [st[key][(g, name)][1] for g in givens_levels]
                a.bar(np.arange(3) + (vi - 1.5) * w, mu, w, yerr=sd, capsize=2, label=name)
            a.set_xticks(range(3), [f"{g} givens" for g in givens_levels]); a.set_yscale("log")
            a.set_ylabel(key); a.set_title(f"Exp 2: {key} ({ordering} ordering)")
            if oi == 0 and ci == 0:
                a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "exp2_csp_techniques.png", dpi=140); plt.close(fig)
    (RESULTS / "exp2_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


# ------------------------------------------------------------------------------ experiment 3
def random_correction(read_grid, checker, rng, budget, max_corr=3):
    """Baseline: try random correction sets (random cells, random new digits) until one is solvable."""
    givens = [i for i, d in enumerate(read_grid) if d]
    sol = checker(list(read_grid))            # try the first reading as is (same start as every other method)
    if sol is not None:
        return sol
    for _ in range(budget - 1):
        k = rng.randint(1, max_corr)
        g = list(read_grid)
        for c in rng.sample(givens, k):
            g[c] = rng.choice([d for d in range(1, 10) if d != read_grid[c]])
        sol = checker(g)
        if sol is not None:
            return sol
    return None


def experiment3(model, T, bank, seeds=5, per_seed=20, givens=30, levels=range(5), random_budget=300):
    methods = ["no correction", "random correction", "NN-guided UCS (raw probs)",
               "NN-guided UCS (calibrated)", "NN-guided UCS (calibrated + unique)"]
    rows = []
    for s in range(seeds):
        puzzles = puzzle_set(s, givens, per_seed)
        for m in levels:
            nrng = np.random.default_rng([TEAM_SEED, 3, s, m])
            rrng = random.Random(f"{TEAM_SEED}-random-{s}-{m}")
            for pid, (p, sol) in enumerate(puzzles):
                mo = build_mosaic(p, sol, m, bank, nrng)
                rd_cal = read_mosaic(model, mo.images, T)
                rd_raw = read_mosaic(model, mo.images, 1.0)
                out = {}
                r0 = solve_without_correction(rd_cal)
                out["no correction"] = (r0.solution, 1)
                chk = GoalChecker(PipelineConfig())
                out["random correction"] = (random_correction(rd_cal.grid, chk, rrng, random_budget), chk.calls)
                for name, rd, cfg in (("NN-guided UCS (raw probs)", rd_raw, PipelineConfig()),
                                      ("NN-guided UCS (calibrated)", rd_cal, PipelineConfig()),
                                      ("NN-guided UCS (calibrated + unique)", rd_cal, PipelineConfig(goal="unique"))):
                    r = solve_reading(rd, cfg)
                    out[name] = (r.solution, r.csp_calls)
                for name in methods:
                    solu, calls = out[name]
                    rows.append(dict(seed=s, puzzle=pid, misreads=m, method=name,
                                     exact=int(solu == sol), solvable_wrong=int(solu is not None and solu != sol),
                                     unreadable=int(solu is None), csp_calls=calls))
    write_csv(RESULTS / "exp3_raw.csv", rows)
    lines = ["| misreads | method | exact recovery % | solvable-but-wrong % | unreadable % | CSP calls |", "|---|---|---|---|---|---|"]
    st = {k: mean_std_over_seeds(rows, ("misreads", "method"), k) for k in ("exact", "solvable_wrong", "unreadable", "csp_calls")}
    summ = []
    for m in levels:
        for name in methods:
            k = (m, name)
            lines.append(f"| {m} | {name} | {fmt(st['exact'][k], True)} | {fmt(st['solvable_wrong'][k], True)} | "
                         f"{fmt(st['unreadable'][k], True)} | {fmt(st['csp_calls'][k])} |")
            summ.append(dict(misreads=m, method=name, **{f"{n}_mean": st[n][k][0] for n in st}, **{f"{n}_std": st[n][k][1] for n in st}))
    write_csv(RESULTS / "exp3_summary.csv", summ)
    (RESULTS / "exp3_table.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    lv = list(levels); w = 0.16
    for mi, name in enumerate(methods):
        mu = [100 * st["exact"][(m, name)][0] for m in lv]
        sd = [100 * st["exact"][(m, name)][1] for m in lv]
        ax.bar(np.arange(len(lv)) + (mi - 2) * w, mu, w, yerr=sd, capsize=2, label=name)
    ax.set_xticks(range(len(lv)), [f"{m} misreads" for m in lv]); ax.set_ylabel("exact solution recovery (%)")
    ax.set_title(f"Exp 3: does the network help? (mean ± std over {seeds} seeds)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "exp3_nn_vs_baselines.png", dpi=140); plt.close(fig)
    print("\n".join(lines))
