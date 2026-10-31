#!/usr/bin/env python
"""Run the three experiments. Usage: python run_experiments.py [--exp 1 2 3] [--seeds 5] [--quick]
Outputs: results/exp*_raw.csv, results/exp*_summary.csv / *_table.md, figures/exp*.png"""
import argparse
import time

from sudoku_rescue.config import FIGURES, RESULTS
from sudoku_rescue.evaluation import experiments as E

ap = argparse.ArgumentParser()
ap.add_argument("--exp", type=int, nargs="+", default=[1, 2, 3])
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--quick", action="store_true", help="tiny smoke run (2 seeds, 4 puzzles)")
a = ap.parse_args()
RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
seeds, per_seed = (2, 4) if a.quick else (a.seeds, 20)
model, T, bank = E.load_context()
for e in a.exp:
    t0 = time.time()
    print(f"\n===== Experiment {e} =====")
    if e == 1:
        E.experiment1(model, T, bank, seeds=seeds, per_seed=per_seed)
    elif e == 2:
        E.experiment2(seeds=seeds, per_seed=4 if a.quick else 10)
    elif e == 3:
        E.experiment3(model, T, bank, seeds=seeds, per_seed=per_seed)
    print(f"(experiment {e} took {time.time() - t0:.0f}s)")
