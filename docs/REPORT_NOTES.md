# Report notes: results, design choices, limits (use these to write YOUR report)

All numbers below come from the shipped `results/` (5 seeds; mean ± std across seeds). Re-run to confirm on your machine.

## 1. Neural network (checklist item 4)

| Model (test set, 9 020 digits 1–9) | Accuracy | ECE (15 bins) | NLL |
|---|---|---|---|
| Logistic regression (baseline), raw | 92.27 % | 0.0204 | 0.2713 |
| Logistic regression, temperature-scaled (T = 0.94) | 92.27 % | 0.0130 | 0.2708 |
| **MLP 784-256-128-9**, raw | **97.92 %** | 0.0071 | 0.0743 |
| **MLP**, temperature-scaled (T = 1.35, fitted on validation) | 97.92 % | **0.0022** | 0.0685 |

Figures: `figures/nn_confusion.png`, `nn_reliability.png`, `nn_training.png`. 188 test digits are misread by the MLP – this is why hard digits must be *injected*.

## 2. Experiment 1 – UCS vs Greedy vs A\* (30 givens, 100 puzzles per misread level = 5 seeds × 20)

| misreads | exact recovery % (UCS / Greedy / A\*) | nodes expanded UCS | Greedy | A\* |
|---|---|---|---|---|
| 0 | 100 / 100 / 100 | 0 | 0 | 0 |
| 1 | 83.0±15.7 (all three) | 15.0±27.1 | 1.0±0.2 | 1.1±0.3 |
| 2 | 69.0 / 68.0 / 69.0 | 17.4±9.4 | 3.1±1.2 | 3.6±0.9 |
| 3 | 53.0 / 54.0 / 53.0 | 76.8±66.2 | 4.7±1.7 | 8.1±2.0 |
| 4 | 0 / 0 / 0 | 607.6±87.6 | 13.7±3.6 | 17.0±3.3 |

* A\* returned **exactly the same cost as UCS in all 448 comparable instances** (the 448 of 500 puzzles that both solved; the other 52 were UNREADABLE for every algorithm), as an admissible heuristic guarantees, while expanding up to ~35× fewer nodes (m = 4: 17 vs 608). Greedy was cheapest in nodes but returned a costlier set in 2 instances.
* The recovery rate is the same for all three because on most puzzles the cheapest solvable set is found by all of them – the algorithms differ in **effort**, not in answer quality here.
* 4 misreads: 0 % recovery by construction (max 3 corrections); the system either says UNREADABLE (48 %) or returns a *solvable but wrong* puzzle (52 %).

## 3. Experiment 2 – BT vs +FC vs +FC+AC-3 (10 puzzles × 5 seeds per givens level, static variable order, cap 200 000 assignments)

| givens | BT | BT+FC | BT+FC+AC-3 | BT+FC+MAC (extra) |
|---|---|---|---|---|
| 25 | 46 084±22 424 (90 % solved within cap) | 12 386±5 790 | 9 190±4 535 | 173±57 |
| 30 | 6 556±6 309 | 1 412±1 057 | 615±480 | 58±4 |
| 35 | 839±496 | 228±108 | 53±5 | 46 |

(assignments, mean ± std.) With **MRV ordering** all variants shrink to roughly 46–360 assignments and FC/AC-3 add almost nothing. Wall-clock: AC-3 at the root costs ~8 000 arc revisions, so for easy 35-given puzzles BT+FC+AC-3 is *slower in time* (1.7 ms) than plain BT (0.8 ms) although it needs fewer assignments. Discuss counters vs time honestly.

## 4. Experiment 3 – is the network useful? (30 givens, 100 puzzles per level)

| misreads | no correction | random correction (300 tries) | NN-guided UCS | + uniqueness goal |
|---|---|---|---|---|
| 1 | 0 % | 24.0±6.5 | 87.0±7.6 | **94.0±2.2** |
| 2 | 0 % | 0 % | 66.0±10.8 | **85.0±7.1** |
| 3 | 0 % | 0 % | 57.0±7.6 | **76.0±7.4** |

(exact recovery.) The uniqueness extension cuts "solvable-but-wrong" answers roughly in half (e.g. 32 % → 13 % at 2 misreads).

## 5. Findings worth a paragraph each in the report

1. **"Solvable" ≠ "correct".** With 30 givens, many wrong corrections still give a solvable (multi-solution) puzzle. The uniqueness goal fixes much of this.
2. **Temperature scaling does not change UCS here.** `ln(p_read/p_new) = (z_read − z_new)/T`: T multiplies *every* cost by the same constant, so the ranking of correction sets is unchanged (raw and calibrated rows in Exp 3 are identical). Calibration would matter if costs also contained a per-correction penalty or a prior on error rates – a natural extension.
3. **A\* heuristic.** Counting conflicts is not a cost; we use disjoint conflicting pairs × the cheapest change of either cell. Dead states (a conflict nobody may fix, or more disjoint conflicts than corrections left) get h = ∞ and are pruned.
4. **Limits.** Only top-2 alternatives and the 10 least confident cells are considered, so a misread whose true digit is ranked ≥ 4th, or whose cell is more confident than ten others, cannot be recovered. Blanks are never "misread". Puzzles are synthetic mosaics of MNIST digits, not photographs. Injected hard digits come from a pool of 172 real misread test images (reused across puzzles). Max 3 corrections.
5. Greedy's priority is the heuristic only; ties are broken by g.

## 6. Still to do by the team (checklist 11–13)

* **Report (≤ 12 pages):** suggested budget – problem & motivation 1 · data & generators 1.5 · CSP design 1.5 · search design (cost, heuristic proof) 2 · NN & calibration 1.5 · integration 1 · experiments 2 · limits & conclusions 1.5.
* **Presentation (10 min) + live demo (5 min):** slide order – problem (1 min) · pipeline figure (1) · NN + calibration (1.5) · CSP techniques (1.5) · search + heuristic (2) · results of 3 experiments (2) · limits (1). Demo script: `python run_demo.py` (search trace and correction log), then `python run_demo.py --misreads 0` (nothing to correct), `python run_demo.py --misreads 4 --seed 1` (the UNREADABLE message with the least confident cells), `python run_demo.py --misreads 4` (default seed: a *solvable-but-wrong* answer – the honest failure mode) and the same with `--unique` (extension catches it), then the CSP counter panel (BT vs FC vs AC-3), then NN-on vs NN-off panel.
* **Individual contribution statements** (≤ 200 words each) and the **git log**: use your own commits; the git history must reflect who actually did what.
* Change `TEAM_SEED`, re-run all commands on your machine with the official MNIST files, and make sure every team member can explain every module (the live demo and questions will test this).
