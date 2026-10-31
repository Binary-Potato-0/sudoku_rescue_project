# P7 – Handwritten Sudoku Reader and Solver (Search + CSP + Neural Network)

A neural network reads a Sudoku written in handwritten MNIST digits. When it misreads a digit and the
puzzle breaks, **AC-3** detects the contradiction, **search** (UCS / Greedy / A\*) finds the most likely set of
corrections (at most 3), and a **CSP solver** (backtracking, forward checking, AC-3) tests each corrected
puzzle. If nothing works the system reports *UNREADABLE* and lists the least confident cells.

```
mosaic image ─► NN (probabilities) ─► AC-3 on first reading ─► correction SEARCH ─► CSP goal test ─► solution
                     │                                              ▲  cost = ln(p_read / p_new)       │
                     └──────────────────────────────────────────────┘                   or "UNREADABLE" + least confident cells
```

## 1. Environment setup (pinned)

Tested with Python 3.12.3 (3.11+ needed).

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                           # numpy==2.4.4, matplotlib==3.10.8, pytest==9.2.0
```

## 2. Exact run commands (in this order)

```bash
python make_data.py            # download + preprocess MNIST (digits 1-9), fixed seed  -> data/processed/mnist19.npz
python train_nn.py             # trains MLP + logistic-regression baseline, temperature scaling -> models/, results/nn_results.json, figures/nn_*.png
python run_demo.py             # the integrated demo -> terminal trace + outputs/demo/demo_overview.png
python run_experiments.py      # the 3 experiments, 5 seeds -> results/exp*_table.md, results/*.csv, figures/exp*.png (~2 min)
python -m pytest -q            # 51 tests (normal / edge / impossible)
```

Useful demo options: `--misreads 0..4`, `--givens 25|30|35`, `--algorithm ucs|greedy|astar`, `--unique`
(extension: accept only a *uniquely* solvable correction), `--seed N`, `--show` (open plot window).
`python run_experiments.py --quick` is a 10-second smoke run. `--exp 1 2 3` selects experiments.

**Team seed:** edit `TEAM_SEED` in `sudoku_rescue/config.py` – it drives every generated puzzle, mosaic and injected misread.

## 3. Layout

| Path | Content |
|---|---|
| `make_data.py` | download (LeCun page → torchvision mirror → GitHub mirror fallback), split 50k/10k, drop digit 0, save |
| `sudoku_rescue/data/` | `mnist.py` loader · `mosaic.py` mosaic generator, hard-digit injection, blank rule, `read_mosaic` |
| `sudoku_rescue/nn/` | `mlp.py` NumPy MLP/logistic regression (Adam, early stopping) · `metrics.py` accuracy, confusion matrix, ECE, NLL, temperature scaling |
| `sudoku_rescue/csp/` | `sudoku.py` BT / FC / AC-3 / MAC with counters · `generator.py` puzzle generator (unique solution) |
| `sudoku_rescue/search/` | `correction.py` UCS, Greedy, A\* with counters, trace and the admissible heuristic |
| `sudoku_rescue/integration/` | `pipeline.py` the whole chain, correction log, "unreadable" report |
| `sudoku_rescue/evaluation/` | `experiments.py` the three experiments |
| `tests/` | 51 tests |
| `results/`, `figures/`, `outputs/demo/` | generated tables, CSVs, plots |
| `DATA_DICTIONARY.md` | every column/array: real vs generated |
| `docs/REPORT_NOTES.md` | results summary, design choices, limits, report/presentation outline |
| `docs/PUBLISHING_ON_GITHUB.md` | step-by-step: publish and work as a team on GitHub |
| `docs/report/`, `docs/slides/` | put the final report (PDF) and presentation here |
| `CONTRIBUTIONS.md` | individual contribution statements (max 200 words each) |

## 4. Design in one paragraph each

* **CSP.** 81 variables, domains as bit masks, `!=` constraints on rows/columns/boxes. `SudokuSolver(use_fc, ac3, ordering)`:
  `use_fc` = forward checking; `ac3="root"` = AC-3 once before search; `ac3="mac"` = AC-3 after every assignment (extra);
  `ordering="static"|"mrv"`. Counters: assignments, backtracks, FC prunes, AC-3 revisions/prunes.
* **Search.** State = set of corrections; action = change one more uncertain cell to one of its top-2 alternatives;
  cost = `ln(p_read/p_new) ≥ 0`; goal = CSP solvable (or uniquely solvable). Candidates: the 10 least confident cells.
  A\* heuristic = sum over vertex-disjoint duplicate pairs of `min(cheapest change of a, cheapest change of b)` –
  a *cost* lower bound, so A\* stays optimal (verified: 0 cost mismatches vs UCS in 448 instances).
* **NN.** MLP 784-256-128-9 trained on digits 1–9; logistic regression = same code with no hidden layer. Blank cells use a brightness rule.
* **Hard-digit injection rule.** Injected cell = a test image of the true digit that the network reads *wrongly* but whose true digit is in its top-3. All other cells show correctly-read test images.

## 5. Data source note

`make_data.py` tries the official LeCun files first. If those hosts are unreachable it falls back to a GitHub mirror of
the same 60 000 + 10 000 images. Class counts match the official ones. Keep the local copy in `data/raw/`.
If the first-run files differ from the ones used to produce the shipped `results/`, expect small numerical differences.

## 6. Mapping to the common checklist

1 modules ✔ · 2 `make_data.py` ✔ · 3 `DATA_DICTIONARY.md` ✔ · 4 `train_nn.py` + `results/nn_results.json` ✔ ·
5 search counters + known-answer test (`tests/test_search.py`) ✔ · 6 CSP switches + counters ✔ · 7 `run_demo.py` ✔ ·
8 this README ✔ · 9 tests: 51 (≥5 normal, ≥3 edge, ≥2 impossible) ✔ · 10 experiments ✔ ·
**11 report, 12 presentation, 13 contribution statements + git log: to be written by the team** (see `docs/REPORT_NOTES.md`).
