# Data dictionary

**REAL** = taken unchanged from MNIST. **GENERATED** = created by our code (seeded). **DERIVED** = computed from other columns.

## A. `data/processed/mnist19.npz` (made by `make_data.py`)

| Array | Shape / type | Real or generated | Meaning |
|---|---|---|---|
| `X_train`, `X_val`, `X_test` | (N, 784) uint8, 0–255 (scaled to 0–1 when loaded) | REAL | flattened 28×28 MNIST digit images, digit 0 removed |
| `y_train`, `y_val`, `y_test` | (N,) uint8, 1–9 | REAL | true digit labels |
| `source` | string | metadata | which MNIST source was used |
| `split_seed` | int (0) | metadata | seed of the 50 000 / 10 000 train/validation split of the original training set |

Sizes: train 45 053, validation 9 024, test 9 020 (after dropping zeros). Test set is never used for training or for fitting the temperature.

## B. Generated Sudoku objects (in memory; seeds in `config.py`)

| Field | Type | Real/generated | How it is made |
|---|---|---|---|
| `solution` | 81 ints 1–9 (index = 9·row + col) | GENERATED | random full grid by randomised backtracking + forward checking |
| `puzzle` | 81 ints, 0 = blank | GENERATED | cells removed in random order, a removal is kept only if the puzzle still has exactly one solution; stops at 25/30/35 givens |
| `Mosaic.images` | (81, 28, 28) float32 | image content REAL, arrangement GENERATED | for each given: a random MNIST **test** image of the true digit; blank cells: Gaussian noise σ=0.02 clipped at 0 |
| `Mosaic.img_index` | (81,) int | GENERATED | test-set index shown in each cell (−1 = blank) |
| `Mosaic.injected` | list of cell indices | GENERATED | cells showing an injected hard digit (rule below) |
| Hard-digit rule | – | GENERATED | for true digit d: test images of d that the network reads wrongly and whose true digit is in its top-3. 172 such images exist (11–37 per digit). Non-injected cells use correctly-read images only, so the number of misreads is exactly the chosen *m*. |
| Blank rule | – | rule | cell mean brightness < 0.02 → blank (no network involved) |

## C. Reading and search objects (DERIVED)

| Field | Meaning |
|---|---|
| `Reading.probs` (81, 9) | network probabilities for digits 1–9 (temperature-scaled; zeros for blank cells) |
| `Reading.grid` | argmax digit per non-blank cell = "first reading" |
| cost of a change | `ln(p_read / p_new)`, ≥ 0 |
| correction log entry | `(cell, read_digit, new_digit, cost)` |
| search counters | `generated`, `expanded`, `goal_tests` (= CSP calls), `max_frontier`, `pruned_by_h` |
| CSP counters | `assignments`, `backtracks`, `fc_prunes`, `ac3_revisions`, `ac3_prunes` |

## D. Result files (`results/`)

`exp1_raw.csv` (one row per puzzle × misread level × algorithm): `seed, puzzle, misreads, algorithm, status` (solved/unreadable), `exact` (1 if the solution equals the true one), `solvable_wrong` (solved but not the true solution), `expanded, generated, goal_tests, pruned_by_h, n_corrections, total_cost, csp_assignments, seconds, injected` (cells), `corrections` (`cell:read->new(cost);…`).

`exp2_raw.csv`: `seed, puzzle, givens, ordering, method, solved, assignments, backtracks, fc_prunes, ac3_revisions, ac3_prunes, seconds`.

`exp3_raw.csv`: `seed, puzzle, misreads, method, exact, solvable_wrong, unreadable, csp_calls`.

`*_summary.csv`, `*_table.md`: mean and sample standard deviation (ddof = 1) **across the 5 seeds**; each seed contributes one rate/average over its 20 puzzles (10 in Exp 2). `nn_results.json`: test accuracy, ECE (15 bins), NLL, fitted temperature for logistic regression and MLP, raw and temperature-scaled.
