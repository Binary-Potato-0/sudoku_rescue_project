import numpy as np
import pytest

from sudoku_rescue.config import MNIST_FILE, MODELS
from sudoku_rescue.data.mosaic import Reading, read_mosaic
from sudoku_rescue.integration.pipeline import PipelineConfig, solve_reading
from sudoku_rescue.nn.metrics import ece, fit_temperature, nll, softmax_T
from sudoku_rescue.nn.mlp import MLP
from tests.common import PUZZLE, SOLUTION, fake_probs


def test_normal_gradient_check():
    rng = np.random.default_rng(0)
    m = MLP([5, 6, 3], seed=1, l2=0.01, dtype=np.float64)
    X, y = rng.standard_normal((8, 5)), rng.integers(0, 3, 8)
    _, gW, gb = m.loss_and_grads(X, y)
    eps = 1e-6
    for layer in range(2):
        for idx in [(0, 0), (1, 2)]:
            m.W[layer][idx] += eps; lp, _, _ = m.loss_and_grads(X, y)
            m.W[layer][idx] -= 2 * eps; lm, _, _ = m.loss_and_grads(X, y)
            m.W[layer][idx] += eps
            assert (lp - lm) / (2 * eps) == pytest.approx(gW[layer][idx], rel=1e-4, abs=1e-8)


def test_normal_mlp_learns_separable_data_and_beats_chance():
    rng = np.random.default_rng(2)
    centers = np.array([[3, 0], [-3, 0], [0, 3]])
    y = rng.integers(0, 3, 600)
    X = centers[y] + rng.standard_normal((600, 2))
    m = MLP([2, 16, 3], seed=0)
    m.fit(X[:400], y[:400], X[400:], y[400:], epochs=30, lr=1e-2, verbose=False)
    assert (m.predict(X[400:]) == y[400:]).mean() > 0.9


def test_normal_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(3)
    z = rng.standard_normal((6000, 4)); z[:, 0] += 1.0
    p = softmax_T(z, 1.0)
    y = np.array([rng.choice(4, p=row) for row in p])
    over = 3.0 * z                                        # overconfident network
    T = fit_temperature(over, y)
    assert 2.4 < T < 3.6
    assert nll(softmax_T(over, T), y) < nll(softmax_T(over, 1.0), y)
    assert ece(softmax_T(over, T), y) < ece(softmax_T(over, 1.0), y)


def test_normal_pipeline_recovers_two_misreads():
    read = list(PUZZLE); read[0] = 3; read[79] = 9
    P = fake_probs(read, {0: {3: 0.52, 5: 0.45}, 79: {9: 0.55, 7: 0.40}})
    res = solve_reading(Reading(read, P, np.array([d == 0 for d in read])))
    assert res.status == "solved" and res.solution == SOLUTION
    assert res.ac3_flagged and len(res.corrections) == 2 and res.total_cost > 0


def test_normal_pipeline_correction_log_has_cell_read_new_cost():
    read = list(PUZZLE); read[0] = 3
    P = fake_probs(read, {0: {3: 0.52, 5: 0.45}})
    res = solve_reading(Reading(read, P, np.array([d == 0 for d in read])))
    cell, old, new, cost = res.corrections[0]
    assert (cell, old, new) == (0, 3, 5) and cost > 0


def test_edge_all_blank_cells_are_detected_without_calling_the_network():
    class Boom:
        def predict_proba(self, *a, **k):
            raise AssertionError("network must not be called for blank cells")
    rd = read_mosaic(Boom(), np.zeros((81, 28, 28), dtype=np.float32))
    assert rd.grid == [0] * 81 and rd.blank.all()


def test_edge_clean_reading_needs_no_correction():
    res = solve_reading(Reading(list(PUZZLE), fake_probs(PUZZLE), np.array([d == 0 for d in PUZZLE])))
    assert res.status == "solved" and res.corrections == [] and not res.ac3_flagged


def test_impossible_four_misreads_report_unreadable_and_list_least_confident_cells():
    g = [0] * 81
    for r in range(4):
        g[r * 9] = g[r * 9 + 1] = r + 1
    P = fake_probs(g, {i: {g[i]: 0.6, (g[i] % 9) + 1: 0.3} for i in (0, 1, 9, 10, 18, 19, 27, 28)})
    res = solve_reading(Reading(g, P, np.array([d == 0 for d in g])))
    assert res.status == "unreadable" and res.solution is None
    assert res.message.startswith("UNREADABLE") and "Least confident cells" in res.message
    assert len(res.least_confident) > 0


@pytest.mark.skipif(not (MNIST_FILE.exists() and (MODELS / "mlp.npz").exists()),
                    reason="run make_data.py and train_nn.py first")
def test_normal_trained_network_beats_logistic_regression_on_test_set():
    from sudoku_rescue.data.mnist import load_mnist19
    d = load_mnist19()
    mlp, _ = MLP.load(MODELS / "mlp.npz")
    lr, _ = MLP.load(MODELS / "logreg.npz")
    acc = lambda m: (m.predict(d["X_test"]) == d["y_test"] - 1).mean()
    assert acc(mlp) > 0.97 and acc(mlp) > acc(lr) + 0.03
