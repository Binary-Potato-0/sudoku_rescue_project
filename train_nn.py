#!/usr/bin/env python
"""Train the MLP classifier and the logistic-regression baseline on MNIST digits 1-9,
fit a temperature on the validation set, and report test results.

Usage: python train_nn.py [--epochs 25] [--seed 0]
Outputs: models/mlp.npz, models/logreg.npz, results/nn_results.json, figures/nn_*.png
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sudoku_rescue.config import FIGURES, MODELS, RESULTS
from sudoku_rescue.data.mnist import load_mnist19
from sudoku_rescue.nn.metrics import (accuracy, confusion_matrix, ece, fit_temperature, nll,
                                      reliability_bins, softmax_T)
from sudoku_rescue.nn.mlp import MLP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = load_mnist19()
    Xtr, ytr, Xva, yva, Xte, yte = (d["X_train"], d["y_train"] - 1, d["X_val"], d["y_val"] - 1,
                                    d["X_test"], d["y_test"] - 1)
    print(f"train {len(ytr)}  val {len(yva)}  test {len(yte)}")
    for p in (MODELS, RESULTS, FIGURES):
        p.mkdir(exist_ok=True, parents=True)

    models = {
        "logreg": MLP([784, 9], seed=args.seed, l2=1e-4),
        "mlp": MLP([784, 256, 128, 9], seed=args.seed, l2=1e-5),
    }
    results, hists = {}, {}
    for name, m in models.items():
        print(f"== training {name}")
        t0 = time.time()
        hists[name] = m.fit(Xtr, ytr, Xva, yva, epochs=args.epochs, lr=1e-3, seed=args.seed)
        lv, lt = m.logits(Xva), m.logits(Xte)
        T = fit_temperature(lv, yva)
        for tag, TT in (("raw", 1.0), ("tempscaled", T)):
            p = softmax_T(lt, TT)
            results[f"{name}_{tag}"] = {"test_accuracy": accuracy(p, yte), "test_ece": ece(p, yte),
                                        "test_nll": nll(p, yte), "temperature": TT}
        m.save(MODELS / f"{name}.npz", temperature=T)
        print(f"   done in {time.time() - t0:.0f}s, fitted T = {T:.3f}")

    print(f"{'model':22s} {'acc':>7s} {'ECE':>7s} {'NLL':>7s}")
    for k, v in results.items():
        print(f"{k:22s} {v['test_accuracy']:7.4f} {v['test_ece']:7.4f} {v['test_nll']:7.4f}")
    (RESULTS / "nn_results.json").write_text(json.dumps(results, indent=2))

    # ---- figures: confusion matrices, reliability diagrams, learning curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, name in zip(ax, ("logreg", "mlp")):
        cm = confusion_matrix(models[name].predict(Xte), yte)
        a.imshow(cm, cmap="Blues")
        a.set_title(f"{name}: test confusion matrix"); a.set_xlabel("predicted digit"); a.set_ylabel("true digit")
        a.set_xticks(range(9), range(1, 10)); a.set_yticks(range(9), range(1, 10))
        for i in range(9):
            for j in range(9):
                if cm[i, j]:
                    a.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                           color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(FIGURES / "nn_confusion.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    mlp = models["mlp"]; T = results["mlp_tempscaled"]["temperature"]
    for a, (tag, TT) in zip(ax, (("raw (T=1)", 1.0), (f"temperature scaled (T={T:.2f})", T))):
        rows = reliability_bins(softmax_T(mlp.logits(Xte), TT), yte)
        xs = [r[1] for r in rows if r[0]]; ys = [r[2] for r in rows if r[0]]
        a.plot([0, 1], [0, 1], "k--", lw=1); a.plot(xs, ys, "o-")
        a.set_title(f"MLP reliability, {tag}"); a.set_xlabel("confidence"); a.set_ylabel("accuracy")
        a.set_xlim(0.2, 1.02); a.set_ylim(0.2, 1.02)
    fig.tight_layout(); fig.savefig(FIGURES / "nn_reliability.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, h in hists.items():
        ax.plot(h["val_loss"], label=f"{name} val loss"); ax.plot(h["train_loss"], "--", label=f"{name} train loss")
    ax.set_xlabel("epoch"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "nn_training.png", dpi=130)
    print("saved models/, results/nn_results.json, figures/nn_*.png")


if __name__ == "__main__":
    main()
