"""A small NumPy neural network (MLP) with softmax output, trained with Adam.

sizes=[784, 9]            -> multinomial logistic regression (the baseline)
sizes=[784, 256, 128, 9]  -> the MLP classifier

Classes are the digits 1..9 stored as indices 0..8.
"""
import numpy as np


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, sizes, seed=0, l2=0.0, dtype=np.float32):
        self.sizes = list(sizes)
        self.l2 = l2
        self.dtype = dtype
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            scale = np.sqrt(2.0 / n_in) if i < len(sizes) - 2 else 0.01   # He init / small output
            self.W.append((rng.standard_normal((n_in, n_out)) * scale).astype(dtype))
            self.b.append(np.zeros(n_out, dtype=dtype))

    # ---------------------------------------------------------------- forward / backward
    def logits(self, X, cache=None):
        a = X.astype(self.dtype, copy=False)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            if cache is not None:
                cache.append(a)
            z = a @ W + b
            a = np.maximum(z, 0) if i < len(self.W) - 1 else z
        return a

    def predict_proba(self, X, T=1.0):
        """Softmax(logits / T). T=1 is the raw network; T>1 softens, T<1 sharpens."""
        return softmax(self.logits(X).astype(np.float64) / T)

    def predict(self, X):
        return self.logits(X).argmax(axis=1)

    def loss_and_grads(self, X, y):
        cache = []
        z = self.logits(X, cache)
        p = softmax(z.astype(np.float64)).astype(self.dtype)
        n = len(y)
        loss = -np.log(np.maximum(p[np.arange(n), y], 1e-12)).mean()
        loss += 0.5 * self.l2 * sum((W.astype(np.float64) ** 2).sum() for W in self.W)
        d = p.copy()
        d[np.arange(n), y] -= 1
        d /= n
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            a_in = cache[i]
            gW[i] = a_in.T @ d + self.l2 * self.W[i]
            gb[i] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.W[i].T) * (cache[i] > 0)   # cache[i] = ReLU output of layer i-1
        return loss, gW, gb

    # ---------------------------------------------------------------- training
    def fit(self, X, y, X_val, y_val, epochs=20, batch=128, lr=1e-3, seed=0, patience=4, verbose=True):
        """Adam + early stopping on validation loss. Returns the training history."""
        rng = np.random.default_rng(seed)
        mW = [np.zeros_like(w) for w in self.W]; vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        t, b1, b2, eps = 0, 0.9, 0.999, 1e-8
        hist = {"train_loss": [], "val_loss": [], "val_acc": []}
        best, best_params, bad = np.inf, None, 0
        for ep in range(epochs):
            perm = rng.permutation(len(y))
            tot = 0.0
            for s in range(0, len(y), batch):
                idx = perm[s:s + batch]
                loss, gW, gb = self.loss_and_grads(X[idx], y[idx])
                tot += loss * len(idx)
                t += 1
                for i in range(len(self.W)):
                    for p, g, m, v in ((self.W, gW, mW, vW), (self.b, gb, mb, vb)):
                        m[i] = b1 * m[i] + (1 - b1) * g[i]
                        v[i] = b2 * v[i] + (1 - b2) * g[i] ** 2
                        mh = m[i] / (1 - b1 ** t); vh = v[i] / (1 - b2 ** t)
                        p[i] -= (lr * mh / (np.sqrt(vh) + eps)).astype(self.dtype)
            pv = self.predict_proba(X_val)
            vl = -np.log(np.maximum(pv[np.arange(len(y_val)), y_val], 1e-12)).mean()
            va = (pv.argmax(1) == y_val).mean()
            hist["train_loss"].append(tot / len(y)); hist["val_loss"].append(float(vl)); hist["val_acc"].append(float(va))
            if verbose:
                print(f"  epoch {ep + 1:2d}  train loss {tot / len(y):.4f}  val loss {vl:.4f}  val acc {va:.4f}")
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_params = ([w.copy() for w in self.W], [b.copy() for b in self.b])
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print("  early stopping")
                    break
        if best_params is not None:
            self.W, self.b = best_params
        return hist

    # ---------------------------------------------------------------- persistence
    def save(self, path, temperature=1.0):
        arrays = {f"W{i}": w for i, w in enumerate(self.W)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.b)})
        np.savez(path, sizes=np.array(self.sizes), temperature=np.array(temperature), **arrays)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        m = cls(z["sizes"].tolist())
        m.W = [z[f"W{i}"] for i in range(len(m.sizes) - 1)]
        m.b = [z[f"b{i}"] for i in range(len(m.sizes) - 1)]
        return m, float(z["temperature"])
