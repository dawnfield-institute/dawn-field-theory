#!/usr/bin/env python3
"""
Exp 04 — Conservation vs Gradient: Noether vs SGD on structured sequences.
===========================================================================

Hypothesis
----------
PAC-Descent (TinyCIMM-Noether) should outperform vanilla SGD on sequence
prediction tasks whose ground-truth generators follow conservation-compatible
structure (power-law, Fibonacci cascade) because the phi-weighted direction
signal and period-gated conservation correction reduce over-fitting to
batch noise while maintaining energy within PAC targets.

Patterns tested
---------------
1. Power-law cascade   : y_t = t^{-0.5}   (scale-free, energy-conserving)
2. Fibonacci cascade   : y_t = F_t / F_{t-1} mod 1  (recursive ratio series)

Design
------
- Sequence length : 200 steps
- Predict next value from previous 8 (sliding window)
- 5 random seeds per condition
- Metric: final 20-step MSE (after 500 training epochs)
- Compared: PACDescent vs plain SGD (numpy, same architecture)
"""

import json
import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from any cwd
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_NOETHER_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _NOETHER_ROOT not in sys.path:
    sys.path.insert(0, _NOETHER_ROOT)

from pac_descent import PACDescent  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PHI = (1 + math.sqrt(5)) / 2
SEQ_LEN = 200
WINDOW = 8
N_EPOCHS = 500
BATCH_SIZE = 32
LAYER_SIZES = [WINDOW, 32, 16, 1]
LR = 0.003
N_SEEDS = 5
EVAL_LAST_N = 20
GRAD_CLIP = 1.0  # clip gradient norm to prevent explosion


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def power_law_sequence(n: int) -> np.ndarray:
    """y_t = (t+1)^{-0.5}, t = 0 … n-1, normalised to [0,1]."""
    t = np.arange(1, n + 1, dtype=float)
    y = t ** -0.5
    return (y - y.min()) / (y.max() - y.min() + 1e-12)


def fibonacci_cascade_sequence(n: int) -> np.ndarray:
    """
    Fibonacci ratio series: r_t = F_{t+2} / F_{t+1} converges to phi.
    We take the fractional part to keep it bounded, then normalise.
    """
    fibs = [1.0, 1.0]
    while len(fibs) < n + 2:
        fibs.append(fibs[-1] + fibs[-2])
    ratios = [fibs[i + 1] / fibs[i] for i in range(n)]
    y = np.array(ratios) % 1.0
    # After the first ~10 steps the ratio is essentially phi mod 1 ≈ 0.618.
    # Inject small perturbations to create non-trivial dynamics.
    rng = np.random.RandomState(0)
    y += rng.randn(n) * 0.02
    y = np.clip(y, 0, 1)
    return y


def make_windows(seq: np.ndarray, window: int):
    """
    Convert a 1-D sequence into (X, y) pairs for next-step prediction.

    X shape: (N - window, window)
    y shape: (N - window, 1)
    """
    X, y = [], []
    for i in range(len(seq) - window):
        X.append(seq[i : i + window])
        y.append([seq[i + window]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Plain SGD baseline (numpy, no PAC)
# ---------------------------------------------------------------------------

class SGDNet:
    """Minimal MLP trained with vanilla SGD and tanh hidden layers."""

    def __init__(self, layer_sizes: list, lr: float = 0.03, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.lr = lr
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            in_d, out_d = layer_sizes[i], layer_sizes[i + 1]
            scale = math.sqrt(2.0 / (in_d + out_d))
            self.weights.append(rng.randn(in_d, out_d) * scale)
            self.biases.append(np.zeros(out_d))

    def _forward(self, x):
        acts = [x]
        h = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ W + b
            if i < len(self.weights) - 1:
                h = np.tanh(z)
            else:
                h = z  # linear output
            acts.append(h)
        return acts

    def step(self, x_batch, y_batch):
        acts = self._forward(x_batch)
        y_pred = acts[-1]
        mse = float(np.mean((y_pred - y_batch) ** 2))

        # Backprop
        delta = y_pred - y_batch  # output delta
        for i in range(len(self.weights) - 1, -1, -1):
            x_in = acts[i]
            dW = x_in.T @ delta
            db = delta.sum(axis=0)
            # Gradient clipping per layer
            dW_norm = float(np.linalg.norm(dW))
            if dW_norm > GRAD_CLIP:
                dW = dW * (GRAD_CLIP / dW_norm)
            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * db
            if i > 0:
                # Propagate through tanh derivative
                delta = (delta @ self.weights[i].T) * (1 - acts[i] ** 2)
                # Clip propagated delta
                d_norm = float(np.linalg.norm(delta))
                if d_norm > GRAD_CLIP:
                    delta = delta * (GRAD_CLIP / d_norm)
        return mse

    def predict(self, x):
        return self._forward(x)[-1]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_and_eval(model, X_train, y_train, X_eval, y_eval,
                   n_epochs=N_EPOCHS, batch_size=BATCH_SIZE):
    """
    Train model for n_epochs full passes over the training data.
    Returns final evaluation MSE.
    """
    n = len(X_train)
    rng = np.random.RandomState(7)

    for _ in range(n_epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = idx[start:end]
            model.step(X_train[batch_idx], y_train[batch_idx])

    y_pred = model.predict(X_eval)
    return float(np.mean((y_pred - y_eval) ** 2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pattern(pattern_name: str, seq_fn):
    """Run one pattern condition; return dict with per-seed and summary stats."""
    seq = seq_fn(SEQ_LEN)
    X, y = make_windows(seq, WINDOW)

    # Use last EVAL_LAST_N windows for evaluation
    split = len(X) - EVAL_LAST_N
    X_train, y_train = X[:split], y[:split]
    X_eval, y_eval = X[split:], y[split:]

    pac_mses, sgd_mses = [], []

    for seed in range(N_SEEDS):
        pac_model = PACDescent(LAYER_SIZES, lr=LR, seed=seed)
        sgd_model = SGDNet(LAYER_SIZES, lr=LR, seed=seed)

        pac_mse = train_and_eval(pac_model, X_train, y_train, X_eval, y_eval)
        sgd_mse = train_and_eval(sgd_model, X_train, y_train, X_eval, y_eval)

        pac_mses.append(pac_mse)
        sgd_mses.append(sgd_mse)

        print(
            f"  [{pattern_name}] seed={seed}  "
            f"PAC={pac_mse:.6f}  SGD={sgd_mse:.6f}  "
            f"ratio={pac_mse/max(sgd_mse, 1e-12):.3f}"
        )

    return {
        "pattern": pattern_name,
        "pac_mse_per_seed": pac_mses,
        "sgd_mse_per_seed": sgd_mses,
        "pac_mse_mean": float(np.mean(pac_mses)),
        "pac_mse_std": float(np.std(pac_mses)),
        "sgd_mse_mean": float(np.mean(sgd_mses)),
        "sgd_mse_std": float(np.std(sgd_mses)),
        "pac_wins": int(sum(p < s for p, s in zip(pac_mses, sgd_mses))),
        "n_seeds": N_SEEDS,
    }


def main():
    print("=" * 60)
    print("Exp 04 — Conservation vs Gradient (Noether vs SGD)")
    print("=" * 60)

    patterns = [
        ("power_law", power_law_sequence),
        ("fibonacci_cascade", fibonacci_cascade_sequence),
    ]

    results = {
        "experiment": "exp_04_conservation_vs_gradient",
        "config": {
            "seq_len": SEQ_LEN,
            "window": WINDOW,
            "n_epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "layer_sizes": LAYER_SIZES,
            "lr": LR,
            "n_seeds": N_SEEDS,
            "eval_last_n": EVAL_LAST_N,
            "phi": PHI,
        },
        "patterns": {},
    }

    for pname, pfn in patterns:
        print(f"\nPattern: {pname}")
        r = run_pattern(pname, pfn)
        results["patterns"][pname] = r
        print(
            f"  Summary: PAC={r['pac_mse_mean']:.6f}±{r['pac_mse_std']:.6f}  "
            f"SGD={r['sgd_mse_mean']:.6f}±{r['sgd_mse_std']:.6f}  "
            f"PAC wins {r['pac_wins']}/{r['n_seeds']}"
        )

    # Save results
    out_path = os.path.join(_NOETHER_ROOT, "results", "exp_04_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # Quick pass/fail summary
    print("\n" + "=" * 60)
    for pname, r in results["patterns"].items():
        verdict = "PAC < SGD" if r["pac_mse_mean"] < r["sgd_mse_mean"] else "PAC >= SGD"
        print(
            f"  {pname:22s}  Noether={r['pac_mse_mean']:.6f}  "
            f"SGD={r['sgd_mse_mean']:.6f}  [{verdict}]"
        )
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
