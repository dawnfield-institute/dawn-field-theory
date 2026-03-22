"""
PAC Descent — Noether-informed gradient descent for TinyCIMM.

Uses PAC (Potential-Actualization Conservation) magnitude targets to
regularize weight norms, and a phi-inverse weighted direction signal
propagated via weight transport (not random projections).

Constants
---------
PHI     = golden ratio  ≈ 1.618
PHI_INV = 1 / PHI       ≈ 0.618

Conservation correction schedule
---------------------------------
Applied every CONSERVATION_PERIOD steps (not every step) and only when
the correction would NOT increase the current batch MSE by more than
CONSERVATION_MSE_GATE (5 %).  This prevents conservation drag from
overwhelming learning in early phases.

Direction signal
----------------
Each layer's update direction is weighted by PHI_INV^depth.  The weight
is NOT normalised to sum=1 (doing so would dilute the first hidden
layer's effective LR by ~4×).  For layers more than one hop from the
output, the error is projected via the actual downstream weight matrix
transposed (weight transport) rather than a fresh random projection.
"""

import math
import numpy as np

# ---------------------------------------------------------------------------
# DFT constants
# ---------------------------------------------------------------------------
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1.0 / PHI  # ≈ 0.618

# Conservation correction is applied only every this many steps.
CONSERVATION_PERIOD = 20

# Skip conservation correction if it would raise batch MSE by more than this.
CONSERVATION_MSE_GATE = 0.05  # 5 %


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error between two arrays."""
    diff = y_pred - y_true
    return float(np.mean(diff * diff))


def _forward(weights: list, x: np.ndarray) -> tuple:
    """
    Simple MLP forward pass with tanh activations on hidden layers,
    linear output.

    Returns
    -------
    activations : list of np.ndarray
        Pre-activation outputs at each layer (including input).
    outputs     : list of np.ndarray
        Post-activation outputs (same indexing; last entry is network output).
    """
    activations = [x]
    outputs = [x]
    h = x
    for i, W in enumerate(weights):
        z = h @ W
        activations.append(z)
        if i < len(weights) - 1:  # hidden layers — tanh
            h = np.tanh(z)
        else:  # output layer — linear
            h = z
        outputs.append(h)
    return activations, outputs


# ---------------------------------------------------------------------------
# PAC magnitude target
# ---------------------------------------------------------------------------

def _pac_target_norm(W: np.ndarray) -> float:
    """
    PAC magnitude target for a weight matrix.

    The target Frobenius norm is  PHI_INV * sqrt(in_dim * out_dim),
    which keeps per-weight energy at the PAC golden-ratio attractor.
    """
    in_dim, out_dim = W.shape
    return PHI_INV * math.sqrt(in_dim * out_dim)


# ---------------------------------------------------------------------------
# Conservation correction (Bug 1 fixed)
# ---------------------------------------------------------------------------

def _conservation_correction(
    weights: list,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    step: int,
) -> list:
    """
    Multiplicative weight rescaling toward PAC magnitude targets.

    Applied only on multiples of CONSERVATION_PERIOD.
    Within those steps, skipped if the rescaling would increase the
    current batch MSE by more than CONSERVATION_MSE_GATE.

    Parameters
    ----------
    weights  : list of np.ndarray — current weight matrices
    x_batch  : current mini-batch inputs
    y_batch  : current mini-batch targets
    step     : global training step index (0-based)

    Returns
    -------
    weights (possibly updated in-place, reference returned for clarity)
    """
    if step % CONSERVATION_PERIOD != 0:
        return weights  # not a correction step

    # Compute current MSE before any correction.
    _, outs = _forward(weights, x_batch)
    mse_before = _mse(outs[-1], y_batch)

    new_weights = []
    for W in weights:
        target = _pac_target_norm(W)
        current = float(np.linalg.norm(W, "fro"))
        if current < 1e-12:
            new_weights.append(W)
            continue
        w_factor = target / current
        new_weights.append(W * w_factor)

    # Gate: only accept the correction if MSE does not worsen by > 5 %.
    _, outs_new = _forward(new_weights, x_batch)
    mse_after = _mse(outs_new[-1], y_batch)

    if mse_before > 0 and (mse_after - mse_before) / mse_before > CONSERVATION_MSE_GATE:
        # Correction would hurt — skip it.
        return weights

    # Accept correction: mutate originals.
    for i, W_new in enumerate(new_weights):
        weights[i][:] = W_new
    return weights


# ---------------------------------------------------------------------------
# Direction signal (Bug 2 fixed)
# ---------------------------------------------------------------------------

def _layer_direction(
    weights: list,
    activations: list,
    outputs: list,
    output_error: np.ndarray,
    layer_idx: int,
) -> np.ndarray:
    """
    Compute the update direction for layer `layer_idx`.

    The signal strength is weighted by PHI_INV^depth (depth = distance
    from output layer, 0-indexed), WITHOUT normalising to sum=1.

    For layers more than one hop from the output the error is propagated
    via actual downstream weight matrices transposed (weight transport),
    not a random projection.

    Parameters
    ----------
    weights      : list of weight matrices [W_0, W_1, ..., W_L]
    activations  : pre-activation values from _forward
    outputs      : post-activation values from _forward
    output_error : error at the output layer (y_pred - y_true)
    layer_idx    : which layer's direction to compute (0 = first hidden)

    Returns
    -------
    direction : np.ndarray with shape == weights[layer_idx].shape
    """
    n_layers = len(weights)
    depth = (n_layers - 1) - layer_idx  # hops from output

    # phi-inverse weighting — NOT normalised to sum=1 (Bug 2 fix)
    phi_weight = PHI_INV ** depth

    if depth == 0:
        # Output layer: direct error
        delta = output_error  # shape: (batch, out_dim)
    elif depth == 1:
        # One hop from output: use output layer weights transposed
        W_next = weights[layer_idx + 1]  # shape: (hidden, out)
        out_dim = W_next.shape[1]
        scale = math.sqrt(PHI_INV / out_dim)
        delta = output_error @ (W_next.T * scale)  # shape: (batch, hidden)
    else:
        # Multiple hops: chain weight transport through all downstream layers
        # Start from the output error and propagate back via W.T at each hop.
        delta = output_error
        for k in range(n_layers - 1, layer_idx, -1):
            W_k = weights[k]  # shape: (in, out)
            out_dim = W_k.shape[1]
            scale = math.sqrt(PHI_INV / out_dim)
            delta = delta @ (W_k.T * scale)

    # Gradient w.r.t. W_layer: x^T @ delta
    x_in = outputs[layer_idx]  # post-activation input to this layer
    direction = x_in.T @ delta  # shape: (in_dim, out_dim)
    direction = direction * phi_weight

    return direction


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PACDescent:
    """
    Noether-informed optimizer for a simple MLP.

    Parameters
    ----------
    layer_sizes : sequence of ints, e.g. [in_dim, 64, 32, out_dim]
    lr          : base learning rate
    seed        : random seed for weight initialisation
    """

    def __init__(self, layer_sizes: list, lr: float = 0.03, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.lr = lr
        self.step_count = 0

        # Xavier-like initialisation
        self.weights = []
        for i in range(len(layer_sizes) - 1):
            in_d, out_d = layer_sizes[i], layer_sizes[i + 1]
            scale = math.sqrt(2.0 / (in_d + out_d))
            W = rng.randn(in_d, out_d) * scale
            self.weights.append(W)

    def step(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """
        Perform one PAC-descent update.

        Returns the batch MSE *before* the gradient step (pre-step loss).
        """
        # --- Forward pass ---
        acts, outs = _forward(self.weights, x_batch)

        y_pred = outs[-1]
        mse = _mse(y_pred, y_batch)

        # Output error (for MSE: dL/dy_pred = 2*(y_pred - y_true)/N, drop 2/N
        # constant — absorbed into lr)
        output_error = y_pred - y_batch  # (batch, out_dim)

        # --- Gradient step for each layer ---
        for i in range(len(self.weights)):
            direction = _layer_direction(
                self.weights, acts, outs, output_error, i
            )
            # Gradient clipping — prevents explosion on early steps
            d_norm = float(np.linalg.norm(direction))
            if d_norm > 1.0:
                direction = direction * (1.0 / d_norm)
            self.weights[i] -= self.lr * direction

        # --- Conservation correction (every CONSERVATION_PERIOD steps, gated) ---
        _conservation_correction(
            self.weights, x_batch, y_batch, self.step_count
        )

        self.step_count += 1
        return mse

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return network output for input x."""
        _, outs = _forward(self.weights, x)
        return outs[-1]
