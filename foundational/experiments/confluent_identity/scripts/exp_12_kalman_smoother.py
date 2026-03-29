"""
exp_12_kalman_smoother.py -- Confluent Identity Phase 4

PURPOSE:
    Replace the ad-hoc exponential backward smoother (exp_06) with a proper
    Kalman filter + Rauch-Tung-Striebel (RTS) smoother. This gives principled
    revision magnitudes with uncertainty estimates.

STATE SPACE MODEL:
    State: x_t = spectral coefficients at time t (R^k)
    Process: x_{t+1} = x_t + w_t,  w ~ N(0, Q)
    Observation: y_t = x_t + v_t,  v ~ N(0, R)

    Q estimated from forward pass: Q = Cov(x_{t+1} - x_t)
    R = 0.1 * Q (observation noise is 10% of process noise)

COMPARISON:
    - RTS revision magnitudes vs exponential smoother (exp_06 style)
    - Uncertainty estimates: 95% of revisions within 2-sigma bounds
    - Temporal gradient preserved

Planck units throughout.
"""

import numpy as np
import json
from datetime import datetime
from scipy import sparse
from scipy.sparse.linalg import eigsh

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _shared import RESULTS_DIR, build_lattice_adjacency
from exp_01_lattice_fluid_baseline import PeriodicLatticeFluid


K_MODES = 10


def compute_identity_fast(C_flat, adjacency, k=K_MODES):
    """Compute spectral identity (reused from exp_06 pattern)."""
    n = len(C_flat)
    W = adjacency
    degrees = np.array(W.sum(axis=1)).ravel()
    L = sparse.diags(degrees) - W

    k_actual = min(k + 1, n - 1)
    eigenvalues, eigvecs = eigsh(L.astype(float), k=k_actual, which='SM',
                                  tol=1e-8, maxiter=5000)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigvecs = eigvecs[:, idx]

    harmonic = float(np.mean(C_flat))
    state_centered = C_flat - harmonic
    coeffs = np.array([float(np.dot(state_centered, eigvecs[:, i]))
                        for i in range(eigvecs.shape[1])])

    return coeffs


def kalman_filter(observations, F, H, Q, R, x0, P0):
    """
    Standard Kalman filter forward pass.

    Args:
        observations: list of y_t vectors
        F: state transition matrix (k x k)
        H: observation matrix (k x k, usually identity)
        Q: process noise covariance
        R: observation noise covariance
        x0: initial state estimate
        P0: initial state covariance

    Returns:
        x_filtered: list of filtered state estimates
        P_filtered: list of filtered covariances
        x_predicted: list of predicted state estimates
        P_predicted: list of predicted covariances
    """
    k = len(x0)
    n = len(observations)

    x_filt = []
    P_filt = []
    x_pred = []
    P_pred = []

    x = x0.copy()
    P = P0.copy()

    for t in range(n):
        # Predict
        x_minus = F @ x
        P_minus = F @ P @ F.T + Q
        x_pred.append(x_minus.copy())
        P_pred.append(P_minus.copy())

        # Update
        y = observations[t]
        innovation = y - H @ x_minus
        S = H @ P_minus @ H.T + R
        K = P_minus @ H.T @ np.linalg.inv(S)

        x = x_minus + K @ innovation
        P = (np.eye(k) - K @ H) @ P_minus

        x_filt.append(x.copy())
        P_filt.append(P.copy())

    return x_filt, P_filt, x_pred, P_pred


def rts_smoother(x_filt, P_filt, x_pred, P_pred, F):
    """
    Rauch-Tung-Striebel backward smoother.

    Returns:
        x_smooth: list of smoothed state estimates
        P_smooth: list of smoothed covariances
    """
    n = len(x_filt)
    k = len(x_filt[0])

    x_smooth = [None] * n
    P_smooth = [None] * n

    x_smooth[-1] = x_filt[-1].copy()
    P_smooth[-1] = P_filt[-1].copy()

    for t in range(n - 2, -1, -1):
        # Smoother gain
        P_pred_inv = np.linalg.inv(P_pred[t + 1])
        G = P_filt[t] @ F.T @ P_pred_inv

        # Smooth
        x_smooth[t] = x_filt[t] + G @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + G @ (P_smooth[t + 1] - P_pred[t + 1]) @ G.T

    return x_smooth, P_smooth


def exponential_smoother(observations, endpoint, alpha=0.3):
    """
    Simple exponential smoother (exp_06 method) for comparison.
    I_smooth(t) = alpha * I_forward(t) + (1-alpha) * I_smooth(t+1)
    """
    n = len(observations)
    k = len(endpoint)
    smoothed = [None] * n

    current = endpoint.copy()
    for t in range(n - 1, -1, -1):
        obs = observations[t]
        min_len = min(len(obs), len(current))
        blended = alpha * obs[:min_len] + (1 - alpha) * current[:min_len]
        current = blended
        smoothed[t] = blended.copy()

    return smoothed


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 4, Experiment 12")
    print("Kalman + RTS Smoother: Principled Backward Revision")
    print("=" * 70)

    # Step 1: Run dynamics and record identity coefficients
    print("\nRunning dynamics and recording identity history...")
    fluid = PeriodicLatticeFluid(
        N=128, total_value=100.0, seed=42,
        n_large_stones=12, n_small_stones=40, gravity=0.005
    )

    C_init = fluid.C
    adjacency = build_lattice_adjacency(C_init)

    MAX_STEPS = 2000
    SAMPLE_INTERVAL = 100
    observations = []
    steps_recorded = []

    for step in range(MAX_STEPS):
        fluid.fluid_step(dt=0.005, viscosity=0.05, sec_threshold=0.1)

        if step % SAMPLE_INTERVAL == 0:
            C = fluid.C
            coeffs = compute_identity_fast(C.ravel(), adjacency)
            observations.append(coeffs)
            steps_recorded.append(step)

            if step % 500 == 0:
                print(f"  Step {step}: ||coeffs||={np.linalg.norm(coeffs):.6f}")

    # Final snapshot
    C_final = fluid.C
    coeffs_final = compute_identity_fast(C_final.ravel(), adjacency)
    observations.append(coeffs_final)
    steps_recorded.append(MAX_STEPS)

    n_obs = len(observations)
    k = len(observations[0])
    print(f"\n  Recorded {n_obs} snapshots, state dimension k={k}")

    # Step 2: Insert perturbation (same as exp_06)
    print("\nInserting perturbation...")
    from scipy.ndimage import uniform_filter, binary_dilation

    grad_x = (np.roll(C_final, -1, axis=0) - np.roll(C_final, 1, axis=0)) / 2.0
    grad_y = (np.roll(C_final, -1, axis=1) - np.roll(C_final, 1, axis=1)) / 2.0
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_smooth = uniform_filter(grad_mag, size=5)
    stone_buffer = binary_dilation(fluid.stone_mask, iterations=8)
    candidate = ~fluid.stone_mask & ~stone_buffer
    grad_candidate = grad_smooth.copy()
    grad_candidate[~candidate] = 0
    high_pos = np.unravel_index(grad_candidate.argmax(), grad_candidate.shape)

    # Apply perturbation
    N = fluid.N
    P_new, A_new = fluid.P.copy(), fluid.A.copy()
    x = np.arange(N)
    X, Y = np.meshgrid(x, x)
    dy = np.minimum(np.abs(Y - high_pos[0]), N - np.abs(Y - high_pos[0]))
    dx = np.minimum(np.abs(X - high_pos[1]), N - np.abs(X - high_pos[1]))
    obstacle_mask = np.sqrt(dx**2 + dy**2) < 3
    obstacle_C = (P_new + A_new).mean() * 3.0
    total_before = P_new.sum() + A_new.sum()
    P_new[obstacle_mask] = 0.1 * obstacle_C
    A_new[obstacle_mask] = 0.9 * obstacle_C
    diff = (P_new.sum() + A_new.sum()) - total_before
    fluid_cells = ~fluid.stone_mask & ~obstacle_mask
    n_fluid = fluid_cells.sum()
    if n_fluid > 0:
        P_new[fluid_cells] -= diff * 0.7 / n_fluid
        A_new[fluid_cells] -= diff * 0.3 / n_fluid

    C_post = P_new + A_new
    adj_post = build_lattice_adjacency(C_post)
    coeffs_post = compute_identity_fast(C_post.ravel(), adj_post)
    print(f"  Post-perturbation: ||coeffs||={np.linalg.norm(coeffs_post):.6f}")

    # Step 3: Estimate noise parameters from forward pass
    print("\nEstimating noise parameters...")
    diffs = []
    for t in range(1, n_obs):
        min_len = min(len(observations[t]), len(observations[t-1]))
        d = observations[t][:min_len] - observations[t-1][:min_len]
        diffs.append(d)

    diffs = np.array(diffs)
    Q = np.cov(diffs.T) if diffs.shape[0] > 1 else np.eye(k) * 1e-6
    # Regularize Q to ensure positive definite
    Q = Q + np.eye(k) * 1e-10
    R = 0.1 * Q  # observation noise is 10% of process noise

    print(f"  Q trace: {np.trace(Q):.6e}")
    print(f"  R trace: {np.trace(R):.6e}")

    # Align all observations to same length
    min_k = min(len(o) for o in observations)
    obs_aligned = [o[:min_k] for o in observations]
    coeffs_post_aligned = coeffs_post[:min_k]
    k = min_k

    # Resize Q and R if needed
    Q = Q[:k, :k]
    R = R[:k, :k]

    # Step 4: Kalman filter + RTS smoother
    print(f"\n{'=' * 70}")
    print("Kalman Filter + RTS Smoother")
    print(f"{'=' * 70}")

    F = np.eye(k)     # state transition: random walk
    H = np.eye(k)     # observation: direct
    x0 = obs_aligned[0]
    P0 = Q * 10       # uncertain initial state

    # Append post-perturbation as final observation
    obs_with_perturbation = obs_aligned + [coeffs_post_aligned]
    steps_with_perturbation = steps_recorded + [MAX_STEPS + 1]

    x_filt, P_filt, x_pred, P_pred = kalman_filter(
        obs_with_perturbation, F, H, Q, R, x0, P0
    )
    x_smooth, P_smooth = rts_smoother(x_filt, P_filt, x_pred, P_pred, F)

    # Step 5: Compare RTS vs exponential smoother
    print(f"\n{'=' * 70}")
    print("Comparison: RTS vs Exponential Smoother")
    print(f"{'=' * 70}")

    exp_smoothed = exponential_smoother(obs_aligned, coeffs_post_aligned, alpha=0.3)

    print(f"\n  {'Step':<8} {'||RTS rev||':<14} {'||Exp rev||':<14} "
          f"{'RTS 2sig':<12} {'Actual in 2sig?':<16}")
    print(f"  {'-'*8} {'-'*14} {'-'*14} {'-'*12} {'-'*16}")

    rts_revisions = []
    exp_revisions = []
    within_2sigma = []

    for t in range(n_obs):  # exclude the post-perturbation point
        # RTS revision: smoothed - filtered
        rts_rev = np.linalg.norm(x_smooth[t] - x_filt[t])
        rts_revisions.append(rts_rev)

        # Exponential revision
        min_len = min(len(exp_smoothed[t]), len(obs_aligned[t]))
        exp_rev = np.linalg.norm(exp_smoothed[t][:min_len] - obs_aligned[t][:min_len])
        exp_revisions.append(exp_rev)

        # 2-sigma bound from smoother covariance
        two_sigma = 2.0 * np.sqrt(np.trace(P_smooth[t]))

        # Is actual revision within 2-sigma?
        in_bound = rts_rev < two_sigma
        within_2sigma.append(in_bound)

        if t % 5 == 0:  # print every 5th for readability
            print(f"  {steps_recorded[t]:<8d} {rts_rev:<14.6f} {exp_rev:<14.6f} "
                  f"{two_sigma:<12.6f} {'Yes' if in_bound else 'NO':<16}")

    # Step 6: Aggregate metrics
    print(f"\n{'=' * 70}")
    print("Aggregate Metrics")
    print(f"{'=' * 70}")

    rts_arr = np.array(rts_revisions)
    exp_arr = np.array(exp_revisions)

    print(f"\n  RTS revision:  mean={rts_arr.mean():.6f}, max={rts_arr.max():.6f}")
    print(f"  Exp revision:  mean={exp_arr.mean():.6f}, max={exp_arr.max():.6f}")

    # Temporal gradient
    n_half = len(rts_revisions) // 2
    rts_early = np.mean(rts_revisions[:n_half])
    rts_late = np.mean(rts_revisions[n_half:])
    exp_early = np.mean(exp_revisions[:n_half])
    exp_late = np.mean(exp_revisions[n_half:])

    print(f"\n  Temporal gradient:")
    print(f"    RTS:  early={rts_early:.6f}, late={rts_late:.6f}, "
          f"ratio={rts_late/rts_early if rts_early > 1e-15 else 0:.2f}x")
    print(f"    Exp:  early={exp_early:.6f}, late={exp_late:.6f}, "
          f"ratio={exp_late/exp_early if exp_early > 1e-15 else 0:.2f}x")

    # Cosine similarity (meaning rotation)
    rts_cosines = []
    exp_cosines = []
    for t in range(n_obs):
        obs = obs_aligned[t]
        # RTS
        smo = x_smooth[t][:len(obs)]
        n1, n2 = np.linalg.norm(obs), np.linalg.norm(smo)
        rts_cos = float(np.dot(obs, smo) / (n1 * n2)) if n1 > 1e-15 and n2 > 1e-15 else 1.0
        rts_cosines.append(rts_cos)
        # Exp
        smo_e = exp_smoothed[t][:len(obs)]
        n2e = np.linalg.norm(smo_e)
        exp_cos = float(np.dot(obs, smo_e) / (n1 * n2e)) if n1 > 1e-15 and n2e > 1e-15 else 1.0
        exp_cosines.append(exp_cos)

    print(f"\n  Cosine(forward, smoothed):")
    print(f"    RTS:  mean={np.mean(rts_cosines):.4f}")
    print(f"    Exp:  mean={np.mean(exp_cosines):.4f}")

    # Step 7: Verification
    print(f"\n{'=' * 70}")
    print("Verification")
    print(f"{'=' * 70}")

    # Test 1: Kalman filter converges (P_t trace stabilizes)
    traces = [np.trace(P_filt[t]) for t in range(n_obs)]
    trace_std = np.std(traces[n_obs//2:])
    trace_mean = np.mean(traces[n_obs//2:])
    converged = trace_std / (trace_mean + 1e-15) < 0.5
    print(f"\n  Test 1: Kalman filter converges?")
    print(f"    Trace(P) late half: mean={trace_mean:.6e}, std={trace_std:.6e}")
    print(f"    {'[VERIFIED]' if converged else '[FAILED]'}")

    # Test 2: RTS produces non-trivial revisions
    nontrivial = rts_arr.mean() > 1e-6
    print(f"\n  Test 2: RTS gives non-trivial revisions?")
    print(f"    Mean RTS revision: {rts_arr.mean():.6e}")
    print(f"    {'[VERIFIED]' if nontrivial else '[FAILED]'}")

    # Test 3: Uncertainty estimates are meaningful (>80% within 2-sigma)
    frac_within = np.mean(within_2sigma)
    meaningful = frac_within > 0.80
    print(f"\n  Test 3: Revisions within 2-sigma bounds?")
    print(f"    Fraction within: {frac_within:.1%}")
    print(f"    {'[VERIFIED]' if meaningful else '[FAILED]'}")

    # Test 4: RTS and exponential agree in direction
    direction_agree = np.corrcoef(rts_arr, exp_arr)[0, 1]
    agrees = direction_agree > 0.3
    print(f"\n  Test 4: RTS and exponential revision correlated?")
    print(f"    Pearson r: {direction_agree:.4f}")
    print(f"    {'[VERIFIED]' if agrees else '[FAILED]'}")

    n_verified = sum([converged, nontrivial, meaningful, agrees])
    print(f"\n  OVERALL: {n_verified}/4 Kalman smoother tests verified")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_12_kalman_smoother',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'Kalman + RTS smoother replaces ad-hoc exponential',
        'n_observations': n_obs,
        'state_dimension': k,
        'noise_parameters': {
            'Q_trace': float(np.trace(Q)),
            'R_trace': float(np.trace(R)),
        },
        'rts_metrics': {
            'mean_revision': float(rts_arr.mean()),
            'max_revision': float(rts_arr.max()),
            'early_revision': float(rts_early),
            'late_revision': float(rts_late),
            'temporal_ratio': float(rts_late / rts_early) if rts_early > 1e-15 else 0,
            'mean_cosine': float(np.mean(rts_cosines)),
        },
        'exponential_metrics': {
            'mean_revision': float(exp_arr.mean()),
            'max_revision': float(exp_arr.max()),
            'early_revision': float(exp_early),
            'late_revision': float(exp_late),
            'temporal_ratio': float(exp_late / exp_early) if exp_early > 1e-15 else 0,
            'mean_cosine': float(np.mean(exp_cosines)),
        },
        'verification': {
            'kalman_converged': bool(converged),
            'rts_nontrivial': bool(nontrivial),
            'uncertainty_meaningful': bool(meaningful),
            'frac_within_2sigma': float(frac_within),
            'methods_correlated': bool(agrees),
            'correlation': float(direction_agree),
            'n_verified': n_verified,
        },
        'temporal_profile': {
            'steps': steps_recorded,
            'rts_revisions': [float(r) for r in rts_revisions],
            'exp_revisions': [float(r) for r in exp_revisions],
            'rts_cosines': [float(c) for c in rts_cosines],
            'exp_cosines': [float(c) for c in exp_cosines],
        },
    }

    output_file = RESULTS_DIR / f'exp_12_kalman_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=lambda o: int(o) if hasattr(o, 'item') else o)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
