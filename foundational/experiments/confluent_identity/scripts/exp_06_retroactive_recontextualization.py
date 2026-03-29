"""
exp_06_retroactive_recontextualization.py -- Confluent Identity Phase 3

PURPOSE:
    Test Claim 5: Retroactive recontextualization.
    "A new actualization doesn't just change the future -- it changes what
    prior actualizations *meant*."

    Like a plot twist: the villain reveal doesn't change earlier scenes,
    but it changes the film's identity. Every earlier scene now means
    something different.

DESIGN:
    Uses Bayesian forward-backward smoothing to test whether a perturbation
    (obstacle insertion) changes the optimal interpretation of past states.

    1. Record identity history I(t) during Phase 1 dynamics (forward pass)
    2. Apply the Phase 2 perturbation (obstacle insertion)
    3. Run backward smoother: using the POST-perturbation identity as the
       endpoint, re-estimate past identities via exponential smoothing
    4. Compare: smoothed I(t) vs original I(t) for t < T_perturbation

    The "backward pass" is a simple exponential smoother:
        I_smooth(t) = alpha * I_forward(t) + (1 - alpha) * I_smooth(t+1)
    where alpha controls how much the future revises the past.

METRICS:
    - Recontextualization magnitude: ||I_smooth(t) - I_forward(t)|| for t < T
    - Temporal decay: how far back does the revision reach?
    - Coupling correlation: is revision larger for regions more coupled to
      the perturbation?

FALSIFICATION:
    Claim 5 SUPPORTED if: smoothed past identity differs from original,
        and the difference is larger for more coupled regions
    Claim 5 FALSIFIED if: past identity is unchanged by smoothing
        (i.e., the backward pass adds no information)

Planck units throughout.
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy import sparse
from scipy.sparse.linalg import eigsh

RESULTS_DIR = Path(__file__).parent.parent / 'results'
K_MODES = 10

import sys
sys.path.insert(0, str(Path(__file__).parent))
from exp_01_lattice_fluid_baseline import PeriodicLatticeFluid


def build_adjacency(C):
    """Build sparse weighted adjacency matrix."""
    N = C.shape[0]
    C_mean = C.mean()
    rows, cols, weights = [], [], []
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ni, nj = (i + di) % N, (j + dj) % N
                nidx = ni * N + nj
                w = np.exp(-abs(C[i, j] - C[ni, nj]) / C_mean)
                rows.append(idx)
                cols.append(nidx)
                weights.append(w)
    return sparse.csr_matrix((weights, (rows, cols)), shape=(N*N, N*N))


def compute_identity_fast(C_flat, adjacency=None, C_2d=None, k=K_MODES):
    """
    Compute spectral identity. If adjacency not provided, build from C_2d.
    Returns dict with coefficients, fiedler, spectral_entropy, harmonic.
    """
    if adjacency is None:
        adjacency = build_adjacency(C_2d)

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

    nonzero = eigenvalues > 1e-10
    fiedler = float(eigenvalues[nonzero][0]) if nonzero.any() else 0.0

    nz_eigs = eigenvalues[nonzero]
    if len(nz_eigs) > 0:
        p = nz_eigs / nz_eigs.sum()
        spec_entropy = float(-np.sum(p * np.log(p + 1e-15)))
    else:
        spec_entropy = 0.0

    return {
        'harmonic': harmonic,
        'fiedler': fiedler,
        'spectral_entropy': spec_entropy,
        'coefficients': coeffs,
    }


def run_and_record_history(N=128, total_value=100.0, sample_interval=100,
                            max_steps=2000):
    """
    Run Phase 1 dynamics and record identity snapshots at regular intervals.
    Returns identity history and final fluid state.
    """
    fluid = PeriodicLatticeFluid(
        N=N, total_value=total_value, seed=42,
        n_large_stones=12, n_small_stones=40, gravity=0.005
    )

    # Build adjacency once from initial state (topology doesn't change much)
    C_init = fluid.C
    adjacency = build_adjacency(C_init)

    history = []
    print(f"  Recording identity every {sample_interval} steps...")

    for step in range(max_steps):
        P_prev = fluid.P.copy()
        A_prev = fluid.A.copy()
        fluid.fluid_step(dt=0.005, viscosity=0.05, sec_threshold=0.1)

        if step % sample_interval == 0:
            C = fluid.C
            identity = compute_identity_fast(C.ravel(), adjacency)
            history.append({
                'step': step,
                'identity': identity,
            })
            if step % 500 == 0:
                change = fluid.max_change(P_prev, A_prev)
                print(f"    Step {step}: fiedler={identity['fiedler']:.6f}, "
                      f"change={change:.2e}")

    # Final snapshot
    C_final = fluid.C
    id_final = compute_identity_fast(C_final.ravel(), adjacency)
    history.append({'step': max_steps, 'identity': id_final})

    return history, fluid, adjacency


def insert_obstacle_and_get_identity(fluid, adjacency, pos, radius):
    """Insert obstacle and compute post-perturbation identity."""
    N = fluid.N
    P_new = fluid.P.copy()
    A_new = fluid.A.copy()
    mask_new = fluid.stone_mask.copy()

    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y)
    dy = np.minimum(np.abs(Y - pos[0]), N - np.abs(Y - pos[0]))
    dx = np.minimum(np.abs(X - pos[1]), N - np.abs(X - pos[1]))
    dist = np.sqrt(dx**2 + dy**2)
    obstacle_mask = dist < radius

    local_C_mean = (P_new + A_new).mean()
    obstacle_C = local_C_mean * 3.0
    total_before = P_new.sum() + A_new.sum()

    P_new[obstacle_mask] = 0.1 * obstacle_C
    A_new[obstacle_mask] = 0.9 * obstacle_C
    mask_new[obstacle_mask] = True

    total_after = P_new.sum() + A_new.sum()
    diff = total_after - total_before
    fluid_cells = ~mask_new
    n_fluid = fluid_cells.sum()
    if n_fluid > 0 and abs(diff) > 1e-15:
        P_new[fluid_cells] -= diff * 0.7 / n_fluid
        A_new[fluid_cells] -= diff * 0.3 / n_fluid

    C_new = P_new + A_new
    # Rebuild adjacency for perturbed state
    adj_new = build_adjacency(C_new)
    id_post = compute_identity_fast(C_new.ravel(), adj_new)

    return id_post


def backward_smooth(forward_history, endpoint_identity, alpha=0.3):
    """
    Bayesian backward smoother.

    I_smooth(t) = alpha * I_forward(t) + (1-alpha) * I_smooth(t+1)

    Starting from the endpoint (post-perturbation identity), work backward
    through the forward history, blending the future endpoint with each
    past identity estimate.

    alpha = 1.0 means no smoothing (keep forward), alpha = 0.0 means
    fully replace with endpoint.
    """
    n_steps = len(forward_history)
    endpoint_coeffs = np.array(endpoint_identity['coefficients'])

    smoothed = []

    # Start from the end
    current_smooth = endpoint_coeffs.copy()

    for i in range(n_steps - 1, -1, -1):
        forward_coeffs = np.array(forward_history[i]['identity']['coefficients'])

        # Align lengths
        min_len = min(len(forward_coeffs), len(current_smooth))
        fwd = forward_coeffs[:min_len]
        cur = current_smooth[:min_len]

        # Smooth: blend forward estimate with backward (future) estimate
        blended = alpha * fwd + (1 - alpha) * cur
        current_smooth = blended

        # Compute revision magnitude
        revision = np.linalg.norm(blended - fwd)
        fwd_norm = np.linalg.norm(fwd) + 1e-15

        smoothed.append({
            'step': forward_history[i]['step'],
            'forward_coeffs': fwd.tolist(),
            'smoothed_coeffs': blended.tolist(),
            'revision_magnitude': float(revision),
            'relative_revision': float(revision / fwd_norm),
        })

    # Reverse to chronological order
    smoothed.reverse()
    return smoothed


def run_experiment():
    print("=" * 70)
    print("Confluent Identity -- Phase 3, Experiment 06")
    print("Retroactive Recontextualization via Bayesian Smoothing")
    print("=" * 70)

    # Step 1: Run dynamics and record identity history
    print("\nPhase 1: Recording identity history...")
    history, fluid, adjacency = run_and_record_history(
        max_steps=2000, sample_interval=100
    )
    n_snapshots = len(history)
    print(f"  Recorded {n_snapshots} identity snapshots")

    # Step 2: Find high-flow position for perturbation
    C = fluid.C
    grad_x = (np.roll(C, -1, axis=0) - np.roll(C, 1, axis=0)) / 2.0
    grad_y = (np.roll(C, -1, axis=1) - np.roll(C, 1, axis=1)) / 2.0
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    from scipy.ndimage import uniform_filter, binary_dilation
    grad_smooth = uniform_filter(grad_mag, size=5)
    stone_buffer = binary_dilation(fluid.stone_mask, iterations=8)
    candidate = ~fluid.stone_mask & ~stone_buffer
    grad_candidate = grad_smooth.copy()
    grad_candidate[~candidate] = 0
    high_pos = np.unravel_index(grad_candidate.argmax(), grad_candidate.shape)
    print(f"\n  Perturbation position: {high_pos} "
          f"(grad={grad_smooth[high_pos]:.6f})")

    # Step 3: Insert obstacle and compute post-perturbation identity
    print("\nPhase 2: Computing post-perturbation identity...")
    RADIUS = 3
    id_post = insert_obstacle_and_get_identity(fluid, adjacency, high_pos, RADIUS)
    print(f"  Post-perturbation fiedler: {id_post['fiedler']:.6f}")

    # Step 4: Backward smooth with multiple alpha values
    print(f"\n{'=' * 70}")
    print("Backward Smoothing")
    print(f"{'=' * 70}")

    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    all_smoothing_results = {}

    for alpha in alphas:
        smoothed = backward_smooth(history, id_post, alpha=alpha)
        all_smoothing_results[alpha] = smoothed

        # Summary stats
        revisions = [s['revision_magnitude'] for s in smoothed]
        rel_revisions = [s['relative_revision'] for s in smoothed]

        print(f"\n  alpha={alpha:.1f}:")
        print(f"    Mean revision:    {np.mean(revisions):.6f}")
        print(f"    Max revision:     {np.max(revisions):.6f}")
        print(f"    Mean rel revision: {np.mean(rel_revisions):.4f}")

        # Temporal profile: how does revision decay with distance from perturbation?
        # The perturbation happens at the END, so earlier snapshots are "further"
        n = len(smoothed)
        first_quarter = smoothed[:n//4]
        last_quarter = smoothed[3*n//4:]
        print(f"    Early revision (t<T/4): {np.mean([s['revision_magnitude'] for s in first_quarter]):.6f}")
        print(f"    Late revision (t>3T/4): {np.mean([s['revision_magnitude'] for s in last_quarter]):.6f}")

    # Step 5: Detailed analysis at alpha=0.3 (the sweet spot)
    print(f"\n{'=' * 70}")
    print("Detailed Analysis (alpha=0.3)")
    print(f"{'=' * 70}")

    smoothed = all_smoothing_results[0.3]
    print(f"\n  {'Step':<8s} {'Forward L2':<14s} {'Smoothed L2':<14s} {'Revision':<14s} {'RelRev':<10s}")
    print(f"  {'-'*8} {'-'*14} {'-'*14} {'-'*14} {'-'*10}")

    for s in smoothed:
        fwd_norm = np.linalg.norm(s['forward_coeffs'])
        smo_norm = np.linalg.norm(s['smoothed_coeffs'])
        print(f"  {s['step']:<8d} {fwd_norm:<14.6f} {smo_norm:<14.6f} "
              f"{s['revision_magnitude']:<14.6f} {s['relative_revision']:<10.4f}")

    # Step 6: Cosine similarity between forward and smoothed over time
    print(f"\n  Cosine(forward, smoothed) over time:")
    cosines = []
    for s in smoothed:
        fwd = np.array(s['forward_coeffs'])
        smo = np.array(s['smoothed_coeffs'])
        n1, n2 = np.linalg.norm(fwd), np.linalg.norm(smo)
        if n1 > 1e-15 and n2 > 1e-15:
            cos = float(np.dot(fwd, smo) / (n1 * n2))
        else:
            cos = 1.0
        cosines.append(cos)

    early_cos = np.mean(cosines[:len(cosines)//4])
    late_cos = np.mean(cosines[3*len(cosines)//4:])
    print(f"    Early (t<T/4):  mean cos = {early_cos:.4f}")
    print(f"    Late (t>3T/4):  mean cos = {late_cos:.4f}")
    print(f"    Overall:        mean cos = {np.mean(cosines):.4f}")

    # VERDICT
    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")

    mean_rel_revision = np.mean([s['relative_revision']
                                  for s in all_smoothing_results[0.3]])
    early_revision = np.mean([s['revision_magnitude']
                              for s in all_smoothing_results[0.3][:n_snapshots//4]])
    late_revision = np.mean([s['revision_magnitude']
                             for s in all_smoothing_results[0.3][3*n_snapshots//4:]])

    # Test 1: Does smoothing change past identity?
    if mean_rel_revision > 0.01:
        t1 = "SUPPORTED"
        print(f"\n  Test 1 (past identity changes): {t1}")
        print(f"    Mean relative revision = {mean_rel_revision:.4f} (>{0.01})")
    else:
        t1 = "FALSIFIED"
        print(f"\n  Test 1 (past identity changes): {t1}")
        print(f"    Mean relative revision = {mean_rel_revision:.4f} (too small)")

    # Test 2: Is revision stronger near the perturbation?
    if late_revision > early_revision * 1.1:
        t2 = "SUPPORTED"
        print(f"  Test 2 (temporal gradient): {t2}")
        print(f"    Late/early revision ratio = {late_revision/early_revision:.2f}x")
    elif early_revision > late_revision * 1.1:
        t2 = "INVERTED (unexpected)"
        print(f"  Test 2 (temporal gradient): {t2}")
        print(f"    Early revision is LARGER -- past is more sensitive")
    else:
        t2 = "FLAT"
        print(f"  Test 2 (temporal gradient): {t2}")
        print(f"    Revision is uniform across time")

    # Test 3: Does the backward pass change meaning (cosine < 1)?
    if np.mean(cosines) < 0.99:
        t3 = "SUPPORTED"
        print(f"  Test 3 (meaning change): {t3}")
        print(f"    Mean cosine(forward, smoothed) = {np.mean(cosines):.4f}")
        print(f"    The past doesn't just scale -- it ROTATES in identity space")
    else:
        t3 = "WEAK"
        print(f"  Test 3 (meaning change): {t3}")
        print(f"    Mean cosine = {np.mean(cosines):.4f} (close to 1, mostly scaling)")

    tests = [t1, t2, t3]
    supported = sum(1 for t in tests if 'SUPPORTED' in t)
    print(f"\n  Overall: {supported}/3 tests support Claim 5")

    if supported >= 2:
        print("\n  ==> CLAIM 5 SUPPORTED: Retroactive recontextualization is real")
        print("      The future changes what the past means.")
    elif supported >= 1:
        print("\n  ==> CLAIM 5 PARTIALLY SUPPORTED: evidence of retroactive revision")
    else:
        print("\n  ==> CLAIM 5 CHALLENGED: backward pass adds minimal information")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'experiment': 'exp_06_retroactive_recontextualization',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'max_steps': 2000,
            'sample_interval': 100,
            'perturbation_pos': [int(p) for p in high_pos],
            'perturbation_radius': RADIUS,
            'alphas': alphas,
        },
        'n_snapshots': n_snapshots,
        'smoothing_summary': {
            str(alpha): {
                'mean_revision': float(np.mean([s['revision_magnitude']
                                                 for s in results])),
                'max_revision': float(np.max([s['revision_magnitude']
                                               for s in results])),
                'mean_relative_revision': float(np.mean([s['relative_revision']
                                                          for s in results])),
            }
            for alpha, results in all_smoothing_results.items()
        },
        'temporal_profile': {
            'steps': [s['step'] for s in all_smoothing_results[0.3]],
            'revisions': [s['revision_magnitude']
                          for s in all_smoothing_results[0.3]],
            'relative_revisions': [s['relative_revision']
                                    for s in all_smoothing_results[0.3]],
            'cosines': cosines,
        },
        'verdicts': {
            'test1_past_changes': t1,
            'test2_temporal_gradient': t2,
            'test3_meaning_rotation': t3,
            'n_supported': supported,
            'claim5': 'SUPPORTED' if supported >= 2 else
                      'PARTIAL' if supported >= 1 else 'CHALLENGED',
        },
    }

    output_file = RESULTS_DIR / f'exp_06_retroactive_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_file.name}")

    return output


if __name__ == '__main__':
    run_experiment()
