"""
exp_20 -- Severance as the Regularity Enforcer

Midnight Initiative -- Milestone R crossover

PRE-REGISTERED: see journals/2026-06-10_exp19-exp20-preregistration.md and the
git commit containing this file. Registered outcomes (steady-state Gini per
condition) are NOT computed before that commit; --selftest checks only the
harness and the conservation invariant.

Background: midnight exp_17 showed the phi-split clamps the per-node maximum
at exactly 1/phi = 0.618 but distribution-wide Gini reaches 0.72 -- the
cascade does not globally self-regulate. Milestone R provides severance
components (stress trigger exp_15, equilibrium-shift ejection exp_07, exact
PAC conservation exp_01) but the direct regulation test has never been run.

HYPOTHESIS: a stress-triggered severance channel propagates the local
phi-bound to the global distribution -- steady-state Gini clamps AT 1/phi.

Registered predictions:
  P1: condition A (no severance) steady-state Gini > 1/phi (violation
      reproduces dynamically; if not, harness void -- stop and report)
  P2: condition B (stress-triggered severance) Gini_steady <= 1/phi
  P3: |Gini_B - 1/phi| < 0.02 (clamps AT the bound, not merely below)
  P4: condition C (random-trigger, rate-matched to B) NOT in 1/phi +/- 0.02

Design (locked): 32-node chain; each generation one erasure event deposits a
unit of potential phi-split along the chain starting at the current
highest-value node (rich-get-richer); 200 generations x 50 trials;
stress = value/mean; node severs when its stress > theta AND all neighbors'
stress > theta (all-edges-overstressed, M-R exp_15); severed excess
(value - mean) leaves as an independent ledger (equilibrium-shift, exp_07).
Theta sweep {1.2,1.4,1.6,1.8,2.0}; primary readout theta = 1.6.
Conservation invariant: retained + severed = injected, every generation.

Outputs: results/exp_20_severance_regulator_YYYYMMDD_HHMMSS.json
"""

import sys
import numpy as np
from pathlib import Path

MIDNIGHT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MIDNIGHT_ROOT / "core"))
from phase_rate import PHI, INV_PHI, LN_PHI, save_midnight_results, _convert_numpy

N_NODES = 32
N_GENERATIONS = 200
N_TRIALS = 50
STEADY_WINDOW = 50          # average Gini over the last 50 generations
THETAS = [1.2, 1.4, 1.6, 1.8, 2.0]
THETA_PRIMARY = 1.6
PHI_BOUND = INV_PHI         # 0.6180
CLAMP_TOL = 0.02


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    s = np.sum(x)
    if s <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * x) / (n * s))


def deposit_phi_split(values, start, rng):
    """One erasure event: unit potential phi-split along the chain from
    `start`, walking toward the longer side (deterministic walk direction
    chosen by available run length; ties broken randomly)."""
    n = len(values)
    left_run, right_run = start, n - 1 - start
    if right_run > left_run:
        step = 1
    elif left_run > right_run:
        step = -1
    else:
        step = rng.choice([-1, 1])
    remaining = 1.0
    i = start
    while 0 <= i < n and remaining > 1e-12:
        share = remaining * INV_PHI
        values[i] += share
        remaining -= share
        i += step
    # deposit any remainder at the last visited in-range node
    j = min(max(i - step, 0), n - 1)
    values[j] += remaining
    return 1.0  # injected amount


def run_condition(condition, theta=THETA_PRIMARY, sever_prob=None, seed0=0):
    """condition: 'A' none, 'B' stress-triggered, 'C' random rate-matched.
    Returns steady Gini per trial, severance event counts, conservation
    residuals."""
    steady_ginis, event_counts, max_resid = [], [], 0.0
    gini_traces = []
    for trial in range(N_TRIALS):
        rng = np.random.RandomState(seed0 + trial)
        values = np.full(N_NODES, 1e-9)
        injected = float(np.sum(values))
        severed_total = 0.0
        events = 0
        trace = []
        for gen in range(N_GENERATIONS):
            start = int(np.argmax(values))
            injected += deposit_phi_split(values, start, rng)

            if condition == 'B':
                mean = np.mean(values)
                stress = values / mean
                over = stress > theta
                for i in range(N_NODES):
                    if not over[i]:
                        continue
                    nbrs = [j for j in (i - 1, i + 1) if 0 <= j < N_NODES]
                    if all(over[j] for j in nbrs):
                        excess = values[i] - mean
                        if excess > 0:
                            values[i] = mean
                            severed_total += excess
                            events += 1
            elif condition == 'C':
                mean = np.mean(values)
                for i in range(N_NODES):
                    if rng.rand() < sever_prob:
                        excess = values[i] - mean
                        if excess > 0:
                            values[i] = mean
                            severed_total += excess
                            events += 1

            resid = abs((np.sum(values) + severed_total) - injected)
            max_resid = max(max_resid, resid)
            trace.append(gini(values))

        steady_ginis.append(float(np.mean(trace[-STEADY_WINDOW:])))
        event_counts.append(events)
        gini_traces.append(trace)

    return {
        'steady_gini_mean': float(np.mean(steady_ginis)),
        'steady_gini_std': float(np.std(steady_ginis)),
        'steady_ginis': steady_ginis,
        'mean_events': float(np.mean(event_counts)),
        'max_conservation_residual': float(max_resid),
        'mean_trace': [float(v) for v in np.mean(np.array(gini_traces), axis=0)],
    }


def run():
    print(f"\n  1/phi bound = {PHI_BOUND:.4f}, clamp tolerance +/- {CLAMP_TOL}")

    # Condition A: baseline
    print("\n  Condition A (no severance):")
    A = run_condition('A', seed0=1000)
    print(f"    steady Gini = {A['steady_gini_mean']:.4f} +/- {A['steady_gini_std']:.4f}")
    print(f"    conservation residual <= {A['max_conservation_residual']:.2e}")
    p1 = A['steady_gini_mean'] > PHI_BOUND
    print(f"    P1 (violation reproduced, Gini > 1/phi): {'PASS' if p1 else 'FAIL'}")
    if not p1:
        print("    HARNESS VOID per registration -- stopping, reporting honestly.")
        return {'experiment': 'exp_20_severance_regulator', 'initiative': 'midnight',
                'condition_A': A, 'harness_void': True, 'score': '0/4',
                'verdict': 'VOID_BASELINE_DOES_NOT_VIOLATE'}

    # Condition B: stress-triggered severance, theta sweep
    print("\n  Condition B (stress-triggered severance):")
    B_sweep = {}
    for theta in THETAS:
        B = run_condition('B', theta=theta, seed0=2000)
        B_sweep[str(theta)] = B
        print(f"    theta={theta}: steady Gini = {B['steady_gini_mean']:.4f} "
              f"+/- {B['steady_gini_std']:.4f}  events/trial={B['mean_events']:.0f}  "
              f"resid<={B['max_conservation_residual']:.2e}")
    B = B_sweep[str(THETA_PRIMARY)]
    p2 = B['steady_gini_mean'] <= PHI_BOUND + 1e-9
    p3 = abs(B['steady_gini_mean'] - PHI_BOUND) < CLAMP_TOL
    print(f"    P2 (Gini_B <= 1/phi): {'PASS' if p2 else 'FAIL'}")
    print(f"    P3 (clamps AT 1/phi +/- {CLAMP_TOL}): "
          f"|{B['steady_gini_mean']:.4f} - {PHI_BOUND:.4f}| = "
          f"{abs(B['steady_gini_mean']-PHI_BOUND):.4f} -> {'PASS' if p3 else 'FAIL'}")

    # Condition C: random-trigger, rate-matched to B at primary theta
    rate = B['mean_events'] / (N_GENERATIONS * N_NODES)
    print(f"\n  Condition C (random severance, rate-matched p={rate:.5f}):")
    C = run_condition('C', sever_prob=rate, seed0=3000)
    print(f"    steady Gini = {C['steady_gini_mean']:.4f} +/- {C['steady_gini_std']:.4f}  "
          f"events/trial={C['mean_events']:.0f}")
    p4 = abs(C['steady_gini_mean'] - PHI_BOUND) >= CLAMP_TOL
    print(f"    P4 (C does NOT land at 1/phi +/- {CLAMP_TOL}): {'PASS' if p4 else 'FAIL'}")

    score = sum([p1, p2, p3, p4])
    if p1 and p2 and p3 and p4:
        verdict = 'SUPPORTED'
    elif p1 and not p2:
        verdict = 'KILLED_no_clamp'
    elif p1 and p2 and not p3:
        verdict = 'PARTIAL_clamps_below_not_at_bound'
    elif p1 and p2 and p3 and not p4:
        verdict = 'PARTIAL_generic_dissipation_not_phi_specific'
    else:
        verdict = 'INCONCLUSIVE'
    print(f"\n  Overall: {score}/4   VERDICT: {verdict}")

    return {
        'experiment': 'exp_20_severance_regulator',
        'initiative': 'midnight',
        'phi_bound': PHI_BOUND,
        'condition_A': A,
        'condition_B_sweep': B_sweep,
        'condition_C': C,
        'P1': p1, 'P2': p2, 'P3': p3, 'P4': p4,
        'verdict': verdict,
        'score': f"{score}/4",
    }


def selftest():
    print("SELFTEST: harness + conservation only (5 generations, 2 trials)")
    global N_GENERATIONS, N_TRIALS, STEADY_WINDOW
    N_GENERATIONS, N_TRIALS, STEADY_WINDOW = 5, 2, 2
    for cond, kw in [('A', {}), ('B', {'theta': 1.6}), ('C', {'sever_prob': 0.01})]:
        r = run_condition(cond, seed0=9000, **kw)
        print(f"  condition {cond}: conservation residual <= "
              f"{r['max_conservation_residual']:.2e}  (must be ~1e-12)")
        assert r['max_conservation_residual'] < 1e-9, "conservation violated"
    print("  OK")


if __name__ == '__main__':
    print("=" * 60)
    print("exp_20: Severance as the Regularity Enforcer")
    print("Midnight Initiative -- pre-registered (clamp at 1/phi)")
    print("=" * 60)
    if '--selftest' in sys.argv:
        selftest()
    else:
        data = run()
        save_midnight_results('exp_20_severance_regulator', _convert_numpy(data))
