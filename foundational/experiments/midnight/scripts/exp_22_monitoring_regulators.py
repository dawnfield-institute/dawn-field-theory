"""
exp_22 -- Regulation as Monitoring: Saturation, the Horizon, and Zeno

Midnight Initiative -- exp_20 follow-up in the decoherence frame

PRE-REGISTERED: journals/2026-06-11_exp21-exp22-preregistration.md (same commit;
--selftest = harness continuity only). Derivation context: P17 note.

exp_20 killed "severance enforces the phi-bound": the collective stress trigger
cannot fire on an isolated rich-get-richer peak (Gini stays 0.94 vs 0.618).
The decoherence reading: an isolated peak is an UN-MONITORED subsystem.
This experiment characterizes the three monitoring-regulator regimes:

  D (saturation): MVAE-style ceiling (M11) -- node values capped at c*mean,
     overflow relaxes to neighbors (internally conserved).
     P22.1 (scored): Gini_steady(D) < Gini_A for every ceiling c.
  E (monitoring horizon): during stress-severance runs, does severance EVER
     fire on the global peak? Measure the peak/neighbor stress contrast.
     P22.2 (scored): a finite critical contrast exists above which the peak
     never severs. Value measured, not predicted [D].
  F (Zeno sweep): stress severance every m-th generation.
     P22.3 (scored): Gini_steady(m) is NON-MONOTONIC in monitoring frequency
     (anti-Zeno analog; exp_20's m=1 edge already exceeded condition A).

Baseline continuity gate: condition A must reproduce Gini_A = 0.930 +/- 0.005.

Outputs: results/exp_22_monitoring_regulators_YYYYMMDD_HHMMSS.json
"""

import sys
import numpy as np
from pathlib import Path

MIDNIGHT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MIDNIGHT_ROOT / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_rate import PHI, INV_PHI, save_midnight_results, _convert_numpy
from exp_20_severance_regulator import (
    gini, deposit_phi_split, run_condition, N_NODES, N_GENERATIONS, N_TRIALS,
    STEADY_WINDOW, THETA_PRIMARY, PHI_BOUND)

GINI_A_REF = 0.9301      # exp_20 baseline (continuity gate +/- 0.005)
CEILINGS = [1.5, 2.0, PHI**2, 3.0, 5.0]
ZENO_M = [1, 2, 5, 10, 20, 50]


# ============================================================
# Condition D: saturation ceiling (internally conserved relaxation)
# ============================================================

def run_condition_D(ceiling_factor, seed0=4000):
    steady, max_resid = [], 0.0
    for trial in range(N_TRIALS):
        rng = np.random.RandomState(seed0 + trial)
        values = np.full(N_NODES, 1e-9)
        injected = float(np.sum(values))
        trace = []
        for gen in range(N_GENERATIONS):
            start = int(np.argmax(values))
            injected += deposit_phi_split(values, start, rng)
            # MVAE ceiling: overflow relaxes to neighbors until no node exceeds
            for _ in range(200):
                mean = np.mean(values)
                cap = ceiling_factor * mean
                over = np.where(values > cap)[0]
                if len(over) == 0:
                    break
                for i in over:
                    excess = values[i] - cap
                    values[i] = cap
                    nbrs = [j for j in (i - 1, i + 1) if 0 <= j < N_NODES]
                    for j in nbrs:
                        values[j] += excess / len(nbrs)
            resid = abs(np.sum(values) - injected)
            max_resid = max(max_resid, resid)
            trace.append(gini(values))
        steady.append(float(np.mean(trace[-STEADY_WINDOW:])))
    return {'ceiling': float(ceiling_factor),
            'steady_gini_mean': float(np.mean(steady)),
            'steady_gini_std': float(np.std(steady)),
            'max_conservation_residual': float(max_resid)}


# ============================================================
# Condition E: the monitoring horizon (instrumented severance)
# ============================================================

def run_condition_E(theta, seed0=5000):
    """Stress severance (exp_20 condition B) instrumented: for every severance
    event record whether it hit the global peak, and the peak's contrast."""
    peak_events, total_events = 0, 0
    contrasts_at_events, peak_contrast_trace = [], []
    for trial in range(min(N_TRIALS, 20)):
        rng = np.random.RandomState(seed0 + trial)
        values = np.full(N_NODES, 1e-9)
        severed_total = 0.0
        for gen in range(N_GENERATIONS):
            start = int(np.argmax(values))
            deposit_phi_split(values, start, rng)
            mean = np.mean(values)
            stress = values / mean
            peak = int(np.argmax(values))
            nbrs_p = [j for j in (peak - 1, peak + 1) if 0 <= j < N_NODES]
            peak_contrast = float(stress[peak] / max(np.mean([stress[j] for j in nbrs_p]), 1e-12))
            peak_contrast_trace.append(peak_contrast)
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
                        total_events += 1
                        if i == peak:
                            peak_events += 1
                            contrasts_at_events.append(peak_contrast)
    pct = np.percentile(peak_contrast_trace, [5, 50, 95])
    return {'theta': float(theta),
            'total_events': int(total_events),
            'peak_events': int(peak_events),
            'peak_sever_fraction': float(peak_events / total_events) if total_events else 0.0,
            'max_contrast_at_peak_severance': (float(np.max(contrasts_at_events))
                                               if contrasts_at_events else None),
            'peak_contrast_p5_p50_p95': [float(x) for x in pct]}


# ============================================================
# Condition F: Zeno sweep (severance every m-th generation)
# ============================================================

def run_condition_F(m, theta=THETA_PRIMARY, seed0=6000):
    steady = []
    for trial in range(N_TRIALS):
        rng = np.random.RandomState(seed0 + trial)
        values = np.full(N_NODES, 1e-9)
        severed_total = 0.0
        trace = []
        for gen in range(N_GENERATIONS):
            start = int(np.argmax(values))
            deposit_phi_split(values, start, rng)
            if gen % m == 0:
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
            trace.append(gini(values))
        steady.append(float(np.mean(trace[-STEADY_WINDOW:])))
    return {'m': int(m), 'steady_gini_mean': float(np.mean(steady)),
            'steady_gini_std': float(np.std(steady))}


# ============================================================
# Main
# ============================================================

def run():
    # Continuity gate: condition A must reproduce exp_20 baseline
    print("\n  Continuity gate: condition A (no severance)")
    A = run_condition('A', seed0=1000)
    gate = abs(A['steady_gini_mean'] - GINI_A_REF) <= 0.005
    print(f"    Gini_A = {A['steady_gini_mean']:.4f} (ref {GINI_A_REF}) "
          f"-> {'OK' if gate else 'VOID'}")
    if not gate:
        return {'experiment': 'exp_22_monitoring_regulators', 'initiative': 'midnight',
                'harness_void': True, 'condition_A': A, 'score': '0/3'}

    # D: saturation ceilings
    print("\n  Condition D (saturation ceiling, internally conserved):")
    D = []
    for c in CEILINGS:
        d = run_condition_D(c)
        D.append(d)
        print(f"    c={c:.3f}: Gini={d['steady_gini_mean']:.4f} "
              f"+/-{d['steady_gini_std']:.4f}  resid<={d['max_conservation_residual']:.1e}")
    p221 = all(d['steady_gini_mean'] < A['steady_gini_mean'] for d in D)
    print(f"    P22.1 (saturation regulates at every ceiling): "
          f"{'PASS' if p221 else 'FAIL'}")

    # E: monitoring horizon
    print("\n  Condition E (monitoring horizon, instrumented severance):")
    Ev = []
    for theta in (1.2, 1.4, 1.6, 1.8, 2.0):
        e = run_condition_E(theta)
        Ev.append(e)
        print(f"    theta={theta}: events={e['total_events']} "
              f"peak_events={e['peak_events']} "
              f"peak_contrast p50={e['peak_contrast_p5_p50_p95'][1]:.1f}")
    total_peak = sum(e['peak_events'] for e in Ev)
    # critical contrast: peak severance never occurs above this contrast
    max_at_sever = [e['max_contrast_at_peak_severance'] for e in Ev
                    if e['max_contrast_at_peak_severance']]
    typical_contrast = float(np.median([e['peak_contrast_p5_p50_p95'][1] for e in Ev]))
    p222 = (total_peak == 0) or (max_at_sever and max(max_at_sever) < typical_contrast)
    crit = max(max_at_sever) if max_at_sever else 0.0
    print(f"    Peak-severance events total: {total_peak}; "
          f"critical contrast <= {crit:.2f} vs typical peak contrast {typical_contrast:.1f}")
    print(f"    P22.2 (finite monitoring horizon exists): {'PASS' if p222 else 'FAIL'}")

    # F: Zeno sweep
    print("\n  Condition F (Zeno sweep, severance every m-th generation):")
    F = []
    for m in ZENO_M:
        f = run_condition_F(m)
        F.append(f)
        print(f"    m={m:>3}: Gini={f['steady_gini_mean']:.4f} +/-{f['steady_gini_std']:.4f}")
    ginis = [f['steady_gini_mean'] for f in F] + [A['steady_gini_mean']]  # m=inf
    interior = ginis[1:-1]
    p223 = (max(interior) > max(ginis[0], ginis[-1]) + 1e-4) or \
           (min(interior) < min(ginis[0], ginis[-1]) - 1e-4)
    print(f"    sequence (m=1..50, A): {[f'{g:.4f}' for g in ginis]}")
    print(f"    P22.3 (non-monotonic in monitoring frequency): "
          f"{'PASS' if p223 else 'FAIL'}")

    score = sum([p221, p222, p223])
    print(f"\n  Score: {score}/3")
    return {
        'experiment': 'exp_22_monitoring_regulators',
        'initiative': 'midnight',
        'condition_A': A,
        'condition_D': D, 'P22_1': p221,
        'condition_E': Ev, 'P22_2': p222,
        'condition_F': F, 'P22_3': p223,
        'score': f"{score}/3",
    }


def selftest():
    print("SELFTEST: harness continuity only (short run)")
    import exp_20_severance_regulator as e20
    saved = e20.N_GENERATIONS, e20.N_TRIALS, e20.STEADY_WINDOW
    e20.N_GENERATIONS, e20.N_TRIALS, e20.STEADY_WINDOW = 10, 2, 3
    r = e20.run_condition('A', seed0=9100)
    print(f"  condition A (short): Gini={r['steady_gini_mean']:.4f}, "
          f"resid<={r['max_conservation_residual']:.1e}")
    e20.N_GENERATIONS, e20.N_TRIALS, e20.STEADY_WINDOW = saved
    print("  OK")


if __name__ == '__main__':
    print("=" * 60)
    print("exp_22: Regulation as Monitoring -- Saturation, Horizon, Zeno")
    print("Midnight Initiative -- pre-registered")
    print("=" * 60)
    if '--selftest' in sys.argv:
        selftest()
    else:
        data = run()
        save_midnight_results('exp_22_monitoring_regulators', _convert_numpy(data))
