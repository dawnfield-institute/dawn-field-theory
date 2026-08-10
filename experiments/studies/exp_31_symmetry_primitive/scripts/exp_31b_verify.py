"""
exp_31b verification: stress-test the v3 results before drawing conclusions.

Questions to answer:
  1. Is the depth-6 convergence to phi real, or seed-dependent?
  2. Is the flat partition result (R=1.612) robust or a fluke?
  3. Are we just numerically confirming an analytical tautology?
     (D_{n+1} = S_n + conservation → phi is KNOWN from exp_01)
  4. Does the convergence PATH matter, or just the endpoint?
"""

import sys
from pathlib import Path
import numpy as np

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXP_ROOT))

from core import PHI, INV_PHI

# Import the tree infrastructure from exp_31b
from exp_31b_attractor_generates_phi import (
    build_random_tree, evolve_scale_invariance, dominant_chain_ratios,
    get_dominant_chain, flat_scale_invariance,
    evolve_target_ratio,
)


def verify_1_large_sample():
    """50 seeds at depth 5 and 6 — is phi convergence robust?"""
    print("=" * 60)
    print("VERIFY 1: Large sample (50 seeds) at depth 5 and 6")
    print("=" * 60)

    for depth in [5, 6]:
        ratios = []
        for seed in range(50):
            rng = np.random.RandomState(seed)
            tree = build_random_tree(100.0, depth, rng=rng)
            history = evolve_scale_invariance(tree, n_steps=5000, alpha=0.03)
            if len(history) > 200:
                eq = np.mean(history[-200:])
                ratios.append(eq)

        arr = np.array(ratios)
        print(f"\n  depth={depth}: n={len(arr)}")
        print(f"    mean  = {np.mean(arr):.6f}")
        print(f"    std   = {np.std(arr):.6f}")
        print(f"    min   = {np.min(arr):.6f}")
        print(f"    max   = {np.max(arr):.6f}")
        print(f"    median= {np.median(arr):.6f}")
        delta = abs(np.mean(arr) - PHI) / PHI
        print(f"    delta_phi = {delta:.4%}")
        # How many are within 5% of phi?
        within_5 = np.sum(np.abs(arr - PHI) / PHI < 0.05)
        within_10 = np.sum(np.abs(arr - PHI) / PHI < 0.10)
        print(f"    within 5% of phi: {within_5}/{len(arr)} ({within_5/len(arr):.0%})")
        print(f"    within 10% of phi: {within_10}/{len(arr)} ({within_10/len(arr):.0%})")


def verify_2_flat_partition_large():
    """30 seeds on flat partition — is R≈phi robust or fluky?"""
    print("\n" + "=" * 60)
    print("VERIFY 2: Flat partition with 30 seeds")
    print("=" * 60)

    ratios = []
    for seed in range(30):
        eq_r = flat_scale_invariance(n=64, n_levels=4, n_steps=3000,
                                     alpha=0.05, seed=seed)
        if np.isfinite(eq_r):
            ratios.append(eq_r)

    arr = np.array(ratios)
    print(f"\n  n={len(arr)}")
    print(f"  mean  = {np.mean(arr):.6f}")
    print(f"  std   = {np.std(arr):.6f}")
    print(f"  min   = {np.min(arr):.6f}")
    print(f"  max   = {np.max(arr):.6f}")
    delta = abs(np.mean(arr) - PHI) / PHI
    print(f"  delta_phi = {delta:.4%}")
    within_5 = np.sum(np.abs(arr - PHI) / PHI < 0.05)
    within_10 = np.sum(np.abs(arr - PHI) / PHI < 0.10)
    print(f"  within 5% of phi: {within_5}/{len(arr)} ({within_5/len(arr):.0%})")
    print(f"  within 10% of phi: {within_10}/{len(arr)} ({within_10/len(arr):.0%})")


def verify_3_tautology_check():
    """
    Is D_{n+1} → S_n just encoding phi directly?

    The analytical result from exp_01:
      conservation + D_{n+1} = S_n → R^2 - R - 1 = 0 → phi

    If the drive DIRECTLY imposes D_{n+1} = S_n, then we're just
    numerically confirming an analytical identity. That's not emergence.

    Test: measure how close D_{n+1} actually gets to S_n at equilibrium.
    If D_{n+1} ≈ S_n exactly, we proved nothing new — just confirmed exp_01.
    The question is whether the drive only PARTIALLY achieves D_{n+1} = S_n
    but phi still emerges (meaning the constraint is an attractor basin,
    not just a fixed point).
    """
    print("\n" + "=" * 60)
    print("VERIFY 3: Tautology check — is D_{n+1} ≈ S_n at equilibrium?")
    print("=" * 60)

    for depth in [4, 5, 6]:
        mismatches = []
        eq_ratios = []
        for seed in range(20):
            rng = np.random.RandomState(seed * 11 + 1)
            tree = build_random_tree(100.0, depth, rng=rng)
            evolve_scale_invariance(tree, n_steps=5000, alpha=0.03)

            # Check D_{n+1} vs S_n along dominant chain
            chain = get_dominant_chain(tree)
            level_mismatches = []
            for i in range(len(chain) - 1):
                P_n, D_n, S_n = chain[i]
                P_n1, D_n1, S_n1 = chain[i + 1]
                if S_n > 1e-15 and D_n1 > 1e-15:
                    mismatch = abs(D_n1 - S_n) / S_n
                    level_mismatches.append(mismatch)
            if level_mismatches:
                mismatches.append(np.mean(level_mismatches))

            ratios = dominant_chain_ratios(tree)
            if ratios:
                eq_ratios.append(np.mean(ratios))

        mean_mismatch = np.mean(mismatches) if mismatches else np.nan
        mean_ratio = np.mean(eq_ratios) if eq_ratios else np.nan
        delta_phi = abs(mean_ratio - PHI) / PHI if np.isfinite(mean_ratio) else np.nan

        print(f"\n  depth={depth}:")
        print(f"    mean |D_{{n+1}} - S_n| / S_n = {mean_mismatch:.6f} ({mean_mismatch:.2%})")
        print(f"    mean R = {mean_ratio:.6f} (delta_phi = {delta_phi:.4%})")

        if mean_mismatch < 0.01:
            print(f"    WARNING: D_{{n+1}} ≈ S_n achieved — this IS the exp_01 tautology")
        else:
            print(f"    D_{{n+1}} ≠ S_n but R still near phi — genuine emergence")


def verify_4_partial_drive():
    """
    What if the drive only WEAKLY pushes D_{n+1} toward S_n?
    If phi still emerges from a weak/incomplete application of the
    constraint, that's more interesting than exact enforcement.
    """
    print("\n" + "=" * 60)
    print("VERIFY 4: Weak drive — does phi emerge with small alpha?")
    print("=" * 60)

    depth = 6
    alphas = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]

    for alpha in alphas:
        ratios = []
        mismatches = []
        for seed in range(15):
            rng = np.random.RandomState(seed * 7 + 2)
            tree = build_random_tree(100.0, depth, rng=rng)
            evolve_scale_invariance(tree, n_steps=5000, alpha=alpha)

            chain = get_dominant_chain(tree)
            for i in range(len(chain) - 1):
                P_n, D_n, S_n = chain[i]
                P_n1, D_n1, S_n1 = chain[i + 1]
                if S_n > 1e-15 and D_n1 > 1e-15:
                    mismatches.append(abs(D_n1 - S_n) / S_n)

            r = dominant_chain_ratios(tree)
            if r:
                ratios.append(np.mean(r))

        mean_r = np.mean(ratios) if ratios else np.nan
        mean_mm = np.mean(mismatches) if mismatches else np.nan
        delta = abs(mean_r - PHI) / PHI if np.isfinite(mean_r) else np.nan

        print(f"  alpha={alpha:.3f}: R={mean_r:.6f}, delta_phi={delta:.4%}, "
              f"mismatch={mean_mm:.4f}")


def verify_5_no_drive_baseline():
    """What ratio does a random tree have WITHOUT any drive?"""
    print("\n" + "=" * 60)
    print("VERIFY 5: Baseline — random tree ratios without any drive")
    print("=" * 60)

    for depth in [4, 5, 6]:
        ratios = []
        for seed in range(50):
            rng = np.random.RandomState(seed)
            tree = build_random_tree(100.0, depth, rng=rng)
            r = dominant_chain_ratios(tree)
            if r:
                ratios.append(np.mean(r))

        mean_r = np.mean(ratios)
        std_r = np.std(ratios)
        delta = abs(mean_r - PHI) / PHI
        print(f"  depth={depth}: R={mean_r:.4f} +/- {std_r:.4f}, delta_phi={delta:.2%}")


def main():
    print("=" * 70)
    print("exp_31b VERIFICATION — stress testing v3 results")
    print("=" * 70)
    print()

    verify_1_large_sample()
    verify_2_flat_partition_large()
    verify_3_tautology_check()
    verify_4_partial_drive()
    verify_5_no_drive_baseline()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
  Key questions:
    1. Is depth-5/6 convergence to phi robust? → Check verify_1
    2. Is flat partition R≈phi a fluke? → Check verify_2
    3. Are we just confirming exp_01 tautologically? → Check verify_3
    4. Does phi emerge even with weak drive? → Check verify_4
    5. What's the baseline without drive? → Check verify_5
    """)


if __name__ == '__main__':
    main()
