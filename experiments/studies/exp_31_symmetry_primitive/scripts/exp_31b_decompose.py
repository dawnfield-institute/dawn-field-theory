"""
exp_31b decomposition: isolate WHAT generates phi.

The v3 verification showed:
  - D_{n+1} = S_n mismatch is 65% at equilibrium
  - Yet R ≈ phi (within 0.3%)
  - Alpha barely matters (0.001 and 0.1 give same result)

This is suspicious. Something other than the explicit drive may be
generating phi. This script isolates the components:

  A. Random noise + conservation (no directed drive)
  B. Scale-invariance drive WITHOUT conservation enforcement
  C. Conservation enforcement only (no drive, no perturbation)
  D. Drive on only ONE level of the chain
  E. Shuffle test: does the DIRECTION of the drive matter?
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

from core import PHI

from exp_31b_attractor_generates_phi import (
    PACNode, build_random_tree, get_dominant_chain, dominant_chain_ratios,
    enforce_conservation_up, rescale_tree,
)


def evolve_random_noise(root, n_steps=5000, alpha=0.03):
    """
    Component A: Random perturbation + conservation enforcement.
    NO directed drive — just noise. If phi still emerges, the
    conservation structure of the tree is selecting phi.
    """
    total = root.value
    ratio_history = []
    rng = np.random.RandomState(999)

    for step in range(n_steps):
        # Random perturbation along dominant chain
        node = root
        while node and node.left and node.right:
            noise = alpha * node.value * rng.uniform(-0.1, 0.1)
            node.left.value += noise
            node.right.value -= noise
            if node.left.value < 1e-10:
                node.left.value = 1e-10
            if node.right.value < 1e-10:
                node.right.value = 1e-10
            node = node.left

        enforce_conservation_up(root)
        rescale_tree(root, total)

        ratios = dominant_chain_ratios(root)
        if ratios:
            ratio_history.append(np.mean(ratios))

    return ratio_history


def evolve_drive_no_conservation(root, n_steps=5000, alpha=0.03):
    """
    Component B: Scale-invariance drive WITHOUT bottom-up conservation.
    Only global rescaling to preserve total.
    """
    total = root.value
    ratio_history = []

    for step in range(n_steps):
        chain = get_dominant_chain(root)
        dom_nodes = []
        node = root
        while node and node.left and node.right:
            dom_nodes.append(node)
            node = node.left

        for n in range(len(dom_nodes) - 1):
            parent = dom_nodes[n]
            child = dom_nodes[n + 1]
            S_n = parent.right.value
            D_n1 = child.left.value if child.left else 0
            if D_n1 < 1e-15 or S_n < 1e-15:
                continue
            delta = alpha * (S_n - D_n1)
            if child.left and child.right:
                child.left.value += delta
                child.right.value -= delta
                if child.left.value < 1e-10:
                    child.left.value = 1e-10
                if child.right.value < 1e-10:
                    child.right.value = 1e-10

        # NO enforce_conservation_up — skip it
        # Only global rescaling
        rescale_tree(root, total)

        ratios = dominant_chain_ratios(root)
        if ratios:
            ratio_history.append(np.mean(ratios))

    return ratio_history


def evolve_conservation_only(root, n_steps=5000):
    """
    Component C: Just conservation enforcement, no perturbation at all.
    Should do nothing (tree already satisfies conservation).
    """
    total = root.value
    ratio_history = []

    for step in range(n_steps):
        enforce_conservation_up(root)
        rescale_tree(root, total)

        ratios = dominant_chain_ratios(root)
        if ratios:
            ratio_history.append(np.mean(ratios))

    return ratio_history


def evolve_reverse_drive(root, n_steps=5000, alpha=0.03):
    """
    Component E: REVERSE drive — push D_{n+1} AWAY from S_n.
    If phi still emerges, the drive direction doesn't matter.
    """
    total = root.value
    ratio_history = []

    for step in range(n_steps):
        dom_nodes = []
        node = root
        while node and node.left and node.right:
            dom_nodes.append(node)
            node = node.left

        for n in range(len(dom_nodes) - 1):
            parent = dom_nodes[n]
            child = dom_nodes[n + 1]
            S_n = parent.right.value
            D_n1 = child.left.value if child.left else 0
            if D_n1 < 1e-15 or S_n < 1e-15:
                continue
            # REVERSE: push AWAY from S_n
            delta = alpha * (D_n1 - S_n)  # opposite sign
            if child.left and child.right:
                child.left.value += delta
                child.right.value -= delta
                if child.left.value < 1e-10:
                    child.left.value = 1e-10
                if child.right.value < 1e-10:
                    child.right.value = 1e-10

        enforce_conservation_up(root)
        rescale_tree(root, total)

        ratios = dominant_chain_ratios(root)
        if ratios:
            ratio_history.append(np.mean(ratios))

    return ratio_history


def evolve_all_nodes_noise(root, n_steps=5000, alpha=0.03):
    """
    Component A2: Random perturbation on ALL nodes (not just dominant chain).
    """
    total = root.value
    ratio_history = []
    rng = np.random.RandomState(42)

    def perturb(node):
        if node.left is None:
            return
        noise = alpha * node.value * rng.uniform(-0.1, 0.1)
        node.left.value += noise
        node.right.value -= noise
        if node.left.value < 1e-10:
            node.left.value = 1e-10
        if node.right.value < 1e-10:
            node.right.value = 1e-10
        perturb(node.left)
        perturb(node.right)

    for step in range(n_steps):
        perturb(root)
        enforce_conservation_up(root)
        rescale_tree(root, total)

        ratios = dominant_chain_ratios(root)
        if ratios:
            ratio_history.append(np.mean(ratios))

    return ratio_history


def run_component(name, evolve_fn, depth=6, n_seeds=20, **kwargs):
    """Run a component test across multiple seeds."""
    ratios = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed * 13 + 1)
        tree = build_random_tree(100.0, depth, rng=rng)

        # Record starting ratio
        start_r = dominant_chain_ratios(tree)
        start_mean = np.mean(start_r) if start_r else np.nan

        history = evolve_fn(tree, **kwargs)
        if len(history) > 200:
            eq = np.mean(history[-200:])
            ratios.append(eq)

    if ratios:
        arr = np.array(ratios)
        mean_r = np.mean(arr)
        std_r = np.std(arr)
        delta_phi = abs(mean_r - PHI) / PHI
        delta_2 = abs(mean_r - 2.0) / 2.0
        return mean_r, std_r, delta_phi
    return np.nan, np.nan, np.nan


def main():
    print("=" * 70)
    print("exp_31b DECOMPOSITION: What generates phi?")
    print("=" * 70)

    depth = 6
    n_seeds = 25

    # Baseline: no drive
    print("\n  BASELINE (no drive, no evolution):")
    baseline_ratios = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed * 13 + 1)
        tree = build_random_tree(100.0, depth, rng=rng)
        r = dominant_chain_ratios(tree)
        if r:
            baseline_ratios.append(np.mean(r))
    bl_mean = np.mean(baseline_ratios)
    bl_delta = abs(bl_mean - PHI) / PHI
    print(f"    R = {bl_mean:.6f}, delta_phi = {bl_delta:.2%}")

    print("\n" + "-" * 60)

    components = [
        ("A: Random noise + conservation",
         evolve_random_noise, {}),
        ("A2: Random noise ALL nodes + conservation",
         evolve_all_nodes_noise, {}),
        ("B: Scale-inv drive, NO conservation",
         evolve_drive_no_conservation, {}),
        ("C: Conservation only (no perturbation)",
         evolve_conservation_only, {}),
        ("D: Scale-inv drive + conservation (v3)",
         None, {}),  # will import from main script
        ("E: REVERSE drive + conservation",
         evolve_reverse_drive, {}),
    ]

    # Import v3 drive for component D
    from exp_31b_attractor_generates_phi import evolve_scale_invariance

    results = {}

    for name, fn, kwargs in components:
        if fn is None:
            # Component D: use v3 drive
            fn = evolve_scale_invariance

        mean_r, std_r, delta_phi = run_component(
            name, fn, depth=depth, n_seeds=n_seeds, **kwargs)

        tag = ""
        if delta_phi < 0.05:
            tag = " *** PHI ***"
        elif delta_phi < 0.15:
            tag = " ~ near phi"

        print(f"\n  {name}:")
        print(f"    R = {mean_r:.6f} +/- {std_r:.6f}")
        print(f"    delta_phi = {delta_phi:.2%}{tag}")

        results[name] = {'mean': mean_r, 'std': std_r, 'delta': delta_phi}

    # ============================================================
    # Analysis
    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS: What component generates phi?")
    print("=" * 70)

    print(f"\n  {'Component':<45} {'R':>8} {'delta_phi':>10}")
    print(f"  {'-'*45} {'-'*8} {'-'*10}")
    print(f"  {'Baseline (no evolution)':<45} {bl_mean:>8.4f} {bl_delta:>9.2%}")
    for name, r in results.items():
        tag = " ***" if r['delta'] < 0.05 else ""
        print(f"  {name:<45} {r['mean']:>8.4f} {r['delta']:>9.2%}{tag}")

    print(f"\n  phi = {PHI:.6f}")

    # Determine what's load-bearing
    print("\n  CONCLUSIONS:")
    a_delta = results.get("A: Random noise + conservation", {}).get('delta', 1)
    b_delta = results.get("B: Scale-inv drive, NO conservation", {}).get('delta', 1)
    d_delta = results.get("D: Scale-inv drive + conservation (v3)", {}).get('delta', 1)
    e_delta = results.get("E: REVERSE drive + conservation", {}).get('delta', 1)

    if a_delta < 0.05:
        print("    -> Random noise + conservation gives phi!")
        print("       The DRIVE DIRECTION doesn't matter.")
        print("       Conservation enforcement on the tree selects phi.")
    elif d_delta < 0.05 and b_delta > 0.15:
        print("    -> Drive + conservation needed (both load-bearing)")
    elif d_delta < 0.05 and e_delta < 0.05:
        print("    -> Drive direction doesn't matter!")
        print("       Any perturbation + conservation → phi")

    if b_delta > 0.15:
        print("    -> Conservation is NECESSARY (drive alone fails)")
    if e_delta < 0.05 and d_delta < 0.05:
        print("    -> Forward AND reverse drives both give phi")
        print("       → it's the perturbation, not the direction")


if __name__ == '__main__':
    main()
