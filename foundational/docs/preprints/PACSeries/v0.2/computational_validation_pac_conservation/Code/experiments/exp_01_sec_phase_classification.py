#!/usr/bin/env python3
"""
SEC Phase Classification — Zero-Parameter Demonstration
========================================================

Demonstrates that SEC phase boundaries (φ², φ, 1/φ) partition
token predictions into four accuracy regimes using only the
golden ratio — no fitted parameters.

Maps to paper §2.2 and §3.1.
"""

import json
import os
import math
import numpy as np
from datetime import datetime

PHI = (1 + math.sqrt(5)) / 2
INV_PHI = PHI - 1  # 0.618...
XI = 1 + math.pi / 55


def classify_sec_phase(ratio):
    """Classify a top-two probability ratio into SEC phase."""
    if ratio > PHI ** 2:
        return "crystallised"
    elif ratio > PHI:
        return "ordered"
    elif ratio > INV_PHI:
        return "transitional"
    else:
        return "chaotic"


def generate_synthetic_logits(n_tokens, model_quality=0.7, seed=42):
    """
    Generate synthetic logit vectors that mimic transformer output.

    model_quality controls how peaked the distribution is:
    - 1.0 = perfect model (always crystallised)
    - 0.0 = random model (uniform distribution)
    """
    rng = np.random.RandomState(seed)
    vocab_size = 50257  # GPT-2 vocabulary

    results = []
    for i in range(n_tokens):
        # Generate logits with controllable peakedness
        logits = rng.randn(vocab_size)

        # Inject signal: make one token dominant with probability ~ model_quality
        if rng.rand() < model_quality:
            correct_idx = rng.randint(vocab_size)
            # Scale the correct token's logit
            dominance = rng.exponential(3.0) + 1.0
            logits[correct_idx] += dominance
            is_correct = True
        else:
            is_correct = False

        # Softmax
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()

        # Sort descending
        sorted_probs = np.sort(probs)[::-1]
        ratio = sorted_probs[0] / max(sorted_probs[1], 1e-30)

        phase = classify_sec_phase(ratio)
        results.append({
            "ratio": float(ratio),
            "phase": phase,
            "correct": is_correct,
            "top1_prob": float(sorted_probs[0]),
        })

    return results


def compute_phase_accuracy(results):
    """Compute accuracy per SEC phase."""
    phase_counts = {}
    phase_correct = {}

    for r in results:
        p = r["phase"]
        phase_counts[p] = phase_counts.get(p, 0) + 1
        if r["correct"]:
            phase_correct[p] = phase_correct.get(p, 0) + 1

    accuracy = {}
    for phase in ["crystallised", "ordered", "transitional", "chaotic"]:
        total = phase_counts.get(phase, 0)
        correct = phase_correct.get(phase, 0)
        accuracy[phase] = {
            "count": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
        }

    return accuracy


def main():
    print("=" * 60)
    print("SEC Phase Classification — Zero-Parameter Demonstration")
    print("=" * 60)

    # Phase boundaries
    print(f"\nPhase boundaries (golden ratio powers, zero fitted parameters):")
    print(f"  Crystallised: ratio > φ² = {PHI**2:.6f}")
    print(f"  Ordered:      φ < ratio ≤ φ² ({PHI:.6f})")
    print(f"  Transitional: 1/φ < ratio ≤ φ ({INV_PHI:.6f})")
    print(f"  Chaotic:      ratio ≤ 1/φ")

    results = {}

    # Test across four model quality levels (simulating 70M → 1B)
    qualities = {
        "70M_analogue": 0.55,
        "160M_analogue": 0.65,
        "410M_analogue": 0.75,
        "1B_analogue": 0.85,
    }

    n_tokens = 5000

    print(f"\nGenerating {n_tokens} synthetic tokens per model quality level...")

    for name, quality in qualities.items():
        tokens = generate_synthetic_logits(n_tokens, model_quality=quality)
        accuracy = compute_phase_accuracy(tokens)

        print(f"\n  {name} (quality={quality}):")
        for phase in ["crystallised", "ordered", "transitional", "chaotic"]:
            a = accuracy[phase]
            pct = a["accuracy"] * 100
            print(f"    {phase:14s}: {pct:6.1f}% ({a['correct']}/{a['count']})")

        results[name] = {
            "model_quality": quality,
            "n_tokens": n_tokens,
            "phase_accuracy": accuracy,
        }

    # Verify monotonicity
    print("\n" + "=" * 60)
    print("Monotonicity check:")
    all_monotonic = True
    for name, data in results.items():
        acc = data["phase_accuracy"]
        values = [
            acc["chaotic"]["accuracy"],
            acc["transitional"]["accuracy"],
            acc["ordered"]["accuracy"],
            acc["crystallised"]["accuracy"],
        ]
        is_monotonic = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
        status = "✓ MONOTONIC" if is_monotonic else "✗ NOT MONOTONIC"
        print(f"  {name}: {status}")
        if not is_monotonic:
            all_monotonic = False

    results["monotonicity_check"] = {
        "all_monotonic": all_monotonic,
        "phase_boundaries": {
            "crystallised_threshold": PHI ** 2,
            "ordered_threshold": PHI,
            "transitional_threshold": INV_PHI,
        },
        "fitted_parameters": 0,
    }

    results["dft_constants"] = {
        "phi": PHI,
        "inv_phi": INV_PHI,
        "phi_squared": PHI ** 2,
        "xi": XI,
    }

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "Data", "results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(out_dir, f"exp_01_sec_phase_{ts}.json")

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
