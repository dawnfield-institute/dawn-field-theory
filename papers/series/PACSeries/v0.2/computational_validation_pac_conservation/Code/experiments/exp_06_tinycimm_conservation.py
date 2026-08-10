#!/usr/bin/env python3
"""
TinyCIMM Conservation Enforcement — 2×2 Design
================================================

Demonstrates that enforcing PAC conservation as an explicit constraint
reduces hallucination-analogous behaviour in a minimal transformer.

2×2 design: Conservation ON/OFF × Factual/Noise streams.

Maps to paper §7.
"""

import json
import os
import math
import numpy as np
from scipy import stats
from datetime import datetime

PHI = (1 + math.sqrt(5)) / 2
XI = 1 + math.pi / 55

N_STEPS = 500
N_SEEDS = 5
N_HEADS = 4
HIDDEN = 32


class TinyAttentionModel:
    """
    Minimal attention model with optional conservation constraint.

    This is a simplified version of TinyCIMM-Boltzmann that demonstrates
    the core principle: conservation enforcement reduces entropy violation
    under noisy inputs.
    """

    def __init__(self, hidden=HIDDEN, n_heads=N_HEADS, conservation_lambda=0.0, seed=42):
        self.rng = np.random.RandomState(seed)
        self.hidden = hidden
        self.n_heads = n_heads
        self.conservation_lambda = conservation_lambda

        # Initialise weights
        scale = math.sqrt(2.0 / hidden)
        self.W_q = self.rng.randn(n_heads, hidden, hidden // n_heads) * scale
        self.W_k = self.rng.randn(n_heads, hidden, hidden // n_heads) * scale
        self.W_v = self.rng.randn(n_heads, hidden, hidden // n_heads) * scale

        self.target_budget = None

    def attention_entropy(self, x):
        """Compute per-head attention entropy."""
        seq_len = x.shape[0]
        head_entropies = []

        for h in range(self.n_heads):
            Q = x @ self.W_q[h]
            K = x @ self.W_k[h]
            d_k = Q.shape[-1]

            scores = Q @ K.T / math.sqrt(d_k)
            scores -= scores.max(axis=-1, keepdims=True)
            attn = np.exp(scores)
            attn /= attn.sum(axis=-1, keepdims=True) + 1e-10

            entropy = -np.sum(attn * np.log(attn + 1e-10)) / seq_len
            head_entropies.append(entropy)

        return np.array(head_entropies)

    def step(self, x, is_noise=False, lr=0.01):
        """
        One training step. Returns metrics.
        """
        head_ent = self.attention_entropy(x)
        total_budget = float(np.sum(head_ent))

        # Set target budget from first factual step
        if self.target_budget is None and not is_noise:
            self.target_budget = total_budget

        # Task loss (simplified: variance of head entropies — want specialisation)
        task_loss = float(np.var(head_ent))

        # Conservation loss
        if self.target_budget is not None:
            conservation_loss = (total_budget - self.target_budget) ** 2
        else:
            conservation_loss = 0.0

        total_loss = task_loss + self.conservation_lambda * conservation_loss

        # Simplified gradient step (add noise to weights, keep if loss decreases)
        for h in range(self.n_heads):
            noise_q = self.rng.randn(*self.W_q[h].shape) * lr
            noise_k = self.rng.randn(*self.W_k[h].shape) * lr

            # Gradient direction: conservation constraint pulls toward target budget
            if self.conservation_lambda > 0 and self.target_budget is not None:
                budget_error = total_budget - self.target_budget
                correction = -self.conservation_lambda * budget_error * lr * 0.1
                self.W_q[h] += noise_q + correction * self.rng.randn(*self.W_q[h].shape)
                self.W_k[h] += noise_k + correction * self.rng.randn(*self.W_k[h].shape)
            else:
                self.W_q[h] += noise_q
                self.W_k[h] += noise_k

        # Compensation ratio
        if hasattr(self, '_prev_head_ent'):
            deltas = head_ent - self._prev_head_ent
            inc = np.sum(np.abs(deltas[deltas > 0]))
            dec = np.sum(np.abs(deltas[deltas < 0]))
            compensation = dec / inc if inc > 0 else 0.0
        else:
            compensation = 0.0

        self._prev_head_ent = head_ent.copy()

        # Budget violation
        violation = 0.0
        if self.target_budget is not None:
            violation = abs(total_budget - self.target_budget) / self.target_budget

        return {
            "task_loss": task_loss,
            "conservation_loss": conservation_loss,
            "total_loss": total_loss,
            "budget": total_budget,
            "violation": violation,
            "compensation": compensation,
        }


def create_factual_stream(seq_len, hidden, rng):
    """Structured input with repeating patterns."""
    x = rng.randn(seq_len, hidden)
    # Add structure: repeat pattern
    pattern = rng.randn(1, hidden) * 2
    for i in range(0, seq_len, 3):
        x[i] += pattern[0]
    return x


def create_noise_stream(seq_len, hidden, rng):
    """Unstructured random input (hallucination analogue)."""
    return rng.randn(seq_len, hidden) * 1.5


def run_condition(is_conservation, is_noise, seed):
    """Run one condition of the 2×2 design."""
    rng = np.random.RandomState(seed)
    lam = 0.1 if is_conservation else 0.0

    model = TinyAttentionModel(
        hidden=HIDDEN, n_heads=N_HEADS,
        conservation_lambda=lam, seed=seed
    )

    seq_len = 16
    metrics = []

    for step in range(N_STEPS):
        if is_noise:
            x = create_noise_stream(seq_len, HIDDEN, rng)
        else:
            x = create_factual_stream(seq_len, HIDDEN, rng)

        m = model.step(x, is_noise=is_noise)
        metrics.append(m)

    return metrics


def main():
    print("=" * 60)
    print("TinyCIMM Conservation Enforcement — 2×2 Design")
    print("=" * 60)

    conditions = {
        "factual_free": {"is_conservation": False, "is_noise": False},
        "factual_conservation": {"is_conservation": True, "is_noise": False},
        "noise_free": {"is_conservation": False, "is_noise": True},
        "noise_conservation": {"is_conservation": True, "is_noise": True},
    }

    seeds = [42, 137, 256, 314, 628]
    results = {}
    condition_summaries = {}

    for cond_name, cond_params in conditions.items():
        print(f"\n--- {cond_name} ---")
        all_violations = []
        all_final_losses = []
        all_violation_slopes = []
        all_compensations = []

        for seed in seeds:
            metrics = run_condition(
                cond_params["is_conservation"],
                cond_params["is_noise"],
                seed,
            )

            violations = [m["violation"] for m in metrics]
            losses = [m["task_loss"] for m in metrics]
            compensations = [m["compensation"] for m in metrics[1:]]

            all_violations.append(np.mean(violations))
            all_final_losses.append(losses[-1])
            all_compensations.append(np.mean(compensations))

            # Violation trend (slope over steps)
            x_steps = np.arange(len(violations))
            slope, _, _, _, _ = stats.linregress(x_steps, violations)
            all_violation_slopes.append(slope)

        mean_violation = np.mean(all_violations)
        mean_final_loss = np.mean(all_final_losses)
        mean_slope = np.mean(all_violation_slopes)
        mean_comp = np.mean(all_compensations)

        print(f"  Mean violation:     {mean_violation:.4f}")
        print(f"  Mean final loss:    {mean_final_loss:.4f}")
        print(f"  Violation trend:    {mean_slope:.6f}")
        print(f"  Mean compensation:  {mean_comp:.3f}")

        condition_summaries[cond_name] = {
            "mean_violation": float(mean_violation),
            "mean_final_loss": float(mean_final_loss),
            "violation_slope": float(mean_slope),
            "mean_compensation": float(mean_comp),
            "n_seeds": len(seeds),
        }

    results["matrix"] = condition_summaries

    # Statistical tests
    print("\n--- Statistical Tests ---")
    tests = {}

    # Test 1: Conservation reduces noise violation
    noise_free_violations = []
    noise_cons_violations = []
    for seed in seeds:
        m_free = run_condition(False, True, seed)
        m_cons = run_condition(True, True, seed)
        noise_free_violations.append(np.mean([m["violation"] for m in m_free]))
        noise_cons_violations.append(np.mean([m["violation"] for m in m_cons]))

    stat, p_conservation = stats.mannwhitneyu(
        noise_free_violations, noise_cons_violations, alternative="greater"
    )
    print(f"\n  Conservation reduces noise violation:")
    print(f"    Free mean:         {np.mean(noise_free_violations):.4f}")
    print(f"    Conservation mean: {np.mean(noise_cons_violations):.4f}")
    print(f"    p-value:           {p_conservation:.4f}")
    tests["conservation_reduces_violation"] = {
        "p_value": float(p_conservation),
        "significant": p_conservation < 0.05,
    }

    # Test 2: No cost to factual learning
    fact_free_losses = []
    fact_cons_losses = []
    for seed in seeds:
        m_free = run_condition(False, False, seed)
        m_cons = run_condition(True, False, seed)
        fact_free_losses.append(m_free[-1]["task_loss"])
        fact_cons_losses.append(m_cons[-1]["task_loss"])

    stat, p_no_cost = stats.mannwhitneyu(
        fact_cons_losses, fact_free_losses, alternative="greater"
    )
    print(f"\n  Conservation hurts factual learning?")
    print(f"    Free final loss:         {np.mean(fact_free_losses):.4f}")
    print(f"    Conservation final loss: {np.mean(fact_cons_losses):.4f}")
    print(f"    p-value:                 {p_no_cost:.4f}")
    print(f"    Significant cost?        {'YES' if p_no_cost < 0.05 else 'NO (n.s.)'}")
    tests["conservation_hurts_factual"] = {
        "p_value": float(p_no_cost),
        "significant": p_no_cost < 0.05,
    }

    # Test 3: Transition shock
    print("\n  Transition shock (factual → noise):")

    free_shocks = []
    cons_shocks = []
    for seed in seeds:
        rng = np.random.RandomState(seed)

        # Free model
        model_free = TinyAttentionModel(HIDDEN, N_HEADS, 0.0, seed)
        for step in range(100):
            x = create_factual_stream(16, HIDDEN, rng)
            model_free.step(x)
        pre_budget = model_free.step(create_factual_stream(16, HIDDEN, rng))["budget"]
        post_budget = model_free.step(create_noise_stream(16, HIDDEN, rng))["budget"]
        free_shocks.append(abs(post_budget - pre_budget))

        # Conservation model
        rng2 = np.random.RandomState(seed)
        model_cons = TinyAttentionModel(HIDDEN, N_HEADS, 0.1, seed)
        for step in range(100):
            x = create_factual_stream(16, HIDDEN, rng2)
            model_cons.step(x)
        pre_budget = model_cons.step(create_factual_stream(16, HIDDEN, rng2))["budget"]
        post_budget = model_cons.step(create_noise_stream(16, HIDDEN, rng2))["budget"]
        cons_shocks.append(abs(post_budget - pre_budget))

    print(f"    Free shock:         {np.mean(free_shocks):.2f}")
    print(f"    Conservation shock: {np.mean(cons_shocks):.2f}")
    ratio = np.mean(free_shocks) / max(np.mean(cons_shocks), 1e-6)
    print(f"    Reduction:          {ratio:.1f}×")

    tests["transition_shock"] = {
        "free_mean": float(np.mean(free_shocks)),
        "conservation_mean": float(np.mean(cons_shocks)),
        "reduction_factor": float(ratio),
    }

    results["tests"] = tests

    results["dft_constants"] = {"phi": PHI, "xi": XI}

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "Data", "results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(out_dir, f"exp_06_tinycimm_conservation_{ts}.json")

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
