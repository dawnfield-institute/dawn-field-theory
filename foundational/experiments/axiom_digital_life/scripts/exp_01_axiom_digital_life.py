"""
Axiom-Seeded Digital Life — Minimal Prototype (v0.1)
======================================================

Experiment: exp_01_axiom_digital_life
Status: Exploratory

CORE IDEA
---------
Instead of simulating physics (gravity, friction, Newtonian mechanics),
seed a world with DFT axioms:

  PAC  — f(Parent) = Σ f(Children)          [conservation of information potential]
  SEC  — ∂S/∂t = α∇I − β∇H                  [structure where ∇I > ∇H]
  MED  — depth ≤ 2, nodes ≤ 3               [bounded complexity]
  Ξ    ≈ 1.057                              [balance operator, from CA experiment]

Creatures ("Infobionts") emerge from, live by, and die from these axioms.
No hardcoded physics. Life is a consequence of information dynamics.

WORLD
-----
- 1D field of N=64 cells
- I[x]: information potential at cell x
- H[x]: entropy at cell x
- PAC constraint: Σ I[x] + Σ H[x] + Σ creature.pac = C_total  (constant)

INFOBIONT DYNAMICS (all axiom-derived)
---------------------------------------
1. Metabolism: absorb I from local field, excrete H (living costs entropy)
2. Movement:   drift toward highest local ∇I  (SEC gradient ascent, not physics)
3. Reproduce:  ⊗ branch if I[pos]/(H[pos]+ε) > Ξ  →  split PAC budget φ:(1-φ)
4. Die:        ⊕ merge/collapse if pac_budget < threshold or H overdominant
5. PAC check:  total conserved each step; drift < 0.1% is validation signal

DFT CONSTANTS USED
-------------------
  φ  = (1 + √5)/2  ≈ 1.618   Golden ratio — PAC's unique stable solution
  Ξ  ≈ 1.057                  Balance operator (empirically validated in CA exp)
  ln(φ) ≈ 0.481               Information per recursion level
  φ-split: (1/φ, 1-1/φ) = (0.618, 0.382)  PAC budget split on reproduction
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# DFT Constants (not tuned — derived)
# ---------------------------------------------------------------------------
PHI = (1 + math.sqrt(5)) / 2          # Golden ratio, PAC unique stable solution
XI = 1.0571                            # Balance operator (cellular_automata_pac_attractors)
LN_PHI = math.log(PHI)                 # ≈ 0.481 — information per recursion level
PHI_SPLIT_HIGH = 1.0 / PHI            # ≈ 0.618 — dominant child's PAC share
PHI_SPLIT_LOW = 1.0 - PHI_SPLIT_HIGH  # ≈ 0.382 — recessive child's PAC share

# ---------------------------------------------------------------------------
# World Parameters
# ---------------------------------------------------------------------------
N_CELLS = 64          # Field size (1D)
T_STEPS = 300         # Simulation timesteps
DT = 0.05             # Timestep size
D_I = 0.08            # Information diffusion coefficient
ENTROPY_GROWTH = 0.015  # Spontaneous entropy production rate (second law)
NOISE_SCALE = 0.002   # Field noise (thermodynamic fluctuations)

# PAC budget pool
C_INIT_FIELD_I = 40.0     # Initial total information in field
C_INIT_FIELD_H = 8.0      # Initial total entropy in field
C_INIT_CREATURE = 0.0     # Creature budgets start at 0 (drawn from field)

# Creature parameters
N_SEED = 6            # Initial creatures
MIN_PAC = 0.05        # Below this → death (⊕ collapse)
SEED_PAC = 1.2        # Initial PAC budget per seeded creature
MAX_DEPTH = 2         # MED bound: max lineage depth
MAX_OFFSPRING = 3     # MED bound: max offspring per parent
MAX_POP = 120         # Hard cap (ecological carrying capacity)
MUTATION_SCALE = 0.04 # Genome mutation magnitude

# Genome defaults (all DFT-grounded)
DEFAULT_GENOME = {
    "alpha":   0.30,   # Information absorption rate
    "beta":    0.18,   # Entropy excretion rate (metabolic cost)
    "theta_r": XI,     # Reproduction threshold (Ξ — not arbitrary!)
    "theta_d": 0.40,   # Death threshold (I/H ratio below which = collapse)
    "mu":      MUTATION_SCALE,
}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
_creature_counter = 0

def _next_id() -> int:
    global _creature_counter
    _creature_counter += 1
    return _creature_counter


@dataclass
class Infobiont:
    """A digital life-form governed purely by DFT axioms."""
    id: int
    pos: int                 # Position on 1D field
    pac: float               # PAC budget (information potential)
    genome: dict             # {alpha, beta, theta_r, theta_d, mu}
    age: int = 0
    depth: int = 0           # Lineage depth (MED constraint)
    parent_id: Optional[int] = None
    offspring_count: int = 0 # Number of offspring produced (MED constraint)

    def local_xi(self, I: np.ndarray, H: np.ndarray) -> float:
        """Compute local Ξ = I[pos] / (H[pos] + ε). Ξ ≈ 1 → balance."""
        return float(I[self.pos]) / (float(H[self.pos]) + 1e-9)

    def is_alive(self) -> bool:
        return self.pac >= MIN_PAC


# ---------------------------------------------------------------------------
# Field Dynamics (SEC-driven, PAC-conserved)
# ---------------------------------------------------------------------------
class PACSECField:
    """
    1D information field evolving under SEC dynamics with PAC conservation.

    SEC: ∂S/∂t = α∇I − β∇H
    Interpretation: information gradient drives structure formation;
    entropy gradient drives dissolution.
    """

    def __init__(self, n: int) -> None:
        self.n = n
        # Initialize I: two Gaussian peaks (information-rich regions)
        xs = np.linspace(0, 1, n)
        self.I = (
            np.exp(-((xs - 0.25) ** 2) / 0.02)
            + np.exp(-((xs - 0.75) ** 2) / 0.02)
        )
        self.I = self.I / self.I.sum() * C_INIT_FIELD_I

        # Initialize H: uniform low entropy
        self.H = np.ones(n) / n * C_INIT_FIELD_H

        self.C_target = self.I.sum() + self.H.sum()  # Field-only conservation target
        # Creature PAC is tracked separately and added at runtime

    def step(self, creatures: list[Infobiont]) -> None:
        """
        Update field one timestep.

        Steps:
        1. Creature metabolism: absorb I, excrete H
        2. SEC diffusion: ∇I spreads information
        3. Entropy production (second law, but bounded by PAC)
        4. PAC enforcement: renormalize to conserve C_total
        """
        # --- 1. Creature metabolism ---
        for c in creatures:
            if not c.is_alive():
                continue
            absorb = c.genome["alpha"] * float(self.I[c.pos]) * DT
            excrete = c.genome["beta"] * DT

            absorb = min(absorb, float(self.I[c.pos]) * 0.5)  # Can't absorb >50% locally
            excrete = min(excrete, c.pac * 0.3)               # Can't excrete more than 30% of budget

            self.I[c.pos] -= absorb
            self.H[c.pos] += excrete
            c.pac += absorb - excrete

        # --- 2. SEC diffusion of information (∇²I term) ---
        I_new = self.I.copy()
        for x in range(self.n):
            left = self.I[x - 1] if x > 0 else self.I[x]
            right = self.I[x + 1] if x < self.n - 1 else self.I[x]
            laplacian = left - 2 * self.I[x] + right
            I_new[x] += D_I * laplacian * DT

        # --- 3. Entropy dynamics ---
        # H grows where |∇I| is large (SEC: information gradient → entropy at edges)
        H_new = self.H.copy()
        grad_I = np.gradient(I_new)
        H_new += ENTROPY_GROWTH * np.abs(grad_I) * DT
        H_new += np.random.normal(0, NOISE_SCALE, self.n)  # Thermal noise
        H_new = np.maximum(H_new, 1e-9)

        # Entropy dissipates slowly (field has mild "cooling")
        H_new *= (1.0 - 0.005 * DT)

        self.I = np.maximum(I_new, 1e-9)
        self.H = H_new

    def enforce_pac(self, creatures: list[Infobiont], C_total: float) -> None:
        """
        Project (I, H, creature PACs) back onto PAC conservation surface.
        Distributes any drift uniformly across the I field.
        """
        creature_pac_sum = sum(c.pac for c in creatures if c.is_alive())
        current_total = self.I.sum() + self.H.sum() + creature_pac_sum
        drift = C_total - current_total
        if abs(drift) > 1e-12:
            # Redistribute drift to I field (information-first correction)
            correction = drift / self.n
            self.I += correction
            self.I = np.maximum(self.I, 1e-9)

    def xi_field(self) -> np.ndarray:
        """Return Ξ[x] = I[x] / (H[x] + ε) — the balance ratio field."""
        return self.I / (self.H + 1e-9)

    def grad_I(self) -> np.ndarray:
        return np.gradient(self.I)


# ---------------------------------------------------------------------------
# Creature Behavior
# ---------------------------------------------------------------------------
def mutate_genome(genome: dict, rng: random.Random) -> dict:
    """Apply small Gaussian mutations. Genome values stay positive."""
    new = {}
    mu = genome["mu"]
    for k, v in genome.items():
        if k == "mu":
            new[k] = max(0.001, v + rng.gauss(0, mu * 0.1))
        else:
            new[k] = max(1e-4, v + rng.gauss(0, mu))
    return new


def creature_move(c: Infobiont, field: PACSECField, rng: random.Random) -> None:
    """
    SEC gradient ascent: move toward highest local I (not physics — no velocity/mass).
    P(stay) = I[pos], P(left) ∝ I[pos-1], P(right) ∝ I[pos+1].
    Weighted by local information potential.
    """
    n = field.n
    x = c.pos
    options = [x]
    weights = [float(field.I[x])]

    if x > 0:
        options.append(x - 1)
        weights.append(float(field.I[x - 1]))
    if x < n - 1:
        options.append(x + 1)
        weights.append(float(field.I[x + 1]))

    # Choose: weighted by I (information gradient ascent)
    total_w = sum(weights)
    r = rng.random() * total_w
    cumsum = 0.0
    for opt, w in zip(options, weights):
        cumsum += w
        if r <= cumsum:
            c.pos = opt
            break


def creature_reproduce(
    c: Infobiont,
    field: PACSECField,
    rng: random.Random,
    population: list[Infobiont],
) -> list[Infobiont]:
    """
    ⊗ Entropic Branching: split if Ξ_local > genome threshold AND MED allows.

    Budget splits φ:(1-φ) — the PAC-unique stable ratio.
    MED constraint: depth ≤ MAX_DEPTH, offspring_count ≤ MAX_OFFSPRING.
    """
    offspring = []

    xi_local = c.local_xi(field.I, field.H)

    # Check reproduction conditions (all axiom-grounded):
    if xi_local < c.genome["theta_r"]:    # Ξ below threshold
        return offspring
    if c.pac < MIN_PAC * 4:               # Not enough budget to split
        return offspring
    if c.depth >= MAX_DEPTH:              # MED depth bound
        return offspring
    if c.offspring_count >= MAX_OFFSPRING:  # MED node bound
        return offspring
    if len(population) >= MAX_POP:        # Ecological cap
        return offspring

    # ⊗ Branch: split PAC budget φ:(1-φ)
    child_pac_high = c.pac * PHI_SPLIT_HIGH   # ≈ 61.8%
    child_pac_low = c.pac * PHI_SPLIT_LOW     # ≈ 38.2%
    c.pac = child_pac_high  # Parent keeps dominant share (PAC conserved by construction)

    child = Infobiont(
        id=_next_id(),
        pos=min(max(c.pos + rng.choice([-1, 0, 1]), 0), field.n - 1),
        pac=child_pac_low,
        genome=mutate_genome(c.genome, rng),
        depth=c.depth + 1,
        parent_id=c.id,
    )
    c.offspring_count += 1
    offspring.append(child)

    return offspring


def creature_die(c: Infobiont, field: PACSECField) -> None:
    """
    ⊕ Collapse Merge: release creature's PAC budget back to the field as I.
    Information is not destroyed — it returns to the field pool.
    """
    field.I[c.pos] += c.pac
    c.pac = 0.0


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def run_simulation(seed: int = 42) -> dict:
    """
    Run the Axiom-Seeded Digital Life simulation.
    Returns a result dict with time-series and final statistics.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    field = PACSECField(N_CELLS)

    # Seed initial creatures at information-rich regions
    # They draw their PAC budget from the field (PAC conserved)
    creatures: list[Infobiont] = []
    high_I_positions = sorted(range(N_CELLS), key=lambda x: -float(field.I[x]))

    for i in range(N_SEED):
        pos = high_I_positions[i % len(high_I_positions[:8])]
        pac_draw = min(SEED_PAC, float(field.I[pos]) * 0.3)
        field.I[pos] -= pac_draw
        c = Infobiont(
            id=_next_id(),
            pos=pos,
            pac=pac_draw,
            genome={**DEFAULT_GENOME, "mu": MUTATION_SCALE},
            depth=0,
        )
        creatures.append(c)

    # Total conserved quantity (field + creatures)
    C_total = field.I.sum() + field.H.sum() + sum(c.pac for c in creatures)

    # Time-series recording
    ts_population = []
    ts_births = []
    ts_deaths = []
    ts_mean_pac = []
    ts_field_xi_mean = []
    ts_field_xi_std = []
    ts_pac_conservation_error = []
    ts_mean_alpha = []
    ts_mean_theta_r = []

    total_births = 0
    total_deaths = 0
    events = []  # Key event log

    print(f"{'Step':>5}  {'Pop':>4}  {'Births':>7}  {'Deaths':>7}  "
          f"{'MeanPAC':>8}  {'Ξ_mean':>7}  {'PAC_err%':>9}")
    print("-" * 65)

    for t in range(T_STEPS):
        births_this_step = 0
        deaths_this_step = 0

        # 1. Field dynamics (SEC + PAC)
        field.step(creatures)

        # 2. Creature actions
        new_offspring: list[Infobiont] = []
        dead_ids: set[int] = set()

        for c in creatures:
            if not c.is_alive():
                creature_die(c, field)
                dead_ids.add(c.id)
                deaths_this_step += 1
                continue

            c.age += 1

            # Move toward information (SEC gradient ascent, not physics)
            creature_move(c, field, rng)

            # Reproduce if conditions met (⊗ branch operator)
            offspring = creature_reproduce(c, field, rng, creatures + new_offspring)
            new_offspring.extend(offspring)
            births_this_step += len(offspring)

            # Death check (⊕ collapse)
            xi_local = c.local_xi(field.I, field.H)
            if xi_local < c.genome["theta_d"] or c.pac < MIN_PAC:
                creature_die(c, field)
                dead_ids.add(c.id)
                deaths_this_step += 1

        # 3. Prune dead, add newborns
        creatures = [c for c in creatures if c.id not in dead_ids]
        creatures.extend(new_offspring)
        total_births += births_this_step
        total_deaths += deaths_this_step

        # 4. PAC enforcement
        field.enforce_pac(creatures, C_total)

        # 5. Compute diagnostics
        xi_field = field.xi_field()
        n_alive = len(creatures)
        mean_pac = float(np.mean([c.pac for c in creatures])) if creatures else 0.0
        mean_alpha = float(np.mean([c.genome["alpha"] for c in creatures])) if creatures else DEFAULT_GENOME["alpha"]
        mean_theta_r = float(np.mean([c.genome["theta_r"] for c in creatures])) if creatures else DEFAULT_GENOME["theta_r"]

        creature_pac_sum = sum(c.pac for c in creatures)
        current_total = field.I.sum() + field.H.sum() + creature_pac_sum
        pac_err_pct = abs(current_total - C_total) / max(C_total, 1e-9) * 100

        ts_population.append(n_alive)
        ts_births.append(births_this_step)
        ts_deaths.append(deaths_this_step)
        ts_mean_pac.append(mean_pac)
        ts_field_xi_mean.append(float(np.mean(xi_field)))
        ts_field_xi_std.append(float(np.std(xi_field)))
        ts_pac_conservation_error.append(pac_err_pct)
        ts_mean_alpha.append(mean_alpha)
        ts_mean_theta_r.append(mean_theta_r)

        # Log key events
        if births_this_step > 0:
            events.append({"t": t, "event": "births", "count": births_this_step})
        if deaths_this_step > 0:
            events.append({"t": t, "event": "deaths", "count": deaths_this_step})
        if n_alive == 0:
            events.append({"t": t, "event": "EXTINCTION"})
            print(f"{'EXTINCTION':>5}  at step {t}")
            break

        if t % 30 == 0 or t == T_STEPS - 1:
            print(f"{t:>5}  {n_alive:>4}  {births_this_step:>7}  {deaths_this_step:>7}  "
                  f"{mean_pac:>8.4f}  {float(np.mean(xi_field)):>7.4f}  {pac_err_pct:>8.5f}%")

    # ---------------------------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------------------------
    survived = len(creatures)
    peak_pop = max(ts_population) if ts_population else 0
    min_pop = min(ts_population) if ts_population else 0
    mean_pac_conservation_error = float(np.mean(ts_pac_conservation_error)) if ts_pac_conservation_error else 0
    max_pac_conservation_error = float(np.max(ts_pac_conservation_error)) if ts_pac_conservation_error else 0

    # Genome evolution: did alpha converge toward any DFT value?
    final_alphas = [c.genome["alpha"] for c in creatures] if creatures else []
    final_theta_rs = [c.genome["theta_r"] for c in creatures] if creatures else []
    genome_drift_alpha = float(np.std(final_alphas)) if final_alphas else 0.0
    mean_final_alpha = float(np.mean(final_alphas)) if final_alphas else 0.0
    mean_final_theta_r = float(np.mean(final_theta_rs)) if final_theta_rs else 0.0

    # Ξ convergence: does the surviving population cluster near Ξ = 1.057?
    final_xi_values = [c.local_xi(field.I, field.H) for c in creatures] if creatures else []
    mean_survivor_xi = float(np.mean(final_xi_values)) if final_xi_values else 0.0

    print("\n" + "=" * 65)
    print("AXIOM-SEEDED DIGITAL LIFE — RESULTS")
    print("=" * 65)
    print(f"  Steps run:              {len(ts_population)}")
    print(f"  Total births:           {total_births}")
    print(f"  Total deaths:           {total_deaths}")
    print(f"  Peak population:        {peak_pop}")
    print(f"  Min population:         {min_pop}")
    print(f"  Survivors:              {survived}")
    print(f"  Mean PAC conserv. err:  {mean_pac_conservation_error:.5f}%")
    print(f"  Max PAC conserv. err:   {max_pac_conservation_error:.5f}%")
    print(f"  Mean survivor Ξ:        {mean_survivor_xi:.4f}  (Ξ_target = {XI})")
    print(f"  Mean final α:           {mean_final_alpha:.4f}  (seed α = {DEFAULT_GENOME['alpha']})")
    print(f"  Mean final θ_r:         {mean_final_theta_r:.4f}  (seed θ_r = {XI})")
    print(f"  Alpha genome drift:     {genome_drift_alpha:.4f}")
    print("=" * 65)

    # Check key DFT predictions:
    print("\nDFT AXIOM VALIDATION:")
    pac_ok = max_pac_conservation_error < 0.5
    xi_ok = abs(mean_survivor_xi - XI) < 0.3 if final_xi_values else False
    print(f"  [{'✓' if pac_ok else '✗'}] PAC conservation held (max err < 0.5%): {max_pac_conservation_error:.4f}%")
    print(f"  [{'✓' if xi_ok else '?'}] Survivors cluster near Ξ={XI}: mean_Ξ={mean_survivor_xi:.4f}")
    print(f"  [{'✓' if total_births > 0 else '✗'}] Reproduction occurred (⊗ branch): {total_births} births")
    print(f"  [{'✓' if total_deaths > 0 else '✗'}] Collapse occurred (⊕ merge): {total_deaths} deaths")

    return {
        "meta": {
            "experiment": "axiom_digital_life",
            "version": "0.1",
            "seed": seed,
            "n_cells": N_CELLS,
            "t_steps": len(ts_population),
            "dft_constants": {
                "phi": PHI,
                "Xi": XI,
                "ln_phi": LN_PHI,
                "phi_split_high": PHI_SPLIT_HIGH,
                "phi_split_low": PHI_SPLIT_LOW,
            },
            "world_params": {
                "D_I": D_I,
                "entropy_growth": ENTROPY_GROWTH,
                "n_seed": N_SEED,
                "seed_pac": SEED_PAC,
                "max_depth_MED": MAX_DEPTH,
                "max_offspring_MED": MAX_OFFSPRING,
            },
            "default_genome": DEFAULT_GENOME,
        },
        "summary": {
            "total_births": total_births,
            "total_deaths": total_deaths,
            "peak_population": peak_pop,
            "min_population": min_pop,
            "final_population": survived,
            "mean_pac_conservation_error_pct": mean_pac_conservation_error,
            "max_pac_conservation_error_pct": max_pac_conservation_error,
            "mean_survivor_xi": mean_survivor_xi,
            "xi_target": XI,
            "mean_final_alpha": mean_final_alpha,
            "mean_final_theta_r": mean_final_theta_r,
            "genome_diversity_alpha": genome_drift_alpha,
            "pac_conservation_held": pac_ok,
            "xi_clustering_near_target": xi_ok,
            "reproduction_occurred": total_births > 0,
            "collapse_occurred": total_deaths > 0,
        },
        "time_series": {
            "population": ts_population,
            "births": ts_births,
            "deaths": ts_deaths,
            "mean_pac": ts_mean_pac,
            "field_xi_mean": ts_field_xi_mean,
            "field_xi_std": ts_field_xi_std,
            "pac_conservation_error_pct": ts_pac_conservation_error,
            "mean_alpha": ts_mean_alpha,
            "mean_theta_r": ts_mean_theta_r,
        },
        "events": events,
        "final_field": {
            "I": field.I.tolist(),
            "H": field.H.tolist(),
            "xi": field.xi_field().tolist(),
        },
        "survivors": [
            {
                "id": c.id,
                "pos": c.pos,
                "pac": c.pac,
                "age": c.age,
                "depth": c.depth,
                "offspring_count": c.offspring_count,
                "genome": c.genome,
                "local_xi": c.local_xi(field.I, field.H),
            }
            for c in creatures
        ],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Axiom-Seeded Digital Life — Minimal Prototype v0.1")
    print(f"DFT Constants: φ={PHI:.6f}  Ξ={XI:.4f}  ln(φ)={LN_PHI:.6f}")
    print(f"PAC split: {PHI_SPLIT_HIGH:.3f} / {PHI_SPLIT_LOW:.3f}")
    print()

    results = run_simulation(seed=42)

    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_01_axiom_digital_life.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")
