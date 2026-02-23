#!/usr/bin/env python3
"""
Experiment 05 — SEC as Navier-Stokes: Structural Equivalence
==============================================================

PACSeries Paper 5, Section 6

The SEC equation:
    ∂S/∂t = α∇I − β∇H

maps term-by-term onto the Navier-Stokes equation:
    ∂v/∂t + (v·∇)v = −(1/ρ)∇p + ν∇²v + f

Under the identification:
    v ↔ ∇I            (information gradient → velocity field)
    p ↔ H              (entropy → pressure)
    ν ↔ β              (entropy diffusion → viscosity)
    f ↔ external SEC   (external information sources)

MED constraint (depth ≤ 2, nodes ≤ 3) may impose regularity
conditions analogous to bounded enstrophy. THIS IS STRUCTURAL
ANALOGY, NOT A PROOF OF NS REGULARITY.

Source: gravity_from_maxwell_pac/scripts/exp_01_sec_navier_stokes.py
"""

import json
import os
import math
import numpy as np
from datetime import datetime


def sec_field_evolution(I, H, alpha, beta, dx, dt, steps):
    """
    Evolve the 1D SEC equation:
        ∂S/∂t = α ∂I/∂x − β ∂H/∂x
    using explicit finite differences.
    """
    N = len(I)
    S_history = []

    for step in range(steps):
        # Compute gradients (central differences, periodic BC)
        dI = np.zeros(N)
        dH = np.zeros(N)
        for i in range(N):
            dI[i] = (I[(i+1) % N] - I[(i-1) % N]) / (2 * dx)
            dH[i] = (H[(i+1) % N] - H[(i-1) % N]) / (2 * dx)

        dS_dt = alpha * dI - beta * dH
        S = alpha * I - beta * H  # Snapshot

        # Update: I evolves via SEC feedback, H via diffusion
        I_new = I + dt * alpha * dI
        H_new = H + dt * beta * np.gradient(np.gradient(H, dx), dx)  # Diffusion

        I = I_new
        H = H_new

        S_history.append(S.copy())

    return np.array(S_history), I, H


def ns_field_evolution(v, p, nu, dx, dt, steps):
    """
    Evolve simplified 1D Burgers-like equation (NS without full nonlinearity):
        ∂v/∂t = −(1/ρ)∂p/∂x + ν ∂²v/∂x²
    """
    N = len(v)
    v_history = []

    for step in range(steps):
        dp = np.zeros(N)
        d2v = np.zeros(N)
        for i in range(N):
            dp[i] = (p[(i+1) % N] - p[(i-1) % N]) / (2 * dx)
            d2v[i] = (v[(i+1) % N] - 2*v[i] + v[(i-1) % N]) / (dx**2)

        dv_dt = -dp + nu * d2v
        v = v + dt * dv_dt
        v_history.append(v.copy())

    return np.array(v_history), v


def main():
    results = {
        'experiment': 'exp_05_sec_navier_stokes',
        'paper': 'PACSeries Paper 5',
        'section': '6',
        'timestamp': datetime.now().isoformat(),
    }

    print("=" * 60)
    print("SEC–Navier-Stokes Structural Equivalence")
    print("=" * 60)
    print()
    print("  Term-by-term mapping:")
    print("    SEC:  ∂S/∂t  =  α ∇I    −  β ∇H")
    print("    NS:   ∂v/∂t  = −∇p/ρ    +  ν ∇²v")
    print()
    print("  Identifications:")
    print("    v ↔ ∇I       (information gradient → velocity)")
    print("    p ↔ H        (entropy → pressure)")
    print("    ν ↔ β        (entropy diffusion → viscosity)")
    print()

    # Setup matching initial conditions
    N = 128
    dx = 2 * np.pi / N
    dt = 0.001
    steps = 200
    x = np.linspace(0, 2*np.pi, N, endpoint=False)

    alpha = 1.0
    beta = 0.01
    nu = beta  # Match viscosity to entropy diffusion

    # SEC: Gaussian information peak, flat entropy
    I0 = np.exp(-((x - np.pi) ** 2) / 0.5)
    H0 = 0.1 * np.ones(N)

    # NS: velocity = gradient of information, pressure = entropy
    v0 = np.gradient(I0, dx)
    p0 = H0.copy()

    # Evolve both
    S_hist, I_f, H_f = sec_field_evolution(I0.copy(), H0.copy(), alpha, beta, dx, dt, steps)
    v_hist, v_f = ns_field_evolution(v0.copy(), p0.copy(), nu, dx, dt, steps)

    # Compare final velocity field with final information gradient
    grad_I_final = np.gradient(I_f, dx)
    correlation = np.corrcoef(grad_I_final, v_f)[0, 1]

    print("=" * 60)
    print("Numerical Comparison (1D)")
    print("=" * 60)
    print(f"  Grid points:          {N}")
    print(f"  Time steps:           {steps}")
    print(f"  α = {alpha}, β = ν = {beta}")
    print()
    print(f"  Correlation(∇I_final, v_final): {correlation:.6f}")
    print(f"  RMS ∇I_final: {np.sqrt(np.mean(grad_I_final**2)):.6f}")
    print(f"  RMS v_final:  {np.sqrt(np.mean(v_f**2)):.6f}")

    # MED regularity analysis
    print()
    print("=" * 60)
    print("MED Regularity Analysis")
    print("=" * 60)
    print()
    print("  MED constraint: depth ≤ 2, nodes ≤ 3")
    print("  For SEC field evolution, this suggests:")
    print("    • Information gradients remain bounded (depth constraint)")
    print("    • Branching limited to ternary (node constraint)")
    print("    • Structurally analogous to bounded enstrophy")
    print()

    # Enstrophy-like quantity: ∫ (∇v)² dx
    enstrophy_sec = []
    enstrophy_ns = []
    for i in range(steps):
        e_sec = np.sum(np.gradient(S_hist[i], dx)**2) * dx
        e_ns = np.sum(np.gradient(v_hist[i], dx)**2) * dx
        enstrophy_sec.append(float(e_sec))
        enstrophy_ns.append(float(e_ns))

    print(f"  SEC enstrophy (initial/final): {enstrophy_sec[0]:.6f} / {enstrophy_sec[-1]:.6f}")
    print(f"  NS  enstrophy (initial/final): {enstrophy_ns[0]:.6f} / {enstrophy_ns[-1]:.6f}")
    print(f"  SEC bounded throughout: {max(enstrophy_sec) < 100}")
    print(f"  NS  bounded throughout: {max(enstrophy_ns) < 100}")
    print()
    print("  CAVEAT: 1D simulation does not capture 3D NS dynamics.")
    print("  This demonstrates structural analogy, not NS regularity proof.")

    results['main_results'] = {
        'term_mapping': {
            'sec_lhs': '∂S/∂t',
            'ns_lhs': '∂v/∂t',
            'v_maps_to': '∇I (information gradient)',
            'p_maps_to': 'H (entropy)',
            'nu_maps_to': 'β (entropy diffusion)',
        },
        'numerical_comparison': {
            'grid_points': N,
            'time_steps': steps,
            'alpha': alpha,
            'beta_nu': beta,
            'correlation_grad_I_v': round(float(correlation), 6),
        },
        'med_regularity': {
            'sec_enstrophy_initial': round(enstrophy_sec[0], 6),
            'sec_enstrophy_final': round(enstrophy_sec[-1], 6),
            'sec_enstrophy_max': round(max(enstrophy_sec), 6),
            'ns_enstrophy_initial': round(enstrophy_ns[0], 6),
            'ns_enstrophy_final': round(enstrophy_ns[-1], 6),
            'ns_enstrophy_max': round(max(enstrophy_ns), 6),
        },
        'caveat': (
            'This demonstrates structural analogy between SEC and NS in 1D. '
            'MED bounds constrain SEC field complexity, which is structurally '
            'analogous to bounded enstrophy. This does NOT constitute a proof '
            'of NS regularity in 3D.'
        ),
    }

    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'results')
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(results_dir, f'exp_05_sec_navier_stokes_{ts}.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {path}")


if __name__ == '__main__':
    main()
