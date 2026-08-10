"""
Möbius–Confluence Simulation + Diagnostics
=========================================

This script simulates coupled fields on a discretized Möbius strip using:
  • SEC (local collapse): gradient descent on E = α||A−P||^2 + β||∇A||^2
  • MED (global smoothness): Laplacian term in the energy
  • Confluence operator 𝒞: P_{t+1}(u,v) = A_t(u+π, 1−v) + diffusion

It outputs CSV metrics and PNG plots into an output directory (default: ./out).

Dependencies: numpy, pandas, matplotlib
Run:  python mobius_confluence_sim.py --help

Notes:
  • Each plot is a single figure (no subplots), and no explicit colors are set.
  • The ‘energy’ plotted is strictly non-negative: α||A−P||^2 + β||∇A||^2.
  • Includes a 2-cycle detector MSE(P_t, P_{t−2}) and a PAC-style residual proxy.
"""

from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- Core Numerics -----------------------------

def laplacian(f: np.ndarray) -> np.ndarray:
    """Discrete Laplacian with periodic BC in u (axis 0) and Neumann in v (axis 1)."""
    f_u_plus = np.roll(f, -1, axis=0)
    f_u_minus = np.roll(f, 1, axis=0)

    fvp = np.empty_like(f)
    fvm = np.empty_like(f)
    fvp[:, :-1] = f[:, 1:]
    fvp[:, -1] = f[:, -1]  # Neumann top
    fvm[:, 1:] = f[:, :-1]
    fvm[:, 0] = f[:, 0]    # Neumann bottom

    return (f_u_plus + f_u_minus + fvp + fvm - 4.0 * f)


def grad_norm_sq(f: np.ndarray) -> np.ndarray:
    """Pointwise ||∇f||^2 with same BCs (forward differences)."""
    du = np.roll(f, -1, axis=0) - f
    dv = np.empty_like(f)
    dv[:, :-1] = f[:, 1:] - f[:, :-1]
    dv[:, -1] = 0.0  # Neumann in v
    return du * du + dv * dv


def sec_step(A: np.ndarray, P: np.ndarray, dt: float, alpha: float, beta: float) -> np.ndarray:
    """One SEC step (local collapse) under the non-negative energy functional."""
    grad = 2 * alpha * (A - P) - 2 * beta * laplacian(A)
    return A - dt * grad


def confluence_update(A: np.ndarray, twist_shift: int, diffuse: float, rng: np.random.Generator) -> np.ndarray:
    """Apply the Möbius confluence: half-loop shift in u and flip across width v, plus small diffusion."""
    A_shifted = np.roll(A, shift=twist_shift, axis=0)
    A_flip = A_shifted[:, ::-1]
    return A_flip + diffuse * rng.normal(size=A.shape)


def init_fields(U: int, V: int, rng: np.random.Generator, n_strands: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Seed potential P with braided Gaussian strands and start A as a noisy copy."""
    u = np.linspace(0, 2 * np.pi, U, endpoint=False)
    v = np.linspace(0, 1, V, endpoint=True)
    Ugrid, Vgrid = np.meshgrid(u, v, indexing="ij")

    # Smooth background
    P0 = 0.2 * np.sin(2 * Ugrid) * (0.5 + 0.5 * np.cos(np.pi * (Vgrid - 0.5)))

    # Add braided strands
    for _ in range(n_strands):
        phase = rng.uniform(0, 2 * np.pi)
        slope = rng.uniform(-2.0, 2.0)
        width = rng.uniform(0.05, 0.12)
        amp = rng.uniform(0.6, 1.2)
        v_center = 0.5 + 0.25 * np.sin(1.5 * Ugrid + phase) + 0.15 * slope * np.cos(0.5 * Ugrid)
        gauss = amp * np.exp(-((Vgrid - v_center) ** 2) / (2 * width ** 2))
        P0 += gauss

    A0 = P0 + 0.1 * rng.normal(size=P0.shape)
    return P0, A0


@dataclass
class SimMetrics:
    iter: List[int]
    E: List[float]
    align: List[float]
    cons: List[float]
    cycle2_mse: List[float]
    pac_resid: List[float]


def run_sim(
    U: int = 128,
    V: int = 32,
    T: int = 120,
    dt: float = 0.15,
    alpha: float = 1.0,
    beta: float = 0.6,
    diffuse: float = 0.01,
    seed: int = 11,
    n_strands: int = 8,
) -> Dict[str, List[float]]:
    rng = np.random.default_rng(seed)
    twist_shift = U // 2
    P, A = init_fields(U, V, rng, n_strands=n_strands)

    metrics = {"iter": [], "E": [], "align": [], "cons": [], "cycle2_mse": [], "pac_resid": []}
    P_history: List[np.ndarray] = [P.copy()]

    for t in range(T):
        # SEC (local collapse)
        A = sec_step(A, P, dt, alpha, beta)

        # Non-negative energy and diagnostics
        E = alpha * float(np.sum((A - P) ** 2)) + beta * float(np.sum(grad_norm_sq(A)))
        align = float(np.linalg.norm(A - P))
        cons = float(np.sum(A) - np.sum(P))

        # PAC-style proxy: column L2 norms across confluence
        col_norm_pre = np.sqrt(np.sum(A * A, axis=1))
        P_next = confluence_update(A, twist_shift, diffuse, rng)
        col_norm_post = np.sqrt(np.sum(P_next * P_next, axis=1))
        pac_resid = float(np.mean(np.abs(col_norm_post - col_norm_pre)))

        # 2-cycle detector: compare current P with P_{t-2}
        if t >= 2:
            cycle2 = float(np.mean((P - P_history[-2]) ** 2))
        else:
            cycle2 = float("nan")

        # Record metrics
        metrics["iter"].append(t)
        metrics["E"].append(E)
        metrics["align"].append(align)
        metrics["cons"].append(cons)
        metrics["cycle2_mse"].append(cycle2)
        metrics["pac_resid"].append(pac_resid)

        # Advance
        P_history.append(P_next.copy())
        if len(P_history) > 4:
            P_history.pop(0)
        P = P_next

    return metrics


# ----------------------------- Utilities -----------------------------

def ensure_out(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def plot_series(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, path: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def heatmap(grid: np.ndarray, title: str, path: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(grid.T, origin="lower", aspect="auto")
    plt.xlabel("u (loop)")
    plt.ylabel("v (width)")
    plt.title(title)
    plt.colorbar()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# ----------------------------- Phase & Noise Sweeps -----------------------------

def phase_diagram(outdir: str, alphas: List[float], betas: List[float], base: dict) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for a in alphas:
        for b in betas:
            m = run_sim(U=base['U'], V=base['V'], T=base['T'], dt=base['dt'], alpha=a, beta=b,
                        diffuse=base['diffuse'], seed=base['seed'], n_strands=base['n_strands'])
            df = pd.DataFrame(m)
            E_end = float(df['E'].iloc[-1])
            align_end = float(df['align'].iloc[-1])
            cons_std = float(df['cons'].std())
            cyc_tail = float(np.nanmean(df['cycle2_mse'].iloc[-20:]))
            pac_tail = float(np.mean(df['pac_resid'].iloc[-20:]))
            records.append({
                'alpha': a, 'beta': b, 'E_end': E_end, 'align_end': align_end,
                'cons_std': cons_std, 'cycle2_mse_tail': cyc_tail, 'pac_resid_tail': pac_tail
            })
    df_phase = pd.DataFrame(records)
    save_csv(df_phase, os.path.join(outdir, 'phase_diagram.csv'))

    # Heatmaps
    def heat(df: pd.DataFrame, field: str, title: str, fname: str):
        grid = np.zeros((len(betas), len(alphas)))
        for i, b in enumerate(betas):
            for j, a in enumerate(alphas):
                val = df[(df['alpha'] == a) & (df['beta'] == b)][field].values
                grid[i, j] = val[0] if len(val) else np.nan
        plt.figure(figsize=(7, 5))
        plt.imshow(grid, origin='lower', aspect='auto')
        plt.xticks(range(len(alphas)), alphas)
        plt.yticks(range(len(betas)), betas)
        plt.xlabel('alpha')
        plt.ylabel('beta')
        plt.title(title)
        plt.colorbar()
        plt.savefig(os.path.join(outdir, fname), bbox_inches='tight')
        plt.close()

    heat(df_phase, 'align_end', 'Phase: final alignment (lower better)', 'phase_align_end.png')
    heat(df_phase, 'cycle2_mse_tail', 'Phase: 2-cycle MSE tail (lower → period-2)', 'phase_cycle2.png')
    heat(df_phase, 'cons_std', 'Phase: conservation jitter (std)', 'phase_conservation_std.png')

    return df_phase


def noise_sweep(outdir: str, noises: List[float], base: dict) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for d in noises:
        m = run_sim(U=base['U'], V=base['V'], T=base['T'], dt=base['dt'], alpha=base['alpha'], beta=base['beta'],
                    diffuse=d, seed=base['seed'], n_strands=base['n_strands'])
        df = pd.DataFrame(m)
        rows.append({
            'diffuse': d,
            'align_end': float(df['align'].iloc[-1]),
            'cons_std': float(df['cons'].std()),
            'cycle2_mse_tail': float(np.nanmean(df['cycle2_mse'].iloc[-20:])),
            'pac_resid_tail': float(np.mean(df['pac_resid'].iloc[-20:])),
        })
    df_noise = pd.DataFrame(rows)
    save_csv(df_noise, os.path.join(outdir, 'noise_sweep.csv'))

    # Combined line plot
    plt.figure(figsize=(8, 6))
    xs = df_noise['diffuse'].values
    plt.plot(xs, df_noise['align_end'].values, marker='o', label='final alignment')
    plt.plot(xs, df_noise['cons_std'].values, marker='o', label='conservation std')
    plt.plot(xs, df_noise['pac_resid_tail'].values, marker='o', label='PAC resid (tail)')
    plt.xlabel('diffuse')
    plt.ylabel('metric')
    plt.title('Noise robustness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, 'noise_sweep.png'), bbox_inches='tight')
    plt.close()

    return df_noise


# ----------------------------- CLI -----------------------------

def main():
    p = argparse.ArgumentParser(description="Möbius–Confluence simulation and diagnostics")
    p.add_argument('--out', type=str, default='out', help='Output directory')
    p.add_argument('--U', type=int, default=128, help='Grid size along loop (u)')
    p.add_argument('--V', type=int, default=32, help='Grid size across width (v)')
    p.add_argument('--T', type=int, default=120, help='Iterations')
    p.add_argument('--dt', type=float, default=0.15, help='SEC step size')
    p.add_argument('--alpha', type=float, default=1.0, help='SEC attachment weight')
    p.add_argument('--beta', type=float, default=0.6, help='MED smoothness weight')
    p.add_argument('--diffuse', type=float, default=0.01, help='Confluence diffusion')
    p.add_argument('--seed', type=int, default=11, help='RNG seed')
    p.add_argument('--n_strands', type=int, default=8, help='Number of braided strands in initialization')
    p.add_argument('--no_phase', action='store_true', help='Skip (alpha,beta) phase diagram')
    p.add_argument('--no_noise', action='store_true', help='Skip noise sweep')
    args = p.parse_args()

    ensure_out(args.out)

    # Main run
    metrics = run_sim(U=args.U, V=args.V, T=args.T, dt=args.dt, alpha=args.alpha, beta=args.beta,
                      diffuse=args.diffuse, seed=args.seed, n_strands=args.n_strands)
    df_main = pd.DataFrame(metrics)
    save_csv(df_main, os.path.join(args.out, 'metrics_main.csv'))

    # Plots
    it = df_main['iter'].values
    plot_series(it, df_main['E'].values, 'Iteration', 'Energy E', 'Energy (non-negative) over iterations', os.path.join(args.out, 'main_energy.png'))
    plot_series(it, df_main['align'].values, 'Iteration', 'Alignment ||A-P||', 'Alignment over iterations', os.path.join(args.out, 'main_alignment.png'))
    plot_series(it, df_main['cons'].values, 'Iteration', 'Conservation sum(A) - sum(P)', 'Conservation proxy over iterations', os.path.join(args.out, 'main_conservation.png'))
    plot_series(it, df_main['cycle2_mse'].values, 'Iteration', 'MSE(P_t, P_{t-2})', '2-cycle detector over iterations', os.path.join(args.out, 'main_2cycle.png'))
    plot_series(it, df_main['pac_resid'].values, 'Iteration', 'PAC-style residual (column L2 drift)', 'PAC-style residual across confluence', os.path.join(args.out, 'main_pac_resid.png'))

    # Phase diagram
    if not args.no_phase:
        base = dict(U=96, V=24, T=80, dt=args.dt, diffuse=args.diffuse, seed=args.seed, n_strands=args.n_strands,
                    alpha=args.alpha, beta=args.beta)
        alphas = [max(0.1, args.alpha - 0.4), args.alpha, args.alpha + 0.4]
        betas  = [max(0.1, args.beta - 0.4), args.beta, args.beta + 0.6]
        phase_diagram(args.out, alphas, betas, base)

    # Noise sweep
    if not args.no_noise:
        base = dict(U=96, V=24, T=80, dt=args.dt, diffuse=args.diffuse, seed=args.seed, n_strands=args.n_strands,
                    alpha=args.alpha, beta=args.beta)
        noises = [0.0, args.diffuse, max(args.diffuse*3, 0.03), max(args.diffuse*6, 0.06)]
        noise_sweep(args.out, noises, base)

    print(f"Saved outputs in: {args.out}")


if __name__ == '__main__':
    main()
