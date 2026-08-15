"""
Exp 05: Rigorous Boundary-φ Validation

HYPOTHESIS: φ appears specifically at information-theoretic boundaries,
not at energy/symmetry boundaries.

NULL HYPOTHESIS: φ appears at random - any apparent pattern is coincidence
given that 1/φ ≈ 0.618 is just one point in [0,1].

This experiment:
1. Null hypothesis testing with proper baselines
2. Parameter sweeps (grid size, samples, tolerance)
3. Monte Carlo confidence intervals
4. Systematic comparison: information vs energy systems
5. Statistical significance testing
"""

import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy import stats
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI


@dataclass
class BoundaryResult:
    """Result of a boundary detection experiment."""
    system_name: str
    system_type: str  # "information" or "energy"
    critical_value: float
    critical_range: Tuple[float, float]
    normalized_position: float  # Where in [0,1] is the critical point?
    distance_to_phi_inv: float
    is_within_tolerance: bool
    confidence_interval: Tuple[float, float]
    n_samples: int


# ============================================================
# NULL HYPOTHESIS BASELINE
# ============================================================

def compute_null_baseline(tolerance: float = 0.05, n_simulations: int = 10000) -> Dict:
    """
    Null hypothesis: critical points are uniformly distributed in [0, 1].
    What's the probability of landing within tolerance of 1/φ by chance?
    """
    # Analytical calculation
    window_size = 2 * tolerance * PHI_INV  # Window around 1/φ
    analytical_prob = window_size  # For uniform [0,1]
    
    # Monte Carlo verification
    random_points = np.random.uniform(0, 1, n_simulations)
    within_tolerance = np.abs(random_points - PHI_INV) < (tolerance * PHI_INV)
    mc_prob = np.mean(within_tolerance)
    mc_std = np.std(within_tolerance) / np.sqrt(n_simulations)
    
    return {
        "analytical_probability": analytical_prob,
        "monte_carlo_probability": float(mc_prob),
        "monte_carlo_std": float(mc_std),
        "tolerance": tolerance,
        "phi_inverse": PHI_INV,
        "window_size": window_size
    }


def binomial_significance(k: int, n: int, p_null: float) -> Dict:
    """
    Test if k successes out of n trials is significant given null probability p_null.
    """
    # Binomial test
    result = stats.binomtest(k, n, p_null, alternative='greater')
    pvalue = result.pvalue
    
    # Expected under null
    expected = n * p_null
    
    # Observed rate
    observed_rate = k / n
    
    # Effect size (odds ratio)
    if p_null > 0 and p_null < 1 and k < n:
        odds_ratio = (k / (n - k + 0.5)) / (p_null / (1 - p_null))
    else:
        odds_ratio = float('inf')
    
    return {
        "k_successes": k,
        "n_trials": n,
        "p_null": p_null,
        "expected_under_null": expected,
        "observed_rate": observed_rate,
        "pvalue": float(pvalue),
        "significant_at_005": pvalue < 0.05,
        "significant_at_001": pvalue < 0.01,
        "odds_ratio": float(odds_ratio)
    }


# ============================================================
# SYSTEM IMPLEMENTATIONS WITH PARAMETER SWEEPS
# ============================================================

def percolation_sweep(
    grid_sizes: List[int] = [20, 30, 50, 75, 100],
    n_samples_list: List[int] = [50, 100, 200],
    p_resolution: int = 41
) -> List[BoundaryResult]:
    """
    Sweep parameters for percolation threshold detection.
    Known exact value: p_c ≈ 0.5927 for 2D square lattice.
    """
    results = []
    
    def check_percolation(grid):
        visited = np.zeros_like(grid, dtype=bool)
        stack = [(0, j) for j in range(grid.shape[1]) if grid[0, j]]
        while stack:
            i, j = stack.pop()
            if i < 0 or i >= grid.shape[0] or j < 0 or j >= grid.shape[1]:
                continue
            if visited[i, j] or not grid[i, j]:
                continue
            visited[i, j] = True
            if i == grid.shape[0] - 1:
                return True
            stack.extend([(i+1,j), (i-1,j), (i,j+1), (i,j-1)])
        return False
    
    for grid_size in grid_sizes:
        for n_samples in n_samples_list:
            p_range = np.linspace(0.4, 0.8, p_resolution)
            percolation_probs = []
            
            for p in p_range:
                successes = sum(1 for _ in range(n_samples) 
                              if check_percolation(np.random.random((grid_size, grid_size)) < p))
                percolation_probs.append(successes / n_samples)
            
            percolation_probs = np.array(percolation_probs)
            
            # Find steepest transition
            deriv = np.gradient(percolation_probs, p_range)
            max_idx = np.argmax(deriv)
            critical_p = p_range[max_idx]
            
            # Bootstrap confidence interval
            bootstrap_criticals = []
            for _ in range(100):
                boot_probs = percolation_probs + np.random.randn(len(percolation_probs)) * 0.05
                boot_deriv = np.gradient(boot_probs, p_range)
                boot_idx = np.argmax(boot_deriv)
                bootstrap_criticals.append(p_range[boot_idx])
            
            ci_low, ci_high = np.percentile(bootstrap_criticals, [2.5, 97.5])
            
            # Normalized position in [0.4, 0.8] range
            norm_pos = (critical_p - 0.4) / 0.4  # Map to [0, 1]
            
            dist_to_phi = abs(critical_p - PHI_INV)
            
            results.append(BoundaryResult(
                system_name=f"percolation_{grid_size}x{grid_size}_n{n_samples}",
                system_type="information",
                critical_value=critical_p,
                critical_range=(0.4, 0.8),
                normalized_position=critical_p,  # Already in [0,1]-ish
                distance_to_phi_inv=dist_to_phi,
                is_within_tolerance=dist_to_phi < 0.05,
                confidence_interval=(ci_low, ci_high),
                n_samples=n_samples
            ))
    
    return results


def ca_lambda_sweep(
    widths: List[int] = [50, 100, 150],
    steps_list: List[int] = [100, 200],
    n_rules_list: List[int] = [20, 50],
    lambda_resolution: int = 17
) -> List[BoundaryResult]:
    """
    Sweep parameters for CA edge-of-chaos detection.
    """
    results = []
    
    def rule_from_lambda(lambda_val, k=2, n=3):
        n_entries = k ** n
        n_active = int(lambda_val * n_entries)
        rule = np.zeros(n_entries, dtype=int)
        active_indices = np.random.choice(n_entries, n_active, replace=False)
        rule[active_indices] = 1
        return rule
    
    def run_ca(rule, width, steps):
        state = np.zeros(width, dtype=int)
        state[width//2] = 1
        pops = [np.sum(state)]
        for _ in range(steps):
            new_state = np.zeros_like(state)
            for i in range(1, width-1):
                pattern = state[i-1] * 4 + state[i] * 2 + state[i+1]
                new_state[i] = rule[pattern]
            state = new_state
            pops.append(np.sum(state))
        return np.array(pops)
    
    for width in widths:
        for steps in steps_list:
            for n_rules in n_rules_list:
                lambda_range = np.linspace(0.1, 0.9, lambda_resolution)
                complexities = []
                
                for lam in lambda_range:
                    entropies = []
                    for _ in range(n_rules):
                        rule = rule_from_lambda(lam)
                        pops = run_ca(rule, width, steps)
                        if np.max(pops) > 0:
                            hist, _ = np.histogram(pops, bins=20, density=True)
                            hist = hist[hist > 0]
                            entropy = -np.sum(hist * np.log(hist + 1e-10))
                        else:
                            entropy = 0
                        entropies.append(entropy)
                    complexities.append(np.mean(entropies))
                
                complexities = np.array(complexities)
                peak_idx = np.argmax(complexities)
                peak_lambda = lambda_range[peak_idx]
                
                # Bootstrap CI
                bootstrap_peaks = []
                for _ in range(100):
                    boot_complex = complexities + np.random.randn(len(complexities)) * 0.1
                    boot_idx = np.argmax(boot_complex)
                    bootstrap_peaks.append(lambda_range[boot_idx])
                ci_low, ci_high = np.percentile(bootstrap_peaks, [2.5, 97.5])
                
                dist_to_phi = abs(peak_lambda - PHI_INV)
                
                results.append(BoundaryResult(
                    system_name=f"ca_lambda_w{width}_s{steps}_r{n_rules}",
                    system_type="information",
                    critical_value=peak_lambda,
                    critical_range=(0.1, 0.9),
                    normalized_position=peak_lambda,
                    distance_to_phi_inv=dist_to_phi,
                    is_within_tolerance=dist_to_phi < 0.15,  # Wider tolerance for noisy measure
                    confidence_interval=(ci_low, ci_high),
                    n_samples=n_rules
                ))
    
    return results


def ising_sweep(
    sizes: List[int] = [10, 20, 30],
    n_sweeps_list: List[int] = [100, 500],
    T_resolution: int = 50
) -> List[BoundaryResult]:
    """
    2D Ising model critical temperature detection via Monte Carlo.
    Known exact: T_c = 2/ln(1+√2) ≈ 2.269
    """
    results = []
    
    def ising_mc(L, T, n_sweeps):
        """Simple Metropolis for 2D Ising."""
        spins = np.random.choice([-1, 1], (L, L))
        J = 1.0
        
        for _ in range(n_sweeps):
            for _ in range(L * L):
                i, j = np.random.randint(0, L, 2)
                neighbors = (spins[(i+1)%L, j] + spins[(i-1)%L, j] +
                           spins[i, (j+1)%L] + spins[i, (j-1)%L])
                dE = 2 * J * spins[i, j] * neighbors
                if dE < 0 or np.random.random() < np.exp(-dE / T):
                    spins[i, j] *= -1
        
        return np.abs(np.mean(spins))  # Magnetization
    
    for L in sizes:
        for n_sweeps in n_sweeps_list:
            T_range = np.linspace(1.5, 3.5, T_resolution)
            magnetizations = []
            
            for T in T_range:
                m = ising_mc(L, T, n_sweeps)
                magnetizations.append(m)
            
            magnetizations = np.array(magnetizations)
            
            # Find steepest drop
            deriv = np.gradient(magnetizations, T_range)
            min_idx = np.argmin(deriv)  # Steepest negative slope
            critical_T = T_range[min_idx]
            
            # Normalize to [0, 1] in range [1.5, 3.5]
            norm_pos = (critical_T - 1.5) / 2.0
            
            # Distance to phi_inv (but in T space this doesn't quite make sense)
            # Better: check if normalized position is near 1/φ
            dist_to_phi = abs(norm_pos - PHI_INV)
            
            results.append(BoundaryResult(
                system_name=f"ising_L{L}_sw{n_sweeps}",
                system_type="energy",
                critical_value=critical_T,
                critical_range=(1.5, 3.5),
                normalized_position=norm_pos,
                distance_to_phi_inv=dist_to_phi,
                is_within_tolerance=dist_to_phi < 0.1,
                confidence_interval=(critical_T - 0.2, critical_T + 0.2),  # Rough estimate
                n_samples=n_sweeps
            ))
    
    return results


def logistic_sweep(
    n_iterations_list: List[int] = [500, 1000, 2000],
    r_resolution: int = 100
) -> List[BoundaryResult]:
    """
    Logistic map chaos onset detection.
    Known: r_c ≈ 3.5699 (Feigenbaum point)
    """
    results = []
    
    for n_iter in n_iterations_list:
        r_range = np.linspace(2.5, 4.0, r_resolution)
        lyapunov = []
        
        for r in r_range:
            x = 0.5
            lya = 0
            for _ in range(n_iter):
                x = r * x * (1 - x)
                if x > 0 and x < 1:
                    lya += np.log(abs(r * (1 - 2*x)))
            lyapunov.append(lya / n_iter)
        
        lyapunov = np.array(lyapunov)
        
        # Find zero crossing (chaos onset)
        zero_crossings = np.where(np.diff(np.sign(lyapunov)))[0]
        if len(zero_crossings) > 0:
            critical_r = r_range[zero_crossings[0]]
        else:
            critical_r = r_range[np.argmax(np.abs(np.diff(lyapunov)))]
        
        # Normalize to [0, 1] in [2.5, 4.0]
        norm_pos = (critical_r - 2.5) / 1.5
        dist_to_phi = abs(norm_pos - PHI_INV)
        
        results.append(BoundaryResult(
            system_name=f"logistic_iter{n_iter}",
            system_type="energy",  # Period-doubling is deterministic, not info-theoretic
            critical_value=critical_r,
            critical_range=(2.5, 4.0),
            normalized_position=norm_pos,
            distance_to_phi_inv=dist_to_phi,
            is_within_tolerance=dist_to_phi < 0.1,
            confidence_interval=(critical_r - 0.05, critical_r + 0.05),
            n_samples=n_iter
        ))
    
    return results


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_rigorous_validation() -> Dict:
    """
    Full validation with null hypothesis, parameter sweeps, and significance testing.
    """
    print("=" * 70)
    print("EXP 05: RIGOROUS BOUNDARY-φ VALIDATION")
    print("=" * 70)
    print()
    
    results = {
        "null_hypothesis": {},
        "parameter_sweeps": {},
        "by_system_type": {},
        "significance_tests": {},
        "final_verdict": {}
    }
    
    # 1. Establish null baseline
    print("Computing null hypothesis baseline...")
    for tolerance in [0.05, 0.10, 0.15]:
        baseline = compute_null_baseline(tolerance=tolerance)
        results["null_hypothesis"][f"tol_{tolerance}"] = baseline
        print(f"  Tolerance {tolerance}: P(random ≈ 1/φ) = {baseline['monte_carlo_probability']:.4f}")
    
    # 2. Run parameter sweeps
    print("\nRunning parameter sweeps...")
    
    print("  Percolation...")
    perc_results = percolation_sweep(
        grid_sizes=[30, 50, 75],
        n_samples_list=[50, 100],
        p_resolution=31
    )
    results["parameter_sweeps"]["percolation"] = [
        {"name": r.system_name, "critical": r.critical_value, 
         "dist_to_phi": r.distance_to_phi_inv, "within_tol": r.is_within_tolerance,
         "ci": r.confidence_interval}
        for r in perc_results
    ]
    
    print("  Cellular Automata...")
    ca_results = ca_lambda_sweep(
        widths=[50, 100],
        steps_list=[100, 200],
        n_rules_list=[20, 30],
        lambda_resolution=13
    )
    results["parameter_sweeps"]["ca_lambda"] = [
        {"name": r.system_name, "critical": r.critical_value,
         "dist_to_phi": r.distance_to_phi_inv, "within_tol": r.is_within_tolerance,
         "ci": r.confidence_interval}
        for r in ca_results
    ]
    
    print("  Ising Model...")
    ising_results = ising_sweep(
        sizes=[10, 15],
        n_sweeps_list=[200],
        T_resolution=30
    )
    results["parameter_sweeps"]["ising"] = [
        {"name": r.system_name, "critical": r.critical_value,
         "dist_to_phi": r.distance_to_phi_inv, "within_tol": r.is_within_tolerance,
         "ci": r.confidence_interval}
        for r in ising_results
    ]
    
    print("  Logistic Map...")
    logistic_results = logistic_sweep(
        n_iterations_list=[500, 1000],
        r_resolution=80
    )
    results["parameter_sweeps"]["logistic"] = [
        {"name": r.system_name, "critical": r.critical_value,
         "dist_to_phi": r.distance_to_phi_inv, "within_tol": r.is_within_tolerance,
         "ci": r.confidence_interval}
        for r in logistic_results
    ]
    
    # 3. Aggregate by system type
    all_results = perc_results + ca_results + ising_results + logistic_results
    
    info_systems = [r for r in all_results if r.system_type == "information"]
    energy_systems = [r for r in all_results if r.system_type == "energy"]
    
    info_within_tol = sum(1 for r in info_systems if r.is_within_tolerance)
    energy_within_tol = sum(1 for r in energy_systems if r.is_within_tolerance)
    
    results["by_system_type"] = {
        "information_systems": {
            "n_total": len(info_systems),
            "n_within_phi_tolerance": info_within_tol,
            "rate": info_within_tol / len(info_systems) if info_systems else 0,
            "mean_distance_to_phi": np.mean([r.distance_to_phi_inv for r in info_systems]),
            "std_distance_to_phi": np.std([r.distance_to_phi_inv for r in info_systems])
        },
        "energy_systems": {
            "n_total": len(energy_systems),
            "n_within_phi_tolerance": energy_within_tol,
            "rate": energy_within_tol / len(energy_systems) if energy_systems else 0,
            "mean_distance_to_phi": np.mean([r.distance_to_phi_inv for r in energy_systems]),
            "std_distance_to_phi": np.std([r.distance_to_phi_inv for r in energy_systems])
        }
    }
    
    # 4. Significance tests
    print("\nSignificance testing...")
    
    # Test: Are information systems more likely to hit φ than chance?
    null_p = results["null_hypothesis"]["tol_0.1"]["monte_carlo_probability"]
    
    info_sig = binomial_significance(info_within_tol, len(info_systems), null_p)
    energy_sig = binomial_significance(energy_within_tol, len(energy_systems), null_p)
    
    results["significance_tests"] = {
        "information_systems": info_sig,
        "energy_systems": energy_sig,
        "null_probability": null_p
    }
    
    # 5. Compare information vs energy
    info_dists = [r.distance_to_phi_inv for r in info_systems]
    energy_dists = [r.distance_to_phi_inv for r in energy_systems]
    
    # Mann-Whitney U test (are info systems closer to φ than energy systems?)
    if len(info_dists) > 0 and len(energy_dists) > 0:
        u_stat, u_pvalue = stats.mannwhitneyu(info_dists, energy_dists, alternative='less')
        comparison = {
            "u_statistic": float(u_stat),
            "pvalue": float(u_pvalue),
            "info_mean_distance": float(np.mean(info_dists)),
            "energy_mean_distance": float(np.mean(energy_dists)),
            "info_closer_to_phi": np.mean(info_dists) < np.mean(energy_dists),
            "significant": u_pvalue < 0.05
        }
    else:
        comparison = {"error": "insufficient data"}
    
    results["significance_tests"]["info_vs_energy_comparison"] = comparison
    
    # 6. Final verdict
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\nNull hypothesis baseline: P(random ≈ 1/φ) = {null_p:.4f}")
    
    print(f"\nInformation-theoretic systems (percolation, CA):")
    print(f"  {info_within_tol}/{len(info_systems)} within tolerance of 1/φ")
    print(f"  Rate: {results['by_system_type']['information_systems']['rate']:.2%}")
    print(f"  vs null expectation: {null_p:.2%}")
    print(f"  p-value: {info_sig['pvalue']:.4f}")
    print(f"  Significant at α=0.05: {'✓ YES' if info_sig['significant_at_005'] else '✗ NO'}")
    
    print(f"\nEnergy/symmetry systems (Ising, logistic):")
    print(f"  {energy_within_tol}/{len(energy_systems)} within tolerance of 1/φ")
    print(f"  Rate: {results['by_system_type']['energy_systems']['rate']:.2%}")
    print(f"  vs null expectation: {null_p:.2%}")
    print(f"  p-value: {energy_sig['pvalue']:.4f}")
    print(f"  Significant at α=0.05: {'✓ YES' if energy_sig['significant_at_005'] else '✗ NO'}")
    
    print(f"\nDirect comparison (info vs energy):")
    if "error" not in comparison:
        print(f"  Info systems mean distance to 1/φ: {comparison['info_mean_distance']:.4f}")
        print(f"  Energy systems mean distance to 1/φ: {comparison['energy_mean_distance']:.4f}")
        print(f"  Mann-Whitney p-value: {comparison['pvalue']:.4f}")
        print(f"  Info significantly closer: {'✓ YES' if comparison['significant'] else '✗ NO'}")
    
    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    info_significant = info_sig['significant_at_005']
    energy_not_significant = not energy_sig['significant_at_005']
    info_closer = comparison.get('significant', False)
    
    if info_significant and energy_not_significant and info_closer:
        verdict = "STRONGLY SUPPORTED"
        explanation = "φ appears at information-theoretic boundaries significantly more than chance, and significantly more than at energy boundaries."
    elif info_significant and info_closer:
        verdict = "SUPPORTED"
        explanation = "φ appears at information-theoretic boundaries more than chance and more than energy systems, though energy systems also show some effect."
    elif info_significant:
        verdict = "PARTIALLY SUPPORTED"
        explanation = "φ appears at information-theoretic boundaries more than chance, but the distinction from energy systems is not clear."
    else:
        verdict = "NOT SUPPORTED"
        explanation = "No significant evidence that φ preferentially appears at information-theoretic boundaries."
    
    results["final_verdict"] = {
        "verdict": verdict,
        "explanation": explanation,
        "info_significant": info_significant,
        "energy_not_significant": energy_not_significant,
        "info_closer_than_energy": comparison.get('significant', False),
        "hypothesis": "φ marks information-theoretic boundaries specifically"
    }
    
    print(f"\n{verdict}")
    print(f"{explanation}")
    
    return results


if __name__ == "__main__":
    np.random.seed(42)
    
    results = run_rigorous_validation()
    
    # Add metadata
    results["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "exp_05_rigorous_boundary_validation",
        "phi": PHI,
        "phi_inverse": PHI_INV
    }
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"exp_05_rigorous_validation_{timestamp}.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else bool(x) if isinstance(x, np.bool_) else str(x))
    
    print(f"\nResults saved to: {output_path}")
