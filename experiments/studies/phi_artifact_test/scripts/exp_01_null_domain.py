"""
Exp 01: Null Domain Test

Question: Does PAC/SEC find φ where no structure exists?

If φ is baked into the framework, it should appear even in random data.
If φ is a genuine domain property, it should appear rarely in random data.
"""

import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path


PHI = (1 + np.sqrt(5)) / 2  # 1.618...
PHI_INVERSE = 1 / PHI       # 0.618...
XI = 1 + np.pi / 55         # 1.0571...

# Detection tolerance
TOLERANCE = 0.05  # 5% tolerance for "finding" φ


def detect_phi_in_ratios(ratios: np.ndarray, tolerance: float = TOLERANCE) -> Dict:
    """
    Detect φ-like ratios in a set of values.
    Returns detection statistics.
    """
    if len(ratios) == 0:
        return {"phi_count": 0, "phi_inverse_count": 0, "total": 0, "phi_rate": 0, "phi_inverse_rate": 0}
    
    # Count ratios near φ
    phi_matches = np.abs(ratios - PHI) < (PHI * tolerance)
    phi_inverse_matches = np.abs(ratios - PHI_INVERSE) < (PHI_INVERSE * tolerance)
    
    return {
        "phi_count": int(np.sum(phi_matches)),
        "phi_inverse_count": int(np.sum(phi_inverse_matches)),
        "total": len(ratios),
        "phi_rate": float(np.mean(phi_matches)),
        "phi_inverse_rate": float(np.mean(phi_inverse_matches)),
        "combined_rate": float(np.mean(phi_matches | phi_inverse_matches))
    }


def pac_analysis(sequence: np.ndarray) -> Dict:
    """
    Apply PAC-style analysis to a sequence.
    Extract consecutive ratios and look for φ patterns.
    """
    if len(sequence) < 2:
        return {"error": "sequence too short"}
    
    # Consecutive ratios
    consecutive_ratios = sequence[1:] / (sequence[:-1] + 1e-10)
    consecutive_ratios = consecutive_ratios[np.isfinite(consecutive_ratios)]
    consecutive_ratios = consecutive_ratios[(consecutive_ratios > 0.1) & (consecutive_ratios < 10)]
    
    # Skip-1 ratios (like Fibonacci)
    if len(sequence) >= 3:
        skip1_ratios = sequence[2:] / (sequence[:-2] + 1e-10)
        skip1_ratios = skip1_ratios[np.isfinite(skip1_ratios)]
        skip1_ratios = skip1_ratios[(skip1_ratios > 0.1) & (skip1_ratios < 10)]
    else:
        skip1_ratios = np.array([])
    
    # Local maxima ratios
    if len(sequence) >= 5:
        local_max_idx = []
        for i in range(2, len(sequence) - 2):
            if sequence[i] > sequence[i-1] and sequence[i] > sequence[i+1]:
                local_max_idx.append(i)
        if len(local_max_idx) >= 2:
            local_max_vals = sequence[local_max_idx]
            local_max_ratios = local_max_vals[1:] / (local_max_vals[:-1] + 1e-10)
        else:
            local_max_ratios = np.array([])
    else:
        local_max_ratios = np.array([])
    
    return {
        "consecutive": detect_phi_in_ratios(consecutive_ratios),
        "skip1": detect_phi_in_ratios(skip1_ratios),
        "local_max": detect_phi_in_ratios(local_max_ratios),
    }


def generate_null_domains(n_samples: int = 1000, n_trials: int = 100) -> Dict[str, List]:
    """Generate various null domain test cases."""
    
    domains = {}
    
    # 1. Gaussian random walk
    domains["gaussian_walk"] = []
    for _ in range(n_trials):
        walk = np.cumsum(np.random.randn(n_samples)) + 100  # Offset to keep positive
        walk = np.abs(walk) + 1  # Ensure positive
        domains["gaussian_walk"].append(walk)
    
    # 2. Uniform random
    domains["uniform_random"] = []
    for _ in range(n_trials):
        uniform = np.random.uniform(1, 100, n_samples)
        domains["uniform_random"].append(uniform)
    
    # 3. Exponential random
    domains["exponential_random"] = []
    for _ in range(n_trials):
        exp_data = np.random.exponential(10, n_samples) + 1
        domains["exponential_random"].append(exp_data)
    
    # 4. Power law random (no structure, just distribution)
    domains["powerlaw_random"] = []
    for _ in range(n_trials):
        power = (np.random.pareto(2, n_samples) + 1) * 10
        domains["powerlaw_random"].append(power)
    
    # 5. Shuffled Fibonacci (breaks correlations)
    domains["shuffled_fibonacci"] = []
    for _ in range(n_trials):
        fib = [1, 1]
        for i in range(n_samples - 2):
            fib.append(fib[-1] + fib[-2])
        fib = np.array(fib, dtype=float)
        np.random.shuffle(fib)  # Break structure
        domains["shuffled_fibonacci"].append(fib)
    
    return domains


def generate_structured_domains(n_samples: int = 1000, n_trials: int = 100) -> Dict[str, List]:
    """Generate domains known to have φ structure."""
    
    domains = {}
    
    # 1. Pure Fibonacci
    domains["fibonacci"] = []
    for _ in range(n_trials):
        fib = [1, 1]
        for i in range(n_samples - 2):
            fib.append(fib[-1] + fib[-2])
        # Add small noise to make it realistic
        fib = np.array(fib, dtype=float) * (1 + 0.01 * np.random.randn(n_samples))
        domains["fibonacci"].append(fib)
    
    # 2. Golden spiral radii
    domains["golden_spiral"] = []
    for _ in range(n_trials):
        theta = np.linspace(0, 20 * np.pi, n_samples)
        r = PHI ** (theta / (2 * np.pi))
        r = r * (1 + 0.01 * np.random.randn(n_samples))
        domains["golden_spiral"].append(r)
    
    # 3. Fibonacci-modulated signal
    domains["fib_modulated"] = []
    for _ in range(n_trials):
        t = np.linspace(0, 10, n_samples)
        signal = np.sin(t * PHI) + np.sin(t / PHI) + 2
        signal = signal * (1 + 0.05 * np.random.randn(n_samples))
        domains["fib_modulated"].append(signal)
    
    return domains


def run_null_domain_test(n_samples: int = 500, n_trials: int = 50) -> Dict:
    """
    Main experiment: Compare φ detection in null vs structured domains.
    """
    print("=" * 60)
    print("EXP 01: NULL DOMAIN TEST")
    print("=" * 60)
    print(f"Testing if PAC/SEC finds φ in random/null data...")
    print(f"Samples per trial: {n_samples}, Trials per domain: {n_trials}")
    print()
    
    results = {
        "null_domains": {},
        "structured_domains": {},
        "comparison": {}
    }
    
    # Generate domains
    print("Generating null domains...")
    null_domains = generate_null_domains(n_samples, n_trials)
    
    print("Generating structured domains...")
    structured_domains = generate_structured_domains(n_samples, n_trials)
    
    # Analyze null domains
    print("\nAnalyzing null domains:")
    for domain_name, trials in null_domains.items():
        combined_rates = []
        for trial_data in trials:
            analysis = pac_analysis(trial_data)
            # Average across analysis types
            rates = []
            for key in ["consecutive", "skip1", "local_max"]:
                if key in analysis and "combined_rate" in analysis[key]:
                    rates.append(analysis[key]["combined_rate"])
            if rates:
                combined_rates.append(np.mean(rates))
        
        mean_rate = np.mean(combined_rates) if combined_rates else 0
        std_rate = np.std(combined_rates) if combined_rates else 0
        
        results["null_domains"][domain_name] = {
            "mean_phi_rate": float(mean_rate),
            "std_phi_rate": float(std_rate),
            "n_trials": n_trials
        }
        print(f"  {domain_name}: φ detection rate = {mean_rate:.4f} ± {std_rate:.4f}")
    
    # Analyze structured domains
    print("\nAnalyzing structured domains:")
    for domain_name, trials in structured_domains.items():
        combined_rates = []
        for trial_data in trials:
            analysis = pac_analysis(trial_data)
            rates = []
            for key in ["consecutive", "skip1", "local_max"]:
                if key in analysis and "combined_rate" in analysis[key]:
                    rates.append(analysis[key]["combined_rate"])
            if rates:
                combined_rates.append(np.mean(rates))
        
        mean_rate = np.mean(combined_rates) if combined_rates else 0
        std_rate = np.std(combined_rates) if combined_rates else 0
        
        results["structured_domains"][domain_name] = {
            "mean_phi_rate": float(mean_rate),
            "std_phi_rate": float(std_rate),
            "n_trials": n_trials
        }
        print(f"  {domain_name}: φ detection rate = {mean_rate:.4f} ± {std_rate:.4f}")
    
    # Compute comparison statistics
    null_rates = [v["mean_phi_rate"] for v in results["null_domains"].values()]
    structured_rates = [v["mean_phi_rate"] for v in results["structured_domains"].values()]
    
    null_mean = np.mean(null_rates)
    structured_mean = np.mean(structured_rates)
    separation_ratio = structured_mean / (null_mean + 1e-10)
    
    results["comparison"] = {
        "null_mean": float(null_mean),
        "structured_mean": float(structured_mean),
        "separation_ratio": float(separation_ratio),
        "artifact_hypothesis_supported": separation_ratio < 3.0,
        "threshold": 3.0
    }
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Mean φ detection in NULL domains:       {null_mean:.4f}")
    print(f"Mean φ detection in STRUCTURED domains: {structured_mean:.4f}")
    print(f"Separation ratio:                       {separation_ratio:.2f}×")
    print()
    
    if separation_ratio < 3.0:
        print("⚠️  ARTIFACT HYPOTHESIS SUPPORTED")
        print(f"   φ appears too frequently in null domains (ratio < 3×)")
        print(f"   This suggests φ may be baked into the analysis method.")
    else:
        print("✓  ARTIFACT HYPOTHESIS REJECTED")
        print(f"   φ detection is {separation_ratio:.1f}× higher in structured domains")
        print(f"   This suggests φ detection reflects genuine domain structure.")
    
    return results


def calculate_baseline_rate():
    """
    Calculate expected φ detection rate by pure chance.
    
    For ratios uniformly distributed in [0.1, 10], what fraction
    fall within 5% of φ (1.618) or 1/φ (0.618)?
    """
    # φ ± 5% = [1.537, 1.699]
    # 1/φ ± 5% = [0.587, 0.649]
    
    # Range width: 10 - 0.1 = 9.9
    phi_range = (1.618 * 1.05) - (1.618 * 0.95)  # 0.162
    phi_inv_range = (0.618 * 1.05) - (0.618 * 0.95)  # 0.062
    
    # But uniform in log-space is more realistic for ratios
    # Using linear for simplicity
    total_range = 9.9
    expected_rate = (phi_range + phi_inv_range) / total_range
    
    print(f"\nBaseline expectation (uniform random ratios in [0.1, 10]):")
    print(f"  Expected φ detection rate: {expected_rate:.4f} ({expected_rate*100:.2f}%)")
    
    return expected_rate


if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Calculate baseline expectation
    baseline = calculate_baseline_rate()
    
    # Run main experiment
    results = run_null_domain_test(n_samples=500, n_trials=50)
    
    # Add metadata
    results["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "exp_01_null_domain",
        "hypothesis": "φ is baked into PAC/SEC framework",
        "baseline_expectation": baseline,
        "tolerance": TOLERANCE,
        "phi": PHI,
        "phi_inverse": PHI_INVERSE
    }
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"exp_01_null_domain_{timestamp}.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    
    print(f"\nResults saved to: {output_path}")
