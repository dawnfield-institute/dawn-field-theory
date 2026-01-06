"""
Experiment 02: QBE Constraint Test

Tests whether QBE (dI/dt + dE/dt = λ·QPL) constrains allowed states.

HYPOTHESIS: The Quantum Balance Equation provides physical constraints
on what information-energy configurations are allowed. Objects violating
QBE balance should not exist.

METHODOLOGY:
1. For each JWST observation, compute dE/dt from Eddington accretion
2. Compute required dI/dt for QBE balance
3. Check if implied mass growth rate is physical
4. Test whether QBE restricts the allowed k-levels

SUCCESS CRITERION: All observed SMBHs are in QBE-allowed configurations.
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.qbe_dynamics import (
    qbe_constrained_k,
    test_qbe_constraints,
    compute_qpl,
    compute_dE_dt_eddington
)
from core.pac_cosmology import JWST_OBSERVATIONS, pac_state_at_z
from core.constants import PHI, XI, LAMBDA_QBE, QPL_OMEGA


def run_experiment():
    """Run Experiment 02: QBE Constraint Test."""
    
    print("=" * 70)
    print("EXPERIMENT 02: QBE CONSTRAINT TEST")
    print("=" * 70)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "experiment": "exp_02_qbe_constraint",
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "QBE constrains allowed information-energy states"
    }
    
    print("\nCore equation: dI/dt + dE/dt = λ·QPL(t)")
    print(f"  λ = {LAMBDA_QBE}")
    print(f"  QPL_ω = {QPL_OMEGA} Hz")
    print(f"  QPL (cosmological average) ≈ {1/np.sqrt(2):.4f}")
    
    # =================================================================
    # Test 1: QPL at different cosmic epochs
    # =================================================================
    print("\n" + "-" * 50)
    print("TEST 1: QPL values across cosmic time")
    print("-" * 50)
    
    print(f"\n{'z':<8} {'Age (Gyr)':<12} {'QPL':<12} {'λ·QPL':<12}")
    print("-" * 44)
    
    qpl_results = []
    for z in [0, 2, 5, 8, 10, 12, 15, 20]:
        state = pac_state_at_z(z)
        qpl = compute_qpl(state.cosmic_age_gyr)
        
        print(f"{z:<8} {state.cosmic_age_gyr:<12.3f} {qpl:<12.4f} {LAMBDA_QBE * qpl:<12.4f}")
        
        qpl_results.append({
            "z": z,
            "age_gyr": state.cosmic_age_gyr,
            "qpl": qpl,
            "lambda_qpl": LAMBDA_QBE * qpl
        })
    
    results["test_1_qpl"] = qpl_results
    
    # =================================================================
    # Test 2: QBE constraints on JWST observations
    # =================================================================
    print("\n" + "-" * 50)
    print("TEST 2: QBE constraints on JWST observations")
    print("-" * 50)
    
    qbe_test = test_qbe_constraints(JWST_OBSERVATIONS)
    
    print(f"\n{'Object':<15} {'z':<8} {'log(M)':<10} {'k':<10} {'dE/dt':<12} {'req dI/dt':<12} {'Physical?':<10}")
    print("-" * 87)
    
    for r in qbe_test["results"]:
        phys = "✓" if r["is_physical"] else "✗"
        print(f"{r['name']:<15} {r['z']:<8.2f} {r['observed_log_mass']:<10.2f} {r['k_from_mass']:<10.2f} {r['dE_dt']:<12.2e} {r['required_dI_dt']:<12.2e} {phys:<10}")
    
    print(f"\nPhysical configurations: {qbe_test['n_physical']}/{qbe_test['n_objects']}")
    print(f"QBE balanced: {qbe_test['n_balanced']}/{qbe_test['n_objects']}")
    
    results["test_2_jwst_qbe"] = {
        "n_objects": qbe_test["n_objects"],
        "n_physical": qbe_test["n_physical"],
        "n_balanced": qbe_test["n_balanced"],
        "all_physical": qbe_test["all_physical"],
        "details": qbe_test["results"]
    }
    
    # =================================================================
    # Test 3: What mass range does QBE allow at each redshift?
    # =================================================================
    print("\n" + "-" * 50)
    print("TEST 3: QBE-allowed mass range at each redshift")
    print("-" * 50)
    
    print("\nFor a given z, what log(M) values are QBE-physical?")
    print("(Physical = positive or near-zero growth rate implied)\n")
    
    range_results = []
    
    for z in [8, 10, 12, 15, 20]:
        # Scan log(M) values
        log_m_range = np.linspace(4, 10, 50)
        physical_masses = []
        
        for log_m in log_m_range:
            qbe_check = qbe_constrained_k(z, log_m)
            if qbe_check["is_physical"]:
                physical_masses.append(log_m)
        
        if physical_masses:
            min_m = min(physical_masses)
            max_m = max(physical_masses)
            print(f"z = {z:<5}: log(M) ∈ [{min_m:.1f}, {max_m:.1f}]")
            range_results.append({
                "z": z,
                "min_log_m": min_m,
                "max_log_m": max_m,
                "range": max_m - min_m
            })
        else:
            print(f"z = {z:<5}: No physical masses found")
            range_results.append({
                "z": z,
                "min_log_m": None,
                "max_log_m": None,
                "range": 0
            })
    
    results["test_3_allowed_range"] = range_results
    
    # =================================================================
    # Test 4: Do observations fall within QBE-allowed range?
    # =================================================================
    print("\n" + "-" * 50)
    print("TEST 4: Do JWST observations fall in QBE-allowed range?")
    print("-" * 50)
    
    in_range_results = []
    
    for obs in JWST_OBSERVATIONS:
        z = obs["z"]
        log_m = obs["log_m_bh"]
        
        # Find allowed range at this z
        allowed = None
        for rr in range_results:
            if abs(rr["z"] - z) < 1:  # Closest z
                allowed = rr
                break
        
        if allowed and allowed["min_log_m"] is not None:
            in_range = allowed["min_log_m"] <= log_m <= allowed["max_log_m"]
        else:
            in_range = False
        
        mark = "✓" if in_range else "✗"
        print(f"{obs['name']:<15} z={z:.1f}, log(M)={log_m:.1f} {mark}")
        
        in_range_results.append({
            "name": obs["name"],
            "z": z,
            "log_m": log_m,
            "in_qbe_range": in_range
        })
    
    n_in_range = sum(1 for r in in_range_results if r["in_qbe_range"])
    results["test_4_in_range"] = {
        "n_in_range": n_in_range,
        "n_total": len(JWST_OBSERVATIONS),
        "all_in_range": n_in_range == len(JWST_OBSERVATIONS),
        "details": in_range_results
    }
    
    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 02 SUMMARY")
    print("=" * 70)
    
    all_tests_pass = qbe_test["all_physical"] and (n_in_range == len(JWST_OBSERVATIONS))
    
    results["success"] = all_tests_pass
    results["conclusion"] = (
        "QBE provides physical constraints - all observations are QBE-consistent"
        if all_tests_pass else
        "QBE constraints partially satisfied - some tension with observations"
    )
    
    print(f"\n{'Test':<50} {'Result':<10}")
    print("-" * 60)
    print(f"{'All observations have physical growth rates':<50} {'✓' if qbe_test['all_physical'] else '✗':<10}")
    print(f"{'All observations in QBE-allowed range':<50} {'✓' if results['test_4_in_range']['all_in_range'] else '✗':<10}")
    
    print(f"\n{'Overall Success':<50} {'✓' if all_tests_pass else '✗':<10}")
    print(f"\nConclusion: {results['conclusion']}")
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"exp_02_qbe_constraint_{timestamp}.json"
    
    with open(results_dir / filename, "w") as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\nResults saved to: results/{filename}")
    
    return results


if __name__ == "__main__":
    run_experiment()
