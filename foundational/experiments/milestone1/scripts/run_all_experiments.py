"""
Run All Milestone 1 Experiments

Master script to execute all experiments in the derivation chain.
Produces a comprehensive summary of results.
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Ensure we're in the scripts directory
SCRIPT_DIR = Path(__file__).parent
os.chdir(SCRIPT_DIR)

# All experiments in order
EXPERIMENTS = [
    # Part I: First Principles
    ("exp_01_pac_conservation.py", "PAC Conservation Law"),
    ("exp_02_sec_dynamics.py", "SEC Dynamics"),
    ("exp_03_med_bounds.py", "MED Bounds"),
    
    # Part II: Golden Ratio
    ("exp_04_phi_emergence.py", "φ Emergence"),
    ("exp_05_fibonacci_integers.py", "Fibonacci Integers"),
    ("exp_06_phi_falsification.py", "φ Falsification"),
    ("exp_07_xi_falsification.py", "Ξ Falsification"),
    
    # Part IV: Electromagnetism (key experiments)
    ("exp_12_alpha_formula.py", "Fine Structure Constant"),
    ("exp_13_alpha_falsification.py", "α Falsification"),
]

def run_experiment(script_name, description):
    """Run a single experiment and return success status."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Script:  {script_name}")
    print('='*70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,  # Show output in real time
            text=True,
            timeout=300  # 5 minute timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  ⚠️ TIMEOUT: {script_name}")
        return False
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False

def main():
    """Run all experiments and produce summary."""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║              MILESTONE 1: PAC/SEC → STANDARD MODEL               ║
    ║                                                                   ║
    ║                    Complete Derivation Chain                      ║
    ║                                                                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Working directory: {SCRIPT_DIR}")
    
    # Track results
    results = []
    all_passed = True
    
    for script, description in EXPERIMENTS:
        if not Path(script).exists():
            print(f"\n⚠️ Script not found: {script} - Skipping")
            results.append({
                "script": script,
                "description": description,
                "status": "SKIPPED",
                "passed": False
            })
            continue
        
        success = run_experiment(script, description)
        results.append({
            "script": script,
            "description": description,
            "status": "PASSED" if success else "FAILED",
            "passed": success
        })
        
        if not success:
            all_passed = False
    
    # Summary
    print("\n" + "="*70)
    print("MILESTONE 1 SUMMARY")
    print("="*70)
    
    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)
    
    print(f"\nResults: {passed_count}/{total_count} experiments passed")
    print("\nDetails:")
    print(f"{'Script':<35} {'Status':<15}")
    print("-"*50)
    
    for r in results:
        status_symbol = "✅" if r["passed"] else "❌" if r["status"] == "FAILED" else "⏭️"
        print(f"{r['description']:<35} {status_symbol} {r['status']}")
    
    # Save summary
    summary = {
        "milestone": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "experiments": results,
        "passed": passed_count,
        "total": total_count,
        "all_passed": all_passed,
        "status": "COMPLETE" if all_passed else "INCOMPLETE"
    }
    
    summary_path = SCRIPT_DIR.parent / "results" / "milestone1_summary.json"
    summary_path.parent.mkdir(exist_ok=True)
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    
    # Final verdict
    print("\n" + "="*70)
    if all_passed:
        print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║                    ✅ MILESTONE 1 COMPLETE                        ║
    ║                                                                   ║
    ║  All experiments passed.                                          ║
    ║  PAC/SEC → Standard Model derivation chain validated.            ║
    ║                                                                   ║
    ╚══════════════════════════════════════════════════════════════════╝
        """)
    else:
        print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║                 ⚠️ MILESTONE 1 INCOMPLETE                         ║
    ║                                                                   ║
    ║  Some experiments failed or were skipped.                        ║
    ║  Review output above for details.                                ║
    ║                                                                   ║
    ╚══════════════════════════════════════════════════════════════════╝
        """)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
