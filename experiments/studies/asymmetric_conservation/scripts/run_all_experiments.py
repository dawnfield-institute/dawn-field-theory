"""
Run All Asymmetric Conservation Experiments

Executes all experiments in sequence and generates summary.
"""

import sys
import json
from datetime import datetime
from pathlib import Path

# Ensure core is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))


def run_all():
    """Run all experiments and collect results."""
    print("=" * 70)
    print(" ASYMMETRIC CONSERVATION - FULL EXPERIMENT SUITE")
    print(f" Started: {datetime.now().isoformat()}")
    print("=" * 70)
    
    results = {}
    
    # Import and run each experiment
    experiments = [
        ('exp_01_sync_baseline', 'Synchronous PAC Baseline'),
        ('exp_02_async_events', 'Asynchronous Event-Driven PAC'),
        ('exp_03_delta_buffer', 'Delta Buffer Dynamics'),
        ('exp_04_frame_asymmetry', 'Frame-Dependent Asymmetry'),
        ('exp_05_xi_from_reconciliation', 'Ξ from Reconciliation'),
        ('exp_06_gaia_integration', 'GAIA Integration'),
        ('exp_07_falsification', 'Falsification Tests'),
    ]
    
    for module_name, description in experiments:
        print(f"\n{'='*70}")
        print(f" Running: {description}")
        print("=" * 70)
        
        try:
            module = __import__(module_name)
            result = module.run_experiment()
            results[module_name] = {
                'status': 'success',
                'summary': result.get('summary', {}),
            }
        except Exception as e:
            print(f"ERROR in {module_name}: {e}")
            import traceback
            traceback.print_exc()
            results[module_name] = {
                'status': 'error',
                'error': str(e),
            }
    
    # Generate summary
    print("\n" + "=" * 70)
    print(" EXPERIMENT SUITE SUMMARY")
    print("=" * 70)
    
    n_success = sum(1 for r in results.values() if r['status'] == 'success')
    n_total = len(results)
    
    print(f"\nExperiments completed: {n_success}/{n_total}")
    
    for name, result in results.items():
        status = '✓' if result['status'] == 'success' else '✗'
        summary = result.get('summary', {})
        key_result = ''
        
        if 'all_tests_passed' in summary:
            key_result = f"passed={summary['all_tests_passed']}"
        elif 'model_validated' in summary:
            key_result = f"validated={summary['model_validated']}"
        elif 'falsified' in summary:
            key_result = f"falsified={summary['falsified']}"
        
        print(f"  {status} {name}: {key_result}")
    
    # Check falsification status
    falsification_result = results.get('exp_07_falsification', {})
    if falsification_result.get('status') == 'success':
        summary = falsification_result.get('summary', {})
        if summary.get('falsified'):
            print("\n⚠️  MODEL FALSIFIED - See exp_07 for details")
        else:
            print("\n✓ MODEL NOT FALSIFIED - Asymmetric conservation validated")
    
    # Save summary
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    summary_file = results_dir / f"suite_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'experiments': results,
            'n_success': n_success,
            'n_total': n_total,
        }, f, indent=2, default=str)
    
    print(f"\nSummary saved to: {summary_file}")
    
    return results


if __name__ == '__main__':
    run_all()
