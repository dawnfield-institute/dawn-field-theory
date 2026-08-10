"""
Shared constants and utilities for asymmetric conservation experiments.
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path

# Dawn Field Constants (derived, not fitted)
PHI = (1 + np.sqrt(5)) / 2          # 1.618033988749895
PHI_INV = 1 / PHI                    # 0.618033988749895
XI = 1 + np.pi / 55                  # 1.0571198664289779
LAMBDA_STAR = 0.618432               # SEC partition threshold
PI = np.pi

# Fibonacci sequence
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]


def print_header(text: str):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def print_subheader(text: str):
    """Print subsection header."""
    print(f"\n--- {text} ---")


def save_results(results: dict, script_name: str):
    """Save results to JSON file in results/ directory."""
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{script_name}_{timestamp}.json"
    filepath = results_dir / filename
    
    # Add metadata
    results["_metadata"] = {
        "script": script_name,
        "timestamp": datetime.now().isoformat(),
        "constants": {
            "PHI": PHI,
            "PHI_INV": PHI_INV,
            "XI": XI,
            "LAMBDA_STAR": LAMBDA_STAR,
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


def format_comparison(name: str, expected: float, actual: float, 
                      tolerance: float = 1e-6) -> str:
    """Format a comparison result."""
    diff = abs(expected - actual)
    status = "✓" if diff < tolerance else "✗"
    pct = (diff / expected * 100) if expected != 0 else 0
    return f"{status} {name}: expected={expected:.10f}, actual={actual:.10f}, diff={diff:.2e} ({pct:.4f}%)"
