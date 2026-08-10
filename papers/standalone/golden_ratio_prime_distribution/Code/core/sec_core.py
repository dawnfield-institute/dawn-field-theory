"""
SEC Core Module — Symbolic Entropy Collapse
============================================

Clean implementation with built-in tracing for reproducibility.

Key components:
- prime_sieve: Generate primes via Sieve of Eratosthenes
- symbolic_entropy: Compute S(n) from factor base divisibility
- entropy_expectation: Local moving average Ŝ(n)
- collapse_impulse: I(n) = Ŝ(n) - S(n)
- stress_field: E(n) = λE(n-1) + I(n)
- compute_sec: Full pipeline with result container
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import json
from datetime import datetime
import subprocess
import sys


# ============================================================================
# CONSTANTS
# ============================================================================

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618033988749895
SQRT5 = np.sqrt(5)

# First 50 primes for factor base experiments
FIRST_50_PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
    157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229
]

def fib(n: int) -> int:
    """Return nth Fibonacci number (1-indexed: F1=1, F2=1, F3=2, ...)"""
    if n <= 0:
        return 0
    if n <= 2:
        return 1
    a, b = 1, 1
    for _ in range(n - 2):
        a, b = b, a + b
    return b

# Precompute Fibonacci numbers
FIBONACCI = {i: fib(i) for i in range(1, 30)}


# ============================================================================
# RESULT CONTAINERS
# ============================================================================

@dataclass
class SECResult:
    """Container for SEC computation results."""
    S: np.ndarray          # Symbolic entropy
    S_hat: np.ndarray      # Expected entropy
    I: np.ndarray          # Collapse impulse
    E: np.ndarray          # Stress field
    prime_mask: np.ndarray # Boolean prime array
    primes: np.ndarray     # List of primes
    n_max: int
    factor_base: List[int]
    window: int
    lam: float


@dataclass  
class ExperimentTrace:
    """Trace container for reproducibility."""
    experiment_id: str
    timestamp: str
    git_commit: str
    parameters: Dict
    results: Dict
    validation: Dict
    dependencies: Dict
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            f.write(self.to_json())


# ============================================================================
# TRACING UTILITIES
# ============================================================================

def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()[:12] if result.returncode == 0 else "unknown"
    except:
        return "unknown"


def get_timestamp() -> str:
    """Get current timestamp in standard format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_dependencies() -> Dict[str, str]:
    """Get versions of key dependencies."""
    deps = {"python": sys.version.split()[0]}
    try:
        import numpy
        deps["numpy"] = numpy.__version__
    except:
        pass
    try:
        import scipy
        deps["scipy"] = scipy.__version__
    except:
        pass
    return deps


def create_trace(experiment_id: str, parameters: Dict, results: Dict, 
                 validation: Dict = None) -> ExperimentTrace:
    """Create a trace object for an experiment."""
    return ExperimentTrace(
        experiment_id=experiment_id,
        timestamp=get_timestamp(),
        git_commit=get_git_commit(),
        parameters=parameters,
        results=results,
        validation=validation or {},
        dependencies=get_dependencies()
    )


# ============================================================================
# CORE SEC FUNCTIONS
# ============================================================================

def prime_sieve(n_max: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate primes up to n_max using Sieve of Eratosthenes."""
    sieve = np.ones(n_max + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n_max**0.5) + 1):
        if sieve[p]:
            sieve[p*p:n_max+1:p] = False
    primes = np.nonzero(sieve)[0]
    return sieve, primes


def symbolic_entropy(n_max: int, factor_base: List[int]) -> np.ndarray:
    """
    Compute symbolic entropy S(n) for integers 0..n_max.
    
    S(n) = (# of factor_base primes dividing n) / len(factor_base)
    
    - Primes not in factor_base → S(n) = 0
    - Highly composite → S(n) approaches 1
    """
    S = np.zeros(n_max + 1, dtype=float)
    k = len(factor_base)
    
    for n in range(2, n_max + 1):
        count = sum(1 for p in factor_base if n % p == 0)
        S[n] = count / k
    
    return S


def entropy_expectation(S: np.ndarray, window: int = 101) -> np.ndarray:
    """
    Compute local moving-average expectation of entropy.
    
    Ŝ(n) = mean of S over [n - window//2, n + window//2]
    """
    n_max = len(S) - 1
    half = window // 2
    
    S_hat = np.zeros_like(S)
    for n in range(2, n_max + 1):
        lo = max(2, n - half)
        hi = min(n_max, n + half)
        S_hat[n] = S[lo:hi+1].mean()
    
    return S_hat


def collapse_impulse(S: np.ndarray, S_hat: np.ndarray) -> np.ndarray:
    """
    Compute collapse impulse I(n) = Ŝ(n) - S(n).
    
    - Positive I(n) → expected complexity > actual → potential collapse
    - Negative I(n) → expected complexity < actual → reinforcement
    """
    return S_hat - S


def stress_field(I: np.ndarray, lam: float = 0.99) -> np.ndarray:
    """
    Compute accumulated stress field E(n).
    
    E(n) = λ·E(n-1) + I(n)
    
    Models tension buildup between prime events.
    """
    E = np.zeros_like(I)
    for n in range(2, len(I)):
        E[n] = lam * E[n-1] + I[n]
    return E


def compute_sec(
    n_max: int = 50000,
    factor_base: Optional[List[int]] = None,
    window: int = 101,
    lam: float = 0.99
) -> SECResult:
    """
    Full SEC computation pipeline.
    
    Args:
        n_max: Maximum integer to analyze
        factor_base: Primes for entropy calculation (default: first 10 primes)
        window: Sliding window size for expectation
        lam: Decay parameter for stress field
    
    Returns:
        SECResult with all computed fields
    """
    if factor_base is None:
        factor_base = FIRST_50_PRIMES[:10]
    
    prime_mask, primes = prime_sieve(n_max)
    S = symbolic_entropy(n_max, factor_base)
    S_hat = entropy_expectation(S, window)
    I = collapse_impulse(S, S_hat)
    E = stress_field(I, lam)
    
    return SECResult(
        S=S, S_hat=S_hat, I=I, E=E,
        prime_mask=prime_mask, primes=primes,
        n_max=n_max, factor_base=factor_base,
        window=window, lam=lam
    )


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def enrichment_analysis(
    scores: np.ndarray,
    prime_mask: np.ndarray,
    quantiles: List[float] = [0.01, 0.05, 0.10]
) -> Dict[float, float]:
    """
    Compute prime enrichment at various score quantiles.
    
    Returns dict mapping quantile -> prime rate in that top fraction.
    """
    order = np.argsort(scores)
    N = len(scores)
    
    results = {}
    for q in quantiles:
        k = max(1, int(N * q))
        top_indices = order[-k:]
        results[q] = float(prime_mask[top_indices].mean())
    
    return results


def compute_phi_threshold(sec: SECResult, restrict_odd: bool = True) -> Dict:
    """
    Compute the stress field partition fraction and compare to 1/φ.
    
    Returns dict with:
    - frac_E_positive: fraction where E(n) > 0
    - error_vs_phi: difference from 1/φ
    - prime_rate_E_pos: prime rate when E > 0
    - prime_rate_E_neg: prime rate when E <= 0
    """
    if restrict_odd:
        idx = np.arange(3, sec.n_max + 1, 2)
    else:
        idx = np.arange(2, sec.n_max + 1)
    
    E = sec.E[idx]
    pm = sec.prime_mask[idx]
    
    frac_positive = float((E > 0).mean())
    prime_rate_pos = float(pm[E > 0].mean()) if (E > 0).any() else 0.0
    prime_rate_neg = float(pm[E <= 0].mean()) if (E <= 0).any() else 0.0
    
    return {
        'frac_E_positive': frac_positive,
        'error_vs_phi': frac_positive - 1/PHI,
        'prime_rate_E_pos': prime_rate_pos,
        'prime_rate_E_neg': prime_rate_neg,
        'ratio': prime_rate_pos / prime_rate_neg if prime_rate_neg > 0 else float('inf')
    }


def run_enrichment_suite(
    sec: SECResult,
    restrict_odd: bool = True
) -> Dict[str, Dict]:
    """
    Run enrichment analysis on multiple SEC-derived scores.
    """
    if restrict_odd:
        idx = np.arange(3, sec.n_max + 1, 2)
    else:
        idx = np.arange(2, sec.n_max + 1)
    
    prime_mask = sec.prime_mask[idx]
    baseline = float(prime_mask.mean())
    
    scores = {
        "abs_I": np.abs(sec.I[idx]),
        "abs_E": np.abs(sec.E[idx]),
        "positive_I": np.clip(sec.I[idx], 0, None),
        "negative_I": np.clip(-sec.I[idx], 0, None),
    }
    
    results = {
        "baseline_prime_rate": baseline,
        "n_analyzed": len(idx),
        "enrichment": {}
    }
    
    for name, arr in scores.items():
        results["enrichment"][name] = enrichment_analysis(arr, prime_mask)
    
    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("SEC Core Module")
    print("=" * 50)
    print(f"φ = {PHI:.10f}")
    print(f"1/φ = {1/PHI:.10f}")
    print(f"First 10 Fibonacci: {[FIBONACCI[i] for i in range(1, 11)]}")
    
    print("\nComputing SEC with default parameters...")
    sec = compute_sec(n_max=50000)
    
    print(f"  n_max: {sec.n_max}")
    print(f"  factor_base: {sec.factor_base}")
    print(f"  primes found: {len(sec.primes)}")
    
    phi_result = compute_phi_threshold(sec)
    print(f"\nφ-threshold analysis:")
    print(f"  frac(E>0): {phi_result['frac_E_positive']:.6f}")
    print(f"  error vs 1/φ: {phi_result['error_vs_phi']:+.6f}")
    print(f"  prime ratio (E>0 / E≤0): {phi_result['ratio']:.2f}x")
