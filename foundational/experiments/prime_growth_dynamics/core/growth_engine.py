"""
Prime Growth Dynamics: Core Engine
===================================

Reusable functions for analyzing prime numbers as base cases and
their role in number line growth dynamics.

Key concepts:
- Primes as injection points (from oscillation_attractor_dynamics)
- Composites as crystallization (PAC conservation)
- Growth models: stack, accretion, slot-in
"""

import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from functools import lru_cache
import math


# =============================================================================
# Prime Generation and Factorization
# =============================================================================

def sieve_of_eratosthenes(limit: int) -> List[int]:
    """Generate all primes up to limit using the Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]


@lru_cache(maxsize=10000)
def prime_factorization(n: int) -> Tuple[Tuple[int, int], ...]:
    """
    Return prime factorization as tuple of (prime, exponent) pairs.
    Cached for efficiency.
    """
    if n < 2:
        return ()
    factors = []
    d = 2
    while d * d <= n:
        exp = 0
        while n % d == 0:
            n //= d
            exp += 1
        if exp > 0:
            factors.append((d, exp))
        d += 1
    if n > 1:
        factors.append((n, 1))
    return tuple(factors)


def prime_factors_flat(n: int) -> List[int]:
    """Return list of prime factors with multiplicity."""
    factors = []
    for p, e in prime_factorization(n):
        factors.extend([p] * e)
    return factors


def is_prime(n: int) -> bool:
    """Check if n is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


# =============================================================================
# PAC Conservation Functions
# =============================================================================

def log_pac(n: int) -> float:
    """
    Logarithmic PAC: log(n) = sum(log(p_i))
    This is trivially conserved by definition of multiplication.
    """
    if n < 2:
        return 0.0
    return math.log(n)


def log_pac_from_factors(factors: List[int]) -> float:
    """Sum of logs of prime factors."""
    return sum(math.log(p) for p in factors if p > 0)


def omega(n: int) -> int:
    """
    Number of distinct prime factors (ω function).
    """
    return len(prime_factorization(n))


def big_omega(n: int) -> int:
    """
    Number of prime factors with multiplicity (Ω function).
    This is the "depth" of factorization.
    """
    return sum(e for _, e in prime_factorization(n))


def kolmogorov_complexity_approx(n: int) -> float:
    """
    Approximate Kolmogorov complexity as log2(n).
    
    Deeper measure: shortest program to produce n.
    For primes, K(p) ≈ log2(p) (need to store the number).
    For composites, K(n) could be smaller if factors compress it.
    """
    if n < 2:
        return 0.0
    return math.log2(n)


def factorization_complexity(n: int) -> float:
    """
    Complexity based on factorization structure.
    K_fac(n) = sum(log2(p_i) + log2(e_i)) for each (p_i, e_i)
    
    This measures the "description length" of n via its factors.
    """
    if n < 2:
        return 0.0
    factors = prime_factorization(n)
    total = 0.0
    for p, e in factors:
        total += math.log2(p) + (math.log2(e) if e > 1 else 0)
    return total


def entropy_of_factorization(n: int) -> float:
    """
    Shannon entropy of the factorization.
    
    Treats factorization as a probability distribution:
    p_i = e_i / Ω(n) for each prime factor.
    """
    if n < 2:
        return 0.0
    factors = prime_factorization(n)
    total_exp = sum(e for _, e in factors)
    if total_exp == 0:
        return 0.0
    entropy = 0.0
    for _, e in factors:
        p = e / total_exp
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


# =============================================================================
# SEC Stress Field (from oscillation_attractor_dynamics)
# =============================================================================

def compute_sec_impulse(n: int, primes_set: set) -> float:
    """
    Compute the SEC impulse I(n).
    
    From oscillation_attractor_dynamics:
    - Primes: I(p) ≈ +0.16 (injection)
    - Composites: I(c) ≈ -0.017 (crystallization)
    
    This is a simplified model; actual values depend on position.
    """
    if n in primes_set:
        # Prime injects structure
        return 0.1595 + 0.01 * np.random.randn()  # With noise
    else:
        # Composite crystallizes
        return -0.0169 + 0.01 * np.random.randn()


def compute_sec_stress_field(limit: int, k: int = 9, lambda_decay: float = 0.98) -> np.ndarray:
    """
    Compute the SEC stress field E(n) for n in [2, limit].
    
    E(n) = λ * E(n-1) + I(n)
    
    Parameters:
        limit: Upper bound for computation
        k: Factor base (default 9 from She-Leveque)
        lambda_decay: Decay rate (default 0.98 near critical)
    
    Returns:
        Array of E values for n = 2, 3, ..., limit
    """
    primes = set(sieve_of_eratosthenes(limit))
    E = np.zeros(limit - 1)  # E[i] corresponds to n = i + 2
    
    for i, n in enumerate(range(2, limit + 1)):
        I_n = compute_sec_impulse(n, primes)
        if i == 0:
            E[i] = I_n
        else:
            E[i] = lambda_decay * E[i-1] + I_n
    
    return E


# =============================================================================
# Growth Models
# =============================================================================

def stack_growth_model(n: int, history: List[int]) -> float:
    """
    Stack growth: structure depends on cumulative history.
    
    Returns a "pressure" measure based on all previous primes.
    """
    primes_below = [p for p in history if is_prime(p) and p < n]
    if not primes_below:
        return 0.0
    return sum(1.0 / (n - p) for p in primes_below)


def accretion_growth_model(n: int, recent_k: int = 10) -> float:
    """
    Accretion growth: structure depends on recent frontier.
    
    Returns a "pressure" measure based on recent primes only.
    """
    recent_primes = [p for p in range(max(2, n - recent_k), n) if is_prime(p)]
    if not recent_primes:
        return 0.0
    return sum(1.0 / (n - p) for p in recent_primes)


def slot_model_prediction(n: int) -> float:
    """
    Slot model: primes occupy predicted positions.
    
    Uses log-density approximation: π(n) ≈ n / ln(n)
    Returns probability that n is prime based on position.
    """
    if n < 2:
        return 0.0
    return 1.0 / math.log(n)


# =============================================================================
# Fibonacci / Mersenne Utilities
# =============================================================================

def fibonacci(n: int) -> int:
    """Return nth Fibonacci number (F_1 = F_2 = 1)."""
    if n <= 0:
        return 0
    if n <= 2:
        return 1
    a, b = 1, 1
    for _ in range(n - 2):
        a, b = b, a + b
    return b


def lucas(n: int) -> int:
    """Return nth Lucas number (L_1 = 1, L_2 = 3)."""
    if n <= 0:
        return 2  # L_0 = 2
    if n == 1:
        return 1
    if n == 2:
        return 3
    a, b = 1, 3
    for _ in range(n - 2):
        a, b = b, a + b
    return b


def mersenne(k: int) -> int:
    """Return kth Mersenne number: M_k = 2^k - 1."""
    return (1 << k) - 1


def is_mersenne_prime(p: int) -> bool:
    """Check if p is a Mersenne prime."""
    if not is_prime(p):
        return False
    # Check if p = 2^k - 1 for some k
    n = p + 1
    return n > 0 and (n & (n - 1)) == 0


def fibonacci_factorization(n: int) -> Optional[List[int]]:
    """
    Try to express n as a product of Fibonacci numbers.
    
    Returns list of Fibonacci indices if found, None otherwise.
    """
    if n <= 0:
        return None
    if n == 1:
        return [1]
    
    # Generate Fibonacci up to n
    fibs = []
    i = 1
    while True:
        f = fibonacci(i)
        if f > n:
            break
        if f > 1:  # Skip F_1 = F_2 = 1
            fibs.append((i, f))
        i += 1
    
    # Simple greedy search (not optimal, but illustrative)
    indices = []
    remaining = n
    for idx, f in reversed(fibs):
        while remaining % f == 0:
            remaining //= f
            indices.append(idx)
    
    if remaining == 1 and indices:
        return sorted(indices)
    return None


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_pac_conservation(limit: int, f: Callable[[int], float]) -> Dict:
    """
    Test if function f is conserved under factorization.
    
    For each composite n, check if f(n) ≈ Σf(p_i) for prime factors.
    
    Returns statistics on conservation.
    """
    primes = set(sieve_of_eratosthenes(limit))
    
    errors = []
    for n in range(4, limit + 1):
        if n in primes:
            continue
        factors = prime_factors_flat(n)
        f_n = f(n)
        f_sum = sum(f(p) for p in factors)
        if f_sum != 0:
            rel_error = abs(f_n - f_sum) / abs(f_sum)
            errors.append(rel_error)
    
    return {
        'mean_error': np.mean(errors) if errors else 0,
        'std_error': np.std(errors) if errors else 0,
        'max_error': max(errors) if errors else 0,
        'n_composites': len(errors),
        'conserved': np.mean(errors) < 0.01 if errors else True
    }


def compute_growth_metrics(limit: int) -> Dict:
    """
    Compute various growth metrics for the number line.
    """
    primes = sieve_of_eratosthenes(limit)
    prime_set = set(primes)
    
    # Gap analysis
    gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
    
    # Local vs global density
    local_densities = []
    global_densities = []
    for n in range(10, limit, 100):
        local_count = sum(1 for p in primes if n - 10 <= p <= n + 10)
        local_densities.append(local_count / 20)
        global_densities.append(n / math.log(n))
    
    return {
        'n_primes': len(primes),
        'mean_gap': np.mean(gaps),
        'std_gap': np.std(gaps),
        'max_gap': max(gaps),
        'local_density_mean': np.mean(local_densities),
        'local_density_std': np.std(local_densities),
    }


# =============================================================================
# Mersenne Analysis (from milestone2/exp_16)
# =============================================================================

def analyze_mersenne_fibonacci_connection(max_k: int = 10) -> Dict:
    """
    Analyze the connection between Mersenne numbers and Fibonacci structure.
    
    From milestone2: Fibonacci structure appears at d = 2^k - 1 for k = 1, 2, 3
    """
    results = {}
    
    for k in range(1, max_k + 1):
        m = mersenne(k)
        
        # Check if m is prime
        m_is_prime = is_prime(m)
        
        # Check Fibonacci factorization
        fib_factors = fibonacci_factorization(m)
        
        results[k] = {
            'mersenne': m,
            'is_prime': m_is_prime,
            'fibonacci_factorization': fib_factors,
            'has_fibonacci_structure': fib_factors is not None,
        }
    
    return results
