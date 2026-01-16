#!/usr/bin/env python3
"""
exp_01_sec_wave_speed.py

Derive the speed of light from SEC (Symbolic Entropy Collapse) parameters.

The SEC wave equation:
    ∂²S/∂t² = (αγ + βδ)∇²S

produces wave speed c_SEC² = αγ + βδ.

We test three hypotheses for SEC parameter structure:
1. Symmetric: α=β, γ=δ
2. Xi-balanced: α/β = Ξ ≈ 1.0571
3. Phi-structured: α/γ = φ

Author: Peter Lorne Groom, Claude (Anthropic)
Date: January 15, 2026
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy.constants import c, pi

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
from constants import (
    PHI, XI, XI_MIN, XI_MEAN,
    FIB, F_7, F_10,
    ALPHA_SYM, GAMMA_SYM,
    xi_balanced_params, phi_structured_params,
    verify_wave_speed
)

# =============================================================================
# EXPERIMENT: SEC PARAMETER MODELS
# =============================================================================

def model_symmetric():
    """
    Hypothesis 1: Full symmetry between information and entropy.
    
    α = β (same coupling to gradients)
    γ = δ (same response to structure)
    
    Then: c² = 2αγ → αγ = c²/2 → α = γ = c/√2
    """
    alpha = c / np.sqrt(2)
    return {
        'name': 'Symmetric',
        'hypothesis': 'α=β, γ=δ (full I↔H symmetry)',
        'alpha': alpha,
        'beta': alpha,
        'gamma': alpha,
        'delta': alpha,
        'ratios': {
            'alpha/beta': 1.0,
            'gamma/delta': 1.0,
            'alpha/gamma': 1.0
        }
    }


def model_xi_balanced():
    """
    Hypothesis 2: Information slightly dominates entropy by factor Ξ.
    
    α/β = Ξ ≈ 1.0571 (from PAC balance operator)
    γ = δ (symmetric response)
    
    Then: c² = βΞγ + βγ = βγ(Ξ+1)
    If γ = β: c² = β²(Ξ+1) → β = c/√(Ξ+1)
    """
    v0 = c / np.sqrt(XI + 1)
    return {
        'name': 'Xi-balanced',
        'hypothesis': 'α/β = Ξ (information dominance)',
        'alpha': XI * v0,
        'beta': v0,
        'gamma': v0,
        'delta': v0,
        'ratios': {
            'alpha/beta': XI,
            'gamma/delta': 1.0,
            'alpha/gamma': XI
        }
    }


def model_phi_structured():
    """
    Hypothesis 3: Golden ratio structure from PAC recursion.
    
    Adjacent PAC levels have ratio φ.
    Perhaps: α/γ = φ (cause→effect ratio)
    
    If α/γ = φ and β/δ = 1/φ:
    c² = αγ + βδ = γ²φ + δ²/φ
    
    With γ = δ: c² = γ²(φ + 1/φ) = γ²(φ² + 1)/φ = γ² × 2.618/1.618
    → γ = c × √(φ/(φ² + 1)) = c × √(φ/(φ+2))
    """
    # φ + 1/φ = φ + (φ-1) = 2φ - 1 ≈ 2.236
    # Actually: φ + 1/φ = (φ² + 1)/φ = (φ+1+1)/φ = (φ+2)/φ
    factor = PHI / (PHI + 2)  # ≈ 0.447
    gamma = c * np.sqrt(factor)
    alpha = PHI * gamma
    delta = gamma
    beta = gamma / PHI
    
    return {
        'name': 'Phi-structured',
        'hypothesis': 'α/γ = φ, β/δ = 1/φ (golden hierarchy)',
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'delta': delta,
        'ratios': {
            'alpha/beta': PHI**2,
            'gamma/delta': 1.0,
            'alpha/gamma': PHI
        }
    }


def model_fibonacci_nested():
    """
    Hypothesis 4: Fibonacci ratios at multiple levels.
    
    Using F₇ = 13 (gauge depth) and F₁₀ = 55 (Xi depth):
    α/β = F₁₀/F₇ ≈ 4.23 ≈ φ³
    """
    ratio_ab = F_10 / F_7  # 55/13 ≈ 4.231
    v0 = c / np.sqrt(ratio_ab + 1)  # ≈ c/2.29
    
    return {
        'name': 'Fibonacci-nested',
        'hypothesis': f'α/β = F₁₀/F₇ = {F_10}/{F_7} ≈ φ³',
        'alpha': ratio_ab * v0,
        'beta': v0,
        'gamma': v0,
        'delta': v0,
        'ratios': {
            'alpha/beta': ratio_ab,
            'gamma/delta': 1.0,
            'alpha/gamma': ratio_ab
        }
    }


def model_xi_mean():
    """
    Hypothesis 5: Use geometric mean Xi.
    
    Ξ_mean = √(Ξ_PAC × Ξ_min) ≈ 1.0289
    """
    v0 = c / np.sqrt(XI_MEAN + 1)
    
    return {
        'name': 'Xi-mean',
        'hypothesis': f'α/β = Ξ_mean = √(Ξ×Ξ_min) ≈ {XI_MEAN:.4f}',
        'alpha': XI_MEAN * v0,
        'beta': v0,
        'gamma': v0,
        'delta': v0,
        'ratios': {
            'alpha/beta': XI_MEAN,
            'gamma/delta': 1.0,
            'alpha/gamma': XI_MEAN
        }
    }


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_model(model):
    """Analyze a SEC parameter model."""
    # Calculate wave speed
    c_sec_sq = model['alpha'] * model['gamma'] + model['beta'] * model['delta']
    c_sec = np.sqrt(c_sec_sq)
    
    # Error from true c
    error = abs(c_sec - c) / c
    
    # Calculate derived quantities
    # ε₀μ₀ = 1/c² in SI
    eps_mu = 1 / c_sec_sq
    
    # Individual "permittivity" and "permeability" analogs
    # If we map: ε₀ ↔ 1/(αγ), μ₀ ↔ 1/(βδ)
    # Then c² = 1/(ε₀μ₀) requires αγ × βδ relation... 
    # Actually simpler: αγ + βδ = c²
    
    return {
        'c_sec': c_sec,
        'c_sec_sq': c_sec_sq,
        'c_true': c,
        'error_abs': c_sec - c,
        'error_rel': error,
        'error_pct': 100 * error,
        'eps_mu_product': eps_mu
    }


def main():
    """Run SEC wave speed experiment."""
    print("=" * 70)
    print("EXP 01: SPEED OF LIGHT FROM SEC PARAMETERS")
    print("=" * 70)
    
    print(f"\nTarget: c = {c:.6e} m/s")
    print(f"Balance operator: Ξ = 1 + π/55 = {XI:.10f}")
    print(f"Golden ratio: φ = {PHI:.10f}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'target_c': c,
        'constants': {
            'PHI': PHI,
            'XI': XI,
            'XI_MEAN': XI_MEAN,
            'F_7': F_7,
            'F_10': F_10
        },
        'models': []
    }
    
    # Test all models
    models = [
        model_symmetric(),
        model_xi_balanced(),
        model_phi_structured(),
        model_fibonacci_nested(),
        model_xi_mean()
    ]
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n{'Model':<20} | {'c_SEC (m/s)':<15} | {'Error':<12} | Key Ratio")
    print("-" * 70)
    
    for model in models:
        analysis = analyze_model(model)
        
        model_result = {
            'name': model['name'],
            'hypothesis': model['hypothesis'],
            'parameters': {
                'alpha': model['alpha'],
                'beta': model['beta'],
                'gamma': model['gamma'],
                'delta': model['delta']
            },
            'ratios': model['ratios'],
            'analysis': analysis
        }
        results['models'].append(model_result)
        
        key_ratio = f"α/β = {model['ratios']['alpha/beta']:.4f}"
        print(f"{model['name']:<20} | {analysis['c_sec']:.6e} | {analysis['error_pct']:.6f}% | {key_ratio}")
    
    # Summary
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    print("""
All models produce c exactly (by construction) - the question is which
parameter structure is physically meaningful.

KEY INSIGHT: The SEC wave equation ∂²S/∂t² = (αγ + βδ)∇²S
automatically produces electromagnetic wave structure IF:
1. α, β, γ, δ have velocity dimensions
2. Their combination satisfies αγ + βδ = c²

The RATIOS between parameters encode physical meaning:
- Symmetric: No preferred direction in I↔H exchange
- Xi-balanced: Information slightly dominates (Ξ ≈ 1.057)
- Phi-structured: Golden hierarchy from PAC recursion
- Fibonacci-nested: F₁₀/F₇ ≈ φ³ connects gauge to balance depth

WHICH IS CORRECT?
Must be determined by other predictions (charge, coupling constants, etc.)
""")
    
    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = results_dir / f'exp_01_sec_wave_speed_{timestamp}.json'
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {filepath}")
    
    return results


if __name__ == '__main__':
    main()
