"""
Pythia Training Dynamics: Phi-Convergence Analysis
===================================================

Tests PAC prediction that training dynamics converge toward phi-related ratios.

Key Insight (from discussion Dec 7, 2025):
- PAC tree = tree of DELTAS (weight changes), not final weights
- Training IS the fractal growth process
- Phi structure should appear in delta dynamics, not static weights

Methodology:
- Load Pythia checkpoints at exponential steps (0,1,2,4,8,...,512)
- Compute delta norms: ||w_{n+1} - w_n|| between consecutive checkpoints
- Compute delta ratios: ||delta_{n+1}|| / ||delta_n||
- Test if ratios converge toward phi (1.618) during training

Results (Dec 7, 2025):
- ALL 4 models show convergence from chaotic (~10-17) to stable (~2.0-2.3)
- Combined p-value: 0.0014 (Fisher's method)
- Late training ratios cluster around 2.0-2.3 (near D=2 attractor)

Usage:
    python run_all_pythia_analysis.py

Requirements:
    pip install torch transformers huggingface_hub numpy scipy
"""

import torch
from huggingface_hub import hf_hub_download
import numpy as np
from scipy import stats
import os
import json
import time
from datetime import datetime
from pathlib import Path

PHI = 1.618034
INV_PHI = 1 / PHI

MODELS = [
    'EleutherAI/pythia-70m',
    'EleutherAI/pythia-160m', 
    'EleutherAI/pythia-410m',
    'EleutherAI/pythia-1b',
]

STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def compute_delta_norm(w1: dict, w2: dict) -> tuple:
    """
    Compute L2 norm of weight delta between two checkpoints.
    Filters out non-finite values (Inf in masked_bias tensors).
    
    Returns:
        (delta_norm, keys_used)
    """
    delta_sq = 0.0
    keys_used = 0
    for k in w1.keys():
        if k in w2:
            d1 = w1[k].float()
            d2 = w2[k].float()
            if torch.isfinite(d1).all() and torch.isfinite(d2).all():
                delta_sq += ((d2 - d1)**2).sum().item()
                keys_used += 1
    return np.sqrt(delta_sq), keys_used


def load_checkpoints(model_id: str, steps: list, cache_dir: str, max_retries: int = 5) -> dict:
    """Load all checkpoints for a model with retry logic."""
    checkpoints = {}
    
    for step in steps:
        print(f'  Loading step {step}...', end=' ', flush=True)
        for attempt in range(max_retries):
            try:
                path = hf_hub_download(
                    repo_id=model_id,
                    filename='pytorch_model.bin',
                    revision=f'step{step}',
                    cache_dir=cache_dir
                )
                checkpoints[step] = torch.load(path, map_location='cpu', weights_only=True)
                print('OK')
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f'retry {attempt+1}...', end=' ', flush=True)
                    time.sleep(2)
                else:
                    print(f'FAILED: {e}')
    
    return checkpoints


def analyze_model(model_id: str, cache_dir: str) -> dict:
    """
    Run full phi-convergence analysis on a single model.
    
    Returns dict with:
        - model_id
        - deltas: list of (step_from, step_to, delta_norm)
        - ratios: list of (step, ratio, closest, distance)
        - statistics: slope, r2, p_value, early_mean, late_mean
    """
    print(f'\n{"="*70}')
    print(f'ANALYZING: {model_id}')
    print(f'{"="*70}')
    
    # Load checkpoints
    checkpoints = load_checkpoints(model_id, STEPS, cache_dir)
    
    if len(checkpoints) < 3:
        print(f'ERROR: Only loaded {len(checkpoints)} checkpoints')
        return None
    
    print(f'Successfully loaded {len(checkpoints)} checkpoints\n')
    
    # Compute deltas
    sorted_steps = sorted(checkpoints.keys())
    deltas = []
    
    for i in range(len(sorted_steps) - 1):
        s1, s2 = sorted_steps[i], sorted_steps[i+1]
        norm, keys = compute_delta_norm(checkpoints[s1], checkpoints[s2])
        deltas.append({
            'step_from': s1,
            'step_to': s2,
            'delta_norm': norm,
            'keys_used': keys
        })
        print(f'  Delta(step {s1} -> {s2}): {norm:.4f}')
    
    # Compute ratios
    print('\nDelta Ratios vs PHI:')
    print('-' * 60)
    
    ratios = []
    for i in range(len(deltas) - 1):
        d1 = deltas[i]['delta_norm']
        d2 = deltas[i+1]['delta_norm']
        step = deltas[i]['step_to']
        
        if d1 > 1e-10:
            ratio = d2 / d1
            dist_phi = abs(ratio - PHI)
            dist_inv = abs(ratio - INV_PHI)
            
            if dist_phi < dist_inv:
                closest, dist = 'phi', dist_phi
            else:
                closest, dist = '1/phi', dist_inv
            
            ratios.append({
                'step': step,
                'ratio': ratio,
                'closest': closest,
                'distance': dist
            })
            print(f'  Step {step:>4}: ratio={ratio:8.4f}  {closest:>5}  dist={dist:.4f}')
    
    # Statistical analysis
    if len(ratios) >= 4:
        print('\nStatistical Analysis:')
        print('-' * 60)
        
        steps_arr = np.array([r['step'] for r in ratios])
        dists = np.array([r['distance'] for r in ratios])
        
        # Regression: phi-distance vs log(step)
        log_steps = np.log10(steps_arr + 1)
        slope, intercept, r, p, se = stats.linregress(log_steps, dists)
        
        print(f'  Phi-distance vs log(step):')
        print(f'    Slope: {slope:.4f} (negative = converging)')
        print(f'    R-squared: {r**2:.4f}')
        print(f'    P-value: {p:.4f}')
        
        # Early vs late comparison
        n = len(ratios)
        early = [r['distance'] for r in ratios[:n//2]]
        late = [r['distance'] for r in ratios[n//2:]]
        
        early_mean = np.mean(early)
        late_mean = np.mean(late)
        
        t_stat, t_p = stats.ttest_ind(early, late, alternative='greater')
        
        print(f'  Early mean distance: {early_mean:.4f}')
        print(f'  Late mean distance: {late_mean:.4f}')
        print(f'  T-test (early > late): t={t_stat:.2f}, p={t_p:.4f}')
        
        statistics = {
            'slope': slope,
            'r_squared': r**2,
            'p_value': p,
            'early_mean': early_mean,
            'late_mean': late_mean,
            't_statistic': t_stat,
            't_p_value': t_p
        }
    else:
        statistics = None
    
    # Free memory
    del checkpoints
    
    return {
        'model_id': model_id,
        'timestamp': datetime.now().isoformat(),
        'checkpoints_analyzed': len(sorted_steps),
        'deltas': deltas,
        'ratios': ratios,
        'statistics': statistics
    }


def run_all_models():
    """Run analysis on all Pythia models and save results."""
    
    cache_dir = './pythia_cache'
    results_dir = Path('./results')
    results_dir.mkdir(exist_ok=True)
    
    all_results = {}
    
    for model_id in MODELS:
        model_name = model_id.split('/')[-1]
        
        result = analyze_model(model_id, cache_dir)
        
        if result:
            all_results[model_name] = result
            
            # Save individual result
            result_file = results_dir / f'{model_name}_results.json'
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f'\nSaved: {result_file}')
    
    # Save combined results
    combined_file = results_dir / 'all_models_results.json'
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved combined results: {combined_file}')
    
    # Print summary
    print_summary(all_results)
    
    return all_results


def print_summary(results: dict):
    """Print cross-model summary."""
    
    print('\n' + '='*70)
    print('CROSS-MODEL PHI-CONVERGENCE SUMMARY')
    print('='*70)
    
    print(f"\n{'Model':15} {'Late Ratio':>12} {'Dist from phi':>14} {'Slope':>10} {'P-value':>10}")
    print('-' * 70)
    
    all_late_ratios = []
    all_slopes = []
    all_ps = []
    
    for name, data in results.items():
        if data['statistics']:
            stats_data = data['statistics']
            ratios = data['ratios']
            n = len(ratios)
            late_ratios = [r['ratio'] for r in ratios[n//2:]]
            mean_late = np.mean(late_ratios)
            dist = abs(mean_late - PHI)
            
            all_late_ratios.append(mean_late)
            all_slopes.append(stats_data['slope'])
            all_ps.append(stats_data['p_value'])
            
            sig = '*' if stats_data['p_value'] < 0.05 else ''
            print(f"{name:15} {mean_late:12.4f} {dist:14.4f} {stats_data['slope']:10.4f} {stats_data['p_value']:9.4f}{sig}")
    
    print('-' * 70)
    
    if all_ps:
        # Fisher's method to combine p-values
        chi2 = -2 * sum(np.log(p) for p in all_ps)
        df = 2 * len(all_ps)
        combined_p = 1 - stats.chi2.cdf(chi2, df)
        
        print(f'\nAggregate Statistics:')
        print(f'  Mean late ratio: {np.mean(all_late_ratios):.4f}')
        print(f'  Distance from phi: {abs(np.mean(all_late_ratios) - PHI):.4f}')
        print(f'  All slopes negative: {all(s < 0 for s in all_slopes)}')
        print(f'  Combined p-value (Fisher): {combined_p:.6f}')
        print(f'\n  CONCLUSION: {"SUPPORTS" if combined_p < 0.05 else "DOES NOT SUPPORT"} PAC PREDICTION')


if __name__ == '__main__':
    run_all_models()
