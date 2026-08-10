"""
Experiment 06: Predictive Power Test

Tests whether the Markov chord model provides better predictions
than a baseline frequency model.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from prime_chords import (
    get_primes, compute_gaps, extract_chords,
    build_transition_matrix
)
import numpy as np
from collections import Counter


def run_experiment(prime_limit: int = 500000, top_k: int = 25):
    """Run predictive power test."""
    
    print("=" * 70)
    print("PRIME HARMONIC MANIFOLD: Predictive Power Test")
    print("=" * 70)
    
    # Generate data
    print(f"\nGenerating primes up to {prime_limit:,}...")
    primes = get_primes(prime_limit)
    gaps = compute_gaps(primes)
    chords = extract_chords(gaps, n_gaps=2)
    print(f"  Primes: {len(primes):,}")
    print(f"  Chords: {len(chords):,}")
    
    # Split into train/test
    split_idx = int(len(chords) * 0.8)
    train_chords = chords[:split_idx]
    test_chords = chords[split_idx:]
    
    print(f"\n  Train: {len(train_chords):,} chords")
    print(f"  Test:  {len(test_chords):,} chords")
    
    # Build models
    print("\nBuilding models...")
    
    # Baseline: frequency-based
    train_counts = Counter(train_chords)
    total = sum(train_counts.values())
    freq_probs = {c: count/total for c, count in train_counts.items()}
    
    # Markov model
    P, top_chord_list = build_transition_matrix(train_chords, top_k=top_k)
    chord_to_idx = {c: i for i, c in enumerate(top_chord_list)}
    other_idx = top_k
    
    # Evaluate on test set
    print("\nEvaluating predictions...")
    
    baseline_ll = 0
    markov_ll = 0
    n_pred = 0
    
    for i in range(len(test_chords) - 1):
        current = test_chords[i]
        next_chord = test_chords[i + 1]
        
        # Baseline: P(next) = freq(next)
        baseline_prob = freq_probs.get(next_chord, 1e-10)
        baseline_ll += np.log(baseline_prob)
        
        # Markov: P(next | current)
        curr_idx = chord_to_idx.get(current, other_idx)
        next_idx = chord_to_idx.get(next_chord, other_idx)
        
        if curr_idx < top_k and next_idx <= top_k:
            markov_prob = P[curr_idx, next_idx]
            if markov_prob < 1e-10:
                markov_prob = 1e-10
        else:
            markov_prob = 1e-10
        
        markov_ll += np.log(markov_prob)
        n_pred += 1
    
    # Compute metrics
    baseline_perplexity = np.exp(-baseline_ll / n_pred)
    markov_perplexity = np.exp(-markov_ll / n_pred)
    
    improvement = (baseline_perplexity - markov_perplexity) / baseline_perplexity * 100
    
    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)
    print(f"  Predictions: {n_pred:,}")
    print(f"  Baseline log-likelihood: {baseline_ll:.2f}")
    print(f"  Markov log-likelihood:   {markov_ll:.2f}")
    print(f"  Baseline perplexity:     {baseline_perplexity:.2f}")
    print(f"  Markov perplexity:       {markov_perplexity:.2f}")
    print(f"  Improvement:             {improvement:+.1f}%")
    
    # Accuracy test (predict most likely next chord)
    print("\n" + "-" * 60)
    print("ACCURACY TEST (predict most likely next)")
    print("-" * 60)
    
    baseline_correct = 0
    markov_correct = 0
    
    for i in range(len(test_chords) - 1):
        current = test_chords[i]
        next_chord = test_chords[i + 1]
        
        # Baseline: predict most common chord
        baseline_pred = train_counts.most_common(1)[0][0]
        if baseline_pred == next_chord:
            baseline_correct += 1
        
        # Markov: predict most likely given current
        curr_idx = chord_to_idx.get(current, other_idx)
        if curr_idx < top_k:
            markov_pred_idx = np.argmax(P[curr_idx, :top_k])
            markov_pred = top_chord_list[markov_pred_idx]
            if markov_pred == next_chord:
                markov_correct += 1
    
    baseline_acc = baseline_correct / n_pred * 100
    markov_acc = markov_correct / n_pred * 100
    
    print(f"  Baseline accuracy: {baseline_acc:.2f}%")
    print(f"  Markov accuracy:   {markov_acc:.2f}%")
    print(f"  Improvement:       {markov_acc - baseline_acc:+.2f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    if improvement > 10:
        print("  Markov model provides SIGNIFICANT predictive improvement.")
        print("  Chord transitions carry information beyond frequency.")
    else:
        print("  Markov model provides modest improvement.")
    
    # Save results
    results = {
        'experiment': 'exp_06_predictive_power',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'prime_limit': prime_limit,
            'top_k': top_k,
            'train_size': len(train_chords),
            'test_size': len(test_chords),
        },
        'results': {
            'n_predictions': n_pred,
            'baseline_ll': baseline_ll,
            'markov_ll': markov_ll,
            'baseline_perplexity': baseline_perplexity,
            'markov_perplexity': markov_perplexity,
            'perplexity_improvement_pct': improvement,
            'baseline_accuracy': baseline_acc,
            'markov_accuracy': markov_acc,
        },
        'conclusion': 'SIGNIFICANT_IMPROVEMENT' if improvement > 10 else 'MODEST_IMPROVEMENT'
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'exp_06_predictive_power_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    return results


if __name__ == '__main__':
    run_experiment()
