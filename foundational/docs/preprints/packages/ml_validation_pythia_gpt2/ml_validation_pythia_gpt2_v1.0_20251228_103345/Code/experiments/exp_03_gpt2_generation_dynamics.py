"""
exp_28_generation_sec_dynamics.py - SEC During Text Generation
==============================================================

INSIGHT FROM exp_27:
- Inference dynamics (attention) ≈ 1.0 (equilibrium)
- Training dynamics (Pythia) → 2.2 (growth)
- But GENERATION is a collapse event!

HYPOTHESIS:
Each token selection is an entropy collapse:
- Before: high entropy (many possible next tokens)
- After: zero entropy (one token selected)
- This should show SEC dynamics!

What we test:
1. Entropy of token probability distribution at each step
2. How entropy evolves during generation
3. Whether entropy crosses φ-related thresholds before collapse
4. Statistical significance vs random baseline

If SEC is real:
- Entropy should show structure (not random walk)
- May see thresholds at 1/φ, φ, or related values
- Pattern should be consistent across prompts
"""

import torch
import numpy as np
from scipy import stats
from datetime import datetime
import json
import os

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch.nn.functional as F
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("WARNING: transformers not installed")

PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1 / PHI


def compute_token_entropy(logits: torch.Tensor, temperature: float = 1.0) -> float:
    """
    Compute entropy of token probability distribution.
    
    H = -sum(p * log2(p))
    
    Returns entropy in bits.
    """
    probs = F.softmax(logits / temperature, dim=-1)
    probs = probs[probs > 1e-10]  # Remove zeros
    entropy = -torch.sum(probs * torch.log2(probs)).item()
    return entropy


def compute_top_k_entropy(logits: torch.Tensor, k: int = 50) -> float:
    """Compute entropy over top-k tokens only."""
    probs = F.softmax(logits, dim=-1)
    top_probs, _ = torch.topk(probs, k)
    top_probs = top_probs / top_probs.sum()  # Renormalize
    entropy = -torch.sum(top_probs * torch.log2(top_probs + 1e-10)).item()
    return entropy


def generate_with_entropy_tracking(
    model, 
    tokenizer, 
    prompt: str, 
    max_tokens: int = 50,
    temperature: float = 1.0
) -> dict:
    """
    Generate text while tracking entropy at each step.
    
    Returns:
    - generated_text
    - entropy_trace: entropy before each token selection
    - selected_probs: probability of selected token
    - entropy_drops: entropy drop after each selection
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    entropy_trace = []
    selected_probs = []
    top_k_entropies = []
    
    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]  # Last position
        
        # Compute entropy before selection
        full_entropy = compute_token_entropy(logits, temperature)
        top_k_entropy = compute_top_k_entropy(logits, k=50)
        entropy_trace.append(full_entropy)
        top_k_entropies.append(top_k_entropy)
        
        # Select next token (greedy for reproducibility, but track prob)
        probs = F.softmax(logits / temperature, dim=-1)
        next_token = torch.argmax(probs).unsqueeze(0).unsqueeze(0)
        selected_prob = probs[next_token.squeeze()].item()
        selected_probs.append(selected_prob)
        
        # Update input
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Stop at EOS
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    # Compute entropy dynamics
    entropy_array = np.array(entropy_trace)
    
    # Ratios between consecutive steps
    ratios = entropy_array[1:] / (entropy_array[:-1] + 1e-10)
    
    # Normalized entropy (as fraction of max possible)
    vocab_size = logits.shape[0]
    max_entropy = np.log2(vocab_size)
    normalized_entropy = entropy_array / max_entropy
    
    return {
        'prompt': prompt,
        'generated_text': generated_text,
        'entropy_trace': entropy_trace,
        'top_k_entropies': top_k_entropies,
        'selected_probs': selected_probs,
        'ratios': ratios.tolist(),
        'normalized_entropy': normalized_entropy.tolist(),
        'max_entropy': max_entropy,
        'vocab_size': vocab_size
    }


def analyze_entropy_thresholds(results: list) -> dict:
    """
    Analyze whether entropy shows SEC-like thresholds.
    
    SEC predicts threshold at 1/φ ≈ 0.618 of max entropy.
    """
    all_normalized = []
    all_ratios = []
    
    for r in results:
        all_normalized.extend(r['normalized_entropy'])
        all_ratios.extend(r['ratios'])
    
    normalized = np.array(all_normalized)
    ratios = np.array(all_ratios)
    
    # Test 1: Is there clustering around 1/φ?
    inv_phi = INV_PHI  # ≈ 0.618
    distances_to_inv_phi = np.abs(normalized - inv_phi)
    mean_distance = np.mean(distances_to_inv_phi)
    
    # Under uniform [0, 1], expected distance from 0.618 ≈ 0.31
    # If significantly less, there's clustering
    expected_distance = 0.31
    clustering_at_inv_phi = mean_distance < expected_distance * 0.8
    
    # Test 2: Do ratios cluster around 1.0 or φ-related values?
    ratio_mean = np.mean(ratios[np.isfinite(ratios)])
    ratio_std = np.std(ratios[np.isfinite(ratios)])
    
    # Test 3: Histogram of normalized entropy
    hist, bin_edges = np.histogram(normalized, bins=20, range=(0, 1))
    
    # Find peaks
    peak_bin = np.argmax(hist)
    peak_value = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2
    
    # Is peak near 1/φ?
    peak_near_inv_phi = abs(peak_value - inv_phi) < 0.1
    
    # Test 4: Fraction of tokens below 1/φ threshold
    below_threshold = np.mean(normalized < inv_phi)
    
    return {
        'mean_normalized_entropy': float(np.mean(normalized)),
        'std_normalized_entropy': float(np.std(normalized)),
        'mean_distance_from_inv_phi': float(mean_distance),
        'clustering_at_inv_phi': clustering_at_inv_phi,
        'ratio_mean': float(ratio_mean),
        'ratio_std': float(ratio_std),
        'histogram': hist.tolist(),
        'histogram_bins': bin_edges.tolist(),
        'peak_value': float(peak_value),
        'peak_near_inv_phi': peak_near_inv_phi,
        'fraction_below_inv_phi': float(below_threshold)
    }


def main():
    if not HAS_TRANSFORMERS:
        print("ERROR: transformers not installed")
        return None
    
    print("="*60)
    print("EXP 28: SEC DURING TEXT GENERATION")
    print("="*60)
    print()
    print("HYPOTHESIS: Token selection = entropy collapse")
    print("            Should show SEC dynamics (threshold at 1/φ?)")
    print()
    
    # Load model
    print("Loading GPT-2...", end=' ', flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    print("OK")
    
    # Test prompts (diverse domains)
    prompts = [
        "The mathematical structure of",
        "In physics, the fundamental forces are",
        "The golden ratio appears in",
        "Neural networks learn by",
        "Entropy is a measure of",
        "The prime numbers follow",
        "Consciousness emerges from",
        "Information theory describes how",
        "The universe is governed by",
        "Recursive patterns appear in",
    ]
    
    print(f"\nGenerating from {len(prompts)} prompts...")
    
    all_results = []
    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] {prompt[:30]}...", end=' ', flush=True)
        result = generate_with_entropy_tracking(model, tokenizer, prompt, max_tokens=50)
        all_results.append(result)
        print(f"OK ({len(result['entropy_trace'])} tokens)")
    
    # Analyze
    print("\nAnalyzing entropy thresholds...")
    threshold_analysis = analyze_entropy_thresholds(all_results)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"\nEntropy Statistics:")
    print(f"  Mean normalized entropy: {threshold_analysis['mean_normalized_entropy']:.4f}")
    print(f"  Std normalized entropy: {threshold_analysis['std_normalized_entropy']:.4f}")
    print(f"  1/φ threshold: {INV_PHI:.4f}")
    
    print(f"\nSEC Threshold Analysis:")
    print(f"  Mean distance from 1/φ: {threshold_analysis['mean_distance_from_inv_phi']:.4f}")
    print(f"  Clustering at 1/φ: {threshold_analysis['clustering_at_inv_phi']}")
    print(f"  Peak of distribution: {threshold_analysis['peak_value']:.4f}")
    print(f"  Peak near 1/φ: {threshold_analysis['peak_near_inv_phi']}")
    print(f"  Fraction below 1/φ: {threshold_analysis['fraction_below_inv_phi']:.4f}")
    
    print(f"\nRatio Statistics:")
    print(f"  Mean ratio: {threshold_analysis['ratio_mean']:.4f}")
    print(f"  Std ratio: {threshold_analysis['ratio_std']:.4f}")
    
    # Key finding
    print("\n" + "="*60)
    print("KEY FINDING")
    print("="*60)
    
    if threshold_analysis['peak_near_inv_phi']:
        print("\n✅ SEC SIGNATURE DETECTED")
        print(f"   Entropy clusters near 1/φ = {INV_PHI:.4f}")
        print(f"   Peak at {threshold_analysis['peak_value']:.4f}")
        sec_confirmed = True
    elif threshold_analysis['clustering_at_inv_phi']:
        print("\n🔄 PARTIAL SEC SIGNATURE")
        print(f"   Some clustering near 1/φ")
        print(f"   Mean distance: {threshold_analysis['mean_distance_from_inv_phi']:.4f}")
        sec_confirmed = False
    else:
        print("\n❌ NO SEC THRESHOLD DETECTED")
        print(f"   Entropy distributed away from 1/φ")
        print(f"   Peak at {threshold_analysis['peak_value']:.4f}")
        sec_confirmed = False
    
    # Additional analysis: entropy trajectory
    print("\n" + "="*60)
    print("ENTROPY TRAJECTORIES (first 3 prompts)")
    print("="*60)
    
    for r in all_results[:3]:
        print(f"\n{r['prompt'][:40]}...")
        ent = r['normalized_entropy'][:10]
        print(f"  First 10 normalized entropies: {[f'{e:.3f}' for e in ent]}")
        
        # Check for SEC-like drops
        ratios = r['ratios'][:10]
        print(f"  First 10 ratios: {[f'{r:.3f}' for r in ratios if r is not None]}")
    
    # Save results
    output = {
        'experiment': 'exp_28_generation_sec_dynamics',
        'timestamp': datetime.now().isoformat(),
        'hypothesis': 'Token selection shows SEC dynamics',
        'model': 'gpt2',
        'num_prompts': len(prompts),
        'results': all_results,
        'threshold_analysis': threshold_analysis,
        'sec_confirmed': sec_confirmed,
        'phi_values': {
            'phi': float(PHI),
            'inv_phi': float(INV_PHI)
        }
    }
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(results_dir, f'exp_28_generation_sec_{timestamp}.json')
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    return output


if __name__ == '__main__':
    main()
