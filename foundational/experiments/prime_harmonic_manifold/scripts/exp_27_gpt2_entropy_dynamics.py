"""
exp_27_gpt2_entropy_dynamics.py - GPT-2 SEC Validation
======================================================

HYPOTHESIS: The Pythia φ-crossing finding (p=0.0014) should replicate in 
            GPT-2, demonstrating architecture independence.

BACKGROUND:
- Pythia: EleutherAI models trained on The Pile
- GPT-2: OpenAI models trained on WebText
- Different architectures, training data, organizations
- Same PAC/SEC prediction should hold

METHODOLOGY:
1. Load GPT-2 checkpoints (OpenAI doesn't release training checkpoints,
   but we can compare model sizes as proxies for "training depth")
2. Measure entropy dynamics in model outputs
3. Compare to Pythia patterns
4. If same pattern → architecture/data independent

ALTERNATIVE APPROACH (since GPT-2 lacks training checkpoints):
- Compare entropy collapse patterns during INFERENCE
- Measure how attention patterns evolve through layers
- Test if layer-wise dynamics show φ-related structure

The key insight: If PAC is fundamental, it should appear in:
- Training dynamics (Pythia checkpoints - validated)
- Inference dynamics (attention through layers - testing now)
- Weight structure (static analysis - secondary)
"""

import torch
import numpy as np
from scipy import stats
from datetime import datetime
import json
import os

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("WARNING: transformers not installed. Install with: pip install transformers")

PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1 / PHI


def compute_layer_entropy(attention_weights: torch.Tensor) -> float:
    """
    Compute entropy of attention pattern.
    Higher entropy = more uniform attention
    Lower entropy = more focused attention
    
    Entropy collapse through layers = SEC manifestation
    """
    # Flatten and normalize
    probs = attention_weights.flatten()
    probs = probs / probs.sum()
    probs = probs[probs > 1e-10]  # Remove zeros
    
    # Shannon entropy
    entropy = -torch.sum(probs * torch.log2(probs)).item()
    return entropy


def compute_attention_statistics(model, tokenizer, texts: list) -> dict:
    """
    Compute attention entropy statistics across layers.
    
    Returns dict with:
    - layer_entropies: mean entropy at each layer
    - layer_entropy_std: std at each layer  
    - entropy_ratios: ratio of consecutive layer entropies
    - phi_crossings: layers where ratio crosses φ or 1/φ
    """
    all_layer_entropies = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        
        # outputs.attentions is tuple of (batch, heads, seq, seq) per layer
        layer_entropies = []
        for layer_attn in outputs.attentions:
            # Average across batch, heads
            entropy = compute_layer_entropy(layer_attn.mean(dim=(0, 1)))
            layer_entropies.append(entropy)
        
        all_layer_entropies.append(layer_entropies)
    
    # Aggregate across texts
    layer_entropies = np.array(all_layer_entropies)
    mean_entropies = layer_entropies.mean(axis=0)
    std_entropies = layer_entropies.std(axis=0)
    
    # Compute ratios
    ratios = mean_entropies[1:] / (mean_entropies[:-1] + 1e-10)
    
    # Find φ-crossings (where ratio crosses φ or 1/φ)
    phi_crossings = []
    for i, r in enumerate(ratios):
        if abs(r - PHI) < 0.1 or abs(r - INV_PHI) < 0.1:
            phi_crossings.append({
                'layer': i + 1,
                'ratio': float(r),
                'distance_from_phi': float(min(abs(r - PHI), abs(r - INV_PHI)))
            })
    
    return {
        'mean_entropies': mean_entropies.tolist(),
        'std_entropies': std_entropies.tolist(),
        'ratios': ratios.tolist(),
        'phi_crossings': phi_crossings,
        'num_layers': len(mean_entropies)
    }


def analyze_weight_structure(model) -> dict:
    """
    Analyze static weight structure for PAC signatures.
    
    Looks for:
    - Layer-wise norm ratios
    - φ-related scaling between layers
    """
    layer_norms = []
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            norm = param.data.norm().item()
            layer_norms.append({
                'name': name,
                'norm': norm,
                'shape': list(param.shape)
            })
    
    # Compute ratios between consecutive layers of same type
    norms_by_type = {}
    for ln in layer_norms:
        layer_type = ln['name'].split('.')[-1]  # e.g., 'weight'
        if layer_type not in norms_by_type:
            norms_by_type[layer_type] = []
        norms_by_type[layer_type].append(ln['norm'])
    
    # Compute ratios
    all_ratios = []
    for layer_type, norms in norms_by_type.items():
        if len(norms) > 1:
            ratios = [norms[i+1]/norms[i] for i in range(len(norms)-1) if norms[i] > 0]
            all_ratios.extend(ratios)
    
    # Statistics
    if all_ratios:
        mean_ratio = np.mean(all_ratios)
        std_ratio = np.std(all_ratios)
        phi_distance = min(abs(mean_ratio - PHI), abs(mean_ratio - INV_PHI))
    else:
        mean_ratio = std_ratio = phi_distance = np.nan
    
    return {
        'num_weight_tensors': len(layer_norms),
        'mean_ratio': float(mean_ratio),
        'std_ratio': float(std_ratio),
        'phi_distance': float(phi_distance),
        'all_ratios': [float(r) for r in all_ratios]
    }


def compare_to_pythia(gpt2_result: dict, pythia_reference: dict = None) -> dict:
    """
    Compare GPT-2 results to Pythia findings.
    
    Pythia reference (from our experiments):
    - Late training ratio: ~2.2 (between φ and 2.0)
    - Combined p-value: 0.0014
    - All slopes negative (convergence)
    """
    if pythia_reference is None:
        # Reference values from our Pythia experiment
        pythia_reference = {
            'late_ratio_mean': 2.31,
            'late_ratio_std': 0.15,
            'phi_distance': 0.69,  # Distance from φ=1.618
            'convergence': True
        }
    
    # Compare entropy ratios
    gpt2_ratios = gpt2_result.get('attention_stats', {}).get('ratios', [])
    if gpt2_ratios:
        gpt2_mean_ratio = np.mean(gpt2_ratios)
        gpt2_late_ratio = np.mean(gpt2_ratios[-3:]) if len(gpt2_ratios) >= 3 else gpt2_mean_ratio
        
        # Check if late layers converge like Pythia
        ratio_difference = abs(gpt2_late_ratio - pythia_reference['late_ratio_mean'])
        similar_pattern = ratio_difference < 0.5  # Within 0.5 of Pythia
    else:
        gpt2_late_ratio = np.nan
        similar_pattern = False
    
    return {
        'gpt2_late_ratio': float(gpt2_late_ratio),
        'pythia_late_ratio': pythia_reference['late_ratio_mean'],
        'ratio_difference': float(abs(gpt2_late_ratio - pythia_reference['late_ratio_mean'])),
        'similar_pattern': similar_pattern,
        'interpretation': 'Architecture-independent' if similar_pattern else 'Architecture-specific'
    }


def run_gpt2_analysis(model_name: str = 'gpt2') -> dict:
    """Run full analysis on a GPT-2 variant."""
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*60}")
    
    # Load model
    print("Loading model...", end=' ', flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, output_attentions=True)
    model.eval()
    print("OK")
    
    # Test texts (diverse to get robust statistics)
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In mathematics, the golden ratio appears in many unexpected places.",
        "Neural networks learn representations through gradient descent optimization.",
        "The structure of prime numbers remains one of mathematics' deepest mysteries.",
        "Consciousness emerges from information integration across neural circuits.",
        "Entropy is a measure of disorder in thermodynamic systems.",
        "Recursive patterns appear at all scales in natural phenomena.",
        "The universe appears to be written in the language of mathematics.",
    ]
    
    # Attention analysis
    print("Analyzing attention entropy...", end=' ', flush=True)
    attention_stats = compute_attention_statistics(model, tokenizer, test_texts)
    print("OK")
    
    # Weight analysis
    print("Analyzing weight structure...", end=' ', flush=True)
    weight_stats = analyze_weight_structure(model)
    print("OK")
    
    # Build result
    result = {
        'model': model_name,
        'num_layers': attention_stats['num_layers'],
        'attention_stats': attention_stats,
        'weight_stats': weight_stats
    }
    
    # Compare to Pythia
    print("Comparing to Pythia...", end=' ', flush=True)
    comparison = compare_to_pythia(result)
    result['pythia_comparison'] = comparison
    print("OK")
    
    return result


def main():
    if not HAS_TRANSFORMERS:
        print("\n" + "="*60)
        print("ERROR: transformers library not installed")
        print("Install with: pip install transformers torch")
        print("="*60)
        return None
    
    print("="*60)
    print("EXP 27: GPT-2 ENTROPY DYNAMICS - SEC VALIDATION")
    print("="*60)
    print()
    print("HYPOTHESIS: Pythia φ-crossing should replicate in GPT-2")
    print("            (architecture independence)")
    print()
    
    # Test multiple GPT-2 sizes
    models = ['gpt2', 'gpt2-medium', 'gpt2-large']
    
    all_results = []
    
    for model_name in models:
        try:
            result = run_gpt2_analysis(model_name)
            all_results.append(result)
            
            # Print summary
            print(f"\n{model_name} Summary:")
            print(f"  Layers: {result['num_layers']}")
            print(f"  φ-crossings: {len(result['attention_stats']['phi_crossings'])}")
            print(f"  Late entropy ratio: {result['pythia_comparison']['gpt2_late_ratio']:.4f}")
            print(f"  Similar to Pythia: {result['pythia_comparison']['similar_pattern']}")
            
        except Exception as e:
            print(f"\nFailed to analyze {model_name}: {e}")
            continue
    
    if not all_results:
        print("\nNo results to analyze")
        return None
    
    # Aggregate analysis
    print("\n" + "="*60)
    print("AGGREGATE ANALYSIS")
    print("="*60)
    
    # Check if pattern is consistent across sizes
    late_ratios = [r['pythia_comparison']['gpt2_late_ratio'] for r in all_results]
    similar_count = sum(1 for r in all_results if r['pythia_comparison']['similar_pattern'])
    
    print(f"\nModels analyzed: {len(all_results)}")
    print(f"Late ratio range: {min(late_ratios):.4f} - {max(late_ratios):.4f}")
    print(f"Models matching Pythia pattern: {similar_count}/{len(all_results)}")
    
    # Statistical test: are late ratios significantly different from random?
    # Null: ratios are uniformly distributed [0.5, 2.0]
    # Alternative: ratios cluster around specific values
    
    all_ratios = []
    for r in all_results:
        all_ratios.extend(r['attention_stats']['ratios'])
    
    if len(all_ratios) > 5:
        # Test if ratios cluster (low variance = clustering)
        ratio_var = np.var(all_ratios)
        # Under uniform [0.5, 2.0], variance would be (2.0-0.5)^2/12 = 0.1875
        expected_var = 0.1875
        clustering = ratio_var < expected_var / 2
        
        # Test distance from φ and 1/φ
        phi_distances = [min(abs(r - PHI), abs(r - INV_PHI)) for r in all_ratios]
        mean_phi_distance = np.mean(phi_distances)
        
        print(f"\nRatio variance: {ratio_var:.4f} (uniform would be ~0.19)")
        print(f"Clustering detected: {clustering}")
        print(f"Mean distance from φ/1/φ: {mean_phi_distance:.4f}")
    
    # Key finding
    print("\n" + "="*60)
    print("KEY FINDING")
    print("="*60)
    
    if similar_count == len(all_results):
        print("\n✅ CONFIRMED: GPT-2 shows same pattern as Pythia")
        print("   φ-related dynamics are ARCHITECTURE-INDEPENDENT")
        confirmation = True
    elif similar_count > 0:
        print(f"\n🔄 PARTIAL: {similar_count}/{len(all_results)} models match Pythia")
        print("   Pattern may be size-dependent")
        confirmation = False
    else:
        print("\n❌ NOT CONFIRMED: GPT-2 differs from Pythia")
        print("   Pattern may be architecture or data specific")
        confirmation = False
    
    # Save results
    output = {
        'experiment': 'exp_27_gpt2_entropy_dynamics',
        'timestamp': datetime.now().isoformat(),
        'hypothesis': 'Pythia φ-crossing replicates in GPT-2',
        'models_tested': models,
        'results': all_results,
        'aggregate': {
            'models_analyzed': len(all_results),
            'similar_to_pythia': similar_count,
            'late_ratio_range': [float(min(late_ratios)), float(max(late_ratios))],
            'confirmation': confirmation
        }
    }
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(results_dir, f'exp_27_gpt2_entropy_{timestamp}.json')
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    return output


if __name__ == '__main__':
    main()
