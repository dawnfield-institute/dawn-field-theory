#!/usr/bin/env python3
"""
Experiment 1: Basic Evolution Engine
=====================================

Demonstrates evolution mechanics with mock fitness evaluation.
This validates the evolutionary framework before integrating real GAIA.

Key tests:
- Mutation system working
- Selection pressure applied
- Genealogy tracked
- Fitness improving over generations

Expected results:
- Fitness improvement over generations
- Constants (φ, Ξ) tracked
- Mutation history preserved
"""

import sys
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
import json

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from worldseed_evolution import (
    ModelConfig,
    evolve_gaia_mock,
)


def main():
    """Run basic evolution experiment."""
    print("="*60)
    print("EXPERIMENT 1: Basic Evolution Engine")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Configuration
    generations = 10
    candidates_per_gen = 3
    mutation_rate = 0.2
    
    print(f"\nConfiguration:")
    print(f"  Generations: {generations}")
    print(f"  Candidates per generation: {candidates_per_gen}")
    print(f"  Mutation rate: {mutation_rate}")
    
    # Create baseline config
    baseline_config = ModelConfig(
        context_size=5,
        concentration_threshold=0.618,
        hot_contexts=10000,
        embedding_dim=768,
        top_k_per_context=100,
        phi=1.618033988749895,
        xi=1.0571,
    )
    
    print(f"\nBaseline configuration:")
    print(f"  Context size: {baseline_config.context_size}")
    print(f"  Concentration: {baseline_config.concentration_threshold}")
    print(f"  Embedding dim: {baseline_config.embedding_dim}")
    print(f"  φ: {baseline_config.phi}")
    print(f"  Ξ: {baseline_config.xi}")
    
    # Run evolution
    print("\n" + "-"*60)
    print("EVOLVING...")
    print("-"*60)
    
    output_dir = Path(__file__).parent.parent.parent / "Data" / "results" / "exp_01"
    
    best_config, history, baseline_fitness = evolve_gaia_mock(
        parent_config=baseline_config,
        generations=generations,
        candidates_per_gen=candidates_per_gen,
        mutation_rate=mutation_rate,
        output_dir=output_dir,
    )
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    final_improvement = history[-1].improvement_percentage
    
    print(f"\nBaseline fitness: {baseline_fitness.overall_fitness:.3f}")
    print(f"Final fitness: {history[-1].best_fitness:.3f}")
    print(f"Improvement: {final_improvement:.1f}%")
    
    print(f"\nEvolved configuration:")
    print(f"  Context size: {best_config.context_size}")
    print(f"  Concentration: {best_config.concentration_threshold:.3f}")
    print(f"  Embedding dim: {best_config.embedding_dim}")
    print(f"  φ: {best_config.phi:.6f} (theory: 1.618034)")
    print(f"  Ξ: {best_config.xi:.6f} (theory: 1.0571)")
    
    print(f"\nMutation history: {best_config.mutation_history[-10:]}")
    
    # Package results
    results = {
        "experiment": "exp_01_basic_evolution",
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "generations": generations,
            "candidates_per_gen": candidates_per_gen,
            "mutation_rate": mutation_rate,
        },
        "baseline_config": asdict(baseline_config),
        "baseline_fitness": asdict(baseline_fitness),
        "best_config": asdict(best_config),
        "best_fitness": history[-1].best_fitness,
        "improvement_percentage": final_improvement,
        "evolution_history": [asdict(s) for s in history],
        "constant_convergence": {
            "phi_final": best_config.phi,
            "phi_theory": 1.618033988749895,
            "phi_error": abs(best_config.phi - 1.618033988749895) / 1.618033988749895 * 100,
            "xi_final": best_config.xi,
            "xi_theory": 1.0571,
            "xi_error": abs(best_config.xi - 1.0571) / 1.0571 * 100,
        }
    }
    
    print("\n✓ Experiment complete")
    return results


if __name__ == "__main__":
    results = main()
    
    # Save results
    output_path = Path(__file__).parent.parent.parent / "Data" / "results"
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_path / f"exp_01_basic_evolution_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
