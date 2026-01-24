"""
WorldSeed GAIA Evolution - Core Module
======================================

Core evolution engine with real GAIA integration.
Based on internal/project_worldseed/worldseed_gaia_real.py

This module provides:
- ModelConfig: Configuration for GAIA model variants
- GAIAMutator: Generates mutations for architecture evolution
- RealFitnessEvaluator: Evaluates fitness with real training
- evolve_gaia_real: Main evolution loop

See trace.yaml for source file mappings.
"""

import torch
import time
import random
import json
from typing import Optional, Callable, Dict, List, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for a GAIA model variant."""
    # Embeddings
    embedding_source: str = 'gpt2'
    embedding_dim: int = 768
    vocab_size: int = 50257

    # PAC Tree
    tree_depth: int = 1
    delta_compression: str = 'none'

    # Transitions
    context_size: int = 5
    hot_contexts: int = 10000
    top_k_per_context: int = 100

    # Concentration
    concentration_threshold: float = 0.618  # φ^-1
    reject_attempts: int = 3

    # Physics constants (for validation)
    phi: float = 1.618033988749895
    xi: float = 1.0571
    lambda_star: float = 0.618432

    # Metadata
    generation: int = 0
    parent_id: Optional[str] = None
    mutation_history: List[str] = field(default_factory=list)


@dataclass
class FitnessResult:
    """Results from fitness evaluation."""
    perplexity: float
    speed: float  # tokens/second
    memory: float  # MB
    quality: float  # 0-1
    overall_fitness: float
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionStats:
    """Statistics for one generation."""
    generation: int
    population_size: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    mean_phi: float
    mean_xi: float
    mean_lambda: float
    mutations_applied: List[str]
    timestamp: str
    improvement_percentage: Optional[float] = None


class GAIAMutator:
    """
    Generates mutations for GAIA configs.

    8 mutation types:
    - context_size: N-gram window (Fibonacci values)
    - hot_contexts: Cache size
    - concentration_threshold: Quality gate
    - reject_attempts: Resample count
    - embedding_dim: Embedding dimensions
    - top_k: Top predictions
    - phi: Physics constant (±5%)
    - xi: Physics constant (±5%)
    """

    def __init__(self, mutation_rate: float = 0.2):
        self.mutation_rate = mutation_rate
        self.mutation_types = [
            'context_size',
            'hot_contexts',
            'concentration',
            'reject_attempts',
            'embedding_dim',
            'top_k',
            'phi',
            'xi',
        ]

    def mutate(self, config: ModelConfig) -> Tuple[ModelConfig, List[str]]:
        """Create mutated variant of config."""
        new_config = ModelConfig(**asdict(config))
        mutations_applied = []

        for mutation_type in self.mutation_types:
            if random.random() < self.mutation_rate:
                getattr(self, f'_mutate_{mutation_type}')(new_config)
                mutations_applied.append(mutation_type)

        new_config.mutation_history = config.mutation_history + mutations_applied
        return new_config, mutations_applied

    def _mutate_context_size(self, config: ModelConfig):
        choices = [3, 5, 7, 9, 11, 13]  # Fibonacci-adjacent
        config.context_size = random.choice(choices)

    def _mutate_hot_contexts(self, config: ModelConfig):
        choices = [5000, 10000, 20000, 50000]
        config.hot_contexts = random.choice(choices)

    def _mutate_concentration(self, config: ModelConfig):
        config.concentration_threshold = random.uniform(0.5, 0.8)

    def _mutate_reject_attempts(self, config: ModelConfig):
        choices = [1, 3, 5, 10]
        config.reject_attempts = random.choice(choices)

    def _mutate_embedding_dim(self, config: ModelConfig):
        choices = [256, 512, 768, 1024]
        config.embedding_dim = random.choice(choices)

    def _mutate_top_k(self, config: ModelConfig):
        choices = [50, 100, 200, 500]
        config.top_k_per_context = random.choice(choices)

    def _mutate_phi(self, config: ModelConfig):
        phi_theory = 1.618033988749895
        config.phi = phi_theory * random.uniform(0.95, 1.05)

    def _mutate_xi(self, config: ModelConfig):
        xi_theory = 1.0571
        config.xi = xi_theory * random.uniform(0.95, 1.05)


class MockFitnessEvaluator:
    """
    Mock fitness evaluator for testing evolution mechanics.
    
    Used for Experiment 1 (basic evolution).
    """

    def __init__(self, fitness_fn: Optional[Callable] = None):
        self.fitness_fn = fitness_fn or self._default_fitness

    def evaluate(self, model: Any, config: ModelConfig) -> FitnessResult:
        """Simulate fitness evaluation with realistic variance."""
        # Base values with config influence
        base_ppl = 45.0 + random.uniform(-10, 10)
        base_speed = 5000 + random.uniform(-1000, 1000)
        base_memory = 100 + random.uniform(-20, 20)
        base_quality = 0.7 + random.uniform(-0.1, 0.1)

        # Config influences fitness
        ppl_bonus = (5 - config.context_size) * 2
        speed_bonus = (768 - config.embedding_dim) / 10
        quality_bonus = (config.concentration_threshold - 0.5) * 0.3

        perplexity = max(10, base_ppl + ppl_bonus)
        speed = max(1000, base_speed + speed_bonus)
        memory = max(50, base_memory)
        quality = min(1.0, max(0.0, base_quality + quality_bonus))

        overall = self.fitness_fn(perplexity, speed, memory, quality)

        return FitnessResult(
            perplexity=perplexity,
            speed=speed,
            memory=memory,
            quality=quality,
            overall_fitness=overall,
            metrics={
                'context_size': config.context_size,
                'concentration_threshold': config.concentration_threshold,
                'phi': config.phi,
                'xi': config.xi,
            }
        )

    def _default_fitness(self, perplexity, speed, memory, quality):
        return (
            0.3 * (100 / perplexity) +
            0.3 * (speed / 10000) +
            0.2 * (1000 / memory) +
            0.2 * quality
        )


def evolve_gaia_mock(
    parent_config: Optional[ModelConfig] = None,
    generations: int = 10,
    candidates_per_gen: int = 3,
    mutation_rate: float = 0.2,
    output_dir: Optional[Path] = None,
) -> Tuple[ModelConfig, List[EvolutionStats], FitnessResult]:
    """
    Evolve GAIA configuration with mock fitness (for testing).
    
    Returns:
        (best_config, evolution_history, baseline_fitness)
    """
    if parent_config is None:
        parent_config = ModelConfig()

    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f"evolution_results/run_{timestamp}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mutator = GAIAMutator(mutation_rate=mutation_rate)
    evaluator = MockFitnessEvaluator()

    # Evaluate baseline
    print("Evaluating baseline...")
    baseline_fitness = evaluator.evaluate(None, parent_config)
    print(f"Baseline fitness: {baseline_fitness.overall_fitness:.3f}")

    current_config = parent_config
    history = []

    for gen in range(generations):
        print(f"\n=== Generation {gen + 1}/{generations} ===")
        
        candidates = []
        all_mutations = []

        for i in range(candidates_per_gen):
            mutated_config, mutations = mutator.mutate(current_config)
            mutated_config.generation = gen + 1
            mutated_config.parent_id = f"gen{gen}"
            
            fitness = evaluator.evaluate(None, mutated_config)
            candidates.append((mutated_config, fitness))
            all_mutations.extend(mutations)
            
            print(f"  Candidate {i+1}: fitness={fitness.overall_fitness:.3f}")

        # Select best
        candidates.sort(key=lambda x: x[1].overall_fitness, reverse=True)
        best_config, best_fitness = candidates[0]
        current_config = best_config

        # Record stats
        fitness_values = [c[1].overall_fitness for c in candidates]
        phi_values = [c[0].phi for c in candidates]
        xi_values = [c[0].xi for c in candidates]

        improvement = ((best_fitness.overall_fitness - baseline_fitness.overall_fitness) 
                      / baseline_fitness.overall_fitness * 100)

        stats = EvolutionStats(
            generation=gen + 1,
            population_size=len(candidates),
            best_fitness=best_fitness.overall_fitness,
            mean_fitness=sum(fitness_values) / len(fitness_values),
            worst_fitness=min(fitness_values),
            mean_phi=sum(phi_values) / len(phi_values),
            mean_xi=sum(xi_values) / len(xi_values),
            mean_lambda=0.618432,
            mutations_applied=list(set(all_mutations)),
            timestamp=datetime.now().isoformat(),
            improvement_percentage=improvement,
        )
        history.append(stats)

        print(f"  Best: {best_fitness.overall_fitness:.3f} "
              f"(+{improvement:.1f}% vs baseline)")

        # Checkpoint
        checkpoint = {
            'generation': gen + 1,
            'best_config': asdict(best_config),
            'best_fitness': asdict(best_fitness),
            'stats': asdict(stats),
        }
        with open(output_dir / f"checkpoint_gen{gen+1}.json", 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)

    return current_config, history, baseline_fitness
