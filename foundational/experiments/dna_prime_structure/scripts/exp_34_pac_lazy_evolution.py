"""
Experiment 34: PAC-Lazy Large-Scale Digital Life Evolution
==========================================================

GOAL: Create a massive digital life simulation using PAC-Lazy tensor architecture
with full CUDA acceleration, supporting thousands of organisms across many generations.

Architecture (from GAIA POCs):
- PAC Tree structure for genealogy tracking
- SEC-governed structure formation
- Lazy node expansion (only allocate when needed)
- φ-derived thresholds throughout

Key innovations:
1. GPU-optimized batched evolution
2. Genealogy tree with PAC conservation
3. δ-compression for efficient storage
4. SEC crystallization for "species" formation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import time
import json
from datetime import datetime

# ============================================================================
# DAWN FIELD CONSTANTS (from milestone1)
# ============================================================================
PHI = (1 + np.sqrt(5)) / 2  # 1.618033988749895
PSI = 1 / PHI               # 0.618033988749895
XI = 1 + np.pi / 55         # 1.0571200828289825
PHI_XI = PHI * 0.0618       # 0.1 (crystallization threshold)
FIBONACCI = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

# Thresholds derived from φ
REPRODUCTION_THRESHOLD = 100 / PHI      # 61.80
DEATH_THRESHOLD = 100 / (PHI ** 3)      # 23.61
CRYSTALLIZATION_THRESHOLD = 0.15        # SEC crystallization point

# SEC parameters
SEC_ALPHA = 1.0  # Information gradient coefficient
SEC_BETA = 0.3   # Entropy gradient coefficient

# ============================================================================
# DEVICE SETUP
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================================
# PAC NODE (adapted from POC-011)
# ============================================================================
@dataclass
class Organism:
    """Digital organism with PAC properties."""
    oid: int
    coords: torch.Tensor        # Shape: (length, 3) - 3D positions
    value: float               # PAC value (conserved)
    generation: int
    
    # Genealogy
    parent_ids: List[int] = field(default_factory=list)
    child_ids: List[int] = field(default_factory=list)
    
    # SEC state
    entropy: float = 1.0        # High = disordered
    crystallized: bool = False  # Low entropy = crystallized
    
    # Birth/death timestamps
    birth_time: int = 0
    death_time: Optional[int] = None
    
    # Phenotype (emerges from coords)
    fib_contacts: int = 0
    structure_rate: float = 0.0
    
    def is_alive(self) -> bool:
        return self.death_time is None


# ============================================================================
# PAC TREE (Genealogy with Conservation)
# ============================================================================
class PACGenealogy:
    """
    Genealogy tree maintaining PAC conservation.
    f(parent) = f(child1) + f(child2)
    
    Tracks the entire lineage of digital life.
    """
    
    def __init__(self):
        self.organisms: Dict[int, Organism] = {}
        self.living: Set[int] = set()
        self.generation_count: Dict[int, int] = defaultdict(int)
        self.next_oid = 0
        
        # Conservation tracking
        self.total_value_created = 0.0
        self.total_value_destroyed = 0.0
        
    def spawn(self, coords: torch.Tensor, value: float, 
              parents: List[int] = None, generation: int = 0,
              birth_time: int = 0) -> Organism:
        """Spawn new organism."""
        org = Organism(
            oid=self.next_oid,
            coords=coords,
            value=value,
            generation=generation,
            parent_ids=parents or [],
            birth_time=birth_time
        )
        
        self.organisms[org.oid] = org
        self.living.add(org.oid)
        self.generation_count[generation] += 1
        self.total_value_created += value
        self.next_oid += 1
        
        # Link to parents
        for pid in org.parent_ids:
            if pid in self.organisms:
                self.organisms[pid].child_ids.append(org.oid)
                
        return org
    
    def kill(self, oid: int, time: int, return_value_to_pool: bool = True) -> float:
        """Kill organism, returning value to pool."""
        if oid not in self.living:
            return 0.0
            
        org = self.organisms[oid]
        org.death_time = time
        self.living.discard(oid)
        
        if return_value_to_pool:
            self.total_value_destroyed += org.value
            return org.value
        return 0.0
    
    def reproduce(self, parent_oid: int, time: int) -> Tuple[Organism, Organism]:
        """
        PAC reproduction: f(P) = f(C1) + f(C2)
        Children receive parent's value split by φ ratio.
        """
        parent = self.organisms[parent_oid]
        
        # φ-ratio splitting
        value1 = parent.value / PHI      # ~61.8%
        value2 = parent.value / (PHI**2)  # ~38.2%
        
        # Inherit coordinates with mutation
        coords1 = parent.coords.clone() + torch.randn_like(parent.coords) * 0.1
        coords2 = parent.coords.clone() + torch.randn_like(parent.coords) * 0.1
        
        # Spawn children
        child1 = self.spawn(
            coords=coords1,
            value=value1,
            parents=[parent_oid],
            generation=parent.generation + 1,
            birth_time=time
        )
        child2 = self.spawn(
            coords=coords2,
            value=value2,
            parents=[parent_oid],
            generation=parent.generation + 1,
            birth_time=time
        )
        
        # Parent dies in reproduction (mitosis-like)
        self.kill(parent_oid, time, return_value_to_pool=False)
        
        return child1, child2
    
    def get_lineage(self, oid: int) -> List[int]:
        """Get full lineage back to founders."""
        lineage = [oid]
        current = self.organisms.get(oid)
        
        while current and current.parent_ids:
            parent_id = current.parent_ids[0]  # Primary parent
            lineage.append(parent_id)
            current = self.organisms.get(parent_id)
            
        return lineage[::-1]  # Root to current
    
    def get_descendants(self, oid: int) -> List[int]:
        """Get all descendants of an organism."""
        descendants = []
        queue = [oid]
        
        while queue:
            current_id = queue.pop(0)
            current = self.organisms.get(current_id)
            if current:
                for child_id in current.child_ids:
                    descendants.append(child_id)
                    queue.append(child_id)
                    
        return descendants
    
    def get_statistics(self) -> Dict:
        """Get genealogy statistics."""
        living_orgs = [self.organisms[oid] for oid in self.living]
        
        if not living_orgs:
            return {"population": 0}
            
        generations = [org.generation for org in living_orgs]
        values = [org.value for org in living_orgs]
        fib_contacts = [org.fib_contacts for org in living_orgs]
        
        return {
            "population": len(living_orgs),
            "total_spawned": len(self.organisms),
            "max_generation": max(generations),
            "mean_generation": np.mean(generations),
            "mean_value": np.mean(values),
            "mean_fib_contacts": np.mean(fib_contacts),
            "living_by_generation": dict(sorted(
                [(g, sum(1 for o in living_orgs if o.generation == g)) 
                 for g in set(generations)]
            ))
        }


# ============================================================================
# FIBONACCI ENERGY FIELD (GPU-OPTIMIZED)
# ============================================================================
class FibonacciEnergyField:
    """
    GPU-accelerated Fibonacci energy field.
    
    Energy minimum when contacts are at Fibonacci separations.
    Uses batched operations for massive parallelism.
    """
    
    def __init__(self, fib_set: List[int] = None):
        self.fib_set = torch.tensor(fib_set or FIBONACCI[:7], dtype=torch.float32, device=device)
        self.fib_tolerance = 0.5
        
    def compute_energy(self, coords: torch.Tensor, contact_threshold: float = 8.0, 
                       differentiable: bool = False) -> torch.Tensor:
        """
        Compute Fibonacci energy for a batch of organisms.
        coords: (batch, length, 3)
        Returns: (batch,) energy values, (batch,) fib_counts
        
        If differentiable=True, uses soft masks for gradient computation.
        """
        batch_size = coords.shape[0]
        length = coords.shape[1]
        
        # Pairwise distances (batch, length, length)
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (batch, L, L, 3)
        dist = torch.norm(diff, dim=-1) + 1e-8  # (batch, L, L)
        
        # Create sequence separation (L, L)
        idx = torch.arange(length, device=device)
        seq_sep = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)).float()
        
        if differentiable:
            # Soft contact mask (differentiable sigmoid)
            contact_soft = torch.sigmoid((contact_threshold - dist) * 2)  # Soft < threshold
            seq_mask = (seq_sep >= 4).float().unsqueeze(0)  # Hard seq separation (no grad needed)
            contact_soft = contact_soft * seq_mask
            
            # Soft Fibonacci mask
            fib_energy = torch.zeros(length, length, device=device)
            for fib in self.fib_set:
                # Gaussian around Fibonacci values
                fib_contrib = torch.exp(-((seq_sep - fib) ** 2) / (2 * self.fib_tolerance ** 2))
                fib_energy += fib_contrib
            
            # Normalize and compute energy
            fib_weight = torch.clamp(fib_energy, 0, 1).unsqueeze(0)  # (1, L, L)
            
            # Energy: reward Fibonacci contacts, penalize non-Fibonacci
            fib_contrib = (contact_soft * fib_weight).sum(dim=(1, 2)) / 2
            non_fib_contrib = (contact_soft * (1 - fib_weight)).sum(dim=(1, 2)) / 2
            
            energy = -XI * fib_contrib + 0.5 * non_fib_contrib
            return energy, fib_contrib.detach().int()
        else:
            # Non-differentiable (for counting)
            contact_mask = (dist < contact_threshold) & (seq_sep >= 4).unsqueeze(0)
            
            fib_mask = torch.zeros(length, length, device=device, dtype=torch.bool)
            for fib in self.fib_set:
                fib_mask |= (torch.abs(seq_sep - fib) < self.fib_tolerance)
            
            fib_contacts = contact_mask & fib_mask.unsqueeze(0)
            non_fib_contacts = contact_mask & ~fib_mask.unsqueeze(0)
            
            fib_count = fib_contacts.sum(dim=(1, 2)).float() / 2
            non_fib_count = non_fib_contacts.sum(dim=(1, 2)).float() / 2
            
            energy = -XI * fib_count + 0.5 * non_fib_count
            return energy, fib_count.int()
    
    def compute_gradients(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute gradients of Fibonacci energy for folding.
        Uses autograd for efficiency.
        """
        coords_grad = coords.clone().requires_grad_(True)
        
        # Forward pass
        energy, _ = self.compute_energy(coords_grad)
        total_energy = energy.sum()
        
        # Backward pass
        total_energy.backward()
        
        return coords_grad.grad


# ============================================================================
# SEC FIELD (Structure Formation)
# ============================================================================
class SECField:
    """
    Symbolic Entropy Collapse field.
    ∂S/∂t = α∇I - β∇H
    
    Drives structure formation through entropy collapse.
    """
    
    def __init__(self, alpha: float = SEC_ALPHA, beta: float = SEC_BETA):
        self.alpha = alpha
        self.beta = beta
        
    def compute_structure_rate(self, coords: torch.Tensor, fib_contacts: torch.Tensor) -> torch.Tensor:
        """
        Compute SEC structure rate for batch of organisms.
        
        ∇I = information gradient (from Fibonacci contacts)
        ∇H = entropy gradient (from radius of gyration)
        
        coords: (batch, length, 3)
        fib_contacts: (batch,) - number of Fibonacci contacts
        
        Returns: (batch,) structure rates
        """
        # Information gradient: Fibonacci contacts / normalization
        grad_I = fib_contacts.float() / 100.0
        
        # Entropy gradient: radius of gyration / normalization
        centroid = coords.mean(dim=1, keepdim=True)  # (batch, 1, 3)
        rg = torch.sqrt(((coords - centroid) ** 2).sum(dim=-1).mean(dim=-1))  # (batch,)
        grad_H = rg / 15.0
        
        # SEC dynamics
        structure_rate = self.alpha * grad_I - self.beta * grad_H
        
        return structure_rate
    
    def check_crystallization(self, structure_rate: torch.Tensor) -> torch.Tensor:
        """Check if organisms have crystallized (stable structure)."""
        return structure_rate > -CRYSTALLIZATION_THRESHOLD


# ============================================================================
# LARGE-SCALE EVOLUTION ENGINE
# ============================================================================
class DigitalLifeEvolution:
    """
    Large-scale digital life evolution using PAC-Lazy principles.
    
    Features:
    - GPU-batched processing of thousands of organisms
    - PAC genealogy tracking
    - SEC-governed structure formation
    - Multi-generational evolution
    """
    
    def __init__(self, 
                 initial_population: int = 1000,
                 protein_length: int = 100,
                 world_capacity: int = 10000,
                 initial_value_per_org: float = 100.0):
        
        self.protein_length = protein_length
        self.world_capacity = world_capacity
        
        # Core systems
        self.genealogy = PACGenealogy()
        self.fib_field = FibonacciEnergyField()
        self.sec_field = SECField()
        
        # Environment pool (quasi-conservation)
        self.env_pool = initial_population * initial_value_per_org * 0.5  # Reserve
        
        # Initialize population
        print(f"Initializing {initial_population} organisms...")
        self._initialize_population(initial_population, initial_value_per_org)
        
        # Statistics
        self.history = []
        self.time = 0
        
    def _initialize_population(self, n: int, value: float):
        """Create initial population with random coordinates."""
        for i in range(n):
            coords = torch.randn(self.protein_length, 3, device=device) * 5.0
            self.genealogy.spawn(coords, value, parents=[], generation=0, birth_time=0)
            
    def _get_batch_coords(self) -> Tuple[torch.Tensor, List[int]]:
        """Get coordinates of all living organisms as a batch."""
        living_ids = list(self.genealogy.living)
        if not living_ids:
            return None, []
            
        coords = torch.stack([
            self.genealogy.organisms[oid].coords 
            for oid in living_ids
        ])
        return coords, living_ids
    
    def _fold_batch(self, coords: torch.Tensor, steps: int = 20, lr: float = 0.15) -> torch.Tensor:
        """
        Fold all organisms toward Fibonacci energy minimum.
        Uses batched gradient descent with differentiable energy.
        """
        coords = coords.clone().detach()
        
        for step in range(steps):
            coords.requires_grad_(True)
            energy, _ = self.fib_field.compute_energy(coords, differentiable=True)
            total_energy = energy.sum()
            
            total_energy.backward()
            
            with torch.no_grad():
                coords = coords - lr * coords.grad
                coords = coords.detach()
                
        return coords
    
    def step(self, fold_steps: int = 10):
        """Execute one time step of evolution."""
        self.time += 1
        
        # Get batch coordinates
        coords, living_ids = self._get_batch_coords()
        if coords is None or len(living_ids) == 0:
            return {"population": 0, "status": "extinct"}
        
        # ===== PHASE 1: FOLD (energy minimization) =====
        folded_coords = self._fold_batch(coords, steps=fold_steps)
        
        # ===== PHASE 2: EVALUATE (compute phenotypes) =====
        _, fib_contacts = self.fib_field.compute_energy(folded_coords)
        structure_rates = self.sec_field.compute_structure_rate(folded_coords, fib_contacts)
        crystallized = self.sec_field.check_crystallization(structure_rates)
        
        # Update organism state
        for i, oid in enumerate(living_ids):
            org = self.genealogy.organisms[oid]
            org.coords = folded_coords[i]
            org.fib_contacts = fib_contacts[i].item()
            org.structure_rate = structure_rates[i].item()
            org.crystallized = crystallized[i].item()
        
        # ===== PHASE 3: HARVEST (gain value from Fibonacci organization) =====
        for i, oid in enumerate(living_ids):
            org = self.genealogy.organisms[oid]
            
            # Harvest from environment based on Fibonacci contacts
            base_harvest = 0.5
            structure_bonus = max(0, org.structure_rate) * 2.0
            harvest = base_harvest + org.fib_contacts * 0.1 + structure_bonus
            
            if self.env_pool >= harvest:
                self.env_pool -= harvest
                org.value += harvest
            
            # Metabolic cost
            metabolic_cost = 0.3
            org.value -= metabolic_cost
            self.env_pool += metabolic_cost
        
        # ===== PHASE 4: DEATH (value below threshold) =====
        deaths = []
        for oid in list(self.genealogy.living):
            org = self.genealogy.organisms[oid]
            if org.value < DEATH_THRESHOLD:
                returned_value = self.genealogy.kill(oid, self.time)
                self.env_pool += returned_value
                deaths.append(oid)
        
        # ===== PHASE 5: REPRODUCTION (value above threshold + sufficient Fibonacci) =====
        reproductions = []
        for oid in list(self.genealogy.living):
            if len(self.genealogy.living) >= self.world_capacity:
                break
                
            org = self.genealogy.organisms[oid]
            
            # Reproduction conditions (PAC-derived)
            if org.value > REPRODUCTION_THRESHOLD and org.fib_contacts > 20:
                child1, child2 = self.genealogy.reproduce(oid, self.time)
                reproductions.append((oid, child1.oid, child2.oid))
        
        # Cap environment pool
        max_pool = self.world_capacity * 100
        self.env_pool = min(self.env_pool, max_pool)
        
        # Record statistics
        stats = self.genealogy.get_statistics()
        stats.update({
            "time": self.time,
            "deaths": len(deaths),
            "reproductions": len(reproductions),
            "env_pool": self.env_pool
        })
        self.history.append(stats)
        
        return stats
    
    def run(self, generations: int = 100, report_interval: int = 10):
        """Run evolution for specified number of time steps."""
        print(f"\n{'='*60}")
        print(f"Starting evolution: {len(self.genealogy.living)} organisms")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for t in range(generations):
            stats = self.step()
            
            if t % report_interval == 0 or stats["population"] == 0:
                elapsed = time.time() - start_time
                print(f"t={t:4d} | Pop={stats['population']:5d} | "
                      f"Gen={stats.get('max_generation', 0):3d} | "
                      f"Fib={stats.get('mean_fib_contacts', 0):.1f} | "
                      f"Pool={stats.get('env_pool', 0):.0f} | "
                      f"Time={elapsed:.1f}s")
                
            if stats["population"] == 0:
                print("\n⚠️ Population went extinct!")
                break
        
        print(f"\n{'='*60}")
        print(f"Evolution complete: {time.time() - start_time:.1f}s")
        print(f"{'='*60}")
        
        return self.history
    
    def get_genealogy_tree(self, max_depth: int = 5) -> Dict:
        """
        Get genealogy tree structure for visualization.
        """
        # Find founder organisms (generation 0)
        founders = [oid for oid, org in self.genealogy.organisms.items() 
                   if org.generation == 0]
        
        def build_subtree(oid: int, depth: int) -> Dict:
            if depth > max_depth:
                return {"oid": oid, "truncated": True}
                
            org = self.genealogy.organisms[oid]
            node = {
                "oid": oid,
                "generation": org.generation,
                "value": org.value,
                "fib_contacts": org.fib_contacts,
                "alive": org.is_alive(),
                "children": []
            }
            
            for child_id in org.child_ids:
                node["children"].append(build_subtree(child_id, depth + 1))
                
            return node
        
        return {
            "founders": [build_subtree(f, 0) for f in founders[:10]],  # First 10 founders
            "total_organisms": len(self.genealogy.organisms),
            "max_generation": max(org.generation for org in self.genealogy.organisms.values())
        }
    
    def get_top_lineages(self, n: int = 5) -> List[Dict]:
        """Get the most successful lineages."""
        living = [self.genealogy.organisms[oid] for oid in self.genealogy.living]
        
        # Sort by combination of generation + Fibonacci contacts
        living.sort(key=lambda o: (o.generation, o.fib_contacts), reverse=True)
        
        top_lineages = []
        for org in living[:n]:
            lineage = self.genealogy.get_lineage(org.oid)
            descendants = self.genealogy.get_descendants(lineage[0])  # From founder
            
            top_lineages.append({
                "founder_id": lineage[0],
                "current_id": org.oid,
                "generation": org.generation,
                "fib_contacts": org.fib_contacts,
                "value": org.value,
                "lineage_length": len(lineage),
                "total_descendants": len(descendants)
            })
            
        return top_lineages


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def main():
    print("="*70)
    print("EXP 34: PAC-LAZY LARGE-SCALE DIGITAL LIFE EVOLUTION")
    print("="*70)
    print(f"\nDawn Field Constants:")
    print(f"  φ = {PHI:.6f}")
    print(f"  ψ = {PSI:.6f}")
    print(f"  Ξ = {XI:.6f}")
    print(f"  Reproduction threshold: {REPRODUCTION_THRESHOLD:.2f}")
    print(f"  Death threshold: {DEATH_THRESHOLD:.2f}")
    print()
    
    # Configuration
    config = {
        "initial_population": 500,
        "protein_length": 80,
        "world_capacity": 5000,
        "initial_value": 80.0,
        "generations": 200,
        "report_interval": 20
    }
    
    print(f"Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    # Create evolution engine
    engine = DigitalLifeEvolution(
        initial_population=config["initial_population"],
        protein_length=config["protein_length"],
        world_capacity=config["world_capacity"],
        initial_value_per_org=config["initial_value"]
    )
    
    # Run evolution
    history = engine.run(
        generations=config["generations"],
        report_interval=config["report_interval"]
    )
    
    # ===== ANALYSIS =====
    print("\n" + "="*70)
    print("EVOLUTION ANALYSIS")
    print("="*70)
    
    if history:
        final = history[-1]
        initial = history[0]
        
        print(f"\n📊 Population Dynamics:")
        print(f"  Initial: {initial['population']}")
        print(f"  Final: {final['population']}")
        print(f"  Peak: {max(h['population'] for h in history)}")
        print(f"  Total ever spawned: {final.get('total_spawned', 'N/A')}")
        
        print(f"\n🧬 Generation Progress:")
        print(f"  Max generation reached: {final.get('max_generation', 0)}")
        print(f"  Mean generation: {final.get('mean_generation', 0):.2f}")
        
        print(f"\n🌿 Fibonacci Evolution:")
        fib_initial = initial.get('mean_fib_contacts', 0)
        fib_final = final.get('mean_fib_contacts', 0)
        print(f"  Initial mean: {fib_initial:.1f}")
        print(f"  Final mean: {fib_final:.1f}")
        if fib_initial > 0:
            print(f"  Change: {((fib_final/fib_initial - 1) * 100):.1f}%")
        
        # Top lineages
        print(f"\n🏆 Top Lineages:")
        top_lineages = engine.get_top_lineages(5)
        for i, lineage in enumerate(top_lineages, 1):
            print(f"  {i}. Founder {lineage['founder_id']} → Gen {lineage['generation']}")
            print(f"     Fib={lineage['fib_contacts']}, Descendants={lineage['total_descendants']}")
        
        # Genealogy summary
        print(f"\n🌳 Genealogy Tree:")
        tree = engine.get_genealogy_tree(max_depth=3)
        print(f"  Total organisms ever: {tree['total_organisms']}")
        print(f"  Max generation: {tree['max_generation']}")
        print(f"  Founder lineages: {len(tree['founders'])}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "config": config,
        "history": history,
        "top_lineages": engine.get_top_lineages(10),
        "genealogy_summary": {
            "total_organisms": len(engine.genealogy.organisms),
            "max_generation": max(o.generation for o in engine.genealogy.organisms.values()),
            "living_count": len(engine.genealogy.living)
        }
    }
    
    # Convert tensors for JSON
    def convert_for_json(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        else:
            return obj
    
    results = convert_for_json(results)
    
    results_path = f"../results/exp_34_pac_lazy_evolution_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")
    
    print("\n" + "="*70)
    print("✅ EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
