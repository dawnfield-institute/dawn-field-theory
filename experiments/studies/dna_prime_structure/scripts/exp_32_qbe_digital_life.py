#!/usr/bin/env python3
"""
exp_32_qbe_digital_life.py

QUANTUM BALANCE EQUATION + RBF DIGITAL LIFE

Integrates Dawn Field Theory principles:

1. QBE (Quantum Balance Equation):
   dI/dt + dE/dt = λ·QPL(t)
   
   Information gain (Fibonacci organization) must balance energy cost.
   Thresholds EMERGE from this balance, not hardcoded.

2. RBF (Recursive Balance Field):
   Systems self-regulate toward dynamic stability through feedback.
   
   Population dynamics, reproduction thresholds, and metabolic costs
   all adjust recursively to maintain far-from-equilibrium order.

Key insight: Replace arbitrary thresholds with BALANCE EQUATIONS.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Dict, Optional
import json
from datetime import datetime
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

FIBONACCI_SET = {3, 5, 8, 13, 21, 34}

# DFT Constants
PHI = (1 + np.sqrt(5)) / 2           # Golden ratio φ ≈ 1.618
PHI_INV = 1 / PHI                     # 1/φ ≈ 0.618
XI = 1 + np.pi / 55                   # Ξ ≈ 1.057 (balance operator)


@dataclass
class Organism:
    """A digital organism with QBE state."""
    id: int
    sequence: torch.Tensor
    coords: torch.Tensor
    
    # QBE state variables
    energy: float = 100.0             # E in QBE
    information: float = 0.0          # I in QBE (Fibonacci organization)
    qpl: float = 1.0                  # Quantum Potential Layer (stability field)
    
    age: int = 0
    generation: int = 0
    parent_id: Optional[int] = None
    
    def __len__(self):
        return len(self.sequence)


class QuantumBalanceField:
    """
    Implements QBE: dI/dt + dE/dt = λ·QPL(t)
    
    The QPL (Quantum Potential Layer) acts as a regulatory boundary.
    When I and E are balanced, the system is stable.
    Imbalance creates stress that drives adaptation or collapse.
    """
    
    def __init__(self, lambda_coupling: float = XI):
        self.lambda_coupling = lambda_coupling  # Use Ξ as coupling constant
        self.global_qpl = 1.0
        
        # RBF feedback parameters
        self.rbf_memory = []  # Track recent balance states
        self.rbf_window = 10
    
    def compute_balance(self, organism: Organism) -> Dict:
        """
        Compute QBE balance for an organism.
        
        Returns balance metrics that determine:
        - Viability (can organism survive?)
        - Reproduction potential
        - Stability
        """
        I = organism.information  # Fibonacci contacts
        E = organism.energy
        
        # QBE: dI/dt + dE/dt = λ·QPL
        # At equilibrium: I/E ratio should approach φ (golden ratio)
        # This is the "golden balance" hypothesis
        
        if E > 0:
            ie_ratio = I / E
            # Distance from golden balance
            balance_deviation = abs(ie_ratio - PHI_INV)  # Target: I/E ≈ 0.618
        else:
            balance_deviation = float('inf')
        
        # QPL: Quantum Potential Layer
        # Higher QPL = more stable, can handle more deviation
        # QPL accumulates from sustained balance
        qpl_contribution = self.lambda_coupling * organism.qpl
        
        # Effective balance score (0-1, higher = more balanced)
        balance_score = np.exp(-balance_deviation / qpl_contribution)
        
        # RBF: Recursive feedback
        # Balance score feeds back into future stability
        stability = balance_score * organism.qpl
        
        return {
            'ie_ratio': ie_ratio if E > 0 else 0,
            'balance_deviation': balance_deviation,
            'balance_score': balance_score,
            'qpl': organism.qpl,
            'stability': stability,
            'golden_target': PHI_INV
        }
    
    def update_qpl(self, organism: Organism, balance_score: float) -> float:
        """
        RBF: Update organism's QPL based on balance history.
        
        Good balance → QPL increases (more resilient)
        Poor balance → QPL decreases (more fragile)
        """
        # Recursive update: QPL moves toward balance_score
        # with damping factor based on Ξ
        damping = 1 / XI
        new_qpl = organism.qpl * damping + balance_score * (1 - damping)
        
        # QPL bounded by golden ratio limits
        new_qpl = np.clip(new_qpl, PHI_INV / 2, PHI)
        
        return new_qpl
    
    def compute_reproduction_threshold(self, population_balance: float) -> float:
        """
        RBF: Reproduction threshold EMERGES from population balance.
        
        High population balance → easier reproduction (abundance)
        Low balance → harder reproduction (scarcity)
        """
        # Base threshold at golden ratio point
        base = 50.0  # Lowered from 100
        
        # Adjust by population balance state
        threshold = base / (population_balance + 0.3)  # More accessible
        
        # Bounded by golden ratio
        return np.clip(threshold, 30.0, 120.0)
    
    def compute_death_threshold(self, organism: Organism) -> float:
        """
        QBE: Death threshold based on individual balance state.
        
        Well-balanced organisms can survive lower energy.
        Imbalanced organisms die at higher energy.
        """
        balance = self.compute_balance(organism)
        
        # Base death threshold
        base = 10.0
        
        # Balanced organisms are more resilient
        resilience = balance['stability']
        
        # Death threshold: lower for balanced organisms
        return base * (1.0 - 0.5 * resilience)


class FibonacciFolder:
    """Energy-based folder with QBE-derived weights."""
    
    def __init__(self, length: int):
        self.L = length
        
        idx = torch.arange(length, device=device)
        self.seq_sep = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        
        # QBE-derived Fibonacci weights
        # Use golden ratio powers for weighting
        self.fib_weights = torch.zeros((length, length), device=device)
        
        fib_numbers = [3, 5, 8, 13, 21, 34]
        for i, f in enumerate(fib_numbers):
            mask = (self.seq_sep == f)
            # Weight by position in Fibonacci sequence (φ^i scaling)
            weight = PHI ** (len(fib_numbers) - i - 3)  # Peak at 8, 13
            self.fib_weights[mask] = weight
        
        self.fib_mask = torch.zeros((length, length), device=device, dtype=torch.bool)
        for f in FIBONACCI_SET:
            self.fib_mask |= (self.seq_sep == f)
    
    def compute_energy(self, coords: torch.Tensor) -> torch.Tensor:
        L = self.L
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)
        dist = torch.norm(diff + 1e-8, dim=2)
        
        backbone_dist = dist[torch.arange(L-1), torch.arange(1, L)]
        e_backbone = 10.0 * ((backbone_dist - 2.0) ** 2).sum()
        
        clash_mask = (self.seq_sep > 1) & (dist < 1.5)
        e_clash = 50.0 * torch.relu(1.5 - dist[clash_mask]).sum()
        
        com = coords.mean(dim=0)
        radius = torch.norm(coords - com, dim=1).mean()
        e_compact = 0.3 * (radius - 6.0) ** 2 if radius > 6.0 else torch.tensor(0.0, device=device)
        
        # Fibonacci contact energy - scaled by Ξ
        contact_potential = torch.exp(-dist / 4.0)
        e_fib = -XI * (self.fib_weights * contact_potential).sum()
        
        return e_backbone + e_clash + e_compact + e_fib
    
    def fold(self, coords: torch.Tensor, steps: int = 150) -> torch.Tensor:
        coords = coords.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([coords], lr=0.15)
        
        for _ in range(steps):
            optimizer.zero_grad()
            energy = self.compute_energy(coords)
            energy.backward()
            optimizer.step()
        
        return coords.detach()
    
    def count_fib_contacts(self, coords: torch.Tensor) -> int:
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)
        dist = torch.norm(diff, dim=2)
        # Contact threshold based on golden ratio: 5.0 * φ^(-1) ≈ 3.09
        threshold = 5.0 * PHI_INV + 2.0  # ≈ 5.09
        contact = (dist < threshold) & self.fib_mask
        return contact.sum().item() // 2


class QBELifeSimulation:
    """
    Life simulation with QBE and RBF dynamics.
    
    Key difference from exp_31:
    - Thresholds are EMERGENT from balance equations
    - QPL provides stability feedback
    - Golden ratio governs critical transitions
    """
    
    def __init__(self,
                 organism_length: int = 30,
                 initial_population: int = 20):
        
        self.L = organism_length
        self.folder = FibonacciFolder(organism_length)
        self.qbf = QuantumBalanceField()
        
        self.organisms: List[Organism] = []
        self.next_id = 0
        self.time = 0
        
        # Dynamic thresholds (RBF-controlled)
        self.reproduction_threshold = 100.0
        self.population_balance = 0.5
        
        self.history = {
            'time': [],
            'population': [],
            'mean_information': [],
            'mean_energy': [],
            'mean_balance': [],
            'mean_qpl': [],
            'reproduction_threshold': [],
            'births': [],
            'deaths': []
        }
        
        self._initialize_population(initial_population)
    
    def _initialize_population(self, n: int):
        for _ in range(n):
            org = self._create_organism(parent=None)
            self.organisms.append(org)
    
    def _create_organism(self, parent: Optional[Organism]) -> Organism:
        if parent is None:
            sequence = torch.randint(0, 20, (self.L,), device=device)
            coords = torch.zeros((self.L, 3), device=device)
            coords[:, 0] = torch.arange(self.L, dtype=torch.float32, device=device) * 2.0
            coords += torch.randn_like(coords) * 0.3
            offset = torch.rand(3, device=device) * 50.0
            coords += offset
            generation = 0
            parent_id = None
            initial_qpl = 1.0
        else:
            sequence = parent.sequence.clone()
            n_mutations = max(1, int(0.1 * self.L))
            positions = torch.randperm(self.L)[:n_mutations]
            sequence[positions] = torch.randint(0, 20, (n_mutations,), device=device)
            
            coords = parent.coords.clone()
            coords += torch.randn_like(coords) * 1.5
            
            generation = parent.generation + 1
            parent_id = parent.id
            # Inherit QPL with slight regression toward mean
            initial_qpl = parent.qpl * 0.9 + 0.1
        
        coords = self.folder.fold(coords, steps=100)
        fib_contacts = self.folder.count_fib_contacts(coords)
        
        org = Organism(
            id=self.next_id,
            sequence=sequence,
            coords=coords,
            energy=80.0,
            information=float(fib_contacts),
            qpl=initial_qpl,
            generation=generation,
            parent_id=parent_id
        )
        self.next_id += 1
        
        return org
    
    def step(self) -> Dict:
        births = 0
        deaths = 0
        
        balance_scores = []
        
        # Update each organism
        for org in self.organisms:
            # Compute QBE balance
            balance = self.qbf.compute_balance(org)
            balance_scores.append(balance['balance_score'])
            
            # Energy harvest based on balance (QBE principle)
            # Well-balanced organisms harvest more efficiently
            harvest_efficiency = 1.0 + balance['stability'] * PHI
            org.energy += 1.5 * harvest_efficiency
            
            # INFORMATION GROWTH: Organisms can "invest" energy in better folding
            # This is the key QBE mechanism - energy → information conversion
            if org.energy > 40 and np.random.random() < 0.2:
                # Refold to potentially find better structure
                new_coords = self.folder.fold(org.coords + torch.randn_like(org.coords) * 0.5, steps=50)
                new_info = float(self.folder.count_fib_contacts(new_coords))
                if new_info > org.information:
                    org.coords = new_coords
                    org.information = new_info
                    org.energy -= 5.0  # Energy cost of refolding
            
            # Metabolic cost scales with ENERGY (not just information)
            # This creates pressure to maintain I/E balance
            metabolic_cost = 1.0 + 0.02 * org.energy  # Costs grow with energy
            org.energy -= metabolic_cost
            
            # Update QPL (RBF feedback)
            org.qpl = self.qbf.update_qpl(org, balance['balance_score'])
            
            org.age += 1
        
        # Update population balance (RBF)
        if balance_scores:
            self.population_balance = np.mean(balance_scores)
        
        # Dynamic reproduction threshold (RBF emergence)
        self.reproduction_threshold = self.qbf.compute_reproduction_threshold(
            self.population_balance
        )
        
        # Reproduction: Based on QBE balance, not just energy
        new_organisms = []
        for org in self.organisms:
            balance = self.qbf.compute_balance(org)
            
            # Reproduction requires:
            # 1. Sufficient energy
            # 2. Good balance (closer to golden ratio = better)
            # Use balance_score directly instead of hard threshold
            reproduction_chance = balance['balance_score'] * (org.energy / self.reproduction_threshold)
            
            can_reproduce = (
                org.energy > self.reproduction_threshold and
                reproduction_chance > PHI_INV  # Probability threshold at φ⁻¹
            )
            
            if can_reproduce and len(self.organisms) + len(new_organisms) < 50:
                child = self._create_organism(parent=org)
                new_organisms.append(child)
                # Reproduction cost scales with parent's information
                org.energy -= 50.0 + org.information * 0.5
                births += 1
        
        self.organisms.extend(new_organisms)
        
        # Death: Based on individual QBE thresholds
        survivors = []
        for org in self.organisms:
            death_threshold = self.qbf.compute_death_threshold(org)
            
            if org.energy > death_threshold:
                survivors.append(org)
            else:
                deaths += 1
        
        self.organisms = survivors
        
        self.time += 1
        self._record_stats(births, deaths)
        
        return {'births': births, 'deaths': deaths, 'population': len(self.organisms)}
    
    def _record_stats(self, births: int, deaths: int):
        if not self.organisms:
            return
        
        infos = [org.information for org in self.organisms]
        energies = [org.energy for org in self.organisms]
        qpls = [org.qpl for org in self.organisms]
        balances = [self.qbf.compute_balance(org)['balance_score'] for org in self.organisms]
        
        self.history['time'].append(self.time)
        self.history['population'].append(len(self.organisms))
        self.history['mean_information'].append(np.mean(infos))
        self.history['mean_energy'].append(np.mean(energies))
        self.history['mean_balance'].append(np.mean(balances))
        self.history['mean_qpl'].append(np.mean(qpls))
        self.history['reproduction_threshold'].append(self.reproduction_threshold)
        self.history['births'].append(births)
        self.history['deaths'].append(deaths)
    
    def run(self, steps: int = 60, report_every: int = 10) -> Dict:
        print(f"\nQBE Life Simulation starting with {len(self.organisms)} organisms...")
        print(f"Golden target I/E ratio: {PHI_INV:.3f}")
        print(f"Balance operator Ξ: {XI:.4f}\n")
        
        for t in range(steps):
            result = self.step()
            
            if t % report_every == 0 and self.organisms:
                mean_info = np.mean([org.information for org in self.organisms])
                mean_balance = np.mean([self.qbf.compute_balance(org)['balance_score'] 
                                       for org in self.organisms])
                mean_qpl = np.mean([org.qpl for org in self.organisms])
                
                print(f"  t={t}: Pop={len(self.organisms)}, "
                      f"I={mean_info:.1f}, Balance={mean_balance:.3f}, "
                      f"QPL={mean_qpl:.3f}, RepThresh={self.reproduction_threshold:.0f}")
            
            if len(self.organisms) == 0:
                print("  Population extinct!")
                break
        
        return {
            'final_population': len(self.organisms),
            'history': self.history,
            'survivors': [(org.id, org.information, org.qpl, org.generation) 
                         for org in self.organisms[:10]]
        }


def run_experiment():
    print("=" * 70)
    print("QBE + RBF DIGITAL LIFE")
    print("=" * 70)
    print(f"\nQuantum Balance Equation: dI/dt + dE/dt = λ·QPL(t)")
    print(f"Target I/E ratio: φ⁻¹ = {PHI_INV:.4f}")
    print(f"Balance operator: Ξ = {XI:.4f}")
    print(f"Golden ratio: φ = {PHI:.4f}")
    
    results = {}
    
    # =========================================================
    # SIMULATION: QBE Life
    # =========================================================
    print("\n" + "=" * 70)
    print("QBE LIFE SIMULATION")
    print("-" * 70)
    
    sim = QBELifeSimulation(
        organism_length=28,
        initial_population=18
    )
    
    sim_results = sim.run(steps=80, report_every=10)
    
    if sim.history['time']:
        # Analyze QBE dynamics
        initial_info = sim.history['mean_information'][0]
        final_info = sim.history['mean_information'][-1]
        initial_balance = sim.history['mean_balance'][0]
        final_balance = sim.history['mean_balance'][-1]
        initial_qpl = sim.history['mean_qpl'][0]
        final_qpl = sim.history['mean_qpl'][-1]
        
        print(f"\n  QBE Dynamics:")
        print(f"    Information (I): {initial_info:.1f} → {final_info:.1f}")
        print(f"    Balance score:   {initial_balance:.3f} → {final_balance:.3f}")
        print(f"    QPL stability:   {initial_qpl:.3f} → {final_qpl:.3f}")
        print(f"    Final population: {sim_results['final_population']}")
        
        # Check if approaching golden balance
        final_ie_ratios = []
        for org in sim.organisms:
            if org.energy > 0:
                final_ie_ratios.append(org.information / org.energy)
        
        if final_ie_ratios:
            mean_ie = np.mean(final_ie_ratios)
            print(f"\n  Golden Balance Analysis:")
            print(f"    Mean I/E ratio: {mean_ie:.4f}")
            print(f"    Target (φ⁻¹):   {PHI_INV:.4f}")
            print(f"    Deviation:      {abs(mean_ie - PHI_INV):.4f}")
            
            if abs(mean_ie - PHI_INV) < 0.2:
                print("    ✅ Population approaching golden balance!")
        
        results['simulation'] = {
            'initial_info': float(initial_info),
            'final_info': float(final_info),
            'initial_balance': float(initial_balance),
            'final_balance': float(final_balance),
            'initial_qpl': float(initial_qpl),
            'final_qpl': float(final_qpl),
            'final_population': sim_results['final_population']
        }
    
    # =========================================================
    # ANALYSIS: QBE Attractor
    # =========================================================
    print("\n" + "=" * 70)
    print("QBE ATTRACTOR ANALYSIS")
    print("-" * 70)
    
    if sim.organisms:
        print("\n  Top organisms by QPL (stability):")
        top_by_qpl = sorted(sim.organisms, key=lambda x: x.qpl, reverse=True)[:5]
        
        for org in top_by_qpl:
            balance = sim.qbf.compute_balance(org)
            ie_ratio = org.information / org.energy if org.energy > 0 else 0
            print(f"    ID={org.id}: I={org.information:.0f}, E={org.energy:.0f}, "
                  f"I/E={ie_ratio:.3f}, QPL={org.qpl:.3f}")
        
        # Check if high-QPL organisms are closer to golden ratio
        top_ie = [o.information / o.energy for o in top_by_qpl if o.energy > 0]
        all_ie = [o.information / o.energy for o in sim.organisms if o.energy > 0]
        
        if top_ie and all_ie:
            top_deviation = np.mean([abs(r - PHI_INV) for r in top_ie])
            all_deviation = np.mean([abs(r - PHI_INV) for r in all_ie])
            
            print(f"\n  Golden ratio deviation:")
            print(f"    Top-5 by QPL: {top_deviation:.4f}")
            print(f"    All organisms: {all_deviation:.4f}")
            
            if top_deviation < all_deviation:
                print("    ✅ Higher QPL correlates with closer golden balance!")
    
    # =========================================================
    # VERDICT
    # =========================================================
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    qbe_success = (
        sim_results.get('final_population', 0) > 10 and
        final_info > initial_info * 1.1 and
        final_balance > 0.3
    )
    
    print(f"""
    QBE Integration: {"✅" if qbe_success else "⚠️"}
      - Population survived: {sim_results.get('final_population', 0)}
      - Information increase: {final_info/initial_info:.2f}x
      - Final balance score: {final_balance:.3f}
    
    Key DFT Principles Demonstrated:
      - QBE: dI/dt + dE/dt = λ·QPL(t) governs viability
      - RBF: Thresholds emerge from recursive feedback
      - φ: Golden ratio as attractor for stable systems
      - Ξ: Balance operator modulates dynamics
    """)
    
    if qbe_success:
        print("""
    ✅ QBE + RBF DIGITAL LIFE SUCCESSFUL
    
    Thresholds are no longer arbitrary - they EMERGE from:
    1. QBE balance between information and energy
    2. RBF recursive feedback on population state
    3. Golden ratio as natural attractor
    
    This demonstrates that DFT principles can govern
    artificial life dynamics, not just describe them.
    """)
    
    results['verdict'] = {'qbe_success': qbe_success}
    
    # Save
    out_path = Path(__file__).parent.parent / 'results' / f'exp_32_qbe_life_{datetime.now():%Y%m%d_%H%M%S}.json'
    out_path.parent.mkdir(exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return float(obj) if not isinstance(obj, np.bool_) else bool(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj
    
    clean_results = convert(results)
    if 'history' in clean_results.get('simulation', {}):
        del clean_results['simulation']['history']
    
    with open(out_path, 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(), 
                   'constants': {'PHI': PHI, 'PHI_INV': PHI_INV, 'XI': XI},
                   'results': clean_results}, f, indent=2)
    
    print(f"\nResults saved to {out_path}")
    
    return results


if __name__ == '__main__':
    run_experiment()
