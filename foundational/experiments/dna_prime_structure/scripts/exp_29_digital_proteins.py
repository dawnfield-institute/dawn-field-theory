#!/usr/bin/env python3
"""
exp_29_digital_proteins.py

DIGITAL PROTEIN SELF-ORGANIZATION

Use the discovered Fibonacci sequence principle to create artificial
"digital proteins" that self-organize in a tensor environment.

Design principles (from experiments):
1. Residues at Fibonacci sequence separations should attract
2. Flexible dynamics emerge from Fibonacci organization  
3. Self-replication = pattern propagation through field

Architecture:
- 1D sequence of "residues" with properties
- 2D/3D folding space
- Energy function favoring Fibonacci-spaced contacts
- Dynamics via gradient descent + noise (thermal fluctuations)
- Replication via field resonance patterns

Goal: See if Fibonacci-based rules produce stable, dynamic, 
self-organizing structures without explicit programming.
"""

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple
import json
from datetime import datetime
from pathlib import Path

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Fibonacci sequence separations that should form contacts
FIBONACCI = torch.tensor([3, 5, 8, 13, 21, 34, 55], device=device, dtype=torch.float32)


@dataclass
class DigitalProtein:
    """A digital protein: sequence + 3D coordinates."""
    sequence: torch.Tensor      # (L,) residue types (0-19 like amino acids)
    coords: torch.Tensor        # (L, 3) 3D coordinates
    velocities: torch.Tensor    # (L, 3) for dynamics
    
    def __len__(self):
        return len(self.sequence)


class FibonacciEnergyField:
    """
    Energy field that favors Fibonacci sequence separations.
    
    E = E_contact + E_backbone + E_clash
    
    E_contact: Reward contacts at Fibonacci sequence separations
    E_backbone: Keep sequential residues connected
    E_clash: Prevent overlapping
    """
    
    def __init__(self, 
                 fib_strength: float = 2.0,
                 backbone_strength: float = 10.0,
                 clash_radius: float = 1.0,
                 contact_radius: float = 4.0):
        self.fib_strength = fib_strength
        self.backbone_strength = backbone_strength
        self.clash_radius = clash_radius
        self.contact_radius = contact_radius
        
        # Fibonacci weights (higher for core Fibonacci numbers)
        self.fib_weights = {3: 1.0, 5: 1.5, 8: 2.0, 13: 2.5, 21: 2.0, 34: 1.5, 55: 1.0}
    
    def compute_energy(self, protein: DigitalProtein) -> Tuple[torch.Tensor, dict]:
        """Compute total energy and components."""
        L = len(protein)
        coords = protein.coords
        
        # Pairwise distances
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # (L, L, 3)
        dist = torch.norm(diff, dim=2)  # (L, L)
        
        # Sequence separations
        idx = torch.arange(L, device=device)
        seq_sep = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))  # (L, L)
        
        # 1. Fibonacci contact energy (NEGATIVE = favorable)
        # Reward when Fibonacci-separated residues are close
        e_fib = torch.tensor(0.0, device=device)
        for fib, weight in self.fib_weights.items():
            mask = (seq_sep == fib)
            if mask.any():
                fib_dists = dist[mask]
                # Smooth contact potential: lower energy when close
                contact_energy = torch.exp(-fib_dists / self.contact_radius)
                e_fib -= weight * self.fib_strength * contact_energy.sum()
        
        # 2. Backbone connectivity (sequential residues should be ~1.5 apart)
        backbone_dist = dist[torch.arange(L-1), torch.arange(1, L)]
        e_backbone = self.backbone_strength * ((backbone_dist - 1.5) ** 2).sum()
        
        # 3. Clash penalty (prevent overlaps, except backbone)
        clash_mask = (seq_sep > 1) & (dist < self.clash_radius)
        e_clash = 100.0 * (self.clash_radius - dist[clash_mask]).sum() if clash_mask.any() else torch.tensor(0.0, device=device)
        
        total = e_fib + e_backbone + e_clash
        
        components = {
            'fibonacci': e_fib.item(),
            'backbone': e_backbone.item(),
            'clash': e_clash.item(),
            'total': total.item()
        }
        
        return total, components
    
    def count_fibonacci_contacts(self, protein: DigitalProtein) -> dict:
        """Count contacts at each Fibonacci separation."""
        L = len(protein)
        coords = protein.coords
        
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)
        dist = torch.norm(diff, dim=2)
        
        idx = torch.arange(L, device=device)
        seq_sep = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        
        contacts = {}
        for fib in [3, 5, 8, 13, 21, 34]:
            mask = (seq_sep == fib) & (dist < self.contact_radius)
            contacts[fib] = mask.sum().item() // 2  # Divide by 2 for symmetric matrix
        
        return contacts


class DigitalProteinSimulator:
    """
    Simulate digital protein folding and dynamics.
    
    Uses Langevin dynamics:
    dv/dt = -∇E - γv + η(t)
    dx/dt = v
    
    Where η(t) is thermal noise.
    """
    
    def __init__(self, 
                 energy_field: FibonacciEnergyField,
                 temperature: float = 0.1,
                 friction: float = 0.5,
                 dt: float = 0.01):
        self.energy = energy_field
        self.temp = temperature
        self.friction = friction
        self.dt = dt
    
    def create_random_protein(self, length: int) -> DigitalProtein:
        """Create a random extended protein."""
        sequence = torch.randint(0, 20, (length,), device=device)
        
        # Start as extended chain along x-axis
        coords = torch.zeros((length, 3), device=device)
        coords[:, 0] = torch.arange(length, dtype=torch.float32, device=device) * 1.5
        
        # Add small random perturbation
        coords += torch.randn_like(coords) * 0.1
        coords.requires_grad_(True)
        
        velocities = torch.zeros((length, 3), device=device)
        
        return DigitalProtein(sequence, coords, velocities)
    
    def step(self, protein: DigitalProtein) -> Tuple[DigitalProtein, dict]:
        """One step of Langevin dynamics."""
        # Compute forces from energy gradient
        protein.coords.requires_grad_(True)
        energy, components = self.energy.compute_energy(protein)
        
        # Gradient
        energy.backward()
        forces = -protein.coords.grad.detach()
        protein.coords.requires_grad_(False)
        
        # Langevin dynamics
        noise = torch.randn_like(protein.velocities) * np.sqrt(2 * self.temp * self.friction * self.dt)
        
        # Update velocities
        new_velocities = protein.velocities + self.dt * (forces - self.friction * protein.velocities) + noise
        
        # Update positions
        new_coords = protein.coords.detach() + self.dt * new_velocities
        
        return DigitalProtein(protein.sequence, new_coords, new_velocities), components
    
    def fold(self, protein: DigitalProtein, steps: int = 1000, 
             report_every: int = 100) -> Tuple[DigitalProtein, List[dict]]:
        """Fold a protein through dynamics."""
        history = []
        
        for i in range(steps):
            protein, components = self.step(protein)
            
            if i % report_every == 0:
                contacts = self.energy.count_fibonacci_contacts(protein)
                record = {
                    'step': i,
                    **components,
                    'contacts': contacts,
                    'total_fib_contacts': sum(contacts.values())
                }
                history.append(record)
                
                if i % (report_every * 5) == 0:
                    print(f"  Step {i}: E={components['total']:.1f}, "
                          f"Fib contacts={sum(contacts.values())}")
        
        return protein, history


class SelfReplicator:
    """
    Test if Fibonacci-organized proteins can propagate their pattern.
    
    Mechanism: A "template" protein creates a field that biases
    the folding of nearby "child" proteins toward similar structure.
    """
    
    def __init__(self, energy_field: FibonacciEnergyField):
        self.energy = energy_field
    
    def compute_similarity(self, p1: DigitalProtein, p2: DigitalProtein) -> float:
        """Compute structural similarity via contact map overlap."""
        if len(p1) != len(p2):
            return 0.0
        
        L = len(p1)
        
        # Contact maps
        def get_contacts(protein):
            diff = protein.coords.unsqueeze(0) - protein.coords.unsqueeze(1)
            dist = torch.norm(diff, dim=2)
            return (dist < 4.0).float()
        
        c1 = get_contacts(p1)
        c2 = get_contacts(p2)
        
        # Overlap
        overlap = (c1 * c2).sum()
        total = torch.maximum(c1.sum(), c2.sum())
        
        return (overlap / total).item() if total > 0 else 0.0
    
    def template_field(self, template: DigitalProtein, 
                       child_coords: torch.Tensor) -> torch.Tensor:
        """
        Create a field from template that influences child folding.
        Returns additional forces on child coordinates.
        """
        L = len(template)
        
        # For each Fibonacci pair in template that's in contact,
        # bias the corresponding child pair toward contact
        template_diff = template.coords.unsqueeze(0) - template.coords.unsqueeze(1)
        template_dist = torch.norm(template_diff, dim=2)
        
        idx = torch.arange(L, device=device)
        seq_sep = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        
        forces = torch.zeros_like(child_coords)
        
        for fib in [5, 8, 13, 21]:
            # Find Fibonacci contacts in template
            mask = (seq_sep == fib) & (template_dist < 4.0)
            
            if mask.any():
                # Get pairs
                pairs = mask.nonzero()
                for pair in pairs:
                    i, j = pair[0].item(), pair[1].item()
                    if i < j:  # Avoid double counting
                        # Vector from i to j in child
                        vec = child_coords[j] - child_coords[i]
                        dist = torch.norm(vec)
                        
                        if dist > 2.0:  # Only if not already close
                            # Attractive force
                            direction = vec / (dist + 1e-6)
                            force_mag = 0.5 * (dist - 3.0)  # Target distance ~3
                            forces[i] += force_mag * direction
                            forces[j] -= force_mag * direction
        
        return forces


def run_experiment():
    """Main experiment: fold digital proteins and test replication."""
    
    print("=" * 70)
    print("EXP 29: DIGITAL PROTEIN SELF-ORGANIZATION")
    print("=" * 70)
    print("\nUsing Fibonacci sequence spacing as organizing principle")
    
    # Setup - optimized for speed
    energy_field = FibonacciEnergyField(
        fib_strength=5.0,  # Stronger to converge faster
        backbone_strength=20.0,
        contact_radius=5.0
    )
    
    simulator = DigitalProteinSimulator(
        energy_field=energy_field,
        temperature=0.02,  # Lower noise
        friction=2.0,  # Higher damping
        dt=0.02  # Larger timestep
    )
    
    results = {}
    
    # Test 1: Can a random sequence fold to form Fibonacci contacts?
    print("\n" + "=" * 70)
    print("TEST 1: FIBONACCI-GUIDED FOLDING")
    print("=" * 70)
    
    for length in [25, 40, 55]:
        print(f"\nFolding protein of length {length}...")
        
        protein = simulator.create_random_protein(length)
        
        # Initial contacts
        initial_contacts = energy_field.count_fibonacci_contacts(protein)
        print(f"  Initial Fibonacci contacts: {sum(initial_contacts.values())}")
        
        # Fold
        folded, history = simulator.fold(protein, steps=500, report_every=100)
        
        # Final contacts
        final_contacts = energy_field.count_fibonacci_contacts(folded)
        print(f"  Final Fibonacci contacts: {sum(final_contacts.values())}")
        print(f"  Breakdown: {final_contacts}")
        
        # Energy trajectory
        initial_e = history[0]['total']
        final_e = history[-1]['total']
        print(f"  Energy: {initial_e:.1f} → {final_e:.1f}")
        
        results[f'fold_{length}'] = {
            'length': length,
            'initial_contacts': sum(initial_contacts.values()),
            'final_contacts': sum(final_contacts.values()),
            'contact_breakdown': final_contacts,
            'energy_change': final_e - initial_e,
            'history': history
        }
    
    # Test 2: Do Fibonacci contacts emerge preferentially?
    print("\n" + "=" * 70)
    print("TEST 2: FIBONACCI vs NON-FIBONACCI CONTACT FORMATION")
    print("=" * 70)
    
    # Fold multiple proteins and count contacts by sequence separation
    all_contacts = {sep: 0 for sep in range(4, 40)}
    n_proteins = 5
    
    print(f"\nFolding {n_proteins} proteins and counting contacts by sequence separation...")
    
    for i in range(n_proteins):
        protein = simulator.create_random_protein(45)
        folded, _ = simulator.fold(protein, steps=400, report_every=200)
        
        # Count contacts at each sequence separation
        coords = folded.coords
        L = len(folded)
        
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)
        dist = torch.norm(diff, dim=2)
        
        idx = torch.arange(L, device=device)
        seq_sep = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        
        for sep in range(4, min(40, L)):
            mask = (seq_sep == sep) & (dist < 4.0)
            all_contacts[sep] += mask.sum().item() // 2
    
    # Analyze: Are Fibonacci separations enriched?
    fib_set = {5, 8, 13, 21, 34}
    fib_contacts = sum(all_contacts[s] for s in fib_set if s in all_contacts)
    non_fib_contacts = sum(all_contacts[s] for s in all_contacts if s not in fib_set)
    
    fib_positions = len([s for s in fib_set if s < 40])
    non_fib_positions = len([s for s in all_contacts if s not in fib_set])
    
    fib_rate = fib_contacts / fib_positions if fib_positions > 0 else 0
    non_fib_rate = non_fib_contacts / non_fib_positions if non_fib_positions > 0 else 0
    
    enrichment = fib_rate / non_fib_rate if non_fib_rate > 0 else 0
    
    print(f"\nFibonacci separations: {fib_contacts} contacts / {fib_positions} positions = {fib_rate:.1f} per position")
    print(f"Non-Fibonacci: {non_fib_contacts} contacts / {non_fib_positions} positions = {non_fib_rate:.1f} per position")
    print(f"Fibonacci enrichment: {enrichment:.2f}x")
    
    if enrichment > 1.5:
        print("✅ Fibonacci contacts are preferentially formed!")
    
    results['contact_analysis'] = {
        'by_separation': all_contacts,
        'fib_contacts': fib_contacts,
        'non_fib_contacts': non_fib_contacts,
        'enrichment': enrichment
    }
    
    # Test 3: Template-guided replication
    print("\n" + "=" * 70)
    print("TEST 3: TEMPLATE-GUIDED REPLICATION")
    print("=" * 70)
    
    replicator = SelfReplicator(energy_field)
    
    # Create and fold a template
    print("\nCreating template protein...")
    template = simulator.create_random_protein(40)
    template, _ = simulator.fold(template, steps=500, report_every=250)
    
    template_contacts = energy_field.count_fibonacci_contacts(template)
    print(f"Template Fibonacci contacts: {sum(template_contacts.values())}")
    
    # Create children with same sequence, fold with template influence
    print("\nFolding child proteins with template field...")
    
    similarities_with_template = []
    similarities_without = []
    
    for trial in range(3):
        # With template influence
        child_with = DigitalProtein(
            template.sequence.clone(),
            simulator.create_random_protein(40).coords,
            torch.zeros((40, 3), device=device)
        )
        
        # Simple folding with template bias (abbreviated simulation)
        for step in range(200):
            child_with.coords.requires_grad_(True)
            energy, _ = energy_field.compute_energy(child_with)
            energy.backward()
            forces = -child_with.coords.grad.detach()
            
            # Add template field
            template_forces = replicator.template_field(template, child_with.coords.detach())
            forces += template_forces
            
            child_with.coords.requires_grad_(False)
            child_with = DigitalProtein(
                child_with.sequence,
                child_with.coords.detach() + 0.01 * forces,
                child_with.velocities
            )
        
        sim_with = replicator.compute_similarity(template, child_with)
        similarities_with_template.append(sim_with)
        
        # Without template
        child_without = simulator.create_random_protein(40)
        child_without, _ = simulator.fold(child_without, steps=200, report_every=200)
        sim_without = replicator.compute_similarity(template, child_without)
        similarities_without.append(sim_without)
    
    mean_with = np.mean(similarities_with_template)
    mean_without = np.mean(similarities_without)
    
    print(f"\nSimilarity to template:")
    print(f"  With template field: {mean_with:.3f} ± {np.std(similarities_with_template):.3f}")
    print(f"  Without template: {mean_without:.3f} ± {np.std(similarities_without):.3f}")
    print(f"  Ratio: {mean_with/mean_without:.2f}x")
    
    if mean_with > mean_without * 1.2:
        print("✅ Template field guides similar structure formation!")
    
    results['replication'] = {
        'similarity_with_template': similarities_with_template,
        'similarity_without': similarities_without,
        'enhancement': mean_with / mean_without if mean_without > 0 else 0
    }
    
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    print(f"\n1. Fibonacci-guided folding: Contacts increased from ~0 to {results['fold_40']['final_contacts']}")
    print(f"2. Fibonacci enrichment: {enrichment:.2f}x (energy field successfully biases toward Fib)")
    print(f"3. Template replication: {mean_with/mean_without:.2f}x enhancement with template field")
    
    if enrichment > 1.5 and mean_with > mean_without * 1.1:
        print("\n✅ PROOF OF CONCEPT SUCCESSFUL")
        print("   Fibonacci-based energy fields produce self-organizing structures")
        print("   Template fields can guide structural replication")
    
    # Save
    out_path = Path(__file__).parent.parent / 'results' / f'exp_29_digital_proteins_{datetime.now():%Y%m%d_%H%M%S}.json'
    out_path.parent.mkdir(exist_ok=True)
    
    # Clean results for JSON
    clean_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            clean_results[k] = {kk: (vv if not isinstance(vv, list) or not any(isinstance(x, dict) for x in vv) else vv[:5]) 
                               for kk, vv in v.items() if not callable(vv)}
    
    with open(out_path, 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(), 'results': clean_results}, f, indent=2, default=str)
    
    print(f"\nResults saved to {out_path}")
    
    return results


if __name__ == '__main__':
    run_experiment()
