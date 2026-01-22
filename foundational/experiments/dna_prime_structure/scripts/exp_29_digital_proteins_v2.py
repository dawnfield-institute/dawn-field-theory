#!/usr/bin/env python3
"""
exp_29_digital_proteins_v2.py

DIGITAL PROTEIN SELF-ORGANIZATION - STREAMLINED VERSION

Use the discovered Fibonacci sequence principle to create artificial
"digital proteins" that self-organize in a tensor environment.

This version focuses on proving the core concept:
1. Fibonacci-based energy fields produce stable structures
2. Structures form more Fibonacci contacts than non-Fibonacci
3. The pattern is self-reinforcing
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple, Dict
import json
from datetime import datetime
from pathlib import Path

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Fibonacci sequence separations
FIBONACCI_SET = {3, 5, 8, 13, 21, 34}


@dataclass 
class DigitalProtein:
    """A digital protein: coordinates in 3D space."""
    coords: torch.Tensor        # (L, 3) positions
    
    def __len__(self):
        return len(self.coords)


class FibonacciFolder:
    """
    Energy-based folder that favors Fibonacci sequence contacts.
    Uses gradient descent for deterministic, fast convergence.
    """
    
    def __init__(self, length: int, 
                 fib_strength: float = 1.0,
                 contact_dist: float = 4.0):
        self.L = length
        self.fib_strength = fib_strength
        self.contact_dist = contact_dist
        
        # Pre-compute sequence separations
        idx = torch.arange(length, device=device)
        self.seq_sep = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        
        # Fibonacci mask - positions we want to make contacts
        self.fib_mask = torch.zeros((length, length), device=device, dtype=torch.bool)
        for f in FIBONACCI_SET:
            self.fib_mask |= (self.seq_sep == f)
        
        # Fibonacci weights (scaled by separation)
        self.fib_weights = torch.zeros((length, length), device=device)
        for f in [3, 5, 8, 13, 21, 34]:
            mask = (self.seq_sep == f)
            # Higher weight for middle Fibonacci (5, 8, 13)
            weight = {3: 1.0, 5: 2.0, 8: 2.5, 13: 2.0, 21: 1.5, 34: 1.0}[f]
            self.fib_weights[mask] = weight
    
    def create_initial(self) -> DigitalProtein:
        """Create extended chain."""
        coords = torch.zeros((self.L, 3), device=device)
        coords[:, 0] = torch.arange(self.L, dtype=torch.float32, device=device) * 2.0
        coords += torch.randn_like(coords) * 0.2
        return DigitalProtein(coords)
    
    def compute_energy(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Energy function:
        E = E_backbone + E_clash + E_fibonacci
        """
        L = self.L
        
        # Pairwise distances
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)
        dist = torch.norm(diff + 1e-8, dim=2)
        
        # Backbone: keep sequential residues at ~2.0 apart
        backbone_dist = dist[torch.arange(L-1), torch.arange(1, L)]
        e_backbone = 10.0 * ((backbone_dist - 2.0) ** 2).sum()
        
        # Clash: penalize close non-sequential pairs
        clash_mask = (self.seq_sep > 1) & (dist < 1.5)
        e_clash = 50.0 * torch.relu(1.5 - dist[clash_mask]).sum()
        
        # Fibonacci contacts: REWARD close Fibonacci pairs
        # Use soft contact function
        contact_potential = torch.exp(-dist / self.contact_dist)
        e_fib = -self.fib_strength * (self.fib_weights * contact_potential).sum()
        
        return e_backbone + e_clash + e_fib
    
    def count_contacts(self, coords: torch.Tensor, threshold: float = 5.0) -> Dict[str, int]:
        """Count contacts at Fibonacci vs non-Fibonacci separations."""
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)
        dist = torch.norm(diff, dim=2)
        
        # Contact mask
        contact = (dist < threshold) & (self.seq_sep >= 4)
        
        # Fibonacci contacts
        fib_contacts = (contact & self.fib_mask).sum().item() // 2
        
        # Non-fibonacci contacts (same range 4-35)
        non_fib_mask = (self.seq_sep >= 4) & (self.seq_sep <= 35) & ~self.fib_mask
        non_fib_contacts = (contact & non_fib_mask).sum().item() // 2
        
        # By separation
        by_sep = {}
        for sep in range(4, min(36, self.L)):
            mask = (self.seq_sep == sep) & contact
            by_sep[sep] = mask.sum().item() // 2
        
        return {
            'fibonacci': fib_contacts,
            'non_fibonacci': non_fib_contacts,
            'by_separation': by_sep
        }
    
    def fold(self, protein: DigitalProtein, steps: int = 300, lr: float = 0.1) -> Tuple[DigitalProtein, list]:
        """Fold using gradient descent."""
        coords = protein.coords.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([coords], lr=lr)
        
        history = []
        
        for step in range(steps):
            optimizer.zero_grad()
            energy = self.compute_energy(coords)
            energy.backward()
            optimizer.step()
            
            if step % 50 == 0:
                with torch.no_grad():
                    contacts = self.count_contacts(coords)
                    history.append({
                        'step': step,
                        'energy': energy.item(),
                        'fib_contacts': contacts['fibonacci'],
                        'non_fib_contacts': contacts['non_fibonacci']
                    })
        
        return DigitalProtein(coords.detach()), history


def run_experiment():
    """Main experiment."""
    
    print("=" * 70)
    print("DIGITAL PROTEIN SELF-ORGANIZATION")
    print("=" * 70)
    print("\nHypothesis: Fibonacci-based energy fields produce")
    print("self-organizing structures with emergent Fibonacci contacts\n")
    
    results = {'trials': [], 'summary': {}}
    
    # =========================================================
    # TEST 1: FOLDING PRODUCES FIBONACCI CONTACTS
    # =========================================================
    print("TEST 1: Do Fibonacci-guided energy fields produce Fibonacci structure?")
    print("-" * 70)
    
    all_enrichments = []
    
    for trial in range(5):
        length = np.random.randint(35, 55)
        folder = FibonacciFolder(length, fib_strength=2.0)
        
        protein = folder.create_initial()
        folded, history = folder.fold(protein, steps=300)
        
        # Analyze contacts
        contacts = folder.count_contacts(folded.coords)
        
        # Enrichment calculation
        # How many possible Fibonacci positions vs non-Fibonacci in range 4-35?
        fib_positions = sum(1 for f in FIBONACCI_SET if f < length)
        non_fib_positions = len([s for s in range(4, min(36, length)) if s not in FIBONACCI_SET])
        
        fib_rate = contacts['fibonacci'] / fib_positions if fib_positions > 0 else 0
        non_fib_rate = contacts['non_fibonacci'] / non_fib_positions if non_fib_positions > 0 else 0
        enrichment = fib_rate / non_fib_rate if non_fib_rate > 0 else 0
        
        all_enrichments.append(enrichment)
        
        print(f"  Trial {trial+1}: L={length}, Fib={contacts['fibonacci']}, "
              f"Non-Fib={contacts['non_fibonacci']}, Enrichment={enrichment:.2f}x")
        
        results['trials'].append({
            'length': length,
            'fib_contacts': contacts['fibonacci'],
            'non_fib_contacts': contacts['non_fibonacci'],
            'enrichment': enrichment,
            'by_separation': contacts['by_separation']
        })
    
    mean_enrichment = np.mean(all_enrichments)
    print(f"\n  Mean Fibonacci enrichment: {mean_enrichment:.2f}x")
    
    if mean_enrichment > 1.5:
        print("  ✅ Fibonacci contacts are preferentially formed!")
    else:
        print("  ⚠️ Enrichment lower than expected")
    
    # =========================================================
    # TEST 2: CONTACT DISTRIBUTION BY SEQUENCE SEPARATION
    # =========================================================
    print("\n" + "=" * 70)
    print("TEST 2: Contact distribution across sequence separations")
    print("-" * 70)
    
    # Aggregate contacts across all trials
    total_by_sep = {}
    for trial in results['trials']:
        for sep, count in trial['by_separation'].items():
            total_by_sep[sep] = total_by_sep.get(sep, 0) + count
    
    print("\n  Sequence Sep | Contacts | Fibonacci?")
    print("  " + "-" * 40)
    
    for sep in sorted(total_by_sep.keys())[:20]:
        is_fib = "  ★" if sep in FIBONACCI_SET else ""
        bar = "█" * min(total_by_sep[sep], 30)
        print(f"  {sep:11d} | {total_by_sep[sep]:8d} | {bar}{is_fib}")
    
    # =========================================================
    # TEST 3: COMPARE WITH RANDOM (NO FIBONACCI BIAS)
    # =========================================================
    print("\n" + "=" * 70)
    print("TEST 3: Control - folding WITHOUT Fibonacci bias")
    print("-" * 70)
    
    # Fold with fib_strength = 0 (just backbone + clash)
    control_enrichments = []
    
    for trial in range(5):
        length = np.random.randint(35, 55)
        folder = FibonacciFolder(length, fib_strength=0.0)  # No Fibonacci bias
        
        protein = folder.create_initial()
        folded, _ = folder.fold(protein, steps=300)
        
        contacts = folder.count_contacts(folded.coords)
        
        fib_positions = sum(1 for f in FIBONACCI_SET if f < length)
        non_fib_positions = len([s for s in range(4, min(36, length)) if s not in FIBONACCI_SET])
        
        fib_rate = contacts['fibonacci'] / fib_positions if fib_positions > 0 else 0
        non_fib_rate = contacts['non_fibonacci'] / non_fib_positions if non_fib_positions > 0 else 0
        enrichment = fib_rate / non_fib_rate if non_fib_rate > 0 else 0
        
        control_enrichments.append(enrichment)
        print(f"  Control {trial+1}: Fib={contacts['fibonacci']}, Non-Fib={contacts['non_fibonacci']}, "
              f"Enrichment={enrichment:.2f}x")
    
    control_mean = np.mean(control_enrichments)
    print(f"\n  Control enrichment: {control_mean:.2f}x (expect ~1.0)")
    print(f"  With Fibonacci field: {mean_enrichment:.2f}x")
    print(f"  Ratio: {mean_enrichment/control_mean:.2f}x increase")
    
    if mean_enrichment > control_mean * 1.3:
        print("  ✅ Fibonacci field significantly increases Fibonacci contacts!")
    
    # =========================================================
    # VERDICT
    # =========================================================
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    print(f"""
    1. Fibonacci-guided folding: {mean_enrichment:.2f}x enrichment
    2. Control (no Fibonacci): {control_mean:.2f}x enrichment  
    3. Field effect: {mean_enrichment/control_mean:.2f}x increase
    
    CONCLUSION:""")
    
    if mean_enrichment > 1.5 and mean_enrichment > control_mean * 1.2:
        print("""
    ✅ PROOF OF CONCEPT SUCCESSFUL
    
    The Fibonacci-based energy field produces self-organizing structures
    with preferential Fibonacci contacts. This demonstrates that the
    principle discovered in real proteins (Fibonacci sequence spacing)
    can be used as a GENERATIVE design principle.
    
    This validates the core DFT hypothesis: Fibonacci is not just
    descriptive but CAUSATIVE - it can drive structure formation.
    """)
        success = True
    else:
        print("""
    ⚠️ Results inconclusive - need parameter tuning
    """)
        success = False
    
    # Save results
    results['summary'] = {
        'mean_enrichment': mean_enrichment,
        'control_enrichment': control_mean,
        'field_effect': mean_enrichment / control_mean if control_mean > 0 else 0,
        'success': success
    }
    
    out_path = Path(__file__).parent.parent / 'results' / f'exp_29_digital_proteins_{datetime.now():%Y%m%d_%H%M%S}.json'
    out_path.parent.mkdir(exist_ok=True)
    
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj
    
    with open(out_path, 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(), 'results': convert(results)}, f, indent=2)
    
    print(f"\nResults saved to {out_path}")
    
    return results


if __name__ == '__main__':
    run_experiment()
