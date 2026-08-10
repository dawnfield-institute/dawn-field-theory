"""
Experiment 34b: Extended Evolution with Genealogy Visualization
===============================================================

Run longer evolution and visualize the genealogy tree.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from exp_34_pac_lazy_evolution import DigitalLifeEvolution, PHI, PSI, XI

def visualize_population_history(history):
    """Plot population dynamics over time."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    times = [h['time'] for h in history]
    pops = [h['population'] for h in history]
    gens = [h.get('max_generation', 0) for h in history]
    fibs = [h.get('mean_fib_contacts', 0) for h in history]
    pools = [h.get('env_pool', 0) for h in history]
    
    # Population
    ax = axes[0, 0]
    ax.plot(times, pops, 'b-', linewidth=2)
    ax.axhline(y=500, color='gray', linestyle='--', alpha=0.5, label='Initial')
    ax.set_xlabel('Time')
    ax.set_ylabel('Population')
    ax.set_title('Population Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Max Generation
    ax = axes[0, 1]
    ax.plot(times, gens, 'g-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Max Generation')
    ax.set_title('Generational Progress')
    ax.grid(True, alpha=0.3)
    
    # Fibonacci Contacts
    ax = axes[1, 0]
    ax.plot(times, fibs, 'r-', linewidth=2)
    ax.axhline(y=fibs[-1], color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean Fibonacci Contacts')
    ax.set_title('Fibonacci Organization Evolution')
    ax.grid(True, alpha=0.3)
    
    # Environment Pool
    ax = axes[1, 1]
    ax.plot(times, pools, 'purple', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Environment Pool')
    ax.set_title('Resource Dynamics')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'PAC-Lazy Digital Life Evolution\nφ={PHI:.3f}, Ξ={XI:.4f}', fontsize=14)
    plt.tight_layout()
    plt.savefig('../results/exp_34b_population_dynamics.png', dpi=150)
    plt.close()
    print("Saved: exp_34b_population_dynamics.png")


def visualize_genealogy(engine, max_lineages=8):
    """Visualize the genealogy tree structure."""
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Get organisms by generation
    gen_buckets = {}
    for oid, org in engine.genealogy.organisms.items():
        if org.generation not in gen_buckets:
            gen_buckets[org.generation] = []
        gen_buckets[org.generation].append(org)
    
    max_gen = max(gen_buckets.keys())
    
    # Assign x positions within each generation
    x_positions = {}
    y_positions = {}
    
    for gen in sorted(gen_buckets.keys()):
        orgs = gen_buckets[gen]
        n = len(orgs)
        for i, org in enumerate(orgs):
            x_positions[org.oid] = (i - n/2) * 0.3  # Spread out
            y_positions[org.oid] = max_gen - gen  # Top = early generations
    
    # Draw edges (parent -> child)
    for oid, org in engine.genealogy.organisms.items():
        if oid not in x_positions:
            continue
        for child_id in org.child_ids:
            if child_id not in x_positions:
                continue
            ax.plot(
                [x_positions[oid], x_positions[child_id]],
                [y_positions[oid], y_positions[child_id]],
                'gray', alpha=0.2, linewidth=0.5
            )
    
    # Draw nodes
    living_ids = engine.genealogy.living
    
    for gen in sorted(gen_buckets.keys()):
        orgs = gen_buckets[gen]
        
        # Split by alive/dead
        alive = [o for o in orgs if o.oid in living_ids]
        dead = [o for o in orgs if o.oid not in living_ids]
        
        if dead:
            ax.scatter(
                [x_positions[o.oid] for o in dead],
                [y_positions[o.oid] for o in dead],
                c='lightgray', s=10, alpha=0.3
            )
        
        if alive:
            colors = [o.fib_contacts for o in alive]
            ax.scatter(
                [x_positions[o.oid] for o in alive],
                [y_positions[o.oid] for o in alive],
                c=colors, cmap='viridis', s=20, alpha=0.8
            )
    
    ax.set_ylabel('Generation (0 = founders)')
    ax.set_title(f'Genealogy Tree: {len(engine.genealogy.organisms)} organisms, {max_gen+1} generations')
    ax.set_yticks(range(max_gen + 1))
    ax.set_yticklabels([str(max_gen - i) for i in range(max_gen + 1)])
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis')
    sm.set_array([o.fib_contacts for o in engine.genealogy.organisms.values()])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
    cbar.set_label('Fibonacci Contacts')
    
    plt.tight_layout()
    plt.savefig('../results/exp_34b_genealogy_tree.png', dpi=150)
    plt.close()
    print("Saved: exp_34b_genealogy_tree.png")


def visualize_lineage_detail(engine, founder_id: int):
    """Detailed visualization of a single successful lineage."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get all descendants
    descendants = engine.genealogy.get_descendants(founder_id)
    lineage_orgs = [engine.genealogy.organisms[founder_id]] + \
                   [engine.genealogy.organisms[d] for d in descendants]
    
    # Generation histogram
    ax = axes[0]
    gens = [o.generation for o in lineage_orgs]
    alive_gens = [o.generation for o in lineage_orgs if o.is_alive()]
    
    ax.hist(gens, bins=range(max(gens)+2), alpha=0.5, label='All', color='blue')
    ax.hist(alive_gens, bins=range(max(gens)+2), alpha=0.7, label='Living', color='green')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Count')
    ax.set_title(f'Lineage from Founder {founder_id}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Fibonacci contacts vs generation
    ax = axes[1]
    for org in lineage_orgs:
        color = 'green' if org.is_alive() else 'gray'
        alpha = 0.8 if org.is_alive() else 0.2
        ax.scatter(org.generation, org.fib_contacts, c=color, alpha=alpha, s=20)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fibonacci Contacts')
    ax.set_title(f'Fibonacci Evolution in Lineage {founder_id}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../results/exp_34b_lineage_{founder_id}.png', dpi=150)
    plt.close()
    print(f"Saved: exp_34b_lineage_{founder_id}.png")


def main():
    print("="*70)
    print("EXP 34b: EXTENDED EVOLUTION WITH VISUALIZATION")
    print("="*70)
    
    # Create engine with more resources
    config = {
        "initial_population": 1000,
        "protein_length": 80,
        "world_capacity": 8000,
        "initial_value": 100.0,
        "generations": 500,
        "report_interval": 50
    }
    
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    engine = DigitalLifeEvolution(
        initial_population=config["initial_population"],
        protein_length=config["protein_length"],
        world_capacity=config["world_capacity"],
        initial_value_per_org=config["initial_value"]
    )
    
    # Run extended evolution
    history = engine.run(
        generations=config["generations"],
        report_interval=config["report_interval"]
    )
    
    # Visualizations
    print("\n📊 Generating visualizations...")
    
    visualize_population_history(history)
    visualize_genealogy(engine)
    
    # Visualize top lineages
    top_lineages = engine.get_top_lineages(3)
    for lineage in top_lineages:
        visualize_lineage_detail(engine, lineage['founder_id'])
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    if history:
        final = history[-1]
        print(f"\n🌍 World State:")
        print(f"  Population: {final['population']}")
        print(f"  Total organisms ever: {final.get('total_spawned', 'N/A')}")
        print(f"  Max generation: {final.get('max_generation', 0)}")
        print(f"  Mean Fibonacci: {final.get('mean_fib_contacts', 0):.1f}")
        
        print(f"\n🏆 Most Successful Lineages:")
        for i, lin in enumerate(top_lineages, 1):
            print(f"  {i}. Founder {lin['founder_id']}: {lin['total_descendants']} descendants")
            print(f"     Max generation: {lin['generation']}, Best Fib: {lin['fib_contacts']}")
    
    print("\n✅ Complete!")


if __name__ == "__main__":
    main()
