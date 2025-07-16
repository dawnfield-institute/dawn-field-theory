import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
from sklearn.metrics import mean_squared_error, r2_score

# Enhanced symbolic collapse simulation with pressure modulation

def init_field(p=0.5, size=1000):
    return np.random.choice(['A', 'B'], size=size, p=[p, 1-p])

def symbolic_collapse_step_prob(field, pressure=0.1):
    """Gradual symbolic alignment with dominant symbol, controlled by pressure (0-1)."""
    symbols, counts = np.unique(field, return_counts=True)
    dominant = symbols[np.argmax(counts)]
    return np.array([
        dominant if np.random.rand() < pressure else sym
        for sym in field
    ])

def compute_entropy(field):
    _, counts = np.unique(field, return_counts=True)
    probs = counts / np.sum(counts)
    return -np.sum(probs * np.log2(probs))

def compute_coherence(field):
    _, counts = np.unique(field, return_counts=True)
    probs = counts / np.sum(counts)
    return np.abs(probs[0] - probs[1]) if len(probs) == 2 else 1.0

# Symbolic stability: max symbol probability (how stable is the dominant symbol?)
def compute_stability(field):
    _, counts = np.unique(field, return_counts=True)
    probs = counts / np.sum(counts)
    max_prob = np.max(probs)
    return max_prob  # strength of the most stable symbol

def simulate_soft_decoherence(p=0.5, steps=50, size=1000, pressure_curve=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    field = init_field(p, size)
    entropy_trace = []
    coherence_trace = []
    stability_trace = []
    if pressure_curve is None:
        pressure_curve = np.linspace(0.05, 1.0, steps)  # Linear increase

    for step in range(steps):
        entropy_trace.append(compute_entropy(field))
        coherence_trace.append(compute_coherence(field))
        stability_trace.append(compute_stability(field))
        field = symbolic_collapse_step_prob(field, pressure=pressure_curve[step])

    return entropy_trace, coherence_trace, stability_trace, list(pressure_curve)

def quantum_decay(t, gamma=0.1):
    return np.exp(-gamma * t)

def run_and_plot_soft_decoherence(seed=42):

    # --- Main (reinforced) run ---
    entropy_trace, coherence_trace, stability_trace, pressures = simulate_soft_decoherence(p=0.5, seed=seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/decoherence_soft_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    t = np.arange(len(entropy_trace))
    quantum = quantum_decay(t)
    mse = mean_squared_error(quantum, coherence_trace)
    r2 = r2_score(quantum, coherence_trace)
    decoherence_threshold_step = next((i for i, c in enumerate(coherence_trace) if c < 0.1), None)

    # --- Control run (no reinforcement: constant low pressure) ---
    control_pressure = 0.05
    control_entropy, control_coherence, control_stability, _ = simulate_soft_decoherence(p=0.5, seed=seed, pressure_curve=[control_pressure]*len(t))
    control_mse = mean_squared_error(quantum, control_coherence)
    control_r2 = r2_score(quantum, control_coherence)

    # --- Statistical analysis: run multiple seeds ---
    n_seeds = 10
    all_coherences = []
    all_entropies = []
    all_stabilities = []
    for s in range(n_seeds):
        e, c, stab, _ = simulate_soft_decoherence(p=0.5, seed=seed+s)
        all_entropies.append(e)
        all_coherences.append(c)
        all_stabilities.append(stab)
    mean_coherence = np.mean(all_coherences, axis=0)
    std_coherence = np.std(all_coherences, axis=0)
    mean_entropy = np.mean(all_entropies, axis=0)
    std_entropy = np.std(all_entropies, axis=0)
    mean_stability = np.mean(all_stabilities, axis=0)
    std_stability = np.std(all_stabilities, axis=0)

    # --- Visualization: spatial field snapshots ---
    # Take snapshots at start, mid, end
    field_snapshots = []
    snapshot_steps = [0, len(t)//2, len(t)-1]
    field = init_field(p=0.5, size=1000)
    pressure_curve = np.linspace(0.05, 1.0, len(t))
    for step in range(len(t)):
        if step in snapshot_steps:
            field_snapshots.append(field.copy())
        field = symbolic_collapse_step_prob(field, pressure=pressure_curve[step])
    for idx, snap in zip(snapshot_steps, field_snapshots):
        plt.figure(figsize=(8,1))
        plt.imshow([snap == 'A'], aspect='auto', cmap='Greys', interpolation='nearest')
        plt.title(f"Field snapshot at step {idx}")
        plt.yticks([])
        plt.xlabel('Field Index')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'field_snapshot_step{idx}.png'))
        plt.close()

    # --- Plotting ---
    # Coherence comparison
    plt.figure()
    plt.plot(t, coherence_trace, label='Symbolic Coherence (main)')
    plt.plot(t, control_coherence, label='Control (no reinforcement)', linestyle=':')
    plt.plot(t, quantum, '--', label='Quantum Coherence (exp)')
    plt.fill_between(t, mean_coherence-std_coherence, mean_coherence+std_coherence, color='blue', alpha=0.2, label='Symbolic Coherence ±1σ')
    plt.xlabel('Collapse Step')
    plt.ylabel('Coherence')
    plt.title('Soft Symbolic vs Quantum Coherence Decay')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'coherence_decay_soft.png'))
    plt.close()

    # Entropy trace
    plt.figure()
    plt.plot(t, entropy_trace, label='Symbolic Entropy (main)')
    plt.plot(t, control_entropy, label='Control (no reinforcement)', linestyle=':')
    plt.fill_between(t, mean_entropy-std_entropy, mean_entropy+std_entropy, color='orange', alpha=0.2, label='Symbolic Entropy ±1σ')
    plt.xlabel('Collapse Step')
    plt.ylabel('Entropy (bits)')
    plt.title('Entropy Over Time (Soft Collapse)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'entropy_trace_soft.png'))
    plt.close()

    # Stability trace
    plt.figure()
    plt.plot(t, stability_trace, label='Symbolic Stability (main)', color='green')
    plt.plot(t, control_stability, label='Control (no reinforcement)', color='red', linestyle=':')
    plt.fill_between(t, mean_stability-std_stability, mean_stability+std_stability, color='green', alpha=0.2, label='Stability ±1σ')
    plt.xlabel('Collapse Step')
    plt.ylabel('Stability (Max Symbol Prob)')
    plt.title('Stability Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'stability_trace_soft.png'))
    plt.close()

    # --- Stepwise mapping: stability_vs_quantum.json ---
    stability_vs_quantum = []
    for i in range(len(t)):
        stability_vs_quantum.append({
            "step": int(t[i]),
            "stability": float(stability_trace[i]),
            "quantum_prob": float(quantum[i]),
            "coherence": float(coherence_trace[i])
        })
    with open(os.path.join(results_dir, 'stability_vs_quantum.json'), 'w') as f:
        json.dump(stability_vs_quantum, f, indent=4)


    # Save metrics
    with open(os.path.join(results_dir, 'decoherence_soft_metrics.json'), 'w') as f:
        json.dump({
            "entropy": entropy_trace,
            "coherence": coherence_trace,
            "stability": stability_trace,
            "final_stability": stability_trace[-1],
            "max_stability_rate": float(np.max(np.diff(stability_trace))),
            "pressures": pressures,
            "mse_vs_quantum": mse,
            "r2_vs_quantum": r2,
            "step_below_coherence_0.1": decoherence_threshold_step,
            "control_entropy": control_entropy,
            "control_coherence": control_coherence,
            "control_stability": control_stability,
            "control_mse_vs_quantum": control_mse,
            "control_r2_vs_quantum": control_r2,
            "mean_coherence": mean_coherence.tolist(),
            "std_coherence": std_coherence.tolist(),
            "mean_entropy": mean_entropy.tolist(),
            "std_entropy": std_entropy.tolist(),
            "mean_stability": mean_stability.tolist(),
            "std_stability": std_stability.tolist()
        }, f, indent=4)

    print(f"Decoherence results saved to: {results_dir}\n"
          f"MSE vs Quantum: {mse:.5f}, R2: {r2:.3f}, First <0.1 coherence at step: {decoherence_threshold_step}\n"
          f"Control MSE: {control_mse:.5f}, Control R2: {control_r2:.3f}")

if __name__ == "__main__":
    run_and_plot_soft_decoherence()
