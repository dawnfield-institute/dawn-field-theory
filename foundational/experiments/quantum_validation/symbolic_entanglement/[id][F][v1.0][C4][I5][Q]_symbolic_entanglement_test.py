import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Enhanced symbolic entanglement emulation
# Publication-ready: multi-seed, parameter sweep, control, extra plots

FIELD_SIZE = 100
TRIALS = 500
ENTANGLEMENT_PAIRS = 50
SEEDS = [42, 123, 2025, 7, 99]
ENTROPY_DECAY_LIST = [0.98, 0.95]
REINFORCEMENT_BOOST_LIST = [0.05, 0.1]
COUPLING_STRENGTH_LIST = [1.0, 0.8, 0.5]

def run_experiment(seed, entropy_decay, reinforcement_boost, coupling_strength, control=False):
    np.random.seed(seed)
    reinforcement = np.zeros(FIELD_SIZE)
    pair_correlations = []
    reinforcement_history = []
    # For Bell/CHSH: simulate four measurement settings (A, A', B, B')
    chsh_trials = []
    for trial in range(TRIALS):
        pair_a = np.random.randint(0, FIELD_SIZE // 2, ENTANGLEMENT_PAIRS)
        pair_b = FIELD_SIZE - 1 - pair_a
        response_a = np.random.choice([0, 1], size=ENTANGLEMENT_PAIRS)
        if control:
            response_b = np.random.choice([0, 1], size=ENTANGLEMENT_PAIRS)
        else:
            response_b = (response_a + np.random.choice([0, 1], size=ENTANGLEMENT_PAIRS, p=[coupling_strength, 1 - coupling_strength])) % 2
        correlation = np.mean(response_a == response_b)
        pair_correlations.append(correlation)
        for i in range(ENTANGLEMENT_PAIRS):
            if response_a[i] == response_b[i] and not control:
                reinforcement[pair_a[i]] += reinforcement_boost
                reinforcement[pair_b[i]] += reinforcement_boost
        reinforcement *= entropy_decay
        reinforcement_history.append(reinforcement.copy())
        # Bell/CHSH: simulate four settings per trial
        # A, A', B, B' are random binary settings
        A = np.random.choice([0, 1], size=ENTANGLEMENT_PAIRS)
        Ap = np.random.choice([0, 1], size=ENTANGLEMENT_PAIRS)
        B = np.random.choice([0, 1], size=ENTANGLEMENT_PAIRS)
        Bp = np.random.choice([0, 1], size=ENTANGLEMENT_PAIRS)
        # Simulate outcomes for each setting
        E_AB = np.mean((A == B))
        E_ABp = np.mean((A == Bp))
        E_ApB = np.mean((Ap == B))
        E_ApBp = np.mean((Ap == Bp))
        # CHSH value: S = |E_AB - E_ABp + E_ApB + E_ApBp|
        S = abs(E_AB - E_ABp + E_ApB + E_ApBp)
        chsh_trials.append(S)
    chsh_mean = float(np.mean(chsh_trials))
    chsh_std = float(np.std(chsh_trials))
    return pair_correlations, reinforcement, np.array(reinforcement_history), chsh_mean, chsh_std

# Run parameter sweeps and multi-seed aggregation
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("results", f"entanglement_emulation_pubready_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

all_results = []
for entropy_decay in ENTROPY_DECAY_LIST:
    for reinforcement_boost in REINFORCEMENT_BOOST_LIST:
        for coupling_strength in COUPLING_STRENGTH_LIST:
            correlations_seeds = []
            chsh_means = []
            chsh_stds = []
            for seed in SEEDS:
                pair_correlations, reinforcement, reinforcement_history, chsh_mean, chsh_std = run_experiment(
                    seed, entropy_decay, reinforcement_boost, coupling_strength, control=False)
                correlations_seeds.append(pair_correlations)
                chsh_means.append(chsh_mean)
                chsh_stds.append(chsh_std)
                # Save per-seed plots
                plt.figure()
                plt.plot(pair_correlations, label='Correlation Over Trials')
                plt.axhline(y=np.mean(pair_correlations), color='r', linestyle='--', label='Mean Correlation')
                plt.title(f"Correlation Trace (seed={seed})")
                plt.xlabel("Trial")
                plt.ylabel("Correlation")
                plt.legend()
                plt.savefig(os.path.join(output_dir, f"correlation_trace_seed{seed}_e{entropy_decay}_r{reinforcement_boost}_c{coupling_strength}.png"))
                plt.close()
                # Histogram
                plt.figure()
                plt.hist(pair_correlations, bins=20, alpha=0.7)
                plt.title(f"Correlation Histogram (seed={seed})")
                plt.xlabel("Correlation")
                plt.ylabel("Frequency")
                plt.savefig(os.path.join(output_dir, f"correlation_hist_seed{seed}_e{entropy_decay}_r{reinforcement_boost}_c{coupling_strength}.png"))
                plt.close()
                # Reinforcement field
                plt.figure()
                plt.plot(reinforcement)
                plt.title(f"Reinforcement Field (seed={seed})")
                plt.xlabel("Field Position")
                plt.ylabel("Reinforcement Level")
                plt.savefig(os.path.join(output_dir, f"reinforcement_field_seed{seed}_e{entropy_decay}_r{reinforcement_boost}_c{coupling_strength}.png"))
                plt.close()
                # Reinforcement evolution
                plt.figure()
                plt.imshow(reinforcement_history.T, aspect='auto', origin='lower', cmap='viridis')
                plt.title(f"Reinforcement Evolution (seed={seed})")
                plt.xlabel("Trial")
                plt.ylabel("Field Position")
                plt.colorbar(label="Reinforcement Level")
                plt.savefig(os.path.join(output_dir, f"reinforcement_evolution_seed{seed}_e{entropy_decay}_r{reinforcement_boost}_c{coupling_strength}.png"))
                plt.close()
            # Aggregate stats
            mean_corr = float(np.mean([np.mean(c) for c in correlations_seeds]))
            std_corr = float(np.std([np.mean(c) for c in correlations_seeds]))
            mean_chsh = float(np.mean(chsh_means))
            std_chsh = float(np.std(chsh_means))
            all_results.append({
                "FIELD_SIZE": FIELD_SIZE,
                "TRIALS": TRIALS,
                "ENTANGLEMENT_PAIRS": ENTANGLEMENT_PAIRS,
                "ENTROPY_DECAY": entropy_decay,
                "REINFORCEMENT_BOOST": reinforcement_boost,
                "COUPLING_STRENGTH": coupling_strength,
                "mean_correlation": mean_corr,
                "std_correlation": std_corr,
                "mean_CHSH": mean_chsh,
                "std_CHSH": std_chsh,
                "seeds": SEEDS
            })
            # Control experiment (no reinforcement)
            control_corrs = []
            control_chsh_means = []
            for seed in SEEDS:
                control_pair_corr, _, _, control_chsh_mean, _ = run_experiment(
                    seed, entropy_decay, reinforcement_boost, coupling_strength, control=True)
                control_corrs.append(np.mean(control_pair_corr))
                control_chsh_means.append(control_chsh_mean)
            control_mean = float(np.mean(control_corrs))
            control_std = float(np.std(control_corrs))
            control_chsh_mean = float(np.mean(control_chsh_means))
            control_chsh_std = float(np.std(control_chsh_means))
            all_results[-1]["control_mean_correlation"] = control_mean
            all_results[-1]["control_std_correlation"] = control_std
            all_results[-1]["control_mean_CHSH"] = control_chsh_mean
            all_results[-1]["control_std_CHSH"] = control_chsh_std

# Save summary
with open(os.path.join(output_dir, "summary_pubready.json"), "w") as f:
    json.dump(all_results, f, indent=2)

print(f"Publication-ready results saved to: {output_dir}")
