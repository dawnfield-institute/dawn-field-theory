
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import entropy as kl_divergence, chi2_contingency, norm
from datetime import datetime
import os
import json

# Symbolic collapse simulation
def symbolic_collapse_trial(p=0.7):
    return 'A' if np.random.rand() < p else 'B'

def run_collapse_trials(p=0.7, N=1000, seed=42):
    np.random.seed(seed)
    results = [symbolic_collapse_trial(p) for _ in range(N)]
    return Counter(results)

def analyze_results(results, p_expected):
    total = sum(results.values())
    observed_p = np.array([results.get('A', 0)/total, results.get('B', 0)/total])
    expected_p = np.array([p_expected, 1 - p_expected])
    abs_error = np.abs(observed_p - expected_p)
    rms_error = np.sqrt(np.mean(abs_error**2))
    kl = kl_divergence(expected_p, observed_p)

    # Chi-squared test
    observed_counts = np.array([results.get('A', 0), results.get('B', 0)])
    expected_counts = expected_p * total
    chi2_stat, chi2_pval = chi2_contingency([observed_counts, expected_counts])[:2]

    return observed_p, abs_error, rms_error, kl, chi2_stat, chi2_pval

def compute_confidence_interval(p_hat, N, alpha=0.05):
    # Normal approximation for binomial confidence interval
    z = norm.ppf(1 - alpha/2)
    ci = z * np.sqrt(p_hat * (1 - p_hat) / N)
    return p_hat - ci, p_hat + ci

def plot_results(observed_p, expected_p, ci, save_path, p_label):
    labels = ['A', 'B']
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width/2, expected_p, width, label='Expected')
    plt.bar(x + width/2, observed_p, width, label='Observed', yerr=[[obs-ci[0] for obs, ci in zip(observed_p, ci)], [ci[1]-obs for obs, ci in zip(observed_p, ci)]], capsize=5)
    plt.xticks(x, labels)
    plt.ylabel('Probability')
    plt.title(f'Born Rule Reproduction: Expected vs Observed (p={p_label})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'born_rule_comparison_{p_label}.png'))
    plt.close()

def plot_entropy(entropy_list, save_path, p_label):
    plt.figure()
    plt.plot(entropy_list)
    plt.xlabel('Trial')
    plt.ylabel('Entropy (bits)')
    plt.title(f'Entropy over Trials (p={p_label})')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'entropy_over_trials_{p_label}.png'))
    plt.close()

if __name__ == "__main__":
    test_values = [0.5, 0.7, 0.8]
    N = 10000
    n_seeds = 10
    seeds = [42 + i for i in range(n_seeds)]
    alpha = 0.05
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    summary = {}

    for p in test_values:
        metrics = {
            "Observed Probabilities": [],
            "Absolute Error": [],
            "RMS Error": [],
            "KL Divergence": [],
            "Chi2 Statistic": [],
            "Chi2 p-value": [],
            "Entropy Over Trials": []
        }
        for seed in seeds:
            results = run_collapse_trials(p=p, N=N, seed=seed)
            observed_p, abs_error, rms_error, kl, chi2_stat, chi2_pval = analyze_results(results, p)
            metrics["Observed Probabilities"].append(observed_p)
            metrics["Absolute Error"].append(abs_error)
            metrics["RMS Error"].append(rms_error)
            metrics["KL Divergence"].append(kl)
            metrics["Chi2 Statistic"].append(chi2_stat)
            metrics["Chi2 p-value"].append(chi2_pval)

            # Track entropy over trials
            trial_results = []
            np.random.seed(seed)
            for i in range(N):
                trial_results.append(symbolic_collapse_trial(p))
            entropy_list = []
            for j in range(1, N+1):
                counts = Counter(trial_results[:j])
                probs = np.array([counts.get('A', 0)/j, counts.get('B', 0)/j])
                entropy_list.append(kl_divergence(probs, base=2))
            metrics["Entropy Over Trials"].append(entropy_list)

        # Aggregate metrics
        obs_probs = np.array(metrics["Observed Probabilities"])
        abs_errs = np.array(metrics["Absolute Error"])
        rms_errs = np.array(metrics["RMS Error"])
        kl_divs = np.array(metrics["KL Divergence"])
        chi2_stats = np.array(metrics["Chi2 Statistic"])
        chi2_pvals = np.array(metrics["Chi2 p-value"])

        mean_obs = obs_probs.mean(axis=0)
        std_obs = obs_probs.std(axis=0)
        ci_obs = [compute_confidence_interval(mean_obs[i], N, alpha) for i in range(2)]

        mean_abs_err = abs_errs.mean(axis=0)
        std_abs_err = abs_errs.std(axis=0)
        mean_rms_err = rms_errs.mean()
        std_rms_err = rms_errs.std()
        mean_kl = kl_divs.mean()
        std_kl = kl_divs.std()
        mean_chi2 = chi2_stats.mean()
        std_chi2 = chi2_stats.std()
        mean_chi2_pval = chi2_pvals.mean()
        std_chi2_pval = chi2_pvals.std()

        summary[str(p)] = {
            "Mean Observed Probabilities": mean_obs.tolist(),
            "Std Observed Probabilities": std_obs.tolist(),
            "Confidence Intervals": [[float(ci_obs[i][0]), float(ci_obs[i][1])] for i in range(2)],
            "Mean Absolute Error": mean_abs_err.tolist(),
            "Std Absolute Error": std_abs_err.tolist(),
            "Mean RMS Error": float(mean_rms_err),
            "Std RMS Error": float(std_rms_err),
            "Mean KL Divergence": float(mean_kl),
            "Std KL Divergence": float(std_kl),
            "Mean Chi2 Statistic": float(mean_chi2),
            "Std Chi2 Statistic": float(std_chi2),
            "Mean Chi2 p-value": float(mean_chi2_pval),
            "Std Chi2 p-value": float(std_chi2_pval)
        }

        # Plot mean observed vs expected with confidence intervals
        plot_results(mean_obs, [p, 1-p], ci_obs, results_dir, p_label=str(p))

        # Plot entropy for first seed
        plot_entropy(metrics["Entropy Over Trials"][0], results_dir, p_label=str(p))

    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"Results saved in: {results_dir}")
