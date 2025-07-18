# CIP-compliant symbolic entropy collapse experiment
# (Renamed from main.py)
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import argparse

# --- Load and process Newick tree into extinction annotation format ---
def extract_leaf_labels_from_newick(newick_str):
    labels = []
    label = ''
    for char in newick_str:
        if char in '(),:;':
            if label:
                labels.append(label.strip())
                label = ''
        else:
            label += char
    if label:
        labels.append(label.strip())
    return sorted(set(filter(None, labels)))

def convert_newick_to_extinction_template(newick_path):
    with open(newick_path) as f:
        newick_str = f.read()
    leaf_labels = extract_leaf_labels_from_newick(newick_str)
    return set(leaf_labels)

# --- Simulate real tree with extinction-aware entropy ---
def create_mock_tree(depth, breadth, extinction_names, current_depth=0):
    if depth == 0:
        label = f"Species_{np.random.randint(10000)}"
        extinct = label in extinction_names
        entropy = np.log1p(current_depth + 1) + (0.3 if extinct else 0.0) + np.random.normal(0, 0.05)
        return {
            'label': label,
            'children': [],
            'entropy': entropy,
            'extinct': extinct
        }
    children = [create_mock_tree(depth - 1, breadth, extinction_names, current_depth + 1) for _ in range(breadth)]
    label = f"Node_{current_depth}_{np.random.randint(1000)}"
    entropy = np.log1p(current_depth + 1) + np.random.normal(0, 0.05)
    return {
        'label': label,
        'children': children,
        'entropy': entropy,
        'extinct': False
    }

# --- Simulate random tree ---
def create_simulated_tree(depth, breadth, current_depth=0):
    if depth == 0:
        entropy = np.log1p(current_depth + 1) + np.random.normal(0, 0.05)
        return {
            'label': f"SimSpecies_{np.random.randint(10000)}",
            'children': [],
            'entropy': entropy,
            'extinct': False
        }
    children = [create_simulated_tree(depth - 1, breadth, current_depth + 1) for _ in range(breadth)]
    label = f"SimNode_{current_depth}_{np.random.randint(1000)}"
    entropy = np.log1p(current_depth + 1) + np.random.normal(0, 0.05)
    return {
        'label': label,
        'children': children,
        'entropy': entropy,
        'extinct': False
    }

# --- Trace entropy wave ---
def trace_entropy(node, depth=0, entropy_trace=None):
    if entropy_trace is None:
        entropy_trace = {}
    if depth not in entropy_trace:
        entropy_trace[depth] = []
    entropy_trace[depth].append(node['entropy'])
    for child in node.get('children', []):
        trace_entropy(child, depth + 1, entropy_trace)
    return entropy_trace

def compute_entropy_metrics(entropy_wave1, entropy_wave2):
    import scipy.stats
    import torch
    arr1 = np.array(entropy_wave1)
    arr2 = np.array(entropy_wave2)
    # KL divergence (softmax for valid probabilities)
    p1 = np.exp(arr1) / np.sum(np.exp(arr1)) if np.sum(np.exp(arr1)) > 0 else np.ones_like(arr1) / len(arr1)
    p2 = np.exp(arr2) / np.sum(np.exp(arr2)) if np.sum(np.exp(arr2)) > 0 else np.ones_like(arr2) / len(arr2)
    kl_div = scipy.stats.entropy(p1, p2) if len(p1) == len(p2) and len(p1) > 1 else 0.0
    js_div = scipy.spatial.distance.jensenshannon(p1, p2) if len(p1) == len(p2) and len(p1) > 1 else 0.0
    wd = scipy.stats.wasserstein_distance(arr1, arr2) if len(arr1) > 1 and len(arr2) > 1 else 0.0
    t1 = torch.tensor(arr1)
    t2 = torch.tensor(arr2)
    if torch.var(t1) == 0 or torch.var(t2) == 0 or len(t1) < 2 or len(t2) < 2:
        qwcs = 0.5
    else:
        try:
            qwcs = 1 - torch.abs(torch.corrcoef(torch.stack([t1, t2]))[0, 1]).item()
        except Exception:
            qwcs = 0.5
    return {
        'KL-Divergence': kl_div,
        'Jensen-Shannon': js_div,
        'Wasserstein': wd,
        'QWCS': qwcs
    }

# --- Utility: Save diagnostics ---
def save_metrics(metrics_by_depth, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_by_depth, f, indent=2)

def save_entropy_traces(traces, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump([{str(k): list(map(float, v)) for k, v in trace.items()} for trace in traces], f, indent=2)

# --- Main experiment logic ---
def run_experiment(
    newick_path,
    extinction_csv,
    depth=3,
    breadth=3,
    extinction_rate=1.0,
    extinction_model="default",
    n_replicates=10,
    seed=42,
    out_dir="output",
    save_plots=True,
    log_provenance=None
):
    """
    Run a batch experiment for symbolic collapse entropy analysis.
    Parameters and results are logged for provenance.
    Supports biological realism via extinction models and rates.
    Performs statistical testing (t-test) for each depth.
    """
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    # Process input data
    try:
        df = pd.read_csv(extinction_csv, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(extinction_csv, encoding='ISO-8859-1')
    extinction_names = set(df['accepted_name'].dropna().astype(str))
    leaf_taxa_from_tree = convert_newick_to_extinction_template(newick_path)
    # Biological realism: extinction rate/model
    if extinction_model == "default":
        filtered_extinctions = extinction_names.intersection(leaf_taxa_from_tree)
    elif extinction_model == "probabilistic":
        leaf_list = list(leaf_taxa_from_tree)
        n_extinct = int(len(leaf_list) * extinction_rate)
        filtered_extinctions = set(np.random.choice(leaf_list, n_extinct, replace=False))
    else:
        filtered_extinctions = extinction_names.intersection(leaf_taxa_from_tree)

    all_metrics = []
    all_real_traces = []
    all_sim_traces = []
    for rep in range(n_replicates):
        np.random.seed(seed + rep)
        real_tree = create_mock_tree(depth=depth, breadth=breadth, extinction_names=filtered_extinctions)
        sim_tree = create_simulated_tree(depth=depth, breadth=breadth)
        real_trace = trace_entropy(real_tree)
        sim_trace = trace_entropy(sim_tree)
        metrics_by_depth = {}
        for d in sorted(real_trace.keys()):
            metrics = compute_entropy_metrics(real_trace[d], sim_trace.get(d, [0]))
            metrics_by_depth[d] = metrics
        all_metrics.append(metrics_by_depth)
        all_real_traces.append(real_trace)
        all_sim_traces.append(sim_trace)

    # Save diagnostics
    save_metrics(all_metrics, os.path.join(out_dir, "metrics.json"))
    save_entropy_traces(all_real_traces, os.path.join(out_dir, "real_entropy_traces.json"))
    save_entropy_traces(all_sim_traces, os.path.join(out_dir, "sim_entropy_traces.json"))

    # Aggregate and plot
    depths = sorted(set().union(*[trace.keys() for trace in all_real_traces]))
    means_real = []
    stds_real = []
    means_sim = []
    stds_sim = []
    for d in depths:
        vals_real = [np.mean(trace[d]) for trace in all_real_traces if d in trace]
        vals_sim = [np.mean(trace[d]) for trace in all_sim_traces if d in trace]
        means_real.append(np.mean(vals_real))
        stds_real.append(np.std(vals_real))
        means_sim.append(np.mean(vals_sim))
        stds_sim.append(np.std(vals_sim))

    plt.figure(figsize=(10, 6))
    plt.errorbar(depths, means_real, yerr=stds_real, fmt='-o', capsize=5, label='Empirical (Extinction-informed)')
    plt.errorbar(depths, means_sim, yerr=stds_sim, fmt='--o', capsize=5, label='Simulated (Random Entropy)')
    plt.title(f"Entropy Waves: Empirical vs Simulated Collapse Trees (Batch)\nDepth={depth}, Breadth={breadth}, ExtRate={extinction_rate}, Model={extinction_model}")
    plt.xlabel("Tree Depth (Collapse Iteration)")
    plt.ylabel("Mean Entropy ± Std (across replicates)")
    plt.legend()
    plt.grid(True)
    if save_plots:
        plt.savefig(os.path.join(out_dir, "entropy_waves.png"), dpi=300)
    plt.close()

    # Statistical testing: t-test for mean entropy at each depth
    from scipy.stats import ttest_ind
    ttest_results = {}
    for i, d in enumerate(depths):
        vals_real = [np.mean(trace[d]) for trace in all_real_traces if d in trace]
        vals_sim = [np.mean(trace[d]) for trace in all_sim_traces if d in trace]
        t_stat, p_val = ttest_ind(vals_real, vals_sim, equal_var=False)
        ttest_results[d] = {'t_stat': t_stat, 'p_value': p_val}
    with open(os.path.join(out_dir, "ttest_results.json"), 'w', encoding='utf-8') as f:
        json.dump(ttest_results, f, indent=2)

    # Provenance logging
    from datetime import datetime
    provenance = {
        'timestamp': datetime.now().isoformat(),
        'newick_path': newick_path,
        'extinction_csv': extinction_csv,
        'depth': depth,
        'breadth': breadth,
        'extinction_rate': extinction_rate,
        'extinction_model': extinction_model,
        'n_replicates': n_replicates,
        'seed': seed,
        'out_dir': out_dir
    }
    with open(os.path.join(out_dir, "provenance.json"), 'w', encoding='utf-8') as f:
        json.dump(provenance, f, indent=2)
    if log_provenance is not None:
        log_provenance.append(provenance)

    print(f"\n[LOG] Batch experiment complete. Metrics, traces, t-tests, and plot saved to '{out_dir}'.")

# --- CLI Entrypoint ---
if __name__ == "__main__":
    from itertools import product
    from datetime import datetime
    parser = argparse.ArgumentParser(description="Symbolic Collapse Entropy Experiment (Batch + Sweep)")
    parser.add_argument('--newick', type=str, default="C:\\Users\\peter\\repos\\dawn-field-theory\\biology_tests\\evolution-symbolic-collapse\\data\\tree.nwk")
    parser.add_argument('--extinctions', type=str, default="C:\\Users\\peter\\repos\\dawn-field-theory\\biology_tests\\evolution-symbolic-collapse\\data\\extinctions.csv")
    parser.add_argument('--depths', type=str, default="3,4,5")
    parser.add_argument('--breadths', type=str, default="3,4,5")
    parser.add_argument('--extinction_rates', type=str, default="1.0,0.7,0.4")
    parser.add_argument('--extinction_model', type=str, default="default", choices=["default", "probabilistic"])
    parser.add_argument('--n_replicates', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default="output")
    parser.add_argument('--no_save_plots', action='store_true')
    args = parser.parse_args()

    depths = [int(x) for x in args.depths.split(",")]
    breadths = [int(x) for x in args.breadths.split(",")]
    extinction_rates = [float(x) for x in args.extinction_rates.split(",")]
    sweep_log = []
    sweep_dir = os.path.join(args.out_dir, f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(sweep_dir, exist_ok=True)
    for depth, breadth, extinction_rate in product(depths, breadths, extinction_rates):
        combo_dir = os.path.join(sweep_dir, f"d{depth}_b{breadth}_e{extinction_rate}")
        run_experiment(
            newick_path=args.newick,
            extinction_csv=args.extinctions,
            depth=depth,
            breadth=breadth,
            extinction_rate=extinction_rate,
            extinction_model=args.extinction_model,
            n_replicates=args.n_replicates,
            seed=args.seed,
            out_dir=combo_dir,
            save_plots=not args.no_save_plots,
            log_provenance=sweep_log
        )
    # Save sweep provenance
    with open(os.path.join(sweep_dir, "sweep_provenance.json"), 'w', encoding='utf-8') as f:
        json.dump(sweep_log, f, indent=2)
    print(f"\n[LOG] Parameter sweep complete. All results saved to '{sweep_dir}'.")
