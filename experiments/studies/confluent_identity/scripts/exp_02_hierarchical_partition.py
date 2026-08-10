"""
exp_02_hierarchical_partition.py — Confluent Identity Phase 1

PURPOSE:
    Partition the steady-state C field from exp_01 into hierarchical regions
    using multi-threshold watershed segmentation. Produces a tree of
    parent-child region relationships for identity analysis.

DESIGN:
    Level 0 (finest): watershed basins from local minima of C field
    Level k (coarser): merge adjacent regions where boundary gradient < threshold_k
    Threshold increases with level => progressively coarser regions

    The hierarchy is a tree: each level-k region is the union of level-(k-1) children.

CONSERVATION CHECK:
    Every cell belongs to exactly one region per level.
    Child regions tile their parent exactly.

Planck units throughout.
"""

import numpy as np
import json
import heapq
from datetime import datetime
from pathlib import Path
from scipy import ndimage


RESULTS_DIR = Path(__file__).parent.parent / 'results'


def load_steady_state():
    """Load exp_01 results."""
    P = np.load(RESULTS_DIR / 'exp_01_P_steady.npy')
    A = np.load(RESULTS_DIR / 'exp_01_A_steady.npy')
    stone_mask = np.load(RESULTS_DIR / 'exp_01_stone_mask.npy')
    return P, A, stone_mask


class UnionFind:
    """Disjoint-set data structure for efficient region merging."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


def watershed_from_minima(C, sigma=2.0, min_filter_size=7):
    """
    Level 0 partition: watershed basins from local minima of smoothed C field.

    Algorithm:
    1. Gaussian smooth C to reduce noise
    2. Find local minima (cells equal to their min-filtered neighborhood)
    3. Label connected minima as seeds
    4. Priority-queue flood fill from seeds (lower C = higher priority)

    Returns: labels array (0 = unassigned should not exist after flooding)
    """
    N = C.shape[0]

    # Smooth
    C_smooth = ndimage.gaussian_filter(C, sigma=sigma, mode='wrap')

    # Find local minima
    C_min = ndimage.minimum_filter(C_smooth, size=min_filter_size, mode='wrap')
    minima = C_smooth == C_min

    # Label connected minima
    seeds, n_seeds = ndimage.label(minima)

    if n_seeds == 0:
        # Fallback: single region
        return np.ones_like(C, dtype=int), 1

    # Priority-queue flood fill
    labels = seeds.copy()
    visited = labels > 0

    # Initialize queue with neighbors of seeds
    pq = []  # (C_value, row, col)
    for i in range(N):
        for j in range(N):
            if visited[i, j]:
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = (i + di) % N, (j + dj) % N
                    if not visited[ni, nj]:
                        heapq.heappush(pq, (C_smooth[ni, nj], ni, nj))

    while pq:
        val, i, j = heapq.heappop(pq)
        if visited[i, j]:
            continue
        # Assign to the label of already-visited neighbor with lowest C
        best_label = 0
        best_val = float('inf')
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = (i + di) % N, (j + dj) % N
            if visited[ni, nj] and labels[ni, nj] > 0:
                if C_smooth[ni, nj] < best_val:
                    best_val = C_smooth[ni, nj]
                    best_label = labels[ni, nj]
        if best_label > 0:
            labels[i, j] = best_label
        else:
            labels[i, j] = 1  # fallback
        visited[i, j] = True

        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = (i + di) % N, (j + dj) % N
            if not visited[ni, nj]:
                heapq.heappush(pq, (C_smooth[ni, nj], ni, nj))

    return labels, n_seeds


def compute_boundary_gradients(labels, C):
    """
    For each pair of adjacent regions, compute the mean boundary gradient.

    Returns: dict {(label_a, label_b): mean_gradient} where label_a < label_b
    """
    N = C.shape[0]
    boundary_sums = {}
    boundary_counts = {}

    for i in range(N):
        for j in range(N):
            for di, dj in [(1, 0), (0, 1)]:  # only right and down to avoid double-counting
                ni, nj = (i + di) % N, (j + dj) % N
                la, lb = labels[i, j], labels[ni, nj]
                if la != lb:
                    key = (min(la, lb), max(la, lb))
                    grad = abs(C[i, j] - C[ni, nj])
                    boundary_sums[key] = boundary_sums.get(key, 0.0) + grad
                    boundary_counts[key] = boundary_counts.get(key, 0) + 1

    return {k: boundary_sums[k] / boundary_counts[k] for k in boundary_sums}


def merge_regions(labels, C, threshold):
    """
    Merge adjacent regions whose boundary gradient is below threshold.
    Uses union-find for efficiency.

    Returns: new labels array, mapping {old_label: new_label}
    """
    unique_labels = np.unique(labels)
    n = unique_labels.max() + 1

    # Build boundary gradients
    boundaries = compute_boundary_gradients(labels, C)

    # Sort by gradient (merge lowest first)
    sorted_pairs = sorted(boundaries.items(), key=lambda x: x[1])

    uf = UnionFind(n)
    for (la, lb), grad in sorted_pairs:
        if grad < threshold:
            uf.union(la, lb)

    # Remap labels
    root_map = {}
    new_id = 1
    new_labels = np.zeros_like(labels)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            root = uf.find(labels[i, j])
            if root not in root_map:
                root_map[root] = new_id
                new_id += 1
            new_labels[i, j] = root_map[root]

    return new_labels, root_map


def build_hierarchy(labels_by_level):
    """
    Build parent-child relationships between levels.

    For each region at level k, find which region at level k+1 contains
    the majority of its cells.

    Returns: dict {(level, region_id): [(level-1, child_id), ...]}
    """
    hierarchy = {}

    for level in range(1, len(labels_by_level)):
        parent_labels = labels_by_level[level]
        child_labels = labels_by_level[level - 1]

        # For each parent region, collect child regions
        parent_ids = np.unique(parent_labels)
        for pid in parent_ids:
            parent_mask = parent_labels == pid
            children_in_parent = np.unique(child_labels[parent_mask])
            hierarchy[(level, int(pid))] = [
                (level - 1, int(cid)) for cid in children_in_parent
            ]

    return hierarchy


def run_experiment():
    """Run hierarchical partition experiment."""

    print("=" * 70)
    print("Confluent Identity — Phase 1, Experiment 02")
    print("Hierarchical Watershed Partition")
    print("=" * 70)

    # Load steady state
    P, A, stone_mask = load_steady_state()
    C = P + A
    N = C.shape[0]
    print(f"\nLoaded steady-state field: {N}x{N}")
    print(f"C range: [{C.min():.6f}, {C.max():.6f}], std={C.std():.6f}")

    # Level 0: finest watershed
    print(f"\nLevel 0: watershed from local minima...")
    labels_0, n_seeds = watershed_from_minima(C, sigma=0.5, min_filter_size=3)
    n_regions_0 = len(np.unique(labels_0))
    print(f"  Seeds: {n_seeds}, Regions: {n_regions_0}")

    # Verification: every cell assigned
    assert (labels_0 > 0).all(), "Some cells unassigned at level 0"
    print(f"  [PASS] All {N*N} cells assigned")

    # Higher levels: merge with increasing thresholds
    # Use percentiles of actual boundary gradient distribution for adaptive thresholds
    boundaries_0 = compute_boundary_gradients(labels_0, C)
    if boundaries_0:
        grad_values = sorted(boundaries_0.values())
        # Merge at 25th, 50th, 75th, 90th percentile of boundary gradients
        percentiles = [25, 50, 75, 90]
        thresholds = [np.percentile(grad_values, p) for p in percentiles]
        print(f"  Boundary gradient range: [{min(grad_values):.6f}, {max(grad_values):.6f}]")
        print(f"  Merge thresholds (p25/p50/p75/p90): "
              f"{', '.join(f'{t:.6f}' for t in thresholds)}")
    else:
        thresholds = [0.001, 0.003, 0.01, 0.03]

    labels_by_level = [labels_0]
    current_labels = labels_0

    for level_idx, thresh in enumerate(thresholds):
        level = level_idx + 1
        print(f"\nLevel {level}: merge threshold = {thresh:.6f}")
        new_labels, _ = merge_regions(current_labels, C, thresh)
        n_regions = len(np.unique(new_labels))
        print(f"  Regions: {n_regions}")

        # Verification
        assert (new_labels > 0).all(), f"Some cells unassigned at level {level}"
        print(f"  [PASS] All cells assigned")

        labels_by_level.append(new_labels)
        current_labels = new_labels

        # Stop if we're down to very few regions
        if n_regions <= 3:
            print(f"  (Stopping: only {n_regions} regions remain)")
            break

    n_levels = len(labels_by_level)
    print(f"\nTotal hierarchy levels: {n_levels}")

    # Build hierarchy
    hierarchy = build_hierarchy(labels_by_level)

    # Statistics
    print(f"\n{'=' * 70}")
    print("Partition Statistics")
    print(f"{'=' * 70}")

    level_stats = []
    for level in range(n_levels):
        labels = labels_by_level[level]
        unique = np.unique(labels)
        sizes = [np.sum(labels == rid) for rid in unique]
        stats = {
            'level': level,
            'n_regions': len(unique),
            'mean_size': float(np.mean(sizes)),
            'std_size': float(np.std(sizes)),
            'min_size': int(np.min(sizes)),
            'max_size': int(np.max(sizes)),
        }
        level_stats.append(stats)
        print(f"  Level {level}: {stats['n_regions']:4d} regions, "
              f"mean_size={stats['mean_size']:.0f}, "
              f"range=[{stats['min_size']}, {stats['max_size']}]")

    # Hierarchy validation
    print(f"\n{'=' * 70}")
    print("Hierarchy Validation")
    print(f"{'=' * 70}")

    for level in range(1, n_levels):
        parent_labels = labels_by_level[level]
        child_labels = labels_by_level[level - 1]

        # Every child region should map to exactly one parent
        child_ids = np.unique(child_labels)
        multi_parent = 0
        for cid in child_ids:
            child_mask = child_labels == cid
            parent_of_child = np.unique(parent_labels[child_mask])
            if len(parent_of_child) > 1:
                multi_parent += 1

        if multi_parent == 0:
            print(f"  Level {level-1} -> {level}: [PASS] All children have exactly one parent")
        else:
            print(f"  Level {level-1} -> {level}: [WARN] {multi_parent} children span multiple parents")

    # Save results
    for level, labels in enumerate(labels_by_level):
        np.save(RESULTS_DIR / f'exp_02_labels_level{level}.npy', labels)

    # Save hierarchy as JSON
    hierarchy_json = {
        f"{level},{rid}": [(cl, int(cid)) for cl, cid in children]
        for (level, rid), children in hierarchy.items()
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'experiment': 'exp_02_hierarchical_partition',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'sigma': 0.5,
            'min_filter_size': 3,
            'thresholds': [float(t) for t in thresholds[:n_levels - 1]],
        },
        'level_stats': level_stats,
        'n_levels': n_levels,
        'hierarchy': hierarchy_json,
    }

    output_file = RESULTS_DIR / f'exp_02_partition_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Labels saved to results/exp_02_labels_level*.npy")
    print(f"  Results saved to: {output_file.name}")

    return results


if __name__ == '__main__':
    run_experiment()
