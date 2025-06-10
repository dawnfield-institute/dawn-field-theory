# Entropy-Based Mutation Detection and Repair Script (QBE-Enhanced)

import math
import random
import matplotlib.pyplot as plt
from collections import Counter

# --- Entropy Functions ---
def shannon_entropy(seq, k=3):
    if len(seq) < k:
        return 0
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    freq = Counter(kmers)
    total = sum(freq.values())
    entropy = -sum((count/total) * math.log2(count/total) for count in freq.values() if count > 0)
    return entropy

def compute_entropy_profile(sequence, window_size=15, step=1, k=3):
    entropy_profile = []
    for i in range(0, len(sequence) - window_size + 1, step):
        window = sequence[i:i+window_size]
        entropy_profile.append((i, shannon_entropy(window, k)))
    return entropy_profile

# --- Mutation Simulation ---
def simulate_mutation(sequence, mutation_rate=0.20):
    amino_acids = sorted(set(sequence))
    mutated = list(sequence)
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            original = mutated[i]
            choices = [aa for aa in amino_acids if aa != original]
            if choices:
                mutated[i] = random.choice(choices)
    return ''.join(mutated)

# --- Delta Entropy Comparison ---
def compute_entropy_deltas(profile1, profile2, threshold=0.02):
    deltas = []
    for (pos1, e1), (pos2, e2) in zip(profile1, profile2):
        if abs(e1 - e2) > threshold:
            deltas.append(pos1)
    return deltas

# --- Diff Detection ---
def find_sequence_differences(seq1, seq2):
    return [i for i, (a, b) in enumerate(zip(seq1, seq2)) if a != b]

# --- QBE Score Function(inhouse physics model) ---
def qbe_score(new_entropy, target_entropy, alt_aa, original_aa, z_factor):
    entropy_balance = (new_entropy - target_entropy) ** 2
    substitution_penalty = abs(ord(alt_aa) - ord(original_aa)) / 26.0
    return z_factor * entropy_balance + 0.05 * substitution_penalty

# --- QBE-Enhanced Repair ---
def entropy_based_repair(mutated_seq, original_seq, target_entropy_profile, window_size=15, step=1, k=3, z_factor=0.75):
    repaired = list(mutated_seq)
    amino_acids = sorted(set(mutated_seq))
    mutated_profile = compute_entropy_profile(mutated_seq, window_size, step, k)
    delta_positions = compute_entropy_deltas(mutated_profile, target_entropy_profile, threshold=0.02)
    diff_positions = find_sequence_differences(mutated_seq, original_seq)

    print("\nRepair Log:")
    if not delta_positions and not diff_positions:
        print("  No significant entropy deviation or sequence difference detected.")
    else:
        print("  Entropy-shifted windows:", delta_positions)
        print("  Direct mutation positions:", diff_positions)

    for pos in diff_positions:
        best_aa = repaired[pos]
        min_score = float('inf')
        for offset in range(-window_size//2, window_size//2 + 1):
            win_start = max(0, pos + offset - window_size//2)
            win_end = min(len(repaired), win_start + window_size)
            if win_end - win_start < window_size:
                continue
            target_window = original_seq[win_start:win_end]
            target_entropy = shannon_entropy(target_window, k)
            for alt in amino_acids:
                if alt == repaired[pos]:
                    continue
                test_seq = list(repaired)
                test_seq[pos] = alt
                new_window = ''.join(test_seq[win_start:win_end])
                new_entropy = shannon_entropy(new_window, k)
                score = qbe_score(new_entropy, target_entropy, alt, original_seq[pos], z_factor)
                if score < min_score:
                    min_score = score
                    best_aa = alt
        if repaired[pos] != best_aa:
            print(f"    Position {pos}: {repaired[pos]} -> {best_aa}")
        repaired[pos] = best_aa
    return ''.join(repaired)

# --- Main Analysis Runner ---
def run_entropy_analysis(sequence, mutation_rate=0.20, z_factor=0.75):
    original_seq = sequence
    mutated_seq = simulate_mutation(original_seq, mutation_rate=mutation_rate)

    original_profile = compute_entropy_profile(original_seq)
    mutated_profile = compute_entropy_profile(mutated_seq)
    repaired_seq = entropy_based_repair(mutated_seq, original_seq, original_profile, z_factor=z_factor)
    repaired_profile = compute_entropy_profile(repaired_seq)

    delta_positions = compute_entropy_deltas(mutated_profile, original_profile)

    print("\nOriginal Sequence:", original_seq)
    print("Mutated Sequence: ", mutated_seq)
    print("Repaired Sequence:", repaired_seq)

    positions = [pos for pos, _ in original_profile]
    orig_entropy = [e for _, e in original_profile]
    mut_entropy = [e for _, e in mutated_profile]
    rep_entropy = [e for _, e in repaired_profile]

    print("\nEntropy Check:")
    print("  Avg Original Entropy:", round(sum(orig_entropy)/len(orig_entropy), 4))
    print("  Avg Mutated  Entropy:", round(sum(mut_entropy)/len(mut_entropy), 4))
    print("  Avg Repaired Entropy:", round(sum(rep_entropy)/len(rep_entropy), 4))

    plt.figure(figsize=(14, 5))
    plt.plot(positions, orig_entropy, label="Original (Healthy)", color='blue', alpha=0.7)
    plt.plot(positions, mut_entropy, label="Mutated", color='orange', alpha=0.7)
    plt.plot(positions, rep_entropy, label="Repaired", linestyle='--', color='green', alpha=0.8)
    for pos in delta_positions:
        plt.axvline(x=pos, color='red', linestyle=':', alpha=0.2)
    plt.title("Entropy-Based Mutation Detection and Repair (QBE Enhanced)")
    plt.xlabel("Position in Sequence")
    plt.ylabel("Shannon Entropy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    test_sequence = "MPGQELRTVNGSQMLLVLLVLSWLPHGGALSLAEASRASFPGPSELHSEDSRFRELRKRY"
    run_entropy_analysis(test_sequence)