from collections import defaultdict
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Utility function to compute symbolic entropy
def symbolic_entropy(text):
    words = text.split()
    unique_words = set(words)
    probs = [words.count(w)/len(words) for w in unique_words]
    return -sum(p * np.log2(p) for p in probs)

# Parameters
initial_entropy_payload = "The structure of intelligence is a balance field."
extended_timesteps = 30
entropy_injection_interval = 5  # How often to inject a new entropy seed
agent_A_novelty_bias = 0.7      # Probability Agent A prefers novel tokens
agent_B_stability_bias = 0.7    # Probability Agent B prefers frequent tokens
min_phrase_length = 2
max_phrase_length = 3

# Set up memory and token tracking
symbol_history_extended = defaultdict(list)
agent_A_memory = []
agent_B_memory = []
history = []
token_pool = list(set(initial_entropy_payload.lower().split()))
all_seen_tokens = set(token_pool)

# Role assignment: driver (first), end (last), context (middle)
def assign_roles(tokens):
    roles = []
    for i, _ in enumerate(tokens):
        if i == 0:
            roles.append('driver')
        elif i == len(tokens) - 1:
            roles.append('end')
        else:
            roles.append('context')
    return roles

# Get most/least frequent tokens for bias
def get_frequent_token(tokens, seen_counts, reverse=True):
    filtered = [t for t in tokens if t.isalpha()]
    if not filtered:
        return random.choice(tokens)
    freq = sorted(filtered, key=lambda x: seen_counts.get(x, 0), reverse=reverse)
    return freq[0] if freq else random.choice(tokens)

# Create phrase from token pool + agent memory
def phrase_mutation(agent_memory, token_pool, seen_counts, role, bias=0.5, prefer_novel=False):
    memory_tokens = [word for phrase in agent_memory for word in phrase.split() if word.isalpha()]
    combined_tokens = list(set(token_pool + memory_tokens))
    phrase_len = random.randint(min_phrase_length, max_phrase_length)
    if not combined_tokens:
        return f"{role}: null"
    chosen_tokens = []
    for _ in range(phrase_len):
        if prefer_novel and random.random() < bias:
            if seen_counts:
                min_seen = min(seen_counts.get(t, 0) for t in combined_tokens)
                rare_tokens = [t for t in combined_tokens if seen_counts.get(t, 0) == min_seen]
                token = random.choice(rare_tokens) if rare_tokens else random.choice(combined_tokens)
            else:
                token = random.choice(combined_tokens)
        elif not prefer_novel and random.random() < bias:
            token = get_frequent_token(combined_tokens, seen_counts, reverse=True)
        else:
            token = random.choice(combined_tokens)
        chosen_tokens.append(token)
    phrase = f"{role}: " + " ".join(chosen_tokens)
    return phrase

# Main simulation loop
seen_counts = defaultdict(int)
unique_token_count_over_time = []
entropy_over_time = []

for t in range(extended_timesteps):
    # Entropy injection: periodically add new unresolved tokens
    if t % entropy_injection_interval == 0 and t != 0:
        entropy_seed = random.choice([
            "synergize", "attenuation", "recursive", "horizon", "potential", "memory", "align", "signal", "fractal", "collapse", "threshold", "field"
        ])
        token_pool.append(entropy_seed)
        all_seen_tokens.add(entropy_seed)

    # Phrase-based agent output
    mutation_A = phrase_mutation(agent_A_memory, token_pool, seen_counts, "A", agent_A_novelty_bias, prefer_novel=True)
    agent_A_memory.append(mutation_A)

    mutation_B = phrase_mutation(agent_B_memory + [mutation_A], token_pool, seen_counts, "B", agent_B_stability_bias, prefer_novel=False)
    agent_B_memory.append(mutation_B)

    combined_output = f"{mutation_A} | {mutation_B}"
    new_tokens = combined_output.lower().split()
    token_pool.extend(new_tokens)
    for token in new_tokens:
        seen_counts[token] += 1
        symbol_history_extended[token].append(t)

    entropy = symbolic_entropy(combined_output)
    entropy_over_time.append(entropy)
    unique_token_count_over_time.append(len(set(token_pool)))

    # Track roles in both agent outputs separately
    a_tokens = mutation_A.lower().split()[1:]
    b_tokens = mutation_B.lower().split()[1:]
    a_roles = assign_roles(a_tokens)
    b_roles = assign_roles(b_tokens)
    roles = {"A_roles": a_roles, "B_roles": b_roles}

    history.append({
        "timestep": t,
        "agent_A_output": mutation_A,
        "agent_B_output": mutation_B,
        "combined_output": combined_output,
        "entropy": entropy,
        "roles": roles
    })

# Package data
symbolic_emergence_df = pd.DataFrame(history)
token_drift_df = pd.DataFrame([
    {"Token": token, "First Seen": min(times), "Last Seen": max(times), "Frequency": len(times)}
    for token, times in symbol_history_extended.items()
]).sort_values(by="Frequency", ascending=False).head(20)

# Visualization: Entropy and unique token count over time
plt.figure(figsize=(12, 5))
plt.plot(entropy_over_time, label='Symbolic Entropy', marker='o')
plt.plot(unique_token_count_over_time, label='Unique Token Count', marker='x')
plt.title('Entropy and Unique Token Count Over Time')
plt.xlabel('Timestep')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Display (or export) results
print(symbolic_emergence_df)
print(token_drift_df)
