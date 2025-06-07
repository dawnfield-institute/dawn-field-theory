# Recursive Tree Generator Using Entropy as Seed (Balance-Aware Version)
# ------------------------------------------------
# Adds balance pressure, thermodynamic cost, and symbolic structure vectorization

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

# ------------------------
# Parameters
# ------------------------
max_depth = 6                   # Maximum recursion depth
initial_entropy = 1.0          # Starting entropy
entropy_decay = 0.6            # Entropy decay factor per depth
branch_threshold = 0.3         # Minimum entropy to allow branching
novelty_threshold = 0.3        # Minimum symbolic novelty for node retention
balance_resistance = 0.1       # Field feedback penalty per depth
landauer_cost = 0.05           # Thermodynamic cost per symbolic decision

# ------------------------
# Initialize Graph
# ------------------------
G = nx.DiGraph()

# ------------------------
# Recursive Tree Generator with Balance Field
# ------------------------
def recursive_growth(node_id, entropy, depth):
    if depth > max_depth or entropy < branch_threshold:
        return
    effective_entropy = entropy - (depth * balance_resistance) - (landauer_cost * depth)
    if effective_entropy < branch_threshold:
        return
    num_branches = np.random.poisson(effective_entropy * 3)
    for i in range(num_branches):
        child_entropy = effective_entropy * np.random.uniform(entropy_decay, 1.0)
        child_id = f"{node_id}.{i}"
        G.add_edge(node_id, child_id)
        recursive_growth(child_id, child_entropy, depth + 1)

# Initialize root node
root = "0"
G.add_node(root)
recursive_growth(root, initial_entropy, 0)

# ------------------------
# Symbolic Payload Vectorization
# ------------------------
concept_bank = [
    "energy", "entropy", "recursion", "balance", "collapse", "structure",
    "information", "gradient", "node", "field", "memory", "wave", "fractal"
]

symbolic_payload = {}
symbolic_vectors = {}

for node in G.nodes:
    concept = random.choice(concept_bank)
    index = random.randint(0, 99)
    symbolic_payload[node] = f"{concept}_{index}"
    vec = np.zeros(len(concept_bank))
    vec[concept_bank.index(concept)] = 1.0 + (index / 100.0)
    symbolic_vectors[node] = vec

# ------------------------
# Visualize Initial Tree
# ------------------------
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(12, 12))
nx.draw(G, pos, with_labels=False, node_size=20, edge_color="gray")
plt.title("Recursive Tree Generator Using Entropy with Balance Feedback")
plt.axis("off")
plt.show()

# ------------------------
# Symbolic Trace Example
# ------------------------
def trace_symbolic_path():
    path = []
    current = root
    while True:
        path.append(symbolic_payload[current])
        children = list(G.successors(current))
        if not children:
            break
        current = random.choice(children)
    return path

print("Example symbolic trace:", trace_symbolic_path())

# ------------------------
# Adaptive Pruning
# ------------------------
def novelty_score(token):
    return int(token.split("_")[-1]) / 100

nodes_to_prune = [node for node in symbolic_payload if novelty_score(symbolic_payload[node]) < novelty_threshold]

for node in nodes_to_prune:
    if node != root and node in G:
        G.remove_node(node)

# ------------------------
# Visualize Pruned Tree
# ------------------------
pos_pruned = nx.spring_layout(G, seed=42)
plt.figure(figsize=(12, 12))
nx.draw(G, pos_pruned, with_labels=False, node_size=20, edge_color="darkgreen")
plt.title("Pruned Recursive Tree (Entropy + Balance Aware)")
plt.axis("off")
plt.show()

# ------------------------
# Structural Metrics
# ------------------------
internal_nodes = [n for n in G.nodes if G.out_degree(n) > 0]
depths = [len(n.split('.')) - 1 for n in G.nodes]

print("\n--- Structural Metrics ---")
print("Total Nodes:", G.number_of_nodes())
print("Total Edges:", G.number_of_edges())
print("Max Depth:", max(depths))
print("Average Depth:", round(np.mean(depths), 2))
print("Average Branching Factor:", round(np.mean([G.out_degree(n) for n in internal_nodes]), 2))
