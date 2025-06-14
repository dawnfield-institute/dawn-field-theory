# Bifractal Symbolic Recursion — 2D Collapse Field with Semantic Ancestry, Entropic Direction, and Structural Intelligence

import numpy as np
import matplotlib.pyplot as plt
import hashlib
import random
import math

# Parameters
grid_size = 128
num_generations = 12
branch_length = 6
inhibition_radius = 5
collision_threshold = 0.75

# Define seed symbols and embeddings
seed_symbols = ["root", "sense"]
symbol_embedding = {
    "r": np.array([1.0, 0.0, 0.0]),
    "s": np.array([0.0, 1.0, 0.0])
}

# Memory and semantic pressure fields
pressure_field = np.zeros((grid_size, grid_size))
inhibition_map = np.zeros((grid_size, grid_size))
semantic_field = np.zeros((grid_size, grid_size))

# Helper functions
def hash_entropy(x, y):
    h = hashlib.sha256(f"{x},{y}".encode()).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    angle = v * 2 * np.pi
    return angle

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def aggregate_embedding(branch):
    embeddings = []
    current = branch
    while current:
        if current.embedding is not None:
            embeddings.append(current.embedding)
        current = current.parent
    if embeddings:
        return np.mean(embeddings, axis=0)
    return np.zeros(3)

# Define Branch object
class Branch:
    def __init__(self, x, y, angle, symbol, generation, parent=None):
        self.x = x
        self.y = y
        self.angle = angle
        self.symbol = symbol
        self.gen = generation
        self.parent = parent
        self.children = []
        self.embedding = symbol_embedding[symbol[0]] if symbol[0] in symbol_embedding else np.zeros(3)

    def extend(self):
        i, j = int(self.x) % grid_size, int(self.y) % grid_size
        if inhibition_map[i, j] > collision_threshold:
            return []

        # Aggregate symbolic ancestry embedding
        ancestry_embedding = aggregate_embedding(self)
        entropy_term = np.linalg.norm(ancestry_embedding)
        angle_variation = 0.4 + semantic_field[i, j] * 0.2 + entropy_term * 0.1

        children = []
        for delta in [-angle_variation, angle_variation]:
            new_angle = self.angle + delta
            nx = self.x + math.cos(new_angle) * branch_length
            ny = self.y + math.sin(new_angle) * branch_length

            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                ni, nj = int(nx), int(ny)
                inhibition_map[ni, nj] += 0.2
                pressure_field[ni, nj] += 0.3

                base_symbol = self.symbol[0]
                other = 's' if base_symbol == 'r' else 'r'
                sim = cosine_similarity(self.embedding, symbol_embedding[other])

                if pressure_field[ni, nj] > 0.6:
                    new_symbol = base_symbol + base_symbol
                elif sim < 0.8:
                    new_symbol = base_symbol + other
                else:
                    new_symbol = base_symbol

                # Semantic alignment with ancestry
                symbol_vector = symbol_embedding[new_symbol[-1]]
                semantic_alignment = cosine_similarity(ancestry_embedding, symbol_vector)
                semantic_field[ni, nj] += semantic_alignment * 0.5

                child = Branch(nx, ny, new_angle, new_symbol, self.gen + 1, self)
                self.children.append(child)
                children.append(child)

        return children

# Initialize root nodes
center = grid_size // 2
entropy_angle = hash_entropy(center, center)
roots = []
for idx, label in enumerate(seed_symbols):
    angle = entropy_angle + idx * np.pi
    root = Branch(center, center, angle, label[0], 0)
    roots.append(root)
    i, j = center, center
    pressure_field[i, j] = 1.0
    inhibition_map[i, j] = 0.2
    semantic_field[i, j] = 1.0

# Recursive expansion
tree_nodes = []
def grow(branch):
    tree_nodes.append(branch)
    if branch.gen < num_generations:
        children = branch.extend()
        for child in children:
            grow(child)

for root in roots:
    grow(root)

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))
for node in tree_nodes:
    if node.parent:
        ax.plot([node.parent.x, node.x], [node.parent.y, node.y], color='black', linewidth=0.6)
    ax.text(node.x, node.y, node.symbol[0], fontsize=6, ha='center', va='center', color='red')

ax.set_title("Bifractal Symbolic Recursion — 2D Tree Collapse with Semantic Intelligence")
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
plt.show()
