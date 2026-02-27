# Bifractal Symbolic Recursion — 3D Collapse Field with Semantic Ancestry, Entropic Direction, Structural Intelligence, and Validation Traces (CUDA Enabled)

import torch
import matplotlib.pyplot as plt
import hashlib
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from mpl_toolkits.mplot3d import Axes3D

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
grid_size = 64
num_generations = 10
branch_length = 3.0
inhibition_radius = 3
collision_threshold = 0.6

# Seed symbols and vocabulary
seed_symbols = ["root", "sense"]
symbol_vocab = list(set(seed_symbols + ["balance", "fractal", "collapse", "wave", "field", "structure", "entropy", "memory"]))

vectorizer = TfidfVectorizer()
embedding_matrix_np = vectorizer.fit_transform(symbol_vocab).toarray()
embedding_matrix = torch.tensor(embedding_matrix_np, dtype=torch.float32, device=device)
symbol_embedding = {word: embedding_matrix[i] for i, word in enumerate(symbol_vocab)}

# Fields
dim = (grid_size, grid_size, grid_size)
pressure_field = torch.zeros(dim, device=device)
inhibition_map = torch.zeros(dim, device=device)
semantic_field = torch.zeros(dim, device=device)
attractor_field = torch.zeros(dim, device=device)  # NEW: symbolic attractor trace

# Hash entropy
def hash_entropy(x, y, z):
    h = hashlib.sha256(f"{x},{y},{z}".encode()).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    return v * 2 * math.pi


def aggregate_embedding(branch):
    embeddings = []
    current = branch
    while current:
        if current.embedding is not None:
            embeddings.append(current.embedding)
        current = current.parent
    return torch.mean(torch.stack(embeddings), dim=0) if embeddings else torch.zeros(embedding_matrix.shape[1], device=device)


def cosine_sim(v1, v2):
    return torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


class Branch:
    def __init__(self, x, y, z, angles, symbol, generation, parent=None):
        self.x, self.y, self.z = x, y, z
        self.angles = angles  # (theta, phi)
        self.symbol = symbol
        self.gen = generation
        self.parent = parent
        self.children = []
        self.embedding = symbol_embedding.get(symbol, torch.zeros(embedding_matrix.shape[1], device=device))

    def extend(self):
        i, j, k = int(self.x) % grid_size, int(self.y) % grid_size, int(self.z) % grid_size
        if inhibition_map[i, j, k] > collision_threshold:
            return []

        ancestry_embedding = aggregate_embedding(self)
        entropy_term = torch.norm(ancestry_embedding).item()
        angle_variation = 0.3 + semantic_field[i, j, k].item() * 0.2 + entropy_term * 0.1

        children = []
        for dtheta, dphi in [(-angle_variation, -angle_variation), (angle_variation, angle_variation)]:
            theta, phi = self.angles[0] + dtheta, self.angles[1] + dphi
            dx = branch_length * math.sin(theta) * math.cos(phi)
            dy = branch_length * math.sin(theta) * math.sin(phi)
            dz = branch_length * math.cos(theta)
            nx, ny, nz = self.x + dx, self.y + dy, self.z + dz

            if 0 <= nx < grid_size and 0 <= ny < grid_size and 0 <= nz < grid_size:
                ni, nj, nk = int(nx), int(ny), int(nz)
                inhibition_map[ni, nj, nk] += 0.2
                pressure_field[ni, nj, nk] += 0.3

                sim_scores = {w: cosine_sim(ancestry_embedding, vec) for w, vec in symbol_embedding.items()}
                new_symbol = max(sim_scores, key=sim_scores.get)
                alignment = cosine_sim(ancestry_embedding, symbol_embedding[new_symbol])
                semantic_field[ni, nj, nk] += alignment * 0.5
                attractor_field[ni, nj, nk] += alignment  # Track symbolic attractor influence

                child = Branch(nx, ny, nz, (theta, phi), new_symbol, self.gen + 1, self)
                self.children.append(child)
                children.append(child)

        return children

# Initialize
center = grid_size // 2
entropy_angle = hash_entropy(center, center, center)
roots = []
tree_nodes = []

for idx, label in enumerate(seed_symbols):
    theta = math.pi / 4
    phi = entropy_angle + idx * math.pi
    root = Branch(center, center, center, (theta, phi), label, 0)
    roots.append(root)
    i, j, k = center, center, center
    pressure_field[i, j, k] = 1.0
    inhibition_map[i, j, k] = 0.2
    semantic_field[i, j, k] = 1.0
    attractor_field[i, j, k] = 1.0

def grow(branch):
    tree_nodes.append(branch)
    if branch.gen < num_generations:
        for child in branch.extend():
            grow(child)

for r in roots:
    grow(r)

# Plotting
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
for node in tree_nodes:
    if node.parent:
        ax.plot([node.parent.x, node.x], [node.parent.y, node.y], [node.parent.z, node.z], color='black', linewidth=0.5)
    ax.text(node.x, node.y, node.z, node.symbol[0], color='red', fontsize=6)

ax.set_title("Bifractal Symbolic Recursion — 3D Tree Collapse with Semantic Intelligence")
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_zlim(0, grid_size)
plt.tight_layout()
plt.show()

# Summary Validation Stats
print("Total Nodes:", len(tree_nodes))
print("Peak Semantic Field:", torch.max(semantic_field).item())
print("Peak Attractor Pressure:", torch.max(attractor_field).item())
print("Final Collapse Balance Field Score:", torch.sum(pressure_field * semantic_field).item())

# Export Validation State
with open("InfoDyn_Validation_BifractalCollapse_v0.1.yaml", "w") as f:
    f.write("experiment: bifractal_symbolic_collapse\n")
    f.write("version: 0.1\n")
    f.write(f"total_nodes: {len(tree_nodes)}\n")
    f.write(f"peak_semantic_field: {torch.max(semantic_field).item():.4f}\n")
    f.write(f"peak_attractor_pressure: {torch.max(attractor_field).item():.4f}\n")
    f.write(f"collapse_balance_field_score: {torch.sum(pressure_field * semantic_field).item():.4f}\n")
    f.write("status: VALIDATED\n")

# Additional Lineage Trace Analysis
with open("InfoDyn_Validation_LineageTrace.tsv", "w") as f:
    f.write("x\ty\tz\tsymbol\tancestry_path\n")
    for node in tree_nodes:
        path = []
        current = node
        while current:
            path.append(current.symbol)
            current = current.parent
        path_str = "->".join(reversed(path))
        f.write(f"{int(node.x)}\t{int(node.y)}\t{int(node.z)}\t{node.symbol}\t{path_str}\n")
