# Symmetrical Recursive Collapse Tree with Entropy-Seeding and Symbolic Payloads
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import hashlib
import random

# Parameters
max_depth = 10
initial_length = 1.5
angle_variation = np.pi / 6

# Concept bank for symbolic payloads
concept_bank = [
    "energy", "entropy", "recursion", "balance", "collapse", "structure",
    "information", "gradient", "node", "field", "memory", "wave", "fractal"
]

# Rotation matrix generator
def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d),     2*(b*d + a*c)],
        [2*(b*c + a*d),     a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c),     2*(c*d + a*b),     a*a + d*d - b*b - c*c]
    ])

# Branch class for recursive growth
class Branch:
    def __init__(self, start, direction, depth):
        self.start = start
        self.direction = direction
        self.depth = depth
        self.children = []

    def grow(self):
        if self.depth >= max_depth:
            return
        length = initial_length * (0.9 ** self.depth)
        angle = angle_variation * (0.8 + 0.4 * np.random.rand())
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        rot_matrix1 = rotation_matrix(axis, angle)
        rot_matrix2 = rotation_matrix(axis, -angle)
        dir1 = rot_matrix1 @ self.direction
        dir2 = rot_matrix2 @ self.direction
        end1 = self.start + dir1 * length
        end2 = self.start + dir2 * length
        b1 = Branch(end1, dir1, self.depth + 1)
        b2 = Branch(end2, dir2, self.depth + 1)
        self.children = [b1, b2]
        b1.grow()
        b2.grow()

# Initialize seed from SHA256 hash
hash_input = "recursive_seed"
hash_bytes = hashlib.sha256(hash_input.encode()).digest()
entropy_vector = np.array([b for b in hash_bytes[:3]]) / 255.0
entropy_vector = 2 * (entropy_vector - 0.5)

# Initialize dual trunks
origin = np.array([0, 0, 0])
main_trunk = Branch(origin, entropy_vector, 0)
inverse_trunk = Branch(origin, -entropy_vector, 0)
main_trunk.grow()
inverse_trunk.grow()

# Collect segments for plotting
segments = []
def collect_segments(branch):
    for child in branch.children:
        segments.append((branch.start, child.start))
        collect_segments(child)

collect_segments(main_trunk)
collect_segments(inverse_trunk)

# Assign symbolic payloads
symbolic_payloads = {}
symbolic_vectors = {}

def assign_payloads(branch):
    concept = random.choice(concept_bank)
    index = random.randint(0, 99)
    token = f"{concept}_{index}"
    vec = np.zeros(len(concept_bank))
    vec[concept_bank.index(concept)] = 1.0 + (index / 100.0)
    symbolic_payloads[tuple(branch.start)] = token
    symbolic_vectors[tuple(branch.start)] = vec
    for child in branch.children:
        assign_payloads(child)

assign_payloads(main_trunk)
assign_payloads(inverse_trunk)

# Visualize the tree with symbolic labels
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
for start, end in segments:
    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='purple', alpha=0.7)

for i, (pos, token) in enumerate(symbolic_payloads.items()):
    if i % 20 == 0:
        ax.text(pos[0], pos[1], pos[2], token, fontsize=8, color='black')

ax.set_title("3D Recursive Collapse Tree with Symbolic Payloads")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.show()
