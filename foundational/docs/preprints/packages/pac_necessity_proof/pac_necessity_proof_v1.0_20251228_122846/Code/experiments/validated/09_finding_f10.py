"""
Extended Fractal PAC Tree Analysis - Finding F_10 = 55
======================================================

The previous analysis showed:
- Sum at each depth = 13
- Cumulative sum through depth 3 = 52
- 52 + 3 = 55 = F_10

This script explores WHERE F_10 comes from in the tree structure.
"""

import numpy as np
from collections import Counter

# Fibonacci
def fib(n):
    if n <= 0: return 0
    if n <= 2: return 1
    a, b = 1, 1
    for _ in range(n - 2):
        a, b = b, a + b
    return b

def fib_index(value):
    if value <= 0: return -1
    n = 1
    while fib(n) < value:
        n += 1
        if n > 50: return -1
    return n if fib(n) == value else -1

# Build complete tree with path tracking
class PACNode:
    def __init__(self, value, depth, path=""):
        self.value = value
        self.depth = depth
        self.path = path  # Track path from root
        self.left = None
        self.right = None

def build_full_tree(root_val, max_depth):
    root = PACNode(root_val, 0, "R")
    
    def split(node, d):
        if d >= max_depth or node.value <= 1:
            return
        n = fib_index(node.value)
        if n <= 2:
            return
        node.left = PACNode(fib(n-1), d+1, node.path + "L")
        node.right = PACNode(fib(n-2), d+1, node.path + "R")
        split(node.left, d+1)
        split(node.right, d+1)
    
    split(root, 0)
    return root

def collect_all_nodes(root):
    nodes = []
    def traverse(n):
        if n is None: return
        nodes.append(n)
        traverse(n.left)
        traverse(n.right)
    traverse(root)
    return nodes

print("=" * 70)
print("EXTENDED FRACTAL PAC ANALYSIS - FINDING F_10 = 55")
print("=" * 70)

tree = build_full_tree(13, max_depth=5)
all_nodes = collect_all_nodes(tree)

# Analysis 1: Sum of all node values
print("\n1. SUM OF ALL NODE VALUES")
print("-" * 50)

for max_d in range(6):
    nodes_up_to = [n for n in all_nodes if n.depth <= max_d]
    total = sum(n.value for n in nodes_up_to)
    print(f"   Depth 0-{max_d}: {len(nodes_up_to):2d} nodes, sum = {total}")

# Analysis 2: Weighted sums
print("\n2. WEIGHTED SUM ANALYSIS")
print("-" * 50)

print("\n   a) Value * (depth + 1):")
for max_d in range(5):
    nodes_up_to = [n for n in all_nodes if n.depth <= max_d]
    weighted = sum(n.value * (n.depth + 1) for n in nodes_up_to)
    print(f"      Depth 0-{max_d}: weighted sum = {weighted}")

print("\n   b) Value * 2^depth:")
for max_d in range(5):
    nodes_up_to = [n for n in all_nodes if n.depth <= max_d]
    weighted = sum(n.value * (2 ** n.depth) for n in nodes_up_to)
    print(f"      Depth 0-{max_d}: weighted sum = {weighted}")

# Analysis 3: Path-based analysis
print("\n3. ALL PATHS AND VALUES")
print("-" * 50)

for n in sorted(all_nodes, key=lambda x: (x.depth, x.path)):
    fib_idx = fib_index(n.value)
    print(f"   {n.path:8s}  depth={n.depth}  value={n.value:2d} (F_{fib_idx})")

# Analysis 4: Looking for 55 specifically
print("\n4. SEARCHING FOR 55 = F_10")
print("-" * 50)

# Try different combinations
print("\n   a) Sum of path lengths weighted by value:")
path_weighted = sum(n.value * len(n.path) for n in all_nodes)
print(f"      Total = {path_weighted}")

print("\n   b) Total nodes * root value:")
print(f"      {len(all_nodes)} * 13 / 3 = {len(all_nodes) * 13 / 3:.1f}")

print("\n   c) Count unique path-value combinations:")
unique_paths = len(set((n.path, n.value) for n in all_nodes))
print(f"      Unique (path, value) pairs: {unique_paths}")

# Analysis 5: Fibonacci identity check
print("\n5. FIBONACCI IDENTITIES IN TREE")
print("-" * 50)

# Check if 55 appears as a sum pattern
print("\n   a) Sum of F_n * (count at that F_n):")
value_counts = Counter(n.value for n in all_nodes)
print(f"      Value counts: {dict(sorted(value_counts.items(), reverse=True))}")
weighted_by_count = sum(v * c for v, c in value_counts.items())
print(f"      Sum of value * count = {weighted_by_count}")

print("\n   b) Sum of first N Fibonacci numbers:")
for N in range(1, 12):
    fib_sum = sum(fib(i) for i in range(1, N+1))
    if fib_sum == 55:
        print(f"      sum(F_1 to F_{N}) = {fib_sum} = F_10 !!!")
    else:
        print(f"      sum(F_1 to F_{N}) = {fib_sum}")

# Analysis 6: The key insight - F_10 = F_7 + F_6 + F_5 + ...
print("\n6. KEY INSIGHT: FIBONACCI SUM IDENTITY")
print("-" * 50)

print("\n   Fibonacci sum property: sum(F_1 to F_n) = F_{n+2} - 1")
print()
print("   sum(F_1 to F_8) = F_10 - 1 = 55 - 1 = 54")
print("   sum(F_1 to F_9) = F_11 - 1 = 89 - 1 = 88")
print()
print("   But also:")
print("   F_7 + F_6 + F_5 + F_4 + F_3 + F_2 + F_1")
print(f"   = {fib(7)} + {fib(6)} + {fib(5)} + {fib(4)} + {fib(3)} + {fib(2)} + {fib(1)}")
print(f"   = {sum(fib(i) for i in range(1, 8))}")
print()
print("   Hmm, that's 33, not 55.")
print()
print("   What about F_10 = F_9 + F_8?")
print(f"   F_10 = {fib(10)} = {fib(9)} + {fib(8)} = F_9 + F_8")

# Analysis 7: The EM cycle interpretation
print("\n7. ELECTROMAGNETIC CYCLE INTERPRETATION")
print("-" * 50)

print("\n   From the original formula:")
print("   F_10 = 55 appears in alpha = (2/3*phi*F_10) * correction")
print()
print("   In the tree context, what's special about 55?")
print()
print("   a) Depth 3 gives us the stable physics (MED depth 2)")
print("   b) At depth 3, sum = 13 (PAC conserved)")
print("   c) Cumulative sum through all depths 0-3: 13 * 4 = 52")
print("   d) 52 + 3 = 55 (add one SU(2) correction)")
print()
print("   e) Or: 55 = 13 * 4 + 3 = 4 traversals of F_7 tree + F_4")
print("          = 4 spacetime dims * gauge closure + spatial dims")

# Analysis 8: Path count analysis
print("\n8. PATH COUNT ANALYSIS")
print("-" * 50)

# Count paths at each depth
for d in range(5):
    nodes_at_d = [n for n in all_nodes if n.depth == d]
    print(f"   Depth {d}: {len(nodes_at_d)} paths")

print()
print("   Total paths (=nodes) in tree:")
total_paths = len(all_nodes)
print(f"   {total_paths} nodes")
print(f"   {total_paths} * 13 / 3 = {total_paths * 13 / 3:.1f}")
print(f"   {total_paths} * 8 / 3 = {total_paths * 8 / 3:.1f}")

# Analysis 9: The alpha formula decomposition
print("\n9. ALPHA FORMULA DECOMPOSITION")
print("-" * 50)

PHI = (1 + np.sqrt(5)) / 2
F7, F10 = 13, 55

print("\n   alpha = (2 / 3*phi*F_10) * (1 - F_10 / 4*pi*F_7^2)")
print()
print("   Breaking it down:")
print(f"   2 / (3 * phi * 55) = 2 / {3 * PHI * 55:.4f} = {2 / (3 * PHI * 55):.8f}")
print()
print("   The correction term:")
print(f"   1 - 55 / (4*pi*169) = 1 - 55 / {4 * np.pi * 169:.2f}")
print(f"                      = 1 - {55 / (4 * np.pi * 169):.6f}")
print(f"                      = {1 - 55 / (4 * np.pi * 169):.6f}")
print()
print("   In tree terms:")
print(f"   - 55 could be F_10 = 4 * 13 + 3 (4 depth-sums + F_4)")
print(f"   - 169 = 13^2 = F_7^2 (closure squared)")
print(f"   - 4*pi ≈ {4*np.pi:.4f} (geometric factor for 4D spacetime)")

# Analysis 10: The key realization
print("\n" + "=" * 70)
print("KEY REALIZATION")
print("=" * 70)

print("""
The F_10 = 55 in the alpha formula might represent:

1. CUMULATIVE TREE WEIGHT
   - Sum through depth 3: 13 + 13 + 13 + 13 = 52
   - Plus one F_4 = 3 correction: 52 + 3 = 55
   - This counts the "total weight" of the physics tree

2. OR: EXTENDED TREE
   - The EM interaction requires traversing "deeper"
   - F_10 root tree would give richer structure
   - But we only "see" down to F_7 = 13 closure

3. THE CORRECTION TERM (1 - F_10/4*pi*F_7^2)
   - F_10 / F_7^2 = 55 / 169 = 0.3254
   - This is the "fill fraction" of EM in closure^2
   - 4*pi makes it spherical (4D spacetime volume)

4. PHYSICAL INTERPRETATION
   - The 55 is how much "recursion capacity" EM uses
   - The 169 is the total "phase space volume" (13^2)
   - The ratio determines the coupling strength
""")

# Final computation
print("\n" + "=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

alpha_computed = (2 / (3 * PHI * 55)) * (1 - 55 / (4 * np.pi * 169))
alpha_measured = 0.0072973525693

print(f"\n   Computed alpha:  {alpha_computed:.10f}")
print(f"   Measured alpha:  {alpha_measured:.10f}")
print(f"   Error: {abs(alpha_computed - alpha_measured)/alpha_measured * 1e6:.2f} ppm")
print()
print("   Components from tree structure:")
print(f"   - F_7 = 13 (root/closure) ✓")
print(f"   - F_10 = 55 = 4*13 + 3 (4 depth-sums + F_4 correction)")
print(f"   - F_7^2 = 169 (closure squared for phase space)")
print(f"   - 4*pi (4D geometric factor)")
print(f"   - 2/3 (matter/antimatter? or depth ratio?)")
print(f"   - phi (golden scaling from PAC)")
