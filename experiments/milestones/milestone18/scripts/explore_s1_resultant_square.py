"""The mechanism: forests are bipartite, so their ADJACENCY spectra are symmetric about 0.
Res(F,G) = prod_{i,j}(lambda_i - mu_j). Pair lambda <-> -lambda and mu <-> -mu:
  (l-m)(l+m)(-l-m)(-l+m) = (l^2 - m^2)^2  -- a square.
So |Res| should be a perfect square for any two forests, whenever it is nonzero.
Our Cartan C = 2I - A, so Res_C = +- Res_A: the same statement."""
import sympy as sp, networkx as nx, random
t = sp.Symbol('t')
def cpA(G):
    n = G.number_of_nodes()
    if n == 0: return sp.Integer(1)
    idx = {v: i for i, v in enumerate(G.nodes())}
    A = sp.zeros(n, n)
    for u, v in G.edges(): A[idx[u], idx[v]] = A[idx[v], idx[u]] = 1
    return sp.expand(A.charpoly(t).as_expr())
def cpC(G):
    n = G.number_of_nodes()
    if n == 0: return sp.Integer(1)
    idx = {v: i for i, v in enumerate(G.nodes())}
    C = sp.zeros(n, n)
    for i in range(n): C[i, i] = 2
    for u, v in G.edges(): C[idx[u], idx[v]] = C[idx[v], idx[u]] = -1
    return sp.expand(C.charpoly(t).as_expr())
def res(a, b): return int(sp.resultant(sp.Poly(a, t), sp.Poly(b, t)))
def forest(n, s):
    if n <= 1: return nx.empty_graph(max(n, 0))
    T = nx.random_labeled_tree(n, seed=s)
    for e in list(T.edges()):                    # randomly drop edges -> forest
        if random.random() < 0.25: T.remove_edge(*e)
    return T
random.seed(5)
print("TEST 1: |Res_A(F, G)| a perfect square for forests F, G (nonzero cases)")
sq = tot = 0; nonsq = []
for _ in range(500):
    F, G = forest(random.randint(1, 9), random.randint(0, 10**6)), forest(random.randint(1, 9), random.randint(0, 10**6))
    r = res(cpA(F), cpA(G))
    if r == 0: continue
    tot += 1
    if sp.integer_nthroot(abs(r), 2)[1]: sq += 1
    else: nonsq.append(r)
print(f"   perfect square: {sq}/{tot}"); print("   non-squares:", nonsq[:8])
print("\nTEST 2: same in the Cartan channel (C = 2I - A), which is what the fold uses")
sq = tot = 0; nonsq = []
for _ in range(500):
    F, G = forest(random.randint(1, 9), random.randint(0, 10**6)), forest(random.randint(1, 9), random.randint(0, 10**6))
    r = res(cpC(F), cpC(G))
    if r == 0: continue
    tot += 1
    if sp.integer_nthroot(abs(r), 2)[1]: sq += 1
    else: nonsq.append(r)
print(f"   perfect square: {sq}/{tot}"); print("   non-squares:", nonsq[:8])
print("\nTEST 3: CONTROL -- non-bipartite graphs should BREAK it")
def anygraph(n, s):
    random.seed(s); G = nx.gnp_random_graph(n, 0.45, seed=s)
    return G
sq = tot = 0; bad = []
for i in range(400):
    F, G = anygraph(random.randint(3, 8), random.randint(0, 10**6)), anygraph(random.randint(3, 8), random.randint(0, 10**6))
    if nx.is_bipartite(F) and nx.is_bipartite(G): continue
    r = res(cpA(F), cpA(G))
    if r == 0: continue
    tot += 1
    if sp.integer_nthroot(abs(r), 2)[1]: sq += 1
    else: bad.append(r)
print(f"   perfect square: {sq}/{tot}  (expect well under 100% if bipartiteness is the cause)")
print("   non-square examples:", bad[:8])
print("\nTEST 4: odd-order forests carry a zero eigenvalue; check both parities separately")
for pF in (0, 1):
    for pG in (0, 1):
        sq = tot = 0
        for _ in range(220):
            nF = random.choice([k for k in range(2, 10) if k % 2 == pF])
            nG = random.choice([k for k in range(2, 10) if k % 2 == pG])
            F, G = forest(nF, random.randint(0, 10**6)), forest(nG, random.randint(0, 10**6))
            r = res(cpA(F), cpA(G))
            if r == 0: continue
            tot += 1
            if sp.integer_nthroot(abs(r), 2)[1]: sq += 1
        print(f"   |F| {'even' if pF==0 else 'odd '}, |G| {'even' if pG==0 else 'odd '}: {sq}/{tot}")
