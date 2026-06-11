"""
exp_01 -- Rapidity as Connection: Arc, Chord, and the First Holonomy

Milestone 15 (The Representative Problem)

PRE-REGISTERED: journals/2026-06-11_m15-exp01-03-preregistration.md (same commit).
Re-poses M13 exp_08 (rapidity composition, 99-292% "errors") as class-level claims:
the chord/arc confusion explains the failure; scalar spectral quantities are exact
(potentials); the genuinely non-exact object is the complement-eigenvector
connection, whose holonomy around affine-A cycles is the framework's first honest
curvature invariant -- IF it is nonzero (registered uncertainty: vertex-transitive
cycles might force triviality; the kill is live).

Tests (all relational):
  T1 [harness]: scalar potential telescopes exactly along paths (A_5..A_8)
  T2: chord-arc deficit is a class quantity (relabeling-invariant multiset;
      constant on automorphism-equivalent pairs; chord=0 iff same orbit)
  T3: affine holonomy on A^_n (n=3..12, k=2): nonzero, labeling-invariant
      angles, rank-stable leading angle (CV < 0.1 over A^_8..A^_12)
  T4: A^ holonomy deficit extremal vs 20 random unicyclic controls per size
      (|V| = 7, 9, 11), >= 80% of sizes; direction reported [D]

Outputs: results/exp_01_rapidity_one_form_YYYYMMDD_HHMMSS.json
"""

import sys
import numpy as np
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from representative import (
    build_path, build_d, build_cycle, random_unicyclic, cycle_basis_single,
    complement_spectrum, vertex_orbits, complement_deformation_rate,
    find_shortest_path, spectral_potential, cycle_holonomy, relabeled,
    save_m15_results, _convert_numpy)

K_PRIMARY = 2
K_ROBUST = 3


def chord(adjacency, u, v):
    s1 = complement_spectrum(adjacency, u)
    s2 = complement_spectrum(adjacency, v)
    m = max(len(s1), len(s2))
    a = np.zeros(m); b = np.zeros(m)
    a[:len(s1)] = s1; b[:len(s2)] = s2
    return float(np.linalg.norm(b - a))


def arc(adjacency, u, v):
    path = find_shortest_path(adjacency, u, v)
    if path is None or len(path) < 2:
        return 0.0
    return float(complement_deformation_rate(adjacency, path)['total'])


def graph_automorphism_pairs(adjacency):
    """Group vertex PAIRS into automorphism-equivalence classes via the orbit
    signature (orbit of u, orbit of v, graph distance)."""
    orbits = vertex_orbits(adjacency)
    orbit_of = {}
    for oi, orb in enumerate(orbits):
        for v in orb:
            orbit_of[v] = oi
    n = adjacency.shape[0]
    classes = {}
    for u, v in combinations(range(n), 2):
        path = find_shortest_path(adjacency, u, v)
        d = len(path) - 1 if path else -1
        key = (tuple(sorted((orbit_of[u], orbit_of[v]))), d)
        classes.setdefault(key, []).append((u, v))
    return classes, orbit_of


def test_T1():
    print("\n  T1 [harness]: scalar potential telescopes exactly")
    max_err = 0.0
    for n in (5, 6, 7, 8):
        a = build_path(n)
        g = [spectral_potential(a, v) for v in range(n)]
        for u, v in combinations(range(n), 2):
            path = find_shortest_path(a, u, v)
            steps = sum(g[path[i + 1]] - g[path[i]] for i in range(len(path) - 1))
            max_err = max(max_err, abs(steps - (g[v] - g[u])))
    passed = max_err < 1e-10
    print(f"    max telescoping error: {max_err:.2e} -> {'PASS' if passed else 'FAIL'}")
    return {'test': 'T1', 'max_err': max_err, 'PASS': passed}


def test_T2():
    print("\n  T2: chord-arc deficit is a class quantity")
    rng = np.random.RandomState(15)
    results = {}
    all_pass = True
    for name, a in (('A_8', build_path(8)), ('D_6', build_d(6))):
        n = a.shape[0]
        deficits = {(u, v): arc(a, u, v) - chord(a, u, v)
                    for u, v in combinations(range(n), 2)}
        base_multiset = np.sort(list(deficits.values()))

        # (a) relabeling invariance of the multiset
        relabel_ok = True
        for _ in range(3):
            perm = rng.permutation(n)
            ap = relabeled(a, perm)
            dp = np.sort([arc(ap, u, v) - chord(ap, u, v)
                          for u, v in combinations(range(n), 2)])
            if np.max(np.abs(dp - base_multiset)) > 1e-9:
                relabel_ok = False

        # (b) constant within automorphism-equivalent pair classes
        classes, orbit_of = graph_automorphism_pairs(a)
        class_ok = True
        for key, pairs in classes.items():
            vals = [deficits[p] for p in pairs]
            if np.max(vals) - np.min(vals) > 1e-9:
                class_ok = False

        # (c) chord = 0 iff same orbit
        lemma_ok = True
        for (u, v) in combinations(range(n), 2):
            same_orbit = orbit_of[u] == orbit_of[v]
            zero_chord = chord(a, u, v) < 1e-9
            if same_orbit != zero_chord:
                lemma_ok = False

        ok = relabel_ok and class_ok and lemma_ok
        all_pass = all_pass and ok
        results[name] = {'relabel_ok': relabel_ok, 'class_ok': class_ok,
                         'lemma_ok': lemma_ok,
                         'n_pair_classes': len(classes)}
        print(f"    {name}: relabel={relabel_ok} class-const={class_ok} "
              f"lemma={lemma_ok}")
    print(f"    -> {'PASS' if all_pass else 'FAIL'}")
    return {'test': 'T2', 'graphs': results, 'PASS': all_pass}


def test_T3():
    print("\n  T3: affine holonomy on A^_n cycles (k=2 primary)")
    rng = np.random.RandomState(151)
    rows = []
    for n in range(3, 13):
        m = n + 1
        a = build_cycle(m)
        cyc = list(range(m))
        h = cycle_holonomy(a, cyc, K_PRIMARY)
        # relabeling invariance of angles
        inv_ok = True
        for _ in range(3):
            perm = rng.permutation(m)
            ap = relabeled(a, perm)
            cyc_p = [int(perm[v]) for v in cyc]
            hp = cycle_holonomy(ap, cyc_p, K_PRIMARY)
            if np.max(np.abs(np.array(hp['angles']) - np.array(h['angles']))) > 1e-8:
                inv_ok = False
        rows.append({'n': n, 'V': m, 'deficit': h['deficit'],
                     'angles': h['angles'], 'det': h['det'],
                     'min_eigengap': h['min_eigengap'], 'relabel_ok': inv_ok})
        print(f"    A^_{n} (C_{m}): deficit={h['deficit']:.6f} "
              f"angles={[f'{x:.4f}' for x in h['angles']]} det={h['det']:+.3f} "
              f"gap={h['min_eigengap']:.3f} inv={inv_ok}")

    nonzero = all(r['deficit'] > 1e-6 for r in rows)
    invariant = all(r['relabel_ok'] for r in rows)
    lead = [max(r['angles']) for r in rows if r['n'] >= 8]
    cv = float(np.std(lead) / np.mean(lead)) if np.mean(lead) > 0 else np.inf
    stable = cv < 0.1
    passed = nonzero and invariant and stable
    print(f"    nonzero={nonzero} invariant={invariant} "
          f"lead-angle CV(A^_8..A^_12)={cv:.4f} -> {'PASS' if passed else 'FAIL'}")

    # robustness at k=3 (reported, not scored)
    rob = []
    for n in (8, 10, 12):
        h3 = cycle_holonomy(build_cycle(n + 1), list(range(n + 1)), K_ROBUST)
        rob.append({'n': n, 'deficit': h3['deficit'], 'angles': h3['angles']})
    return {'test': 'T3', 'rows': rows, 'lead_angle_cv': cv,
            'k3_robustness': rob, 'nonzero': nonzero, 'invariant': invariant,
            'stable': stable, 'PASS': passed}


def test_T4():
    print("\n  T4: A^ holonomy extremal among random unicyclic controls")
    rng = np.random.RandomState(152)
    per_size = []
    for m in (7, 9, 11):
        a = build_cycle(m)
        h_ade = cycle_holonomy(a, list(range(m)), K_PRIMARY)['deficit']
        controls = []
        for _ in range(20):
            g = random_unicyclic(m, rng)
            try:
                cyc = cycle_basis_single(g)
                if len(cyc) < 3:
                    continue
                controls.append(cycle_holonomy(g, cyc, K_PRIMARY)['deficit'])
            except Exception:
                continue
        lo, hi = (min(controls), max(controls)) if controls else (0, 0)
        is_max = h_ade >= hi - 1e-12
        is_min = h_ade <= lo + 1e-12
        per_size.append({'V': m, 'ade_deficit': h_ade, 'n_controls': len(controls),
                         'control_min': lo, 'control_max': hi,
                         'extremal': bool(is_max or is_min),
                         'direction': 'max' if is_max else ('min' if is_min else 'interior')})
        print(f"    |V|={m}: A^ deficit={h_ade:.6f} controls=[{lo:.6f},{hi:.6f}] "
              f"({len(controls)}) -> {per_size[-1]['direction']}")
    frac = np.mean([1.0 if r['extremal'] else 0.0 for r in per_size])
    passed = frac >= 0.8
    print(f"    extremal fraction: {frac:.2f} -> {'PASS' if passed else 'FAIL'}")
    return {'test': 'T4', 'per_size': per_size, 'extremal_fraction': float(frac),
            'PASS': passed}


def selftest():
    print("SELFTEST: builders + transport only (no registered quantities)")
    a = build_cycle(6)
    assert int(a.sum()) == 12, "cycle edges wrong"
    assert abs(a.sum() / 2 - a.shape[0]) < 1e-9, "unicyclic |E| != |V|"
    from representative import edge_transport
    T, gap = edge_transport(a, 0, 1, 2)
    err = np.max(np.abs(T @ T.T - np.eye(2)))
    print(f"  transport orthogonality error: {err:.2e}, eigengap: {gap:.3f}")
    assert err < 1e-10
    rng = np.random.RandomState(0)
    g = random_unicyclic(8, rng)
    cyc = cycle_basis_single(g)
    print(f"  random unicyclic |V|=8: cycle length {len(cyc)}")
    assert len(cyc) >= 3
    print("  OK")


if __name__ == '__main__':
    print("=" * 60)
    print("exp_01: Rapidity as Connection -- Arc, Chord, First Holonomy")
    print("Milestone 15 -- pre-registered")
    print("=" * 60)
    if '--selftest' in sys.argv:
        selftest()
    else:
        t1, t2, t3, t4 = test_T1(), test_T2(), test_T3(), test_T4()
        score = sum(t['PASS'] for t in (t1, t2, t3, t4))
        killed = (not t2['PASS']) or (not t3['nonzero'])
        verdict = ('SUPPORTED' if score == 4 else
                   'KILLED' if killed else 'PARTIAL')
        print(f"\n  Overall: {score}/4  VERDICT: {verdict}")
        save_m15_results('exp_01_rapidity_one_form', _convert_numpy({
            'experiment': 'exp_01_rapidity_one_form', 'milestone': 'M15',
            'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4,
            'score': f"{score}/4", 'verdict': verdict}))
