#!/usr/bin/env python3
# pac_lab.py
# A compact lab to test PAC residuals, global balance projection, and emergent lattice reconfiguration.

from __future__ import annotations
import argparse, dataclasses, json, math, os, sys, time
from typing import Dict, List, Tuple
import numpy as np

# Optional SciPy for fast Poisson solve (lattice mode). Fallback to Jacobi if unavailable.
try:
    from scipy.fft import fft2, ifft2
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ---------------------------
# Utils
# ---------------------------

def mk_run_dir(tag: str = "") -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"runs/{ts}{('_' + tag) if tag else ''}"
    os.makedirs(name, exist_ok=True)
    return name

def seed_all(seed: int = 42):
    np.random.seed(seed)

def save_csv(path: str, header: List[str], rows: List[List[float]]):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

def pretty(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=True)


# ---------------------------
# PART A: Graph PAC (discrete)
# ---------------------------

@dataclasses.dataclass
class GraphPAC:
    """
    Discrete PAC on a directed acyclic graph (DAG) or general directed graph.
    Nodes are 0..N-1.
    f: node values (information/energy payloads)
    edges: list of (parent, child)
    alphas: ownership weights for each (parent, child), in [0,1]
    """
    f: np.ndarray                  # shape (N,)
    edges: List[Tuple[int, int]]   # (p, c)
    alphas: Dict[Tuple[int, int], float]

    def N(self) -> int:
        return self.f.shape[0]

    def build_A(self) -> np.ndarray:
        """A v = residuals, where A[v,v]=1 and A[v,u]=-alpha_{v->u} for u in D(v)."""
        N = self.N()
        A = np.eye(N, dtype=float)
        # For each parent, subtract weighted children
        for (p, c) in self.edges:
            A[p, c] -= self.alphas[(p, c)]
        return A

    def residuals(self, f: np.ndarray | None = None) -> np.ndarray:
        """r(v) = f(v) - sum_{u in D(v)} alpha_{v->u} f(u)"""
        if f is None: f = self.f
        N = f.shape[0]
        r = np.copy(f)
        for (p, c) in self.edges:
            r[p] -= self.alphas[(p, c)] * f[c]
        return r

    def project_pinv(self, f: np.ndarray | None = None, rtol: float = 1e-12) -> Tuple[np.ndarray, float]:
        """
        Minimal-norm global correction using Moore–Penrose pseudoinverse.
        Returns (f_corrected, post_residual_norm).
        """
        if f is None: f = self.f
        A = self.build_A()
        r = A @ f
        pinv = np.linalg.pinv(A, rcond=rtol)
        delta = - pinv @ r
        f_corr = f + delta
        post = np.linalg.norm(A @ f_corr)
        return f_corr, float(post)

    def project_gauss_seidel(self, iters: int = 200, tol: float = 1e-9) -> Tuple[np.ndarray, List[float]]:
        """
        Iterative local projection: for each parent p, rescale its children so that
        f(p) = sum alpha_{p->u} f(u). Works best if children do not overlap too much.
        Returns (f_corrected, residual_norm_trace).
        """
        f = self.f.copy()
        N = self.N()
        # Precompute children list per parent
        children: Dict[int, List[int]] = {i: [] for i in range(N)}
        for (p, c) in self.edges:
            children[p].append(c)

        trace = []
        A = self.build_A()
        for _ in range(iters):
            # sweep
            for p in range(N):
                if not children[p]:  # leaf
                    continue
                denom = sum(self.alphas[(p, c)] for c in children[p])
                if denom == 0.0:
                    continue
                target = f[p]
                weighted_sum = sum(self.alphas[(p, c)] * f[c] for c in children[p])
                if weighted_sum == 0.0:
                    # distribute proportionally to alphas
                    base = target / (denom + 1e-12)
                    for c in children[p]:
                        f[c] = base  # scaled by alpha inside the equation next loop
                else:
                    ratio = target / (weighted_sum + 1e-12)
                    for c in children[p]:
                        f[c] *= ratio  # rescale children to meet parent sum
            rn = float(np.linalg.norm(A @ f))
            trace.append(rn)
            if rn < tol:
                break
        return f, trace

    def run_perturb_resolve(self, node: int, eps: float = 1.0, mode: str = "pinv") -> Dict:
        """
        Perturb one node by eps, then re-project to conservation using chosen mode.
        Returns diagnostics dict.
        """
        f0 = self.f.copy()
        self.f[node] += eps
        r_before = self.residuals()
        if mode == "pinv":
            f_corr, post = self.project_pinv()
            traj = None
        else:
            f_corr, traj = self.project_gauss_seidel()
            post = float(np.linalg.norm(self.build_A() @ f_corr))

        r_after = self.residuals(f_corr)
        out = dict(
            mode=mode,
            node=node,
            eps=float(eps),
            pre_residual=float(np.linalg.norm(r_before)),
            post_residual=float(np.linalg.norm(r_after)),
            traj=traj,
            f0=f0.tolist(),
            f_corr=f_corr.tolist()
        )
        # restore original f
        self.f = f0
        return out


def demo_graph_small(seed: int = 7) -> GraphPAC:
    """
    Build a small graph with one shared child to test weighted ownership.
        P(0) -> A(1), B(2)
        A(1) -> X(3), A2(4)
        B(2) -> X(3), B2(5), B3(6)
    Ownership: alpha_{A->X}=0.6, alpha_{B->X}=0.4, others = 1.0 to unique children normalized per parent.
    """
    rng = np.random.default_rng(seed)
    N = 7
    f = rng.uniform(10, 50, size=N).astype(float)
    edges = [(0,1),(0,2),(1,3),(1,4),(2,3),(2,5),(2,6)]

    # Set alphas so for each parent p, sum alpha_{p->u} over its children is 1.0
    alphas = {}
    # P -> A, B
    alphas[(0,1)] = 0.5
    alphas[(0,2)] = 0.5
    # A -> X, A2
    alphas[(1,3)] = 0.6
    alphas[(1,4)] = 0.4
    # B -> X, B2, B3 (normalize to 1.0)
    raw = [0.4, 0.35, 0.25]
    s = sum(raw)
    alphas[(2,3)] = raw[0]/s
    alphas[(2,5)] = raw[1]/s
    alphas[(2,6)] = raw[2]/s

    return GraphPAC(f=f, edges=edges, alphas=alphas)


# ---------------------------
# PART B: Lattice “emergent gravity” demo
# ---------------------------

@dataclasses.dataclass
class LatticeModel:
    """
    2D periodic lattice of a scalar field f.
    Local residual r = f - alpha * (sum of 4 neighbors).
    Solve Poisson: Laplacian(psi) = -r, then update f by f += eta * psi + noise.
    """
    N: int = 128
    alpha: float = 0.25
    eta: float = 0.2
    noise: float = 0.005
    seed: int = 1
    use_fft: bool = True

    def __post_init__(self):
        seed_all(self.seed)
        self.f = np.random.normal(loc=0.0, scale=1.0, size=(self.N, self.N))
        if SCIPY_OK and self.use_fft:
            kx = np.fft.fftfreq(self.N) * 2 * np.pi
            ky = np.fft.fftfreq(self.N) * 2 * np.pi
            self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')
            self.lap_symbol = -(self.KX**2 + self.KY**2)
            self.lap_symbol[0,0] = 1.0  # avoid division by 0 (we'll zero mean)

    def residual(self) -> np.ndarray:
        up = np.roll(self.f, -1, axis=0)
        down = np.roll(self.f, 1, axis=0)
        left = np.roll(self.f, -1, axis=1)
        right = np.roll(self.f, 1, axis=1)
        child = self.alpha * (up + down + left + right)
        return self.f - child

    def solve_poisson(self, rhs: np.ndarray) -> np.ndarray:
        # Laplace psi = rhs, periodic BCs
        if SCIPY_OK and self.use_fft:
            rhs_hat = fft2(rhs)
            psi_hat = rhs_hat.copy()
            psi_hat[0,0] = 0.0
            psi_hat /= self.lap_symbol
            psi = np.real(ifft2(psi_hat))
            return psi
        # Jacobi fallback (slower)
        psi = np.zeros_like(rhs)
        for _ in range(200):
            psi = 0.25 * (np.roll(psi,1,0) + np.roll(psi,-1,0) +
                          np.roll(psi,1,1) + np.roll(psi,-1,1) - rhs)
        return psi

    @staticmethod
    def entropy_proxy(field: np.ndarray, bins: int = 100) -> float:
        hist, _ = np.histogram(field.flatten(), bins=bins, density=True)
        hist = hist + 1e-12
        return float(-np.sum(hist * np.log(hist)) * (1.0 / bins))

    def step(self) -> Dict[str, float]:
        r = self.residual()
        psi = self.solve_poisson(-r)
        self.f += self.eta * psi
        if self.noise > 0:
            self.f += self.noise * np.random.normal(size=self.f.shape)
        # diagnostics
        ent = self.entropy_proxy(self.f)
        rn = float(np.linalg.norm(r))
        thr = np.mean(self.f) + 1.0 * np.std(self.f)
        cluster_frac = float(np.mean(self.f > thr))
        return dict(residual_norm=rn, entropy=ent, cluster_frac=cluster_frac)

    def run(self, steps: int = 200) -> List[Dict[str, float]]:
        logs = []
        for t in range(steps):
            d = self.step()
            d["t"] = t
            logs.append(d)
            if t % 20 == 0:
                print(f"[t={t:03d}] residual={d['residual_norm']:.4e}  entropy={d['entropy']:.4f}  clusters={d['cluster_frac']:.4f}")
        return logs


# ---------------------------
# CLI Entrypoints
# ---------------------------

def run_graph(args):
    seed_all(args.seed)
    g = demo_graph_small(seed=args.seed)
    run_dir = mk_run_dir("graph")
    print("Graph initial f:", np.array2string(g.f, precision=3))

    # Baseline residual
    r0 = g.residuals()
    print("Initial residual norm:", np.linalg.norm(r0))

    # Exact projection
    f_corr, post = g.project_pinv()
    print("PINV post-residual:", post)

    # Iterative projection
    f_it, traj = g.project_gauss_seidel(iters=args.iters, tol=args.tol)
    print("GS post-residual:", np.linalg.norm(g.build_A() @ f_it))

    # Perturb & resolve demo
    report = g.run_perturb_resolve(node=args.perturb_node, eps=args.eps, mode="pinv")
    print("Perturb/resolve (PINV):", pretty({k: report[k] for k in ("node","eps","pre_residual","post_residual")}))

    report2 = g.run_perturb_resolve(node=args.perturb_node, eps=args.eps, mode="gs")
    print("Perturb/resolve (GS):", pretty({k: report2[k] for k in ("node","eps","pre_residual","post_residual")}))

    # Save logs
    rows = []
    for i, r in enumerate(traj):
        rows.append([i, r])
    save_csv(os.path.join(run_dir, "gs_trace.csv"), ["iter","residual_norm"], rows)

    meta = dict(
        seed=args.seed, iters=args.iters, tol=args.tol,
        initial_f=g.f.tolist(),
        pinv_post=post,
        gs_post=float(np.linalg.norm(g.build_A() @ f_it)),
        perturb_pin=report, perturb_gs=report2
    )
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved outputs to", run_dir)


def run_lattice(args):
    model = LatticeModel(N=args.N, alpha=args.alpha, eta=args.eta,
                         noise=args.noise, seed=args.seed, use_fft=not args.no_fft)
    run_dir = mk_run_dir("lattice")
    logs = model.run(steps=args.steps)
    save_csv(os.path.join(run_dir, "lattice_metrics.csv"),
             ["t","residual_norm","entropy","cluster_frac"],
             [[d["t"], d["residual_norm"], d["entropy"], d["cluster_frac"]] for d in logs])
    # Save final field as .npy for quick visualization elsewhere
    np.save(os.path.join(run_dir, "final_field.npy"), model.f)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(model) | {"steps": args.steps}, f, indent=2)
    print("Saved outputs to", run_dir)
    print("Tip: visualize final_field.npy with matplotlib imshow for clustering.")


def main():
    p = argparse.ArgumentParser(description="PAC Lab: Graph projection & Lattice emergence")
    sub = p.add_subparsers(dest="mode", required=True)

    pg = sub.add_parser("graph", help="Discrete PAC on a small graph")
    pg.add_argument("--seed", type=int, default=7)
    pg.add_argument("--iters", type=int, default=400)
    pg.add_argument("--tol", type=float, default=1e-9)
    pg.add_argument("--perturb-node", type=int, default=3, help="Node index to perturb")
    pg.add_argument("--eps", type=float, default=4.0, help="Perturbation amount")
    pg.set_defaults(func=run_graph)

    pl = sub.add_parser("lattice", help="Emergent lattice reconfiguration")
    pl.add_argument("--N", type=int, default=128)
    pl.add_argument("--steps", type=int, default=200)
    pl.add_argument("--alpha", type=float, default=0.25, help="Neighbor weight")
    pl.add_argument("--eta", type=float, default=0.2, help="Reconfiguration rate")
    pl.add_argument("--noise", type=float, default=0.005, help="Thermal/noise term")
    pl.add_argument("--seed", type=int, default=1)
    pl.add_argument("--no-fft", action="store_true", help="Disable FFT Poisson solver (use Jacobi)")
    pl.set_defaults(func=run_lattice)

    args = p.parse_args()
    os.makedirs("runs", exist_ok=True)
    args.func(args)

if __name__ == "__main__":
    main()
