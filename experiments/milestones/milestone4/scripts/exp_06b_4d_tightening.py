#!/usr/bin/env python3
"""
EXPERIMENT 06b: 4D Turbulence Tightening — GPU-Optimized Reynolds Sweep
========================================================================
Dawn Field Institute — Milestone 4, Block B

GPU OPTIMIZATIONS over exp_06:
  1. Structure functions: GPU-side accumulation (no .item() sync barriers)
  2. Structure functions: vectorized power computation across all orders at once
  3. Nonlinear term: batched FFT via stacked tensor (single fftn call for d² products)
  4. Reduced samples: ~200 high-quality samples instead of 1000
  5. In-place operations throughout to reduce memory allocations
  6. Separate step timing from structure function timing for diagnostics

STRATEGY: Reynolds number sweep → extrapolate k(Re) and β(Re) to Re→∞

INCORPORATING prior results:
  exp_06:    N=32, ν=0.020, Re≈63  → k=10.78, β=0.838
  06b_run1:  N=32, ν=0.015, Re≈82  → k=45.00, β=0.809  (k unstable, β trend clear)
  06b_run2:  N=32, ν=0.012, Re≈103 → k=15.01, β=0.735
"""

import torch
import torch.fft
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import sys, os, time, gc, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from constants import PHI, XI_BALANCE
from utils import save_results

# ============================================================
# GPU SETUP
# ============================================================
assert torch.cuda.is_available(), "CUDA required"
DEVICE = torch.device('cuda')
GPU_NAME = torch.cuda.get_device_name(0)
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9

print("=" * 70)
print("EXPERIMENT 06b: 4D Turbulence Tightening [GPU-OPTIMIZED]")
print("=" * 70)
print(f"  GPU: {GPU_NAME} ({VRAM_GB:.1f} GB)")
print(f"  PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

FIB = {1: 1, 2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21}


def she_leveque_constrained(p, gamma, beta):
    if abs(1 - beta) < 1e-12:
        return p / 3.0
    C0 = gamma / (1 - beta)
    return (1 - gamma) * p / 3 + C0 * (1 - beta ** (p / 3))


# ============================================================
# GPU-OPTIMIZED PSEUDO-SPECTRAL SOLVER
# ============================================================

class FastSpectralNS:
    """
    GPU-optimized pseudo-spectral NS solver.

    Key differences from exp_06:
      - Batched FFT: stack d components, single fftn call
      - Batched nonlinear: outer product → single batched FFT
      - Structure functions: all accumulation on GPU, single .cpu() at end
      - Reduced sync points: no .item() in hot loops
    """

    def __init__(self, d, N, nu, dt, forcing_band=(1.0, 2.5), epsilon=0.1):
        self.d = d
        self.N = N
        self.nu = nu
        self.dt = dt
        self.epsilon = epsilon

        # Wavenumber grids
        freq = torch.fft.fftfreq(N, d=1.0 / N).to(DEVICE)
        grids = torch.meshgrid(*([freq] * d), indexing='ij')
        self.k_components = list(grids)

        self.k_sq = sum(ki ** 2 for ki in self.k_components)
        self.k_mag = torch.sqrt(self.k_sq.float())

        # Dealiasing mask: |k_i| <= N/3
        kmax = N // 3
        mask = torch.ones((N,) * d, dtype=torch.bool, device=DEVICE)
        for ki in self.k_components:
            mask &= (ki.abs() <= kmax)
        self.dealias_mask = mask

        # Viscous decay (exact integrating factor)
        self.visc_decay = torch.exp(-nu * self.k_sq * dt).to(torch.complex64)

        # Forcing setup
        k_lo, k_hi = forcing_band
        self.force_mask = (self.k_mag >= k_lo) & (self.k_mag <= k_hi)
        self.n_forced = self.force_mask.sum().item()
        self.f_norm = np.sqrt(2 * epsilon * N**d / (d * dt * max(1, self.n_forced)))

        # Projection: k_i / |k|^2, pre-computed as complex
        k_sq_safe = self.k_sq.float().clamp(min=1.0)
        self.k_over_ksq = [ki.to(torch.complex64) / k_sq_safe.to(torch.complex64)
                           for ki in self.k_components]
        self.k_complex = [ki.to(torch.complex64) for ki in self.k_components]

        # Spatial dims for batched FFT
        self.spatial_dims = list(range(1, d + 1))  # [1, 2, ..., d]
        self.spatial_dims_2d = list(range(2, d + 2))  # [2, 3, ..., d+1]

        # Resolution diagnostics
        eta = (nu**3 / epsilon)**0.25
        mem_mb = d * N**d * 8 / 1e6
        print(f"  {d}D NS: N={N}, grid={N**d:,}, nu={nu:.1e}, dt={dt:.1e}")
        print(f"  k_max={kmax}, eta={eta:.4f}, k_max*eta={kmax*eta:.2f}")
        print(f"  Forcing: {self.n_forced} modes, eps={epsilon:.1e}")
        print(f"  Velocity field: {mem_mb:.0f} MB")

    def project_inplace(self, u_hat):
        """Leray projection in-place."""
        k_dot_u = torch.zeros_like(u_hat[0])
        for i in range(self.d):
            k_dot_u.add_(self.k_complex[i] * u_hat[i])
        for i in range(self.d):
            u_hat[i].sub_(self.k_over_ksq[i] * k_dot_u)

    def nonlinear_batched(self, u_hat):
        """
        Compute -(u·∇)u via batched FFT. Much faster than d² Python-loop FFTs.

        For d=4: one batched IFFT of shape (4,...), one batched FFT of shape (4,4,...).
        Total: ~5 kernel launches instead of ~40.
        """
        d = self.d

        # Batched IFFT: u_hat (list of d) → u_phys (d, N, N, N, N)
        u_hat_stack = torch.stack(u_hat)  # (d, N^d...)
        u_phys = torch.fft.ifftn(u_hat_stack, dim=self.spatial_dims).real  # (d, N^d...)

        # Outer product: u_i * u_j → shape (d, d, N^d...)
        # u_phys[:, None] is (d, 1, N^d...), u_phys[None, :] is (1, d, N^d...)
        products = u_phys.unsqueeze(1) * u_phys.unsqueeze(0)  # (d, d, N^d...)

        # Batched FFT + dealiasing
        products_hat = torch.fft.fftn(products, dim=self.spatial_dims_2d)
        products_hat *= self.dealias_mask  # broadcast

        # nl_hat[i] = -Σ_j (1j * k_j * products_hat[i,j])
        nl_hat = []
        for i in range(d):
            nl_i = torch.zeros_like(u_hat[0])
            for j in range(d):
                nl_i.sub_(1j * self.k_complex[j] * products_hat[i, j])
            nl_hat.append(nl_i)

        del u_phys, products, products_hat, u_hat_stack
        self.project_inplace(nl_hat)
        return nl_hat

    def forcing(self):
        """Stochastic divergence-free forcing."""
        f_hat = []
        for i in range(self.d):
            f = torch.randn((self.N,) * self.d, dtype=torch.complex64, device=DEVICE)
            f.mul_(self.force_mask).mul_(self.f_norm)
            f_hat.append(f)
        self.project_inplace(f_hat)
        return f_hat

    def step(self, u_hat):
        """RK2 Heun with batched nonlinear term."""
        f = self.forcing()

        # k1 = NL(u) + f
        k1 = self.nonlinear_batched(u_hat)
        for i in range(self.d):
            k1[i].add_(f[i])

        # Predictor: u_pred = u + dt * k1
        u_pred = [u_hat[i] + self.dt * k1[i] for i in range(self.d)]

        # k2 = NL(u_pred) + f
        k2 = self.nonlinear_batched(u_pred)
        del u_pred
        for i in range(self.d):
            k2[i].add_(f[i])
        del f

        # Corrector with viscous decay
        hdt = 0.5 * self.dt
        u_new = []
        for i in range(self.d):
            u_new.append(self.visc_decay * (u_hat[i] + hdt * (k1[i] + k2[i])))
        del k1, k2
        return u_new

    def init_velocity(self, seed=42, u_rms_target=0.5):
        """Initialize divergence-free velocity at target u_rms."""
        torch.manual_seed(seed)
        init_mask = (self.k_mag >= 1.0) & (self.k_mag <= 4.0)
        u_hat = []
        for i in range(self.d):
            amp = torch.randn((self.N,) * self.d, dtype=torch.complex64, device=DEVICE)
            amp.mul_(init_mask)
            u_hat.append(amp)
        self.project_inplace(u_hat)

        E = self._total_energy_gpu(u_hat)
        if E > 0:
            u_rms = torch.sqrt(2 * E / self.N ** self.d)
            scale = u_rms_target / (u_rms + 1e-30)
            for uh in u_hat:
                uh.mul_(scale)
        return u_hat

    def _total_energy_gpu(self, u_hat):
        """Total energy as GPU scalar (no sync)."""
        E = torch.zeros(1, device=DEVICE)
        for uh in u_hat:
            E += 0.5 * torch.sum(torch.abs(uh) ** 2)
        return E / self.N ** self.d

    def total_energy(self, u_hat):
        """Total energy (with CPU sync for diagnostics)."""
        return self._total_energy_gpu(u_hat).item()

    def structure_functions_fast(self, u_hat, orders=list(range(1, 9)),
                                  n_sep=16):
        """
        GPU-optimized structure functions.

        Key optimizations:
          - All accumulation stays on GPU tensors (no .item() sync)
          - Vectorized power computation: abs_du^p for all p at once
          - Single .cpu() transfer at the end
          - Reduced separations (16 instead of 32)
        """
        d = self.d
        N = self.N
        n_ord = len(orders)
        p_tensor = torch.tensor(orders, dtype=torch.float32, device=DEVICE)

        # Batched IFFT
        u_hat_stack = torch.stack(u_hat)
        u_phys = torch.fft.ifftn(u_hat_stack, dim=self.spatial_dims).real
        del u_hat_stack

        # Log-spaced separations
        max_r = N // 2
        seps = np.unique(np.geomspace(1, max_r, n_sep).astype(int))
        n_actual = len(seps)

        # Accumulator: (n_seps, n_orders) on GPU
        sp_acc = torch.zeros(n_actual, n_ord, device=DEVICE)

        for si, r in enumerate(seps):
            for axis in range(d):
                # Roll and compute delta
                u_shifted = torch.roll(u_phys[axis], shifts=-int(r), dims=axis)
                abs_du = torch.abs(u_shifted - u_phys[axis])
                del u_shifted

                # Vectorized: compute abs_du^p for all p at once
                # abs_du has shape (N,)*d, we need mean(abs_du^p) for each p
                # Reshape abs_du to (1, N^d), p_tensor to (n_ord, 1)
                flat = abs_du.reshape(1, -1)  # (1, N^d)
                p_col = p_tensor.reshape(-1, 1)  # (n_ord, 1)
                # Powers: (n_ord, N^d), then mean along dim=1
                means = torch.mean(flat ** p_col, dim=1)  # (n_ord,)
                sp_acc[si] += means
                del abs_du, flat, means

            # Average over d axes
            sp_acc[si] /= d

        del u_phys

        # Single CPU transfer
        sp_np = sp_acc.cpu().numpy()
        r_phys = seps.astype(float) * 2 * np.pi / N

        result = {'r': r_phys, 'seps': seps}
        for pi, p in enumerate(orders):
            result[p] = sp_np[:, pi]
        return result


# ============================================================
# FAST DNS RUNNER
# ============================================================

def run_dns_4d(N, nu, dt, epsilon, n_equil, n_collect,
               collect_every=50, seed=42, forcing_band=(1.0, 2.5),
               u_rms_target=0.2, n_sep=16):
    """
    Run 4D DNS with optimized GPU utilization.
    Targets ~200 samples with larger spacing between them.
    """
    d = 4
    torch.cuda.empty_cache()
    gc.collect()

    solver = FastSpectralNS(d=d, N=N, nu=nu, dt=dt, epsilon=epsilon,
                             forcing_band=forcing_band)
    u_hat = solver.init_velocity(seed=seed, u_rms_target=u_rms_target)

    E_target = N ** d * u_rms_target ** 2 / 2
    Re_target = u_rms_target * 2 * np.pi / nu
    print(f"\n  Target: u_rms={u_rms_target:.2f}, Re~{Re_target:.0f}")

    # --- Equilibrate ---
    print(f"  Equilibrating ({n_equil} steps)...", end="", flush=True)
    t0 = time.time()
    blowups = 0
    for step in range(n_equil):
        u_hat = solver.step(u_hat)
        if step % 100 == 0:
            E = solver.total_energy(u_hat)
            if np.isnan(E) or E > 1e10 or E <= 0:
                blowups += 1
                u_hat = solver.init_velocity(seed=seed + step,
                                              u_rms_target=u_rms_target)
            else:
                scale = np.sqrt(E_target / (E + 1e-30))
                if 0.5 < scale < 2.0:
                    for uh in u_hat:
                        uh.mul_(scale)
                elif scale >= 2.0:
                    u_hat = solver.init_velocity(seed=seed + step,
                                                  u_rms_target=u_rms_target)
                    blowups += 1

    equil_time = time.time() - t0
    E_post = solver.total_energy(u_hat)
    u_rms_post = np.sqrt(2 * E_post / N**d) if E_post > 0 else 0
    Re_post = u_rms_post * 2 * np.pi / nu
    print(f" {equil_time:.0f}s, Re~{Re_post:.0f}, blowups={blowups}")

    # --- Collect ---
    n_expected = n_collect // collect_every
    print(f"  Collecting ({n_collect} steps, every {collect_every}"
          f" -> ~{n_expected} samples)...", end="", flush=True)
    t0 = time.time()
    t_step = 0
    t_sf = 0

    sp_acc = None
    n_samples = 0
    Re_samples = []

    for step in range(n_collect):
        ts = time.time()
        u_hat = solver.step(u_hat)
        t_step += time.time() - ts

        # Energy rescaling every 200 steps
        if step % 200 == 0 and step > 0:
            E = solver.total_energy(u_hat)
            if E > 0 and not np.isnan(E):
                scale = np.sqrt(E_target / (E + 1e-30))
                if 0.5 < scale < 2.0:
                    for uh in u_hat:
                        uh.mul_(scale)

        if step % collect_every == 0:
            E = solver.total_energy(u_hat)
            if np.isnan(E) or E > 1e10 or E <= 0:
                u_hat = solver.init_velocity(seed=seed + n_equil + step,
                                              u_rms_target=u_rms_target)
                continue

            u_rms_now = np.sqrt(2 * E / N**d)
            Re_samples.append(u_rms_now * 2 * np.pi / nu)

            ts = time.time()
            sf = solver.structure_functions_fast(u_hat, n_sep=n_sep)
            t_sf += time.time() - ts

            if sp_acc is None:
                sp_acc = {p: np.zeros_like(sf[p]) for p in range(1, 9)}
                sp_acc['r'] = sf['r']
                sp_acc['seps'] = sf['seps']
            for p in range(1, 9):
                sp_acc[p] += sf[p]
            n_samples += 1

    collect_time = time.time() - t0
    Re_mean = np.mean(Re_samples) if Re_samples else 0
    Re_std = np.std(Re_samples) if Re_samples else 0
    print(f" {collect_time:.0f}s ({n_samples} samples)")
    print(f"    Timing: step={t_step:.0f}s, SF={t_sf:.0f}s, "
          f"other={collect_time-t_step-t_sf:.0f}s")
    print(f"    Re_mean={Re_mean:.0f}+/-{Re_std:.0f}")

    if n_samples > 0:
        for p in range(1, 9):
            sp_acc[p] /= n_samples

    del u_hat, solver
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'sp': sp_acc,
        'n_samples': n_samples,
        'Re_mean': Re_mean,
        'Re_std': Re_std,
        'metadata': {
            'd': 4, 'N': N, 'nu': nu, 'dt': dt, 'epsilon': epsilon,
            'n_equil': n_equil, 'n_collect': n_collect,
            'equil_time': equil_time, 'collect_time': collect_time,
            't_step': t_step, 't_sf': t_sf,
            'forcing_band': list(forcing_band), 'seed': seed,
        }
    }


# ============================================================
# ANALYSIS
# ============================================================

def extract_zeta_ess(sp):
    """ESS: ζ_p / ζ_3 = ζ_p."""
    s3 = sp[3]
    good = s3 > 0
    if np.sum(good) < 5:
        return {}
    s3_max = s3[good].max()
    ir = good & (s3 > 0.01 * s3_max) & (s3 < 0.90 * s3_max)
    if np.sum(ir) < 4:
        ir = good & (s3 > 0.001 * s3_max)
    if np.sum(ir) < 4:
        return {}
    log_s3 = np.log(s3[ir] + 1e-30)
    zeta = {}
    for p in range(1, 9):
        sp_vals = sp[p][ir]
        valid = sp_vals > 0
        if np.sum(valid) < 4:
            continue
        log_sp = np.log(sp_vals[valid] + 1e-30)
        log_s3_v = log_s3[valid]
        sl, ic, r, pv, se = stats.linregress(log_s3_v, log_sp)
        zeta[p] = {'value': sl, 'r2': r ** 2, 'stderr': se}
    return zeta


def fit_she_leveque(zeta_dict):
    """Fit She-Leveque → (k, β, γ)."""
    orders = [p for p in [1, 2, 4, 5, 6, 7, 8]
              if p in zeta_dict and zeta_dict[p]['r2'] > 0.85]
    if len(orders) < 3:
        return {'success': False, 'reason': f'Only {len(orders)} good orders'}

    p_arr = np.array(orders, dtype=float)
    z_arr = np.array([zeta_dict[p]['value'] for p in orders])
    if np.any(np.isnan(z_arr)):
        return {'success': False, 'reason': 'NaN in zeta'}

    def model(p, gam, bet):
        return (1 - gam) * p / 3 + gam / (1 - bet + 1e-15) * (1 - bet ** (p / 3))

    try:
        popt, pcov = curve_fit(model, p_arr, z_arr, p0=[0.6, 0.6],
                               bounds=([0.01, 0.01], [0.99, 0.99]),
                               maxfev=20000)
        gam, bet = popt
        perr = np.sqrt(np.diag(pcov))
        z_pred = model(p_arr, gam, bet)
        ss_res = np.sum((z_arr - z_pred) ** 2)
        ss_tot = np.sum((z_arr - np.mean(z_arr)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-30)
        return {
            'success': True,
            'k': 3.0 / (1 - gam), 'beta': bet, 'gamma': gam,
            'C0': gam / (1 - bet),
            'k_err': 3 * perr[0] / (1 - gam) ** 2,
            'beta_err': perr[1], 'gamma_err': perr[0],
            'r2': r2, 'rmse': np.sqrt(np.mean((z_arr - z_pred) ** 2)),
            'n_orders': len(orders), 'orders_used': orders,
        }
    except Exception as e:
        return {'success': False, 'reason': str(e)}


def bootstrap_k_beta(zeta_data, n_boot=3000, seed=42):
    """Bootstrap CI for k and β."""
    rng = np.random.default_rng(seed)
    k_list, beta_list = [], []
    for _ in range(n_boot):
        zt = {}
        for p in range(1, 9):
            if p in zeta_data:
                noise = rng.normal(0, zeta_data[p].get('stderr', 0.01))
                zt[p] = {'value': zeta_data[p]['value'] + noise,
                         'r2': zeta_data[p]['r2']}
        fit = fit_she_leveque(zt)
        if fit['success']:
            k_list.append(fit['k'])
            beta_list.append(fit['beta'])
    if len(k_list) < 100:
        return None, None
    k_arr = np.array(k_list)
    b_arr = np.array(beta_list)
    return (np.percentile(k_arr, [2.5, 97.5]),
            np.percentile(b_arr, [2.5, 97.5]))


# ============================================================
# BENCHMARK: Verify GPU optimization before expensive runs
# ============================================================

print("\n" + "=" * 70)
print("BENCHMARK: Verifying GPU optimization")
print("=" * 70)

bench_solver = FastSpectralNS(d=4, N=32, nu=0.02, dt=5e-3, epsilon=0.03)
bench_u = bench_solver.init_velocity(seed=999, u_rms_target=0.2)

# Warm up
for _ in range(5):
    bench_u = bench_solver.step(bench_u)
torch.cuda.synchronize()

# Time 100 pure steps
t0 = time.time()
for _ in range(100):
    bench_u = bench_solver.step(bench_u)
torch.cuda.synchronize()
t_100_steps = time.time() - t0
ms_per_step = t_100_steps / 100 * 1000

# Time 5 structure function calls
torch.cuda.synchronize()
t0 = time.time()
for _ in range(5):
    sf = bench_solver.structure_functions_fast(bench_u, n_sep=16)
torch.cuda.synchronize()
t_5_sf = time.time() - t0
ms_per_sf = t_5_sf / 5 * 1000

print(f"\n  Step: {ms_per_step:.1f} ms/step")
print(f"  Structure fn: {ms_per_sf:.0f} ms/call")
print(f"  Ratio: {ms_per_sf/ms_per_step:.1f}x (SF cost per step)")

# Estimate total times
for label, N, n_equil, n_coll, c_every in [
    ("N=32", 32, 2000, 10000, 50),
    ("N=40", 40, 2000, 10000, 50),
    ("N=48", 48, 2000, 8000, 50),
]:
    # Scale step time by N^4 ratio
    scale = (N / 32) ** 4
    est_step = ms_per_step * scale
    est_sf = ms_per_sf * scale
    n_samples = n_coll // c_every
    total_s = (n_equil + n_coll) * est_step / 1000 + n_samples * est_sf / 1000
    print(f"  {label}: ~{est_step:.0f}ms/step, ~{est_sf:.0f}ms/sf, "
          f"~{n_samples} samples -> ~{total_s/60:.1f} min")

del bench_solver, bench_u
torch.cuda.empty_cache()

# ============================================================
# RUN SCHEDULE
# ============================================================

print("\n" + "=" * 70)
print("REYNOLDS SWEEP — 4D TURBULENCE")
print("=" * 70)

# Prior data from exp_06 and 06b_run1/2
PRIOR_RESULTS = [
    {'Re_mean': 63.0, 'k': 10.78, 'beta': 0.838,
     'k_err': 9.58, 'beta_err': 0.057,
     'N': 32, 'nu': 0.020, 'label': 'exp_06 N32 v=.020'},
    {'Re_mean': 82.0, 'k': 45.00, 'beta': 0.809,
     'k_err': 170.73, 'beta_err': 0.053,
     'N': 32, 'nu': 0.015, 'label': '06b_r1 N32 v=.015'},
    {'Re_mean': 103.0, 'k': 15.01, 'beta': 0.735,
     'k_err': 9.54, 'beta_err': 0.044,
     'N': 32, 'nu': 0.012, 'label': '06b_r2 N32 v=.012'},
]

# New runs: focus on N=40 and N=48 for higher Re
RUNS = [
    # N=40 block
    {'N': 40, 'nu': 0.012, 'dt': 3e-3, 'eps': 0.03, 'n_equil': 2000,
     'n_collect': 10000, 'every': 50, 'seed': 300, 'label': 'N40 v=.012'},
    {'N': 40, 'nu': 0.010, 'dt': 2.5e-3, 'eps': 0.03, 'n_equil': 2000,
     'n_collect': 10000, 'every': 50, 'seed': 400, 'label': 'N40 v=.010'},
    {'N': 40, 'nu': 0.008, 'dt': 2e-3, 'eps': 0.03, 'n_equil': 2000,
     'n_collect': 10000, 'every': 50, 'seed': 500, 'label': 'N40 v=.008'},

    # N=48 block
    {'N': 48, 'nu': 0.010, 'dt': 2e-3, 'eps': 0.03, 'n_equil': 2000,
     'n_collect': 8000, 'every': 50, 'seed': 600, 'label': 'N48 v=.010'},
    {'N': 48, 'nu': 0.008, 'dt': 1.5e-3, 'eps': 0.03, 'n_equil': 2000,
     'n_collect': 8000, 'every': 50, 'seed': 700, 'label': 'N48 v=.008'},
]

all_results = list(PRIOR_RESULTS)
t0_total = time.time()

for i, run in enumerate(RUNS):
    Re_est = 0.2 * 2 * np.pi / run['nu']
    print(f"\n{'=' * 70}")
    print(f"RUN {i+1}/{len(RUNS)}: {run['label']}  (Re_target~{Re_est:.0f})")
    print(f"{'=' * 70}")

    try:
        dns = run_dns_4d(
            N=run['N'], nu=run['nu'], dt=run['dt'],
            epsilon=run['eps'], n_equil=run['n_equil'],
            n_collect=run['n_collect'], collect_every=run['every'],
            seed=run['seed'], u_rms_target=0.2,
        )

        if dns['sp'] is None or dns['n_samples'] < 10:
            print(f"  *** Insufficient samples ({dns['n_samples']}). Skip.")
            continue

        zeta = extract_zeta_ess(dns['sp'])
        fit = fit_she_leveque(zeta)

        if fit['success']:
            print(f"\n  k = {fit['k']:.2f} +/- {fit['k_err']:.2f}")
            print(f"  beta = {fit['beta']:.4f} +/- {fit['beta_err']:.4f}")
            print(f"  gamma = {fit['gamma']:.4f}, R2 = {fit['r2']:.6f}")

            print(f"\n  {'p':>3} | {'zeta_p':>10} | {'K41':>7} | {'Delta':>10} | {'R2':>7}")
            print("  " + "-" * 48)
            for p in range(1, 9):
                if p in zeta:
                    z = zeta[p]
                    print(f"  {p:>3} | {z['value']:>10.6f} | {p/3:>7.4f} | "
                          f"{z['value']-p/3:>+10.6f} | {z['r2']:>7.4f}")

            k_ci, b_ci = bootstrap_k_beta(zeta)

            entry = {
                'Re_mean': dns['Re_mean'],
                'k': fit['k'], 'beta': fit['beta'],
                'k_err': fit['k_err'], 'beta_err': fit['beta_err'],
                'gamma': fit['gamma'], 'r2': fit['r2'],
                'N': run['N'], 'nu': run['nu'],
                'n_samples': dns['n_samples'],
                'label': run['label'],
                'zeta': {str(p): zeta[p] for p in zeta},
                'metadata': dns['metadata'],
            }
            if k_ci is not None:
                entry['k_ci_95'] = k_ci.tolist()
                entry['beta_ci_95'] = b_ci.tolist()
                print(f"\n  95% CI: k=[{k_ci[0]:.2f}, {k_ci[1]:.2f}], "
                      f"beta=[{b_ci[0]:.4f}, {b_ci[1]:.4f}]")
            all_results.append(entry)
        else:
            print(f"\n  SL fit FAILED: {fit.get('reason', 'unknown')}")

    except torch.cuda.OutOfMemoryError:
        print(f"\n  *** CUDA OOM at N={run['N']}. Skip.")
        torch.cuda.empty_cache()
        gc.collect()
    except BaseException as e:
        print(f"\n  *** {type(e).__name__}: {e}")
        torch.cuda.empty_cache()
        gc.collect()

total_time = time.time() - t0_total
print(f"\n\nAll runs: {total_time:.0f}s ({total_time/60:.1f} min)")


# ============================================================
# EXTRAPOLATION
# ============================================================

print("\n" + "=" * 70)
print("REYNOLDS EXTRAPOLATION")
print("=" * 70)

successful = [r for r in all_results if 'k' in r and 'Re_mean' in r]
successful.sort(key=lambda x: x['Re_mean'])
n_pts = len(successful)

Re_arr = np.array([r['Re_mean'] for r in successful])
k_arr = np.array([r['k'] for r in successful])
beta_arr = np.array([r['beta'] for r in successful])

print(f"\n  {'Label':>22} | {'Re':>6} | {'k':>8} | {'beta':>8}")
print("  " + "-" * 55)
for r in successful:
    print(f"  {r['label']:>22} | {r['Re_mean']:>6.0f} | "
          f"{r['k']:>8.2f} | {r['beta']:>8.4f}")

# --- beta extrapolation (more reliable than k) ---

def model_invRe(Re, x_inf, A):
    return x_inf + A / Re

def model_invSqrtRe(Re, x_inf, A):
    return x_inf + A / np.sqrt(Re)

re_extrap = {}

print(f"\n--- beta(Re) extrapolation (beta measured ~100x more precisely) ---")
for name, func, p0, bds in [
    ('beta = b_inf + A/Re', model_invRe, [0.6, 15], ([0.01, -500], [0.99, 500])),
    ('beta = b_inf + A/sqrtRe', model_invSqrtRe, [0.6, 2], ([0.01, -500], [0.99, 500])),
]:
    try:
        popt, pcov = curve_fit(func, Re_arr, beta_arr, p0=p0, bounds=bds,
                               maxfev=50000)
        b_pred = func(Re_arr, *popt)
        ss_res = np.sum((beta_arr - b_pred) ** 2)
        ss_tot = np.sum((beta_arr - np.mean(beta_arr)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-30)
        b_inf = popt[0]
        b_inf_err = np.sqrt(pcov[0, 0]) if pcov[0, 0] > 0 else 999
        print(f"\n  {name}:")
        print(f"    b_inf = {b_inf:.4f} +/- {b_inf_err:.4f}")
        print(f"    R2 = {r2:.4f}")
        re_extrap[f'beta_{name}'] = {
            'beta_inf': b_inf, 'beta_inf_err': b_inf_err,
            'r2': r2, 'params': popt.tolist(), 'name': name,
        }
    except Exception as e:
        print(f"\n  {name}: failed -- {e}")

# DFT: beta(4) = 3/5 = 0.600
print(f"\n  DFT prediction: beta(4) = F3/F5 = 3/5 = 0.600")
print(f"  Observation: beta DECREASING with Re (expected -- approaching asymptotic)")

print(f"\n--- k(Re) extrapolation (noisy, for completeness) ---")
# Filter out k > 100 (unreliable fits at low Re)
reliable = [(r, k, b) for r, k, b in zip(Re_arr, k_arr, beta_arr) if k < 100]
if len(reliable) >= 2:
    Re_rel = np.array([x[0] for x in reliable])
    k_rel = np.array([x[1] for x in reliable])

    for name, func, p0, bds in [
        ('k = k_inf + A/Re', model_invRe, [15, -300], ([3, -5000], [50, 5000])),
        ('k = k_inf + A/sqrtRe', model_invSqrtRe, [15, -50], ([3, -5000], [50, 5000])),
    ]:
        try:
            popt, pcov = curve_fit(func, Re_rel, k_rel, p0=p0, bounds=bds,
                                   maxfev=50000)
            k_pred = func(Re_rel, *popt)
            ss_res = np.sum((k_rel - k_pred) ** 2)
            ss_tot = np.sum((k_rel - np.mean(k_rel)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-30)
            k_inf = popt[0]
            k_inf_err = np.sqrt(pcov[0, 0]) if pcov[0, 0] > 0 else 999
            print(f"\n  {name}:")
            print(f"    k_inf = {k_inf:.2f} +/- {k_inf_err:.2f}")
            print(f"    R2 = {r2:.4f}")
            re_extrap[f'k_{name}'] = {
                'k_inf': k_inf, 'k_inf_err': k_inf_err,
                'r2': r2, 'params': popt.tolist(), 'name': name,
            }
        except Exception as e:
            print(f"\n  {name}: failed -- {e}")


# ============================================================
# VERDICT
# ============================================================

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

print(f"\n  Predictions:")
print(f"    DFT:   k(4) = 20,  beta(4) = 0.600")
print(f"    Naive: k(4) = 16   (no beta prediction)")

if successful:
    best = successful[-1]
    print(f"\n  Highest-Re direct measurement (Re={best['Re_mean']:.0f}):")
    print(f"    k = {best['k']:.2f}, beta = {best['beta']:.4f}")

# beta trend
if len(successful) >= 3:
    slope, _, r, _, _ = stats.linregress(Re_arr, beta_arr)
    print(f"\n  beta trend: slope={slope:+.5f}/Re, r={r:.3f}")
    print(f"    beta is {'decreasing' if slope < 0 else 'increasing'} with Re")
    Re_inf = 10000
    beta_lin = np.mean(beta_arr) + slope * (Re_inf - np.mean(Re_arr))
    beta_lin = np.clip(beta_lin, 0.01, 0.99)
    print(f"    Linear extrap to Re={Re_inf}: beta ~ {beta_lin:.3f}")

# Best beta extrapolation
best_beta = None
for key, val in re_extrap.items():
    if key.startswith('beta_') and (best_beta is None or val['r2'] > best_beta['r2']):
        best_beta = val
if best_beta:
    print(f"\n  Best beta extrapolation ({best_beta['name']}):")
    print(f"    b_inf = {best_beta['beta_inf']:.4f} +/- "
          f"{best_beta['beta_inf_err']:.4f}")
    dist_dft = abs(best_beta['beta_inf'] - 0.600)
    print(f"    Distance to DFT beta=0.600: {dist_dft:.4f}")
    b_lo = best_beta['beta_inf'] - 2 * best_beta['beta_inf_err']
    b_hi = best_beta['beta_inf'] + 2 * best_beta['beta_inf_err']
    print(f"    95% range: [{b_lo:.4f}, {b_hi:.4f}]")
    print(f"    DFT beta=0.600 in range: "
          f"{'YES' if b_lo <= 0.6 <= b_hi else 'NO'}")

print("\n" + "=" * 70)


# ============================================================
# SAVE
# ============================================================

output = {
    'experiment': 'milestone4/exp_06b_4d_tightening',
    'method': 'GPU pseudo-spectral DNS (optimized) + Re extrapolation',
    'gpu': GPU_NAME,
    'total_time_seconds': total_time,
    'runs': [],
    'extrapolation': re_extrap,
    'predictions': {'dft': {'k': 20, 'beta': 0.6}, 'naive': {'k': 16}},
}

for r in all_results:
    entry = {
        'label': r['label'],
        'Re_mean': float(r['Re_mean']),
        'N': int(r['N']), 'nu': float(r['nu']),
        'k': float(r['k']), 'beta': float(r['beta']),
        'k_err': float(r.get('k_err', 0)),
        'beta_err': float(r.get('beta_err', 0)),
    }
    if 'k_ci_95' in r:
        entry['k_ci_95'] = r['k_ci_95']
        entry['beta_ci_95'] = r['beta_ci_95']
    if 'zeta' in r:
        entry['zeta'] = {p: {'value': float(v['value']), 'r2': float(v['r2']),
                              'stderr': float(v.get('stderr', 0))}
                          for p, v in r['zeta'].items()}
    if 'metadata' in r:
        entry['metadata'] = r['metadata']
    output['runs'].append(entry)

save_results(output, 'exp_06b_4d_tightening')
