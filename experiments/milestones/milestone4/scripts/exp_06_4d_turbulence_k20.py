#!/usr/bin/env python3
"""
EXPERIMENT 06: 4D Turbulence — Testing k(4) = 20
============================================================
Dawn Field Institute — Milestone 4, Block B

GPU-ACCELERATED (PyTorch CUDA) pseudo-spectral Navier-Stokes DNS
in 2D, 3D, and 4D to directly measure She-Leveque intermittency
exponents and test the DFT dimensional prediction.

PREDICTION (PACSeries Paper 5 §6.3):
    k(d) = d × F_{d+1}  where F_n is Fibonacci.
        k(2) = 4,  β(2) = 1/2 = 0.500   [verified]
        k(3) = 9,  β(3) = 2/3 = 0.667   [verified]
        k(4) = 20, β(4) = 3/5 = 0.600   [PREDICTION]
    Competing: k = d²  → k(4) = 16.
    Both agree for d ≤ 3. DIVERGE at d = 4: 20 vs 16.

METHOD:
    Pseudo-spectral Navier-Stokes on periodic box [0,2π]^d.
    - GPU-accelerated via PyTorch CUDA (RTX 3070 Ti).
    - Exponential integrator for viscous term (exact, unconditionally stable).
    - RK2 (Heun) for nonlinear + forcing.
    - 2/3 dealiasing rule.
    - Stochastic large-scale forcing (shells |k| ∈ [1,2]).
    - Structure functions via ESS (Extended Self-Similarity) for
      robust ζ_p/ζ_3 extraction even at modest Reynolds numbers.

    Pipeline: Run d-dim DNS → extract S_p(k) → ESS → fit She-Leveque → (k, β)
    Calibrate on d=2 (k=4) and d=3 (k=9), then measure d=4.

FALSIFICATION:
    If d=2 and d=3 calibrations fail → method broken.
    If d=4 gives k ≈ 16 → naive d² wins, DFT formula falsified.
    If d=4 gives k ≈ 20 → DFT formula k = d × F_{d+1} validated.
"""

import torch
import torch.fft
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import sys, os, time, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from constants import PHI, XI_BALANCE
from utils import save_results, bootstrap_ci

# ============================================================
# GPU SETUP
# ============================================================
assert torch.cuda.is_available(), "CUDA required for this experiment"
DEVICE = torch.device('cuda')
GPU_NAME = torch.cuda.get_device_name(0)
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9

print("=" * 70)
print("EXPERIMENT 06: 4D Turbulence — Testing k(4) = 20")
print("Dawn Field Institute — Milestone 4  [GPU-ACCELERATED]")
print("=" * 70)
print(f"  GPU: {GPU_NAME}")
print(f"  VRAM: {VRAM_GB:.1f} GB")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.version.cuda}")

FIB = {1: 1, 2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21}


# ============================================================
# PART 1: ANALYTICAL PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("PART 1: Analytical Predictions")
print("=" * 70)


def she_leveque_constrained(p, gamma, beta):
    """ζ_p = (1-γ)p/3 + γ/(1-β) × (1-β^{p/3}). Automatically ζ₃=1."""
    if abs(1 - beta) < 1e-12:
        return p / 3.0
    C0 = gamma / (1 - beta)
    return (1 - gamma) * p / 3 + C0 * (1 - beta ** (p / 3))


print("\n--- DFT: k(d) = d × F_{d+1} ---")
print(f"{'d':>3} | {'k':>4} | {'beta':>8} | {'gamma':>8} | {'C0':>8}")
print("-" * 45)
predictions = {}
for d in [2, 3, 4]:
    k = d * FIB[d + 1]
    beta = FIB[d] / FIB[d + 1]
    gamma = 1 - 3.0 / k
    C0 = gamma / (1 - beta)
    print(f"{d:>3} | {k:>4} | {beta:>8.4f} | {gamma:>8.4f} | {C0:>8.4f}")
    predictions[d] = {'k': k, 'beta': beta, 'gamma': gamma, 'C0': C0}

print(f"\n  d=4 divergence: DFT k=20  vs  naive k=d^2=16")

print(f"\n--- zeta_p predictions for d=4 ---")
print(f"{'p':>3} | {'DFT k=20':>10} | {'Naive k=16':>10} | {'Diff':>8}")
print("-" * 40)
gamma_20 = 1 - 3.0 / 20
gamma_16 = 1 - 3.0 / 16
beta_4d = 3 / 5
for p in range(1, 9):
    z20 = she_leveque_constrained(p, gamma_20, beta_4d)
    z16 = she_leveque_constrained(p, gamma_16, beta_4d)
    print(f"{p:>3} | {z20:>10.6f} | {z16:>10.6f} | {z20 - z16:>+8.6f}")


# ============================================================
# PART 2: GPU PSEUDO-SPECTRAL NAVIER-STOKES SOLVER
# ============================================================

print("\n\n" + "=" * 70)
print("PART 2: GPU Pseudo-Spectral Navier-Stokes")
print("=" * 70)


class SpectralNS:
    """
    Pseudo-spectral Navier-Stokes solver in d dimensions on [0,2pi]^d.

    Uses:
      - Exponential integrator for exact viscous decay (unconditionally stable)
      - RK2 (Heun) for nonlinear advection + forcing
      - 2/3 dealiasing via spectral truncation
      - Stochastic band-limited forcing in low wavenumbers
    """

    def __init__(self, d, N, nu, dt, forcing_band=(1.0, 2.5), epsilon=0.1,
                 device=DEVICE):
        self.d = d
        self.N = N
        self.nu = nu
        self.dt = dt
        self.epsilon = epsilon
        self.forcing_band = forcing_band
        self.device = device

        # Wavenumber grids: k_i in {0, 1, ..., N/2, -N/2+1, ..., -1}
        freq = torch.fft.fftfreq(N, d=1.0 / N).to(device)  # integer wavenumbers
        # Build d-dimensional wavenumber mesh
        grids = torch.meshgrid(*([freq] * d), indexing='ij')
        self.k_components = list(grids)  # [k_x, k_y, ...] each shape (N,)*d

        # |k|^2 for Laplacian
        self.k_sq = sum(ki ** 2 for ki in self.k_components)

        # |k| for shell binning
        self.k_mag = torch.sqrt(self.k_sq.float())

        # Dealiasing mask: keep |k_i| <= N/3 in each direction
        kmax = N // 3
        mask = torch.ones((N,) * d, dtype=torch.bool, device=device)
        for ki in self.k_components:
            mask = mask & (ki.abs() <= kmax)
        self.dealias_mask = mask

        # Viscous decay factor per timestep (exact integrating factor)
        self.visc_decay = torch.exp(-nu * self.k_sq * dt).to(torch.complex64)

        # Forcing mask: shells with |k| in forcing_band
        k_lo, k_hi = forcing_band
        self.force_mask = (self.k_mag >= k_lo) & (self.k_mag <= k_hi)
        self.n_forced = self.force_mask.sum().item()

        # Projection: P_{ij}(k) = delta_ij - k_i k_j / |k|^2
        self.k_sq_safe = self.k_sq.clone().float()
        self.k_sq_safe[self.k_sq_safe == 0] = 1.0  # avoid /0 at k=0

        # Forcing amplitude: compute Fourier norm for target energy injection rate
        # epsilon ≈ d * n_f * dt * f_norm^2 / (2 * N^d)
        self.f_norm = np.sqrt(2 * epsilon * N**d / (d * dt * max(1, self.n_forced)))

        print(f"  {d}D NS: N={N}, grid={N**d}, nu={nu:.1e}, dt={dt:.1e}")
        print(f"  Dealias: |k_i|<={kmax}, forcing |k| in {forcing_band}, "
              f"{self.n_forced} forced modes")
        print(f"  epsilon={epsilon:.1e}, f_norm={self.f_norm:.1f}")
        mem_mb = d * N**d * 8 / 1e6  # complex64 = 8 bytes per element
        print(f"  Velocity field: {d} x {N}^{d} complex64 = {mem_mb:.0f} MB")

    def project(self, u_hat):
        """Leray projection: remove compressible part. u_i - k_i(k.u)/|k|^2"""
        k_dot_u = sum(self.k_components[i].to(torch.complex64) * u_hat[i]
                       for i in range(self.d))
        result = []
        for i in range(self.d):
            result.append(
                u_hat[i] - self.k_components[i].to(torch.complex64)
                * k_dot_u / self.k_sq_safe.to(torch.complex64)
            )
        return result

    def nonlinear(self, u_hat):
        """
        Compute -(u . grad)u in Fourier space via pseudo-spectral method.
        Returns hat of the nonlinear term (dealiased, projected).
        """
        # Transform velocity to physical space
        u_phys = [torch.fft.ifftn(uh).real for uh in u_hat]

        # Compute d_j(u_i u_j) = i k_j FT(u_i u_j)  (with minus for advection)
        nl_hat = [torch.zeros_like(u_hat[0]) for _ in range(self.d)]

        for j in range(self.d):
            for i in range(self.d):
                prod = u_phys[i] * u_phys[j]
                prod_hat = torch.fft.fftn(prod)
                prod_hat = prod_hat * self.dealias_mask
                kj = self.k_components[j].to(torch.complex64)
                nl_hat[i] = nl_hat[i] - 1j * kj * prod_hat

        nl_hat = self.project(nl_hat)
        return nl_hat

    def forcing(self):
        """Stochastic divergence-free forcing in the low-wavenumber band."""
        f_hat = []
        for i in range(self.d):
            phase = torch.randn((self.N,) * self.d, dtype=torch.complex64,
                                device=self.device)
            fi = self.f_norm * phase * self.force_mask
            f_hat.append(fi)
        f_hat = self.project(f_hat)
        return f_hat

    def step(self, u_hat):
        """
        One timestep: RK2 (Heun) for nonlinear + forcing, exact viscous decay.
        u_new = visc * [u + dt/2 * (NL1 + NL2)]
        """
        f = self.forcing()

        # Stage 1
        nl1 = self.nonlinear(u_hat)
        rhs1 = [nl1[i] + f[i] for i in range(self.d)]

        # Predictor
        u_pred = [u_hat[i] + self.dt * rhs1[i] for i in range(self.d)]

        # Stage 2
        nl2 = self.nonlinear(u_pred)
        rhs2 = [nl2[i] + f[i] for i in range(self.d)]

        # Corrector + exact viscous decay
        u_new = []
        for i in range(self.d):
            u_new.append(
                self.visc_decay * (u_hat[i] + 0.5 * self.dt * (rhs1[i] + rhs2[i]))
            )
        return u_new

    def init_velocity(self, seed=42, u_rms_target=0.5):
        """Initialize with random divergence-free velocity at target u_rms."""
        torch.manual_seed(seed)
        u_hat = []
        init_mask = (self.k_mag >= 1.0) & (self.k_mag <= 4.0)
        for i in range(self.d):
            amp = torch.randn((self.N,) * self.d, dtype=torch.complex64,
                              device=self.device)
            amp = amp * init_mask
            u_hat.append(amp)
        u_hat = self.project(u_hat)

        # Rescale to achieve target u_rms
        E = self.total_energy(u_hat)
        if E > 0:
            u_rms_curr = np.sqrt(2 * E / self.N ** self.d)
            scale = u_rms_target / (u_rms_curr + 1e-30)
            u_hat = [uh * scale for uh in u_hat]

        return u_hat

    def dissipation_rate(self, u_hat):
        """Compute total dissipation rate: 2*nu * sum(k^2 * |u_hat|^2) / N^d."""
        enstrophy = sum(torch.sum(self.k_sq * torch.abs(uh) ** 2).item()
                        for uh in u_hat)
        return 2 * self.nu * enstrophy / self.N ** self.d

    def energy_spectrum(self, u_hat, n_bins=None):
        """Compute isotropic energy spectrum E(k) by shell averaging."""
        if n_bins is None:
            n_bins = self.N // 2

        e_k = torch.zeros_like(self.k_mag)
        for uh in u_hat:
            e_k = e_k + 0.5 * torch.abs(uh) ** 2

        k_max = self.k_mag.max().item()
        bin_edges = torch.linspace(0.5, k_max + 0.5, n_bins + 1, device=self.device)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        spectrum = torch.zeros(n_bins, device=self.device)
        k_flat = self.k_mag.flatten()
        e_flat = e_k.flatten()

        for b in range(n_bins):
            in_bin = (k_flat >= bin_edges[b]) & (k_flat < bin_edges[b + 1])
            if in_bin.any():
                spectrum[b] = e_flat[in_bin].sum()

        return bin_centers.cpu().numpy(), spectrum.cpu().numpy()

    def structure_functions(self, u_hat, orders=range(1, 9)):
        """
        Real-space longitudinal structure functions via velocity increments.
        S_p(r) = <|u_i(x + r*e_i) - u_i(x)|^p> averaged over axes and grid.
        Uses torch.roll for periodic shifts. Float64 for high-order precision.
        """
        # Physical velocity — use float64 for high-order precision on small grids,
        # but stay float32 for large 4D grids to conserve VRAM
        use_double = (self.N ** self.d) <= 2_000_000  # roughly N=32 in 4D
        u_phys = [torch.fft.ifftn(uh).real for uh in u_hat]
        if use_double:
            u_phys = [up.double() for up in u_phys]

        # Log-spaced separations from 1 to N/2 grid cells
        max_r = self.N // 2
        n_sep = min(max_r, 32)
        seps = np.unique(np.geomspace(1, max_r, n_sep).astype(int))

        sp = {p: np.zeros(len(seps)) for p in orders}

        for si, r in enumerate(seps):
            for axis in range(self.d):
                u_shifted = torch.roll(u_phys[axis], shifts=-int(r), dims=axis)
                delta_u = u_shifted - u_phys[axis]
                abs_du = torch.abs(delta_u)

                for p in orders:
                    sp[p][si] += torch.mean(abs_du ** p).item()

            # Average over d directions
            for p in orders:
                sp[p][si] /= self.d

        r_phys = seps.astype(float) * 2 * np.pi / self.N

        result = {'r': r_phys, 'seps': seps}
        for p in orders:
            result[p] = sp[p]
        return result

    def total_energy(self, u_hat):
        """Total kinetic energy (normalized)."""
        E = 0.0
        for uh in u_hat:
            E = E + 0.5 * torch.sum(torch.abs(uh) ** 2).item()
        return E / self.N ** self.d


def run_dns(d, N, nu, dt, epsilon, n_equil, n_collect, collect_every=50,
            seed=42, forcing_band=(1.0, 2.5), u_rms_target=0.5):
    """
    Run d-dimensional DNS and collect time-averaged structure functions.
    Initializes velocity at target u_rms and maintains energy during equilibration.
    """
    solver = SpectralNS(d=d, N=N, nu=nu, dt=dt, epsilon=epsilon,
                         forcing_band=forcing_band)
    u_hat = solver.init_velocity(seed=seed, u_rms_target=u_rms_target)

    E_target = N ** d * u_rms_target ** 2 / 2
    Re_target = u_rms_target * 2 * np.pi / nu
    print(f"\n  Target: u_rms={u_rms_target:.2f}, E_target={E_target:.1f}, Re~{Re_target:.0f}")

    print(f"  Equilibrating ({n_equil} steps, with energy rescaling)...", end="", flush=True)
    t0 = time.time()
    blowups = 0
    for step in range(n_equil):
        u_hat = solver.step(u_hat)

        # Energy rescaling every 100 steps to maintain target energy
        if step % 100 == 0:
            E = solver.total_energy(u_hat)
            if np.isnan(E) or E > 1e10 or E <= 0:
                blowups += 1
                u_hat = solver.init_velocity(seed=seed + step,
                                              u_rms_target=u_rms_target)
            else:
                scale = np.sqrt(E_target / (E + 1e-30))
                if 0.5 < scale < 2.0:  # Only moderate rescaling
                    u_hat = [uh * scale for uh in u_hat]
                elif scale >= 2.0:  # Energy too low — reinitialize
                    u_hat = solver.init_velocity(seed=seed + step,
                                                  u_rms_target=u_rms_target)
                    blowups += 1
    equil_time = time.time() - t0
    E_final = solver.total_energy(u_hat)
    u_rms = np.sqrt(2 * E_final / N**d) if E_final > 0 else 0
    Re_est = u_rms * 2 * np.pi / nu
    eps_diss = solver.dissipation_rate(u_hat)
    print(f" done ({equil_time:.1f}s, E={E_final:.4e}, u_rms={u_rms:.3e}, Re~{Re_est:.0f}, eps_diss={eps_diss:.2e}, blowups={blowups})")

    print(f"  Collecting ({n_collect} steps, sample every {collect_every})...",
          end="", flush=True)
    t0 = time.time()

    sp_acc = None
    n_samples = 0
    E_history = []

    rescale_every = 200  # maintain energy during collection
    for step in range(n_collect):
        u_hat = solver.step(u_hat)

        # Periodic energy rescaling to prevent decay
        if step % rescale_every == 0 and step > 0:
            E = solver.total_energy(u_hat)
            if E > 0 and not np.isnan(E):
                scale = np.sqrt(E_target / (E + 1e-30))
                if 0.5 < scale < 2.0:
                    u_hat = [uh * scale for uh in u_hat]

        if step % collect_every == 0:
            E = solver.total_energy(u_hat)
            E_history.append(E)

            if np.isnan(E) or E > 1e10 or E <= 0:
                u_hat = solver.init_velocity(seed=seed + n_equil + step,
                                              u_rms_target=u_rms_target)
                continue

            sf = solver.structure_functions(u_hat)
            if sp_acc is None:
                sp_acc = {p: np.zeros_like(sf[p]) for p in range(1, 9)}
                sp_acc['r'] = sf['r']
                sp_acc['seps'] = sf['seps']
            for p in range(1, 9):
                sp_acc[p] += sf[p]
            n_samples += 1

    collect_time = time.time() - t0
    E_end = solver.total_energy(u_hat)
    u_rms_end = np.sqrt(2 * E_end / N**d) if E_end > 0 else 0
    print(f" done ({collect_time:.1f}s, {n_samples} samples, u_rms_end={u_rms_end:.3f})")

    if n_samples > 0:
        for p in range(1, 9):
            sp_acc[p] /= n_samples

    # Final spectrum
    k_spec, E_spec = solver.energy_spectrum(u_hat)

    del u_hat, solver
    torch.cuda.empty_cache()

    return {
        'sp': sp_acc,
        'E_spectrum': {'k': k_spec.tolist(), 'E': E_spec.tolist()},
        'E_history': np.array(E_history),
        'n_samples': n_samples,
        'metadata': {
            'd': d, 'N': N, 'nu': nu, 'dt': dt, 'epsilon': epsilon,
            'n_equil': n_equil, 'n_collect': n_collect,
            'equil_time': equil_time, 'collect_time': collect_time,
            'forcing_band': list(forcing_band),
        }
    }


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def extract_zeta_ess(sp):
    """
    Extract zeta_p / zeta_3 via Extended Self-Similarity (ESS).
    Since zeta_3 = 1 exactly (K41 4/5-law), slope = zeta_p.
    """
    s3 = sp[3]

    # Inertial range: where S_3 is significant but not at peak
    good = s3 > 0
    if np.sum(good) < 5:
        return {}
    s3_good = s3[good]
    s3_max = s3_good.max()
    # Use middle decades
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


def fit_she_leveque(zeta):
    """Fit She-Leveque form to extracted zeta_p. Returns (k, beta, gamma, C0)."""
    orders = [p for p in [1, 2, 4, 5, 6, 7, 8]
              if p in zeta and zeta[p]['r2'] > 0.85]
    if len(orders) < 3:
        return {'success': False, 'reason': f'Only {len(orders)} good orders'}

    p_arr = np.array(orders, dtype=float)
    z_arr = np.array([zeta[p]['value'] for p in orders])

    if np.any(np.isnan(z_arr)):
        return {'success': False, 'reason': 'NaN in zeta'}

    def model(p, gam, bet):
        return (1 - gam) * p / 3 + gam / (1 - bet + 1e-15) * (1 - bet ** (p / 3))

    try:
        popt, pcov = curve_fit(model, p_arr, z_arr,
                               p0=[0.6, 0.6],
                               bounds=([0.01, 0.01], [0.99, 0.99]),
                               maxfev=20000)
        gam, bet = popt
        perr = np.sqrt(np.diag(pcov))
        z_pred = model(p_arr, gam, bet)
        residuals = z_arr - z_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((z_arr - np.mean(z_arr)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-30)
        k_val = 3.0 / (1 - gam)
        C0 = gam / (1 - bet)
        return {
            'success': True, 'k': k_val, 'beta': bet, 'gamma': gam, 'C0': C0,
            'k_err': 3 * perr[0] / (1 - gam) ** 2,
            'beta_err': perr[1], 'gamma_err': perr[0],
            'r2': r2, 'rmse': np.sqrt(np.mean(residuals ** 2)),
            'n_orders': len(orders), 'orders_used': orders,
        }
    except Exception as e:
        return {'success': False, 'reason': str(e)}


def analyze_dimension(d, dns_result, target_k, target_beta, label=""):
    """Full analysis pipeline for one dimension."""
    print(f"\n  --- {label} (d={d}) ---")
    sp = dns_result['sp']
    meta = dns_result['metadata']
    E_hist = dns_result['E_history']

    print(f"  Grid: {meta['N']}^{d}, nu={meta['nu']:.1e}, "
          f"samples={dns_result['n_samples']}")
    if len(E_hist) > 0:
        E_mean = np.mean(E_hist)
        E_std = np.std(E_hist)
        frac = E_std / (E_mean + 1e-30)
        print(f"  Energy: mean={E_mean:.4e}, std/mean={frac:.1%}")

    # ESS extraction
    zeta = extract_zeta_ess(sp)

    # Reference She-Leveque for this dimension
    sl_ref = {p: she_leveque_constrained(p, 1 - 3.0/target_k, target_beta)
              for p in range(1, 9)}

    print(f"\n  {'p':>3} | {'zeta(ESS)':>10} | {'SL ref':>8} | {'K41':>7} "
          f"| {'err':>7} | {'R2':>7}")
    print("  " + "-" * 58)

    for p in range(1, 9):
        if p in zeta:
            z = zeta[p]
            ref = sl_ref[p]
            err_pct = abs(z['value'] - ref) / (abs(ref) + 1e-10) * 100
            print(f"  {p:>3} | {z['value']:>10.4f} | {ref:>8.4f} | {p/3:>7.4f} "
                  f"| {err_pct:>5.1f}%  | {z['r2']:>7.4f}")
        else:
            print(f"  {p:>3} | {'---':>10} | {sl_ref[p]:>8.4f} | {p/3:>7.4f} "
                  f"| {'---':>7} | {'---':>7}")

    # She-Leveque fit
    fit = fit_she_leveque(zeta)
    cal_pass = False
    if fit['success']:
        k_err_pct = abs(fit['k'] - target_k) / target_k * 100
        b_err_pct = abs(fit['beta'] - target_beta) / target_beta * 100
        print(f"\n  SL fit: k={fit['k']:.2f} (target {target_k}), "
              f"beta={fit['beta']:.4f} (target {target_beta:.4f})")
        print(f"  gamma={fit['gamma']:.4f}, C0={fit['C0']:.3f}, "
              f"R2={fit['r2']:.6f}")
        print(f"  Errors: k {k_err_pct:.1f}%, beta {b_err_pct:.1f}%")
        cal_pass = k_err_pct < 35 and b_err_pct < 30
        tag = 'PASS' if cal_pass else 'CHECK NEEDED'
        print(f"  CALIBRATION: {tag}")
    else:
        print(f"\n  SL fit FAILED: {fit.get('reason', 'unknown')}")

    return {'zeta': zeta, 'fit': fit, 'calibration_pass': cal_pass,
            'sl_ref': sl_ref}


# ============================================================
# PART 3: 2D CALIBRATION
# ============================================================

print("\n\n" + "=" * 70)
print("PART 3: 2D Calibration (k=4, beta=0.500)")
print("=" * 70)

t0_total = time.time()
t0_2d = time.time()
dns_2d = run_dns(d=2, N=256, nu=5e-4, dt=1e-3, epsilon=0.003,
                 n_equil=5000, n_collect=15000, collect_every=20,
                 forcing_band=(1.0, 3.0), u_rms_target=0.3)
result_2d = analyze_dimension(2, dns_2d, target_k=4, target_beta=0.5,
                               label="2D Enstrophy Cascade")
print(f"  Total 2D time: {time.time() - t0_2d:.1f}s")


# ============================================================
# PART 4: 3D CALIBRATION
# ============================================================

print("\n\n" + "=" * 70)
print("PART 4: 3D Calibration (k=9, beta=0.667)")
print("=" * 70)

t0_3d = time.time()
dns_3d = run_dns(d=3, N=64, nu=5e-3, dt=2e-3, epsilon=0.01,
                 n_equil=3000, n_collect=15000, collect_every=20,
                 forcing_band=(1.0, 2.5), u_rms_target=0.3)
result_3d = analyze_dimension(3, dns_3d, target_k=9, target_beta=2/3,
                               label="3D Energy Cascade")
print(f"  Total 3D time: {time.time() - t0_3d:.1f}s")


# ============================================================
# PART 5: 4D — THE KEY TEST
# ============================================================

print("\n\n" + "=" * 70)
print("PART 5: 4D Turbulence — k=20 vs k=16")
print("=" * 70)

t0_4d = time.time()
dns_4d = run_dns(d=4, N=32, nu=2e-2, dt=5e-3, epsilon=0.03,
                 n_equil=2000, n_collect=10000, collect_every=10,
                 forcing_band=(1.0, 2.5), u_rms_target=0.2)
result_4d = analyze_dimension(4, dns_4d, target_k=20, target_beta=0.6,
                               label="4D CASCADE — KEY TEST")

# Compare to naive prediction
boot_k_arr = np.array([])
boot_beta_arr = np.array([])
k_ci = None
b_ci = None

fit_4d = result_4d['fit']
if fit_4d['success']:
    k_meas = fit_4d['k']
    dist_dft = abs(k_meas - 20)
    dist_naive = abs(k_meas - 16)

    print(f"\n  === KEY RESULT ===")
    print(f"  Measured k = {k_meas:.2f}")
    print(f"  DFT prediction: k = 20  (distance = {dist_dft:.2f})")
    print(f"  Naive prediction: k = 16  (distance = {dist_naive:.2f})")
    winner = 'DFT (k=20)' if dist_dft < dist_naive else 'Naive (k=16)'
    print(f"  Closer to: {winner}")

    beta_meas = fit_4d['beta']
    print(f"\n  Measured beta = {beta_meas:.4f}")
    print(f"  DFT prediction: beta = 0.6000 "
          f"(distance = {abs(beta_meas - 0.6):.4f})")

    # Bootstrap CI by perturbing zeta_p within their stderr
    boot_k_list = []
    boot_beta_list = []
    rng = np.random.default_rng(42)
    zeta_data = result_4d['zeta']

    for _ in range(2000):
        zeta_perturbed = {}
        for p in range(1, 9):
            if p in zeta_data:
                noise = rng.normal(0, zeta_data[p].get('stderr', 0.01))
                zeta_perturbed[p] = {
                    'value': zeta_data[p]['value'] + noise,
                    'r2': zeta_data[p]['r2']
                }
        fit_b = fit_she_leveque(zeta_perturbed)
        if fit_b['success']:
            boot_k_list.append(fit_b['k'])
            boot_beta_list.append(fit_b['beta'])

    if len(boot_k_list) > 100:
        boot_k_arr = np.array(boot_k_list)
        boot_beta_arr = np.array(boot_beta_list)
        k_ci = np.percentile(boot_k_arr, [2.5, 97.5])
        b_ci = np.percentile(boot_beta_arr, [2.5, 97.5])
        print(f"\n  Bootstrap 95%% CI for k: [{k_ci[0]:.2f}, {k_ci[1]:.2f}]")
        print(f"  Bootstrap 95%% CI for beta: [{b_ci[0]:.4f}, {b_ci[1]:.4f}]")
        k20_in = k_ci[0] <= 20 <= k_ci[1]
        k16_in = k_ci[0] <= 16 <= k_ci[1]
        print(f"    k=20 in CI: {'YES' if k20_in else 'NO'}")
        print(f"    k=16 in CI: {'YES' if k16_in else 'NO'}")

print(f"\n  Total 4D time: {time.time() - t0_4d:.1f}s")


# ============================================================
# PART 6: EXTENDED 4D — SKIPPED (VRAM limitation)
# ============================================================

print("\\n\\n" + "=" * 70)
print("PART 6: Extended 4D — SKIPPED")
print("=" * 70)
print("  N=48 (5.3M grid) causes CUDA hard crash on 8.6GB VRAM.")
print("  Would require ~3GB during RK2 nonlinear computation.")
print("  Skipping to save results from Parts 3-5.")

t0_4d_ext = time.time()
result_4d_ext = None
dns_4d_ext = None
# Part 6 disabled: N=48 4D DNS exceeds RTX 3070 Ti VRAM during RK2 nonlinear computation

print(f"  Total extended time: {time.time() - t0_4d_ext:.1f}s")


# ============================================================
# PART 7: SUMMARY
# ============================================================

total_time = time.time() - t0_total

print("\n\n" + "=" * 70)
print("EXPERIMENT 06 SUMMARY")
print("=" * 70)

print(f"\n  GPU: {GPU_NAME} ({VRAM_GB:.1f} GB)")
print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"\n  --- DFT Predictions ---")
print(f"  k(d) = d x F(d+1):  k(2)=4, k(3)=9, k(4)=20")
print(f"  beta(d) = F_d/F(d+1): beta(2)=0.5, beta(3)=0.667, beta(4)=0.6")
print(f"  Competing: k = d^2:  k(2)=4, k(3)=9, k(4)=16")

for d, label, result in [(2, '2D', result_2d), (3, '3D', result_3d),
                           (4, '4D', result_4d)]:
    fit = result['fit']
    cal = result['calibration_pass']
    if fit['success']:
        pred_k = predictions[d]['k']
        pred_beta = predictions[d]['beta']
        tag = 'PASS' if cal else 'CHECK'
        print(f"\n  d={d} ({label}): k={fit['k']:.2f} (DFT:{pred_k}), "
              f"beta={fit['beta']:.4f} (DFT:{pred_beta:.4f}), "
              f"R2={fit['r2']:.4f} [{tag}]")
    else:
        print(f"\n  d={d} ({label}): FIT FAILED - {fit.get('reason', 'unknown')}")

# Extended 4D
if result_4d_ext is not None and result_4d_ext['fit']['success']:
    fe = result_4d_ext['fit']
    print(f"\n  d=4 (N=48): k={fe['k']:.2f}, beta={fe['beta']:.4f}, "
          f"R2={fe['r2']:.4f}")

# Final verdict
if fit_4d['success']:
    k4 = fit_4d['k']
    dft_closer = abs(k4 - 20) < abs(k4 - 16)
    print(f"\n  {'=' * 50}")
    print(f"  4D VERDICT: k = {k4:.2f}")
    print(f"    DFT (k=20): distance = {abs(k4-20):.2f}")
    print(f"    Naive (k=16): distance = {abs(k4-16):.2f}")
    tag = 'DFT k=d*F(d+1)' if dft_closer else 'Naive k=d^2'
    print(f"    >>> {tag} is CLOSER")
    if k_ci is not None:
        print(f"    95%% CI: [{k_ci[0]:.2f}, {k_ci[1]:.2f}]")
    print(f"  {'=' * 50}")

print("=" * 70)


# ============================================================
# SAVE RESULTS
# ============================================================

def safe_zeta(zeta):
    """Convert zeta dict for JSON serialization."""
    return {str(p): {'value': float(v['value']), 'r2': float(v['r2']),
                      'stderr': float(v.get('stderr', 0))}
            for p, v in zeta.items()}


def safe_fit(fit):
    """Convert fit dict for JSON serialization."""
    out = {}
    for k, v in fit.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (list, tuple)):
            out[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x
                      for x in v]
        else:
            out[k] = v
    return out


output = {
    'experiment': 'milestone4/exp_06_4d_turbulence_k20',
    'method': 'GPU pseudo-spectral DNS (PyTorch CUDA)',
    'gpu': GPU_NAME,
    'total_time_seconds': total_time,

    'predictions': {
        'dft': {str(d): {'k': predictions[d]['k'],
                          'beta': float(predictions[d]['beta'])}
                for d in [2, 3, 4]},
        'naive_k4': 16,
    },

    'd2': {
        'metadata': dns_2d['metadata'],
        'n_samples': dns_2d['n_samples'],
        'zeta': safe_zeta(result_2d['zeta']),
        'fit': safe_fit(result_2d['fit']),
        'calibration_pass': result_2d['calibration_pass'],
    },

    'd3': {
        'metadata': dns_3d['metadata'],
        'n_samples': dns_3d['n_samples'],
        'zeta': safe_zeta(result_3d['zeta']),
        'fit': safe_fit(result_3d['fit']),
        'calibration_pass': result_3d['calibration_pass'],
    },

    'd4': {
        'metadata': dns_4d['metadata'],
        'n_samples': dns_4d['n_samples'],
        'zeta': safe_zeta(result_4d['zeta']),
        'fit': safe_fit(result_4d['fit']),
        'calibration_pass': result_4d['calibration_pass'],
    },

    'result': {
        'k4_measured': float(fit_4d['k']) if fit_4d['success'] else None,
        'beta4_measured': float(fit_4d['beta']) if fit_4d['success'] else None,
        'dft_closer': bool(abs(fit_4d['k'] - 20) < abs(fit_4d['k'] - 16))
            if fit_4d['success'] else None,
    },
}

# Extended 4D
if result_4d_ext is not None and result_4d_ext['fit']['success']:
    output['d4_extended'] = {
        'metadata': dns_4d_ext['metadata'],
        'n_samples': dns_4d_ext['n_samples'],
        'zeta': safe_zeta(result_4d_ext['zeta']),
        'fit': safe_fit(result_4d_ext['fit']),
    }

# Bootstrap CI
if k_ci is not None:
    output['result']['k4_ci_95'] = [float(k_ci[0]), float(k_ci[1])]
    output['result']['beta4_ci_95'] = [float(b_ci[0]), float(b_ci[1])]
    output['result']['k20_in_ci'] = bool(k_ci[0] <= 20 <= k_ci[1])
    output['result']['k16_in_ci'] = bool(k_ci[0] <= 16 <= k_ci[1])

save_results(output, 'exp_06_4d_turbulence_k20')
