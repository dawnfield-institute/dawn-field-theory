"""
GPU-Accelerated PAC Energy Cascade Engine (PyTorch + CUDA)

Batched version of utils.energy_cascade() for massive Monte Carlo null tests.
Runs B independent cascade simulations in parallel on GPU, falling back to
CPU transparently when CUDA is unavailable.

Usage:
    from gpu_cascade import get_device, energy_cascade_gpu, measure_exponent_batch

    device = get_device()
    # Run 256 independent cascades in one GPU call
    all_results = energy_cascade_gpu(1.0, 25, n_modes=8, batch_size=256, device=device)
    exponents = measure_exponent_batch(all_results)
"""

import math
import numpy as np
from scipy import stats

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

_LANDAUER_MIN = math.log(2)


def get_device():
    """Return best available torch device, or None if torch unavailable."""
    if not HAS_TORCH:
        return None
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_device_info(device):
    """Print GPU info if available."""
    if device is None:
        print("  Backend: NumPy (CPU only, torch not installed)")
        return
    print(f"  Backend: PyTorch {torch.__version__}")
    print(f"  Device:  {device}")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU:     {props.name}")
        print(f"  VRAM:    {props.total_memory / 1e9:.1f} GB")
    else:
        print("  GPU:     not available (CPU fallback)")


# ============================================================
# GPU-BATCHED CASCADE ENGINE
# ============================================================

def _build_coupling_matrix_batch(n_modes, coupling_decay, batch_size, device):
    """Build B identical structured coupling matrices on device.

    C[i,j] = exp(-|i-j| * coupling_decay)
    Returns: (B, n_modes, n_modes) tensor
    """
    idx = torch.arange(n_modes, device=device, dtype=torch.float32)
    diff = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))  # (M, M)
    C = torch.exp(-diff * coupling_decay)  # (M, M)
    return C.unsqueeze(0).expand(batch_size, -1, -1).clone()  # (B, M, M)


def _build_random_coupling_batch(n_modes, batch_size, device, generator=None):
    """Build B random symmetric PD coupling matrices (Wishart-like).

    Returns: (B, n_modes, n_modes) tensor
    """
    A = torch.randn(batch_size, n_modes, n_modes, device=device,
                     generator=generator)
    C = torch.bmm(A, A.transpose(1, 2)) / n_modes
    C = (C + C.transpose(1, 2)) / 2

    # Ensure positive definite
    eigs = torch.linalg.eigvalsh(C)  # (B, M)
    min_eigs = eigs.min(dim=1).values  # (B,)
    needs_shift = min_eigs < 1e-10
    if needs_shift.any():
        shift = torch.zeros(batch_size, device=device)
        shift[needs_shift] = torch.abs(min_eigs[needs_shift]) + 1e-6
        C = C + shift.view(-1, 1, 1) * torch.eye(n_modes, device=device)

    return C


def energy_cascade_gpu(injection_energy, n_scales, n_modes=8,
                       n_samples=15000, coupling_decay=0.3,
                       nonlinear_strength=0.3, batch_size=1,
                       device=None, random_coupling=False,
                       seed=None):
    """
    GPU-batched PAC energy cascade.

    Runs `batch_size` independent cascades in parallel. Each cascade is
    identical to utils.energy_cascade() in logic — structured coupling
    matrix, eigenvalue-based org fraction, Landauer floor, 2% dissipation.

    Parameters
    ----------
    injection_energy : float
    n_scales : int
    n_modes : int
    n_samples : int
        Monte Carlo samples per scale per cascade.
    coupling_decay : float
    nonlinear_strength : float
    batch_size : int
        Number of independent cascades to run in parallel.
    device : torch.device or None
        If None, falls back to NumPy CPU path.
    random_coupling : bool
        If True, use random PSD coupling matrices instead of structured.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    list of list of dicts  (batch_size x n_scales)
        Same dict format as utils.energy_cascade().
    """
    # Fallback: no torch or CPU-only small batch → use NumPy path
    if device is None or (device.type == 'cpu' and batch_size <= 4):
        return _cascade_numpy_batch(
            injection_energy, n_scales, n_modes, n_samples,
            coupling_decay, nonlinear_strength, batch_size,
            random_coupling, seed)

    return _cascade_torch_batch(
        injection_energy, n_scales, n_modes, n_samples,
        coupling_decay, nonlinear_strength, batch_size,
        device, random_coupling, seed)


def _cascade_torch_batch(injection_energy, n_scales, n_modes, n_samples,
                         coupling_decay, nonlinear_strength, batch_size,
                         device, random_coupling, seed):
    """PyTorch GPU implementation of batched cascade."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    B, M, S = batch_size, n_modes, n_samples

    # Initialize power per batch element
    P = torch.full((B,), injection_energy, device=device, dtype=torch.float64)
    prev_dominant = None  # (B, M) or None

    all_results = [[] for _ in range(B)]

    for k_idx in range(n_scales):
        wavenumber = 2 ** (k_idx + 1)
        alive_mask = P > 1e-18  # (B,)

        # Dead cascades
        for b in range(B):
            if not alive_mask[b]:
                all_results[b].append({
                    'k_index': k_idx, 'wavenumber': wavenumber,
                    'P_input': 0.0, 'org_fraction': 0.0, 'alive': False
                })

        if not alive_mask.any():
            continue

        # Build coupling matrices
        if random_coupling:
            C = _build_random_coupling_batch(M, B, device, generator)
        else:
            C = _build_coupling_matrix_batch(M, coupling_decay, B, device)

        # Nonlinear feedback from previous dominant eigenvector
        if prev_dominant is not None and nonlinear_strength > 0:
            bias = torch.bmm(
                prev_dominant.unsqueeze(2),  # (B, M, 1)
                prev_dominant.unsqueeze(1)   # (B, 1, M)
            )  # (B, M, M)
            bias_max = bias.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-15)
            bias = bias / bias_max
            C = C + bias.float() * nonlinear_strength

        # Ensure symmetric PD
        C = (C + C.transpose(1, 2)) / 2
        eigs_check = torch.linalg.eigvalsh(C.float())
        min_eigs = eigs_check.min(dim=1).values
        needs_shift = min_eigs < 1e-10
        if needs_shift.any():
            shift = torch.zeros(B, device=device)
            shift[needs_shift] = torch.abs(min_eigs[needs_shift]) + 1e-6
            C = C + shift.view(-1, 1, 1) * torch.eye(M, device=device)

        C_float = C.float()

        # Process each batch element (sampling + eigendecomposition)
        # Note: batched multivariate normal is tricky with varying P,
        # so we loop over alive elements but keep GPU for matrix ops
        for b in range(B):
            if not alive_mask[b]:
                continue

            p_val = P[b].item()
            C_b = C_float[b]  # (M, M)

            # Energy distribution across modes
            mode_idx = torch.arange(M, device=device, dtype=torch.float32)
            means = p_val * torch.exp(-mode_idx * coupling_decay)
            means = means * (p_val / means.sum())

            try:
                sf = p_val / (C_b.trace() / M) * 0.2
                cov_sample = C_b * sf
                # Ensure PD for MultivariateNormal
                cov_sample = (cov_sample + cov_sample.T) / 2
                eigs_s = torch.linalg.eigvalsh(cov_sample)
                if eigs_s.min() < 1e-10:
                    cov_sample = cov_sample + torch.eye(M, device=device) * (
                        abs(eigs_s.min().item()) + 1e-6)

                dist = torch.distributions.MultivariateNormal(
                    means, covariance_matrix=cov_sample)
                samples = dist.sample((S,)).abs()  # (S, M)
            except Exception:
                samples = torch.distributions.Exponential(
                    M / p_val).sample((S, M)).to(device)

            # Covariance + eigendecomposition on GPU
            samples_centered = samples - samples.mean(dim=0, keepdim=True)
            cov = (samples_centered.T @ samples_centered) / (S - 1)
            eigenvalues = torch.linalg.eigvalsh(cov).clamp(min=1e-30)
            total_var = eigenvalues.sum()
            org_frac = (eigenvalues[-1] / total_var).item()

            E_org = p_val * org_frac
            E_transfer = p_val * (1 - org_frac)

            if E_transfer < _LANDAUER_MIN and p_val > _LANDAUER_MIN:
                E_transfer = _LANDAUER_MIN
                E_org = p_val - E_transfer
                org_frac = E_org / p_val

            # Dominant eigenvector for feedback
            _, eigvecs = torch.linalg.eigh(cov)
            if prev_dominant is None:
                prev_dominant = torch.zeros(B, M, device=device, dtype=torch.float64)
            prev_dominant[b] = eigvecs[:, -1].double()

            participation = (eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum()).item()

            all_results[b].append({
                'k_index': k_idx,
                'wavenumber': wavenumber,
                'P_input': p_val,
                'org_fraction': org_frac,
                'E_organized': E_org,
                'E_transfer': E_transfer,
                'participation_ratio': participation,
                'alive': True
            })

            P[b] = E_transfer * 0.98

    return all_results


def _cascade_numpy_batch(injection_energy, n_scales, n_modes, n_samples,
                         coupling_decay, nonlinear_strength, batch_size,
                         random_coupling, seed):
    """NumPy fallback for small batches or no-GPU environments."""
    from utils import energy_cascade

    all_results = []
    for b in range(batch_size):
        b_seed = (seed + b * 7919) if seed is not None else None
        if b_seed is not None:
            np.random.seed(b_seed)

        if random_coupling:
            # Use the random coupling variant
            res = _energy_cascade_random_numpy(
                injection_energy, n_scales, n_modes, n_samples,
                coupling_decay, nonlinear_strength, b_seed)
        else:
            res = energy_cascade(
                injection_energy, n_scales, n_modes, n_samples,
                coupling_decay, nonlinear_strength)
        all_results.append(res)

    return all_results


def _energy_cascade_random_numpy(injection_energy, n_scales, n_modes,
                                 n_samples, coupling_decay, nonlinear_strength,
                                 seed=None):
    """NumPy cascade with random PSD coupling (for null tests)."""
    rng = np.random.default_rng(seed)
    results = []
    P = injection_energy

    for k_idx in range(n_scales):
        if P < 1e-18:
            results.append({
                'k_index': k_idx, 'wavenumber': 2 ** (k_idx + 1),
                'P_input': 0, 'org_fraction': 0, 'alive': False
            })
            continue

        A = rng.standard_normal((n_modes, n_modes))
        C = A @ A.T / n_modes
        C = (C + C.T) / 2
        eigs_C = np.linalg.eigvalsh(C)
        if np.min(eigs_C) < 1e-10:
            C += np.eye(n_modes) * (abs(np.min(eigs_C)) + 1e-6)

        means = np.full(n_modes, P / n_modes)

        try:
            sf = P / (np.trace(C) / n_modes) * 0.2
            samples = np.abs(rng.multivariate_normal(means, C * sf, size=n_samples))
        except Exception:
            samples = rng.exponential(P / n_modes, (n_samples, n_modes))

        cov = np.cov(samples.T)
        eigenvalues = np.maximum(np.linalg.eigvalsh(cov), 1e-30)
        total_var = np.sum(eigenvalues)
        org_frac = eigenvalues[-1] / total_var

        E_org = P * org_frac
        E_transfer = P * (1 - org_frac)

        if E_transfer < _LANDAUER_MIN and P > _LANDAUER_MIN:
            E_transfer = _LANDAUER_MIN
            E_org = P - E_transfer
            org_frac = E_org / P

        results.append({
            'k_index': k_idx, 'wavenumber': 2 ** (k_idx + 1),
            'P_input': P, 'org_fraction': org_frac,
            'E_organized': E_org, 'E_transfer': E_transfer,
            'participation_ratio': np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2),
            'alive': True
        })

        P = E_transfer * 0.98

    return results


# ============================================================
# BATCH EXPONENT MEASUREMENT
# ============================================================

def measure_exponent_batch(all_results, trim=2):
    """
    Extract spectral exponents from a batch of cascade results.

    Parameters
    ----------
    all_results : list of list of dicts
        Output of energy_cascade_gpu (B x n_scales).
    trim : int
        Scales to trim from each end.

    Returns
    -------
    list of tuples (slope, r_squared, avg_org_fraction, std_error)
        One per batch element. None-tuples for insufficient data.
    """
    from utils import measure_exponent
    return [measure_exponent(res, trim) for res in all_results]
