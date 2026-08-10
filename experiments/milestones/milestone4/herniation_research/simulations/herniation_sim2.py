"""
Herniation Boundary Simulation - Experiment 2 (Revised)
========================================================
The Hamiltonian approach from Exp 1 produced clear discrete bound states.
The dynamical simulation was overdamped. 

This revision:
1. Deep analysis of the Hamiltonian bound state spectrum
2. Better dynamical model with conserved energy
3. Test whether bound state ratios show PAC-relevant structure

Peter McNally / Dawn Field Institute, 2026
"""

import numpy as np
from scipy.linalg import eigh
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# =============================================================================
# PART 1: Deep Hamiltonian Analysis
# =============================================================================

def herniation_hamiltonian(N=500, coupling=1.0, boundary_width=10, asymmetry=0.0):
    """
    Hamiltonian for the order-chaos boundary.
    
    The potential well represents the herniation: where order and chaos
    interface, there's an energy minimum that can trap bound states.
    
    Asymmetry parameter: if > 0, the well isn't symmetric — order side
    is different from chaos side. This models the fact that the two
    fields have different character.
    """
    dx = 0.5
    x = np.arange(N) * dx - N * dx / 2  # centered at 0
    
    # Herniation potential: Gaussian well with optional asymmetry
    V = -coupling * np.exp(-x**2 / (2 * boundary_width**2))
    
    # Asymmetry: order side (x < 0) is steeper than chaos side (x > 0)
    if asymmetry > 0:
        V += asymmetry * x * np.exp(-x**2 / (2 * boundary_width**2))
    
    # Kinetic energy operator
    T = np.zeros((N, N))
    for i in range(N):
        T[i, i] = -2.0
        if i + 1 < N:
            T[i, i+1] = 1.0
        if i - 1 >= 0:
            T[i, i-1] = 1.0
    T *= -0.5 / dx**2
    
    H = T + np.diag(V)
    eigenvalues, eigenvectors = eigh(H)
    
    # Bound states
    bound_mask = eigenvalues < 0
    bound_E = eigenvalues[bound_mask]
    bound_psi = eigenvectors[:, bound_mask]
    
    return {
        'energies': bound_E,
        'wavefunctions': bound_psi,
        'n_bound': len(bound_E),
        'potential': V,
        'x': x,
        'all_eigenvalues': eigenvalues
    }


def analyze_spectrum(energies, label=""):
    """
    Deep analysis of the bound state energy spectrum.
    Look for structure in gaps, ratios, and relationships to constants.
    """
    if len(energies) < 2:
        print(f"  [{label}] Only {len(energies)} bound state(s), skipping analysis")
        return {}
    
    E = np.array(energies)
    
    # Energy gaps
    gaps = np.diff(E)
    
    # Gap ratios (consecutive)
    gap_ratios = gaps[:-1] / gaps[1:] if len(gaps) > 1 else np.array([])
    
    # Energy ratios to ground state
    E_ratios = E / E[0]
    
    # Level spacing statistics (key diagnostic)
    # For quantum chaos: Wigner-Dyson distribution
    # For integrable: Poisson distribution
    # For our system: ?
    normalized_gaps = gaps / np.mean(gaps)
    gap_variance = np.var(normalized_gaps)
    
    # Constants
    phi = (1 + np.sqrt(5)) / 2
    xi_pac = 1.0571
    
    # Check if gap ratios converge
    if len(gap_ratios) > 3:
        late_ratios = gap_ratios[len(gap_ratios)//2:]
        convergence_value = np.mean(late_ratios)
        convergence_std = np.std(late_ratios)
    else:
        convergence_value = np.mean(gap_ratios) if len(gap_ratios) > 0 else 0
        convergence_std = np.std(gap_ratios) if len(gap_ratios) > 0 else 0
    
    results = {
        'n_states': len(E),
        'energies': E.tolist(),
        'gaps': gaps.tolist(),
        'gap_ratios': gap_ratios.tolist(),
        'energy_ratios': E_ratios.tolist(),
        'gap_ratio_convergence': float(convergence_value),
        'gap_ratio_std': float(convergence_std),
        'gap_variance': float(gap_variance),
        'normalized_gaps': normalized_gaps.tolist()
    }
    
    print(f"\n  [{label}] {len(E)} bound states")
    print(f"  Ground state energy: {E[0]:.6f}")
    print(f"  First 5 energies: {[f'{e:.4f}' for e in E[:5]]}")
    print(f"  First 5 gaps: {[f'{g:.4f}' for g in gaps[:5]]}")
    if len(gap_ratios) > 0:
        print(f"  First 5 gap ratios: {[f'{r:.4f}' for r in gap_ratios[:5]]}")
        print(f"  Gap ratio converges to: {convergence_value:.4f} ± {convergence_std:.4f}")
        print(f"  Compare: φ={phi:.4f}, 1/φ={1/phi:.4f}, ξ_PAC={xi_pac:.4f}")
        print(f"  Gap variance (Poisson=1, Wigner≈0.27): {gap_variance:.4f}")
    
    return results


# =============================================================================
# PART 2: Conserved Dynamical Model
# =============================================================================

def run_wave_simulation(
    N=256, T=10000, dt=0.002, 
    coupling=2.0, boundary_width=10,
    seed=42
):
    """
    Wave equation approach instead of diffusion.
    Two coupled wave fields — order and chaos — that can exchange
    energy at the boundary without dissipation.
    
    ∂²ψ_o/∂t² = v_o² ∂²ψ_o/∂x² + g * mask * ψ_c
    ∂²ψ_c/∂t² = v_c² ∂²ψ_c/∂x² - g * mask * ψ_o
    
    This is conservative (energy sloshes between fields, no damping).
    """
    rng = np.random.default_rng(seed)
    dx = 1.0
    x = np.arange(N) * dx
    center = N // 2
    
    # Speeds: order propagates slower than chaos
    v_order = 0.5
    v_chaos = 1.0
    
    # Fields and velocities
    psi_o = np.exp(-((x - center*dx)**2) / (2 * 20**2)) * np.cos(2*np.pi*x/(N*dx) * 3)
    psi_c = rng.normal(0, 0.3, N)
    dpsi_o = np.zeros(N)
    dpsi_c = np.zeros(N)
    
    # Boundary
    mask = np.exp(-((x - x[center])**2) / (2 * boundary_width**2))
    
    # Record at boundary center
    n_record = min(T, 10000)
    record_interval = max(1, T // n_record)
    history_o = []
    history_c = []
    history_bound = []
    total_energy = []
    
    for t in range(T):
        # Laplacians
        lap_o = np.roll(psi_o, 1) + np.roll(psi_o, -1) - 2*psi_o
        lap_c = np.roll(psi_c, 1) + np.roll(psi_c, -1) - 2*psi_c
        
        # Accelerations with coupling
        ddpsi_o = v_order**2 * lap_o / dx**2 + coupling * mask * psi_c
        ddpsi_c = v_chaos**2 * lap_c / dx**2 - coupling * mask * psi_o
        
        # Leapfrog integration
        dpsi_o += ddpsi_o * dt
        dpsi_c += ddpsi_c * dt
        psi_o += dpsi_o * dt
        psi_c += dpsi_c * dt
        
        # Record
        if t % record_interval == 0:
            history_o.append(psi_o[center])
            history_c.append(psi_c[center])
            history_bound.append(psi_o[center] * psi_c[center])  # bound state signal
            
            # Total energy (kinetic + potential + coupling)
            KE = 0.5 * np.sum(dpsi_o**2 + dpsi_c**2) * dx
            PE = 0.5 * (v_order**2 * np.sum((np.roll(psi_o,1) - psi_o)**2) + 
                        v_chaos**2 * np.sum((np.roll(psi_c,1) - psi_c)**2)) / dx
            CE = -coupling * np.sum(mask * psi_o * psi_c) * dx
            total_energy.append(KE + PE + CE)
    
    return {
        'history_o': np.array(history_o),
        'history_c': np.array(history_c),
        'history_bound': np.array(history_bound),
        'total_energy': np.array(total_energy),
        'dt_effective': dt * record_interval,
        'psi_o_final': psi_o,
        'psi_c_final': psi_c,
        'mask': mask,
        'x': x
    }


def analyze_wave_spectrum(result, label=""):
    """Frequency analysis of the wave simulation at the boundary."""
    
    # Use the bound state signal (product of both fields at boundary)
    signal = result['history_bound']
    dt = result['dt_effective']
    
    # Skip transient
    start = len(signal) // 5
    signal = signal[start:]
    
    # Remove DC
    signal = signal - np.mean(signal)
    
    # FFT
    spectrum = np.abs(fft(signal))**2
    freqs = fftfreq(len(signal), dt)
    pos = freqs > 0
    freqs_pos = freqs[pos]
    spec_pos = spectrum[pos]
    
    # Normalize
    spec_pos = spec_pos / spec_pos.max() if spec_pos.max() > 0 else spec_pos
    
    # Find peaks
    peaks, props = find_peaks(spec_pos, height=0.01, distance=5, prominence=0.005)
    
    if len(peaks) > 0:
        peak_f = freqs_pos[peaks]
        peak_h = spec_pos[peaks]
        
        # Sort by height
        idx = np.argsort(peak_h)[::-1]
        peak_f = peak_f[idx]
        peak_h = peak_h[idx]
        
        # Ratios
        if len(peak_f) > 1:
            ratios = peak_f / peak_f[0]
            consec = peak_f[:-1] / peak_f[1:]
        else:
            ratios = np.array([1.0])
            consec = np.array([])
        
        print(f"\n  [{label}] Wave simulation: {len(peak_f)} frequency peaks")
        print(f"  Top frequencies: {[f'{f:.4f}' for f in peak_f[:8]]}")
        print(f"  Ratios to fundamental: {[f'{r:.3f}' for r in ratios[:8]]}")
        if len(consec) > 0:
            print(f"  Consecutive ratios: {[f'{r:.3f}' for r in consec[:7]]}")
        
        # Energy conservation check
        E = result['total_energy']
        E_drift = (E[-1] - E[0]) / abs(E[0]) * 100
        print(f"  Energy conservation: {E_drift:+.2f}% drift")
        
        return {
            'peak_frequencies': peak_f.tolist(),
            'peak_heights': peak_h.tolist(),
            'ratios': ratios.tolist(),
            'consecutive_ratios': consec.tolist(),
            'n_peaks': len(peak_f),
            'energy_drift_pct': float(E_drift)
        }
    else:
        print(f"\n  [{label}] No peaks found in wave simulation")
        return {'n_peaks': 0}


# =============================================================================
# PART 3: Double-Well (Order-Chaos Duality)
# =============================================================================

def double_well_hamiltonian(N=500, coupling=2.0, well_sep=20, well_width=8, asymmetry=0.3):
    """
    Double-well potential representing the order-chaos duality.
    
    One well = order (information-dominated)
    Other well = chaos (energy-dominated)
    
    The bound states that span BOTH wells are the quantum locks —
    information and energy bound together.
    
    Asymmetry represents the different nature of order vs chaos.
    """
    dx = 0.3
    x = np.arange(N) * dx - N * dx / 2
    
    # Two Gaussian wells separated by well_sep
    V_order = -coupling * np.exp(-((x + well_sep/2)**2) / (2 * well_width**2))
    V_chaos = -(coupling - asymmetry) * np.exp(-((x - well_sep/2)**2) / (2 * well_width**2))
    V = V_order + V_chaos
    
    # Kinetic
    T = np.zeros((N, N))
    for i in range(N):
        T[i, i] = -2.0
        if i + 1 < N: T[i, i+1] = 1.0
        if i - 1 >= 0: T[i, i-1] = 1.0
    T *= -0.5 / dx**2
    
    H = T + np.diag(V)
    eigenvalues, eigenvectors = eigh(H)
    
    bound_mask = eigenvalues < 0
    bound_E = eigenvalues[bound_mask]
    bound_psi = eigenvectors[:, bound_mask]
    
    # Classify: does the wavefunction span both wells?
    left_region = x < 0
    right_region = x > 0
    
    spanning_states = []
    localized_order = []
    localized_chaos = []
    
    for i in range(len(bound_E)):
        psi = bound_psi[:, i]
        left_weight = np.sum(psi[left_region]**2)
        right_weight = np.sum(psi[right_region]**2)
        total = left_weight + right_weight
        
        balance = min(left_weight, right_weight) / max(left_weight, right_weight) if total > 0 else 0
        
        if balance > 0.3:  # spans both wells
            spanning_states.append(i)
        elif left_weight > right_weight:
            localized_order.append(i)
        else:
            localized_chaos.append(i)
    
    return {
        'energies': bound_E,
        'wavefunctions': bound_psi,
        'n_bound': len(bound_E),
        'spanning_indices': spanning_states,
        'order_indices': localized_order,
        'chaos_indices': localized_chaos,
        'potential': V,
        'x': x,
        'n_spanning': len(spanning_states),
        'n_order': len(localized_order),
        'n_chaos': len(localized_chaos)
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    
    print("=" * 70)
    print("HERNIATION BOUNDARY SIMULATION — REVISED")
    print("=" * 70)
    
    # ================================================================
    # EXPERIMENT A: Deep Hamiltonian spectrum analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Hamiltonian Bound State Spectrum")
    print("=" * 70)
    
    # Baseline
    ham = herniation_hamiltonian(coupling=3.0, boundary_width=10)
    spec_A = analyze_spectrum(ham['energies'], "baseline g=3 w=10")
    
    # Narrow boundary (tight herniation)
    ham_narrow = herniation_hamiltonian(coupling=3.0, boundary_width=3)
    spec_narrow = analyze_spectrum(ham_narrow['energies'], "narrow g=3 w=3")
    
    # Wide boundary
    ham_wide = herniation_hamiltonian(coupling=3.0, boundary_width=25)
    spec_wide = analyze_spectrum(ham_wide['energies'], "wide g=3 w=25")
    
    # Strong coupling
    ham_strong = herniation_hamiltonian(coupling=10.0, boundary_width=10)
    spec_strong = analyze_spectrum(ham_strong['energies'], "strong g=10 w=10")
    
    # Asymmetric boundary
    ham_asym = herniation_hamiltonian(coupling=3.0, boundary_width=10, asymmetry=1.0)
    spec_asym = analyze_spectrum(ham_asym['energies'], "asymmetric g=3 w=10 a=1")
    
    # ================================================================
    # EXPERIMENT B: Wave dynamics
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Conservative Wave Dynamics")
    print("=" * 70)
    
    for g in [0.5, 1.0, 2.0, 5.0]:
        wave = run_wave_simulation(coupling=g, T=15000, dt=0.001)
        analyze_wave_spectrum(wave, f"wave g={g}")
    
    # ================================================================
    # EXPERIMENT C: Double-well (order-chaos duality)
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT C: Double-Well Order-Chaos Bound States")
    print("=" * 70)
    
    print("\n  --- Separation sweep ---")
    for sep in [5, 10, 15, 20, 30, 40]:
        dw = double_well_hamiltonian(well_sep=sep, coupling=3.0, asymmetry=0.3)
        n_span = dw['n_spanning']
        n_ord = dw['n_order']
        n_cha = dw['n_chaos']
        print(f"  Sep={sep:2d}: {dw['n_bound']} bound, "
              f"{n_span} spanning (LOCKS), "
              f"{n_ord} order-localized, {n_cha} chaos-localized")
        
        if n_span > 1:
            span_E = dw['energies'][dw['spanning_indices']]
            span_gaps = np.diff(span_E)
            print(f"          Lock energies: {[f'{e:.4f}' for e in span_E[:5]]}")
            if len(span_gaps) > 0:
                print(f"          Lock gaps: {[f'{g:.4f}' for g in span_gaps[:4]]}")
    
    print("\n  --- Asymmetry sweep (sep=15) ---")
    for asym in [0.0, 0.1, 0.3, 0.5, 1.0, 1.5]:
        dw = double_well_hamiltonian(well_sep=15, coupling=3.0, asymmetry=asym)
        n_span = dw['n_spanning']
        print(f"  Asym={asym:.1f}: {dw['n_bound']} bound, "
              f"{n_span} spanning (LOCKS), "
              f"split: {dw['n_order']}O/{dw['n_chaos']}C")
        
        if n_span > 0:
            span_E = dw['energies'][dw['spanning_indices']]
            print(f"          Lock energies: {[f'{e:.4f}' for e in span_E[:5]]}")
    
    # ================================================================
    # EXPERIMENT D: Detailed lock state analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT D: Detailed Quantum Lock Analysis")
    print("=" * 70)
    
    # Optimal parameters from sweeps
    dw = double_well_hamiltonian(well_sep=12, coupling=3.0, well_width=8, asymmetry=0.3)
    
    print(f"\n  Total bound states: {dw['n_bound']}")
    print(f"  Spanning (quantum locks): {dw['n_spanning']}")
    print(f"  Order-localized: {dw['n_order']}")
    print(f"  Chaos-localized: {dw['n_chaos']}")
    
    if dw['n_spanning'] > 1:
        span_idx = dw['spanning_indices']
        span_E = dw['energies'][span_idx]
        
        print(f"\n  QUANTUM LOCK ENERGIES:")
        for i, (idx, e) in enumerate(zip(span_idx, span_E)):
            psi = dw['wavefunctions'][:, idx]
            left_w = np.sum(psi[dw['x'] < 0]**2)
            right_w = np.sum(psi[dw['x'] > 0]**2)
            balance = min(left_w, right_w) / max(left_w, right_w)
            print(f"    Lock {i}: E={e:.6f}, balance={balance:.3f} (1.0=perfect)")
        
        span_gaps = np.diff(span_E)
        if len(span_gaps) > 1:
            span_gap_ratios = span_gaps[:-1] / span_gaps[1:]
            
            print(f"\n  LOCK ENERGY GAPS: {[f'{g:.6f}' for g in span_gaps[:6]]}")
            print(f"  GAP RATIOS: {[f'{r:.4f}' for r in span_gap_ratios[:5]]}")
            
            phi = (1 + np.sqrt(5)) / 2
            xi = 1.0571
            
            print(f"\n  COMPARISON TO PAC CONSTANTS:")
            print(f"  φ = {phi:.4f}, 1/φ = {1/phi:.4f}")
            print(f"  ξ_PAC = {xi:.4f}")
            print(f"  4/π = {4/np.pi:.4f}")
            
            for i, r in enumerate(span_gap_ratios[:5]):
                comparisons = [
                    ('φ', phi), ('1/φ', 1/phi), ('ξ_PAC', xi),
                    ('4/π', 4/np.pi), ('π/2', np.pi/2), ('√2', np.sqrt(2)),
                    ('2', 2.0), ('3', 3.0), ('1', 1.0)
                ]
                closest = min(comparisons, key=lambda c: abs(r - c[1]))
                dev = abs(r - closest[1])
                print(f"    Ratio {i}: {r:.4f} ≈ {closest[0]} ({closest[1]:.4f}), dev={dev:.4f}")
    
    # All bound state spectrum
    if len(dw['energies']) > 2:
        all_gaps = np.diff(dw['energies'])
        all_gap_ratios = all_gaps[:-1] / all_gaps[1:]
        
        print(f"\n  FULL SPECTRUM GAP RATIOS (first 10):")
        for i, r in enumerate(all_gap_ratios[:10]):
            print(f"    Level {i}-{i+1}/{i+1}-{i+2}: {r:.4f}")
        
        if len(all_gap_ratios) > 5:
            mean_late = np.mean(all_gap_ratios[len(all_gap_ratios)//2:])
            std_late = np.std(all_gap_ratios[len(all_gap_ratios)//2:])
            print(f"\n  Late gap ratio convergence: {mean_late:.4f} ± {std_late:.4f}")
    
    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("OVERALL FINDINGS")
    print("=" * 70)
    print("""
    KEY QUESTIONS AND RESULTS:
    
    Q1: Do discrete bound states emerge at the order-chaos boundary?
    Q2: Do bound states that SPAN both wells (quantum locks) exist?
    Q3: Does the spectrum depend on topology (boundary shape)?
    Q4: Does the spectrum show structure related to PAC constants?
    Q5: Do conservative wave dynamics produce discrete frequencies?
    """)
