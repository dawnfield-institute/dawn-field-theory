"""
Herniation Boundary Simulation - Experiment 1
==============================================
Model: Order-chaos boundary where information and energy fields couple.

The idea:
- "Order" field: structured, low-entropy, information-carrying
- "Chaos" field: high-entropy, energy-carrying
- Boundary: where they interface and can quantum-lock

We model this as two coupled 1D fields on a lattice:
- ψ_order(x, t): information field (tends toward structure/pattern)
- ψ_chaos(x, t): energy field (tends toward disorder/diffusion)
- Coupling term: nonlinear binding at the boundary

Question: Do stable bound states with discrete frequencies emerge
at the interface? If so, what determines the frequency spectrum?

Peter McNally / Dawn Field Institute, 2026
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.linalg import eigh
import json

# =============================================================================
# PART 1: Coupled Field Dynamics
# =============================================================================

def run_coupled_field_simulation(
    N=256,           # lattice sites
    T=5000,          # time steps
    dt=0.005,        # time step size
    dx=1.0,          # spatial step
    D_order=0.1,     # diffusion coefficient for order field
    D_chaos=0.5,     # diffusion coefficient for chaos field (higher = more diffusive)
    coupling=1.0,    # coupling strength at boundary
    boundary_width=10,  # width of the interaction zone
    seed=42
):
    """
    Simulate two coupled fields:
    - Order field: diffuses slowly, has self-organizing nonlinearity
    - Chaos field: diffuses quickly, has dispersive nonlinearity
    - They couple nonlinearly in a boundary region
    """
    rng = np.random.default_rng(seed)
    
    # Initialize fields
    # Order: structured initial condition (standing wave pattern)
    x = np.arange(N) * dx
    psi_order = np.cos(2 * np.pi * x / N) * 0.5
    
    # Chaos: random/thermal initial condition
    psi_chaos = rng.normal(0, 0.5, N)
    
    # Boundary region: centered in the lattice
    center = N // 2
    boundary_mask = np.exp(-((x - x[center])**2) / (2 * boundary_width**2))
    
    # Storage for time evolution at boundary
    boundary_history = np.zeros((T, boundary_width * 4))
    energy_history = np.zeros(T)
    info_history = np.zeros(T)
    coupling_history = np.zeros(T)
    
    # Sample region around boundary
    sample_start = center - boundary_width * 2
    sample_end = center + boundary_width * 2
    
    for t in range(T):
        # Laplacian (periodic boundary conditions)
        lap_order = np.roll(psi_order, 1) + np.roll(psi_order, -1) - 2 * psi_order
        lap_chaos = np.roll(psi_chaos, 1) + np.roll(psi_chaos, -1) - 2 * psi_chaos
        
        # Self-interaction terms
        # Order field: tends to form patterns (cubic nonlinearity like Ginzburg-Landau)
        self_order = psi_order - psi_order**3
        
        # Chaos field: tends to disperse (anti-pattern)
        self_chaos = -0.1 * psi_chaos**3
        
        # Coupling at boundary: order and chaos bind
        # The coupling creates a potential well where both fields lock together
        coupling_term = coupling * boundary_mask * psi_order * psi_chaos
        
        # Landauer cost: binding has minimum energy cost
        landauer_cost = 0.01 * boundary_mask * np.sign(psi_order * psi_chaos)
        
        # Update equations
        dpsi_order = dt * (D_order * lap_order / dx**2 + self_order + coupling_term - landauer_cost)
        dpsi_chaos = dt * (D_chaos * lap_chaos / dx**2 + self_chaos - coupling_term + landauer_cost)
        
        psi_order += dpsi_order
        psi_chaos += dpsi_chaos
        
        # Record boundary region
        boundary_history[t, :] = (psi_order[sample_start:sample_end] + 
                                   psi_chaos[sample_start:sample_end])
        
        # Track energies
        energy_history[t] = np.sum(psi_chaos**2) / N
        info_history[t] = np.sum(psi_order**2) / N
        coupling_history[t] = np.sum(np.abs(coupling_term)) / N
    
    return {
        'boundary_history': boundary_history,
        'energy_history': energy_history,
        'info_history': info_history,
        'coupling_history': coupling_history,
        'psi_order_final': psi_order,
        'psi_chaos_final': psi_chaos,
        'boundary_mask': boundary_mask,
        'x': x,
        'params': {
            'N': N, 'T': T, 'dt': dt, 'coupling': coupling,
            'D_order': D_order, 'D_chaos': D_chaos,
            'boundary_width': boundary_width
        }
    }


# =============================================================================
# PART 2: Frequency Analysis of Bound States
# =============================================================================

def analyze_bound_states(result):
    """
    Look for discrete frequency modes in the boundary region.
    If quantum locking produces stable bound states, we should see
    discrete peaks in the frequency spectrum.
    """
    bh = result['boundary_history']
    T = bh.shape[0]
    dt = result['params']['dt']
    
    # Skip transient (first 20%)
    start = T // 5
    bh_steady = bh[start:, :]
    
    # Frequency analysis at each boundary point
    freqs = fftfreq(bh_steady.shape[0], dt)
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    
    # Average power spectrum across boundary points
    avg_spectrum = np.zeros(pos_mask.sum())
    for i in range(bh_steady.shape[1]):
        spectrum = np.abs(fft(bh_steady[:, i]))**2
        avg_spectrum += spectrum[pos_mask]
    avg_spectrum /= bh_steady.shape[1]
    
    # Normalize
    avg_spectrum /= avg_spectrum.max()
    
    # Find peaks (discrete modes)
    peaks, properties = find_peaks(avg_spectrum, height=0.05, distance=10, prominence=0.02)
    
    peak_freqs = freqs_pos[peaks]
    peak_heights = avg_spectrum[peaks]
    
    # Sort by height
    sort_idx = np.argsort(peak_heights)[::-1]
    peak_freqs = peak_freqs[sort_idx]
    peak_heights = peak_heights[sort_idx]
    
    return {
        'frequencies': freqs_pos,
        'spectrum': avg_spectrum,
        'peak_frequencies': peak_freqs,
        'peak_heights': peak_heights,
        'n_modes': len(peak_freqs)
    }


# =============================================================================
# PART 3: Frequency Ratio Analysis
# =============================================================================

def analyze_frequency_ratios(peak_freqs):
    """
    If bound states are 'string-like', their frequencies should show
    structure — harmonic ratios, or ratios related to coupling topology.
    
    Compare to:
    - Harmonic series (n:1 ratios)
    - Known mass ratios of fundamental particles
    - Golden ratio relationships
    """
    if len(peak_freqs) < 2:
        return {'ratios': [], 'analysis': 'Too few peaks'}
    
    # Take top modes
    top = peak_freqs[:min(10, len(peak_freqs))]
    base = top[0]
    
    ratios = top / base
    
    # Check for harmonic structure
    nearest_int = np.round(ratios)
    harmonic_deviation = np.abs(ratios - nearest_int) / nearest_int
    
    # Check consecutive ratios
    consecutive_ratios = top[:-1] / top[1:] if len(top) > 1 else np.array([])
    
    # Golden ratio check
    phi = (1 + np.sqrt(5)) / 2
    phi_deviation = np.min(np.abs(consecutive_ratios - phi)) if len(consecutive_ratios) > 0 else float('inf')
    
    # Known particle mass ratios (rough, for reference)
    # up quark / down quark ≈ 0.47
    # electron / up quark ≈ 0.24
    # muon / electron ≈ 206.8
    # tau / muon ≈ 16.8
    
    return {
        'ratios_to_fundamental': ratios.tolist(),
        'consecutive_ratios': consecutive_ratios.tolist(),
        'harmonic_deviation': harmonic_deviation.tolist(),
        'mean_harmonic_deviation': float(np.mean(harmonic_deviation)),
        'phi_nearest': float(phi_deviation),
        'is_harmonic': bool(np.mean(harmonic_deviation) < 0.1),
        'is_phi_related': bool(phi_deviation < 0.05)
    }


# =============================================================================
# PART 4: Coupling Strength Sweep
# =============================================================================

def coupling_sweep(coupling_values=None, seeds=5):
    """
    Sweep coupling strength and see how bound state spectrum changes.
    This tests whether different coupling strengths produce different
    'particle' spectra — analogous to different lock frequencies
    producing different particle types.
    """
    if coupling_values is None:
        coupling_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    results = []
    
    for g in coupling_values:
        seed_results = []
        for s in range(seeds):
            sim = run_coupled_field_simulation(coupling=g, seed=s*100+42, T=3000)
            analysis = analyze_bound_states(sim)
            seed_results.append({
                'n_modes': analysis['n_modes'],
                'peak_freqs': analysis['peak_frequencies'][:5].tolist() if len(analysis['peak_frequencies']) > 0 else [],
                'peak_heights': analysis['peak_heights'][:5].tolist() if len(analysis['peak_heights']) > 0 else []
            })
        
        # Average across seeds
        avg_n_modes = np.mean([r['n_modes'] for r in seed_results])
        
        # Collect all peak frequencies across seeds
        all_freqs = []
        for r in seed_results:
            all_freqs.extend(r['peak_freqs'])
        
        results.append({
            'coupling': g,
            'avg_n_modes': float(avg_n_modes),
            'mode_counts': [r['n_modes'] for r in seed_results],
            'all_peak_freqs': sorted(all_freqs),
            'seed_details': seed_results
        })
        
        print(f"  Coupling g={g:.1f}: avg {avg_n_modes:.1f} modes across {seeds} seeds")
    
    return results


# =============================================================================
# PART 5: Boundary Width Sweep (topology dependence)
# =============================================================================

def boundary_sweep(widths=None, seeds=3):
    """
    Sweep boundary width — this is the 'herniation' size.
    Different widths = different topological configurations.
    Question: Does the spectrum depend on topology?
    """
    if widths is None:
        widths = [3, 5, 8, 10, 15, 20, 30, 50]
    
    results = []
    
    for w in widths:
        seed_results = []
        for s in range(seeds):
            sim = run_coupled_field_simulation(boundary_width=w, seed=s*100+42, T=3000)
            analysis = analyze_bound_states(sim)
            seed_results.append({
                'n_modes': analysis['n_modes'],
                'peak_freqs': analysis['peak_frequencies'][:5].tolist() if len(analysis['peak_frequencies']) > 0 else [],
            })
        
        avg_n_modes = np.mean([r['n_modes'] for r in seed_results])
        
        results.append({
            'boundary_width': w,
            'avg_n_modes': float(avg_n_modes),
            'mode_counts': [r['n_modes'] for r in seed_results],
            'seed_details': seed_results
        })
        
        print(f"  Width w={w}: avg {avg_n_modes:.1f} modes across {seeds} seeds")
    
    return results


# =============================================================================
# PART 6: Hamiltonian Analysis (eigenvalue approach)
# =============================================================================

def hamiltonian_bound_states(N=200, coupling=1.0, boundary_width=10):
    """
    Alternative approach: construct an effective Hamiltonian for the
    order-chaos boundary and solve for eigenvalues directly.
    
    This gives us the allowed energy levels (frequencies) of bound
    states at the herniation boundary.
    """
    dx = 1.0
    x = np.arange(N) * dx
    center = N // 2
    
    # Potential well from order-chaos coupling
    # The boundary creates a potential well where binding can occur
    V_coupling = -coupling * np.exp(-((x - x[center])**2) / (2 * boundary_width**2))
    
    # Kinetic energy (second derivative operator)
    T_mat = np.zeros((N, N))
    for i in range(N):
        T_mat[i, i] = -2.0
        T_mat[i, (i+1) % N] = 1.0
        T_mat[i, (i-1) % N] = 1.0
    T_mat *= -0.5 / dx**2
    
    # Hamiltonian
    H = T_mat + np.diag(V_coupling)
    
    # Solve for eigenvalues
    eigenvalues, eigenvectors = eigh(H)
    
    # Bound states have negative energy (below the potential well)
    bound_mask = eigenvalues < 0
    bound_energies = eigenvalues[bound_mask]
    bound_states = eigenvectors[:, bound_mask]
    
    # Energy ratios
    if len(bound_energies) > 1:
        ratios = bound_energies / bound_energies[0]
    else:
        ratios = np.array([1.0]) if len(bound_energies) > 0 else np.array([])
    
    return {
        'n_bound_states': int(bound_mask.sum()),
        'bound_energies': bound_energies.tolist(),
        'energy_ratios': ratios.tolist(),
        'bound_wavefunctions': bound_states,
        'potential': V_coupling,
        'x': x
    }


# =============================================================================
# MAIN: Run all experiments
# =============================================================================

if __name__ == '__main__':
    
    print("=" * 70)
    print("HERNIATION BOUNDARY SIMULATION")
    print("Order-Chaos Quantum Locking → Bound State Frequencies")
    print("=" * 70)
    
    # ---- Experiment 1: Single simulation, detailed analysis ----
    print("\n[Exp 1] Single simulation with default parameters...")
    sim1 = run_coupled_field_simulation(T=5000)
    freq1 = analyze_bound_states(sim1)
    ratios1 = analyze_frequency_ratios(freq1['peak_frequencies'])
    
    print(f"  Found {freq1['n_modes']} discrete frequency modes at boundary")
    if len(freq1['peak_frequencies']) > 0:
        print(f"  Top 5 peak frequencies: {freq1['peak_frequencies'][:5]}")
        print(f"  Ratios to fundamental: {ratios1['ratios_to_fundamental'][:5]}")
        print(f"  Consecutive ratios: {ratios1['consecutive_ratios'][:5]}")
        print(f"  Harmonic? {ratios1['is_harmonic']} (mean dev: {ratios1['mean_harmonic_deviation']:.4f})")
        print(f"  φ-related? {ratios1['is_phi_related']} (nearest dev: {ratios1['phi_nearest']:.4f})")
    
    # ---- Experiment 2: Coupling sweep ----
    print(f"\n[Exp 2] Coupling strength sweep...")
    coupling_results = coupling_sweep()
    
    # ---- Experiment 3: Boundary width sweep ----
    print(f"\n[Exp 3] Boundary width (herniation size) sweep...")
    boundary_results = boundary_sweep()
    
    # ---- Experiment 4: Hamiltonian eigenvalue analysis ----
    print(f"\n[Exp 4] Hamiltonian bound state analysis...")
    
    print("\n  Coupling sweep (Hamiltonian):")
    ham_coupling_results = []
    for g in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
        ham = hamiltonian_bound_states(coupling=g)
        ham_coupling_results.append({
            'coupling': g,
            'n_bound': ham['n_bound_states'],
            'energies': ham['bound_energies'][:5],
            'ratios': ham['energy_ratios'][:5]
        })
        print(f"    g={g:.1f}: {ham['n_bound_states']} bound states, "
              f"energies={[f'{e:.4f}' for e in ham['bound_energies'][:3]]}")
    
    print("\n  Boundary width sweep (Hamiltonian):")
    ham_width_results = []
    for w in [3, 5, 10, 15, 20, 30, 50]:
        ham = hamiltonian_bound_states(boundary_width=w)
        ham_width_results.append({
            'width': w,
            'n_bound': ham['n_bound_states'],
            'energies': ham['bound_energies'][:5],
            'ratios': ham['energy_ratios'][:5]
        })
        print(f"    w={w}: {ham['n_bound_states']} bound states, "
              f"energies={[f'{e:.4f}' for e in ham['bound_energies'][:3]]}")
    
    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n1. DISCRETE MODES EXIST: {freq1['n_modes'] > 0}")
    print(f"   Number of modes found: {freq1['n_modes']}")
    
    print(f"\n2. MODES ARE COUPLING-DEPENDENT:")
    for r in coupling_results:
        print(f"   g={r['coupling']:.1f} → {r['avg_n_modes']:.1f} modes")
    
    print(f"\n3. MODES ARE TOPOLOGY-DEPENDENT:")
    for r in boundary_results:
        print(f"   w={r['boundary_width']} → {r['avg_n_modes']:.1f} modes")
    
    print(f"\n4. HAMILTONIAN BOUND STATES:")
    for r in ham_coupling_results:
        print(f"   g={r['coupling']:.1f} → {r['n_bound']} bound states")
    
    if ratios1['consecutive_ratios']:
        print(f"\n5. FREQUENCY STRUCTURE:")
        print(f"   Harmonic: {ratios1['is_harmonic']}")
        print(f"   φ-related: {ratios1['is_phi_related']}")
        print(f"   Consecutive ratios: {[f'{r:.3f}' for r in ratios1['consecutive_ratios'][:5]]}")
    
    # ---- Detailed Hamiltonian analysis at optimal coupling ----
    print(f"\n6. DETAILED BOUND STATE SPECTRUM (g=5.0, w=10):")
    ham_detail = hamiltonian_bound_states(coupling=5.0, boundary_width=10)
    if ham_detail['n_bound_states'] > 1:
        energies = np.array(ham_detail['bound_energies'])
        gaps = np.diff(energies)
        gap_ratios = gaps[:-1] / gaps[1:] if len(gaps) > 1 else np.array([])
        
        print(f"   Bound state energies: {[f'{e:.4f}' for e in energies[:8]]}")
        print(f"   Energy gaps: {[f'{g:.4f}' for g in gaps[:7]]}")
        if len(gap_ratios) > 0:
            print(f"   Gap ratios: {[f'{r:.4f}' for r in gap_ratios[:6]]}")
            print(f"   Mean gap ratio: {np.mean(gap_ratios):.4f}")
            
            # Check for interesting constants
            phi = (1 + np.sqrt(5)) / 2
            pi = np.pi
            print(f"\n   Comparison to constants:")
            print(f"   φ = {phi:.4f}")
            print(f"   π = {pi:.4f}")
            print(f"   4/π = {4/pi:.4f}")
            
            for i, r in enumerate(gap_ratios[:6]):
                closest = min(
                    ('φ', abs(r - phi)),
                    ('1/φ', abs(r - 1/phi)),
                    ('π', abs(r - pi)),
                    ('4/π', abs(r - 4/pi)),
                    ('2', abs(r - 2)),
                    ('3', abs(r - 3)),
                    ('√2', abs(r - np.sqrt(2))),
                    key=lambda x: x[1]
                )
                print(f"   Gap ratio {i}: {r:.4f} (closest: {closest[0]}, dev={closest[1]:.4f})")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
    If discrete bound states emerge at the order-chaos boundary with
    frequencies that depend on coupling strength and boundary topology,
    this supports the hypothesis that:
    
    1. 'Quantum locking' at the herniation boundary produces stable
       bound states — information-energy pairs locked together.
    
    2. The frequency/energy of the lock determines the 'type' of
       bound state — analogous to different particle types.
    
    3. The spectrum is determined by topology (boundary geometry)
       and coupling strength — not by spatial dimensions.
    
    4. These bound states are inherently 1D (one info bit + one
       energy bit = one edge in the PAC tree) — they ARE strings,
       but strings whose vibration is set by the actualization
       topology, not by extra spatial dimensions.
    """)
