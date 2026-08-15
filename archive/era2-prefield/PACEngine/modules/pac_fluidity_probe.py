"""
PAC Mesh Fluidity Probe
======================

Investigating: Does the PAC mesh structure itself exhibit fluid-like behavior?

Key questions:
1. Does spacing cascade like turbulent energy?
2. Does the mesh satisfy continuity-like constraints?
3. Is there a Kolmogorov-like spectrum in the structure?
4. Does XI-balanced subdivision create turbulence-compliant geometry?
"""

import numpy as np
from bifractal_pac_mesh import BiFractalPACMesh, XI


def probe_fluid_behavior():
    print("="*70)
    print("PROBING FLUID-LIKE BEHAVIOR IN PAC MESH STRUCTURE")
    print("="*70)

    H = 1.0
    mesh_gen = BiFractalPACMesh(H=H)

    # Generate deep mesh to see cascade
    depth = 8
    pac_mesh = mesh_gen.generate_bifractal_mesh(depth)
    spacings = np.diff(pac_mesh)

    print(f"\nMesh depth: {depth}, Points: {len(pac_mesh)}")
    print(f"Balance operator XI = {XI:.4f}")

    # 1. Check for power-law scaling in spacing distribution
    print("\n1. SPACING DISTRIBUTION (looking for power-law)")
    print("-"*50)

    # Histogram of spacings
    hist, bin_edges = np.histogram(spacings, bins=20)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Log-log plot data
    valid = hist > 0
    if np.sum(valid) > 3:
        log_spacing = np.log10(bin_centers[valid])
        log_count = np.log10(hist[valid])
        
        # Fit power law: count ~ spacing^alpha
        coeffs = np.polyfit(log_spacing, log_count, 1)
        alpha = coeffs[0]
        
        print(f"Power-law exponent alpha = {alpha:.3f}")
        print(f"(Kolmogorov turbulence has -5/3 = -1.67 for energy spectrum)")

    # 2. Check for self-similarity across scales
    print("\n2. SELF-SIMILARITY ACROSS SCALES")
    print("-"*50)

    # Compute ratio of spacings at consecutive tree levels
    level_spacings = []
    for d in range(3, depth+1):
        mesh_d = mesh_gen.generate_bifractal_mesh(d)
        sp = np.diff(mesh_d)
        level_spacings.append({
            "depth": d,
            "mean": np.mean(sp),
            "min": np.min(sp),
            "max": np.max(sp)
        })

    print("Level-to-level scaling:")
    for i in range(1, len(level_spacings)):
        prev = level_spacings[i-1]
        curr = level_spacings[i]
        ratio = prev["mean"] / curr["mean"]
        expected_xi = 1 + XI
        print(f"  Depth {prev['depth']}->{curr['depth']}: ratio = {ratio:.4f} (binary=2.0, XI-based={expected_xi:.3f})")

    # 3. Check spacing gradient - does it behave like velocity field?
    print("\n3. SPACING AS VELOCITY FIELD")
    print("-"*50)

    # Treat spacing as a velocity at each point
    y_mid = (pac_mesh[:-1] + pac_mesh[1:]) / 2  # midpoints
    v = spacings  # velocity = local spacing

    # Compute divergence - rate of change of spacing
    dv_dy = np.gradient(v, y_mid)

    print(f"Spacing field statistics:")
    print(f"  Mean velocity: {np.mean(v):.4f}")
    print(f"  Max divergence:  {np.max(np.abs(dv_dy)):.4f}")
    print(f"  Mean divergence: {np.mean(dv_dy):.6f} (should be ~0 for incompressible)")
    
    # Check symmetry (fluid-like flows often have symmetry)
    left_half = v[:len(v)//2]
    right_half = v[len(v)//2:][::-1]  # Reversed
    symmetry_error = np.mean(np.abs(left_half - right_half[:len(left_half)]))
    print(f"  Symmetry error: {symmetry_error:.6f} (0 = perfect reflection)")

    # 4. Energy-like cascade
    print("\n4. ENERGY CASCADE STRUCTURE")
    print("-"*50)

    # Define energy at each scale as 1/spacing^2 (like kinetic energy ~ v^2)
    energy = 1.0 / spacings**2

    # Compute spectrum by binning
    n_bins = 10
    bin_edges_e = np.linspace(y_mid.min(), y_mid.max(), n_bins+1)
    energy_spectrum = []

    for i in range(n_bins):
        mask = (y_mid >= bin_edges_e[i]) & (y_mid < bin_edges_e[i+1])
        if np.sum(mask) > 0:
            energy_spectrum.append(np.mean(energy[mask]))
        else:
            energy_spectrum.append(0)

    energy_spectrum = np.array(energy_spectrum)
    print(f"Energy distribution (boundary -> center -> boundary):")
    print(f"  Left boundary:  {energy_spectrum[:3]}")
    print(f"  Center:         {energy_spectrum[4:6]}")
    print(f"  Right boundary: {energy_spectrum[-3:]}")
    if energy_spectrum[5] > 0:
        print(f"  Ratio boundary/center: {energy_spectrum[0]/energy_spectrum[5]:.2f}x")

    # 5. CRITICAL: Does it satisfy Reynolds number scaling?
    print("\n5. REYNOLDS-LIKE SCALING")
    print("-"*50)
    
    # In turbulence: Re = UL/ν
    # In our mesh: Re_mesh = (spacing_max * domain_size) / spacing_min^2
    # This measures the "range of scales" in the mesh
    
    L = 2 * H  # Domain size
    Re_mesh = (np.max(spacings) * L) / (np.min(spacings)**2)
    print(f"Mesh Reynolds number: Re_mesh = {Re_mesh:.1f}")
    print(f"  (Higher = more scale separation, like higher Re turbulence)")
    
    # Kolmogorov microscale ratio: η/L ~ Re^(-3/4)
    # In our mesh: spacing_min/L should follow similar scaling
    eta_ratio = np.min(spacings) / L
    predicted_ratio = Re_mesh**(-3/4)
    print(f"  Smallest scale ratio: {eta_ratio:.4f}")
    print(f"  Kolmogorov prediction Re^(-3/4): {predicted_ratio:.4f}")
    
    return {
        "power_law_exponent": alpha if "alpha" in dir() else None,
        "mean_divergence": np.mean(dv_dy),
        "symmetry_error": symmetry_error,
        "re_mesh": Re_mesh
    }


def probe_turbulence_compliance():
    """
    Does the PAC mesh structure satisfy turbulence constraints?
    
    Kolmogorov's theory says:
    - Energy cascades from large to small scales
    - E(k) ~ k^(-5/3) in inertial range
    - Dissipation at Kolmogorov scale η = (ν³/ε)^(1/4)
    
    Question: Does XI create a structure that naturally 
    respects these constraints?
    """
    print("\n" + "="*70)
    print("TURBULENCE COMPLIANCE CHECK")
    print("="*70)
    
    H = 1.0
    mesh_gen = BiFractalPACMesh(H=H)
    
    # Generate meshes at different depths
    print("\nWavenumber spectrum analysis:")
    print("-"*50)
    
    for depth in [5, 6, 7, 8]:
        pac_mesh = mesh_gen.generate_bifractal_mesh(depth)
        spacings = np.diff(pac_mesh)
        
        # Convert spacings to "wavenumbers" k = 1/spacing
        k = 1.0 / spacings
        
        # Energy at each wavenumber E(k) ~ spacing^2 = 1/k^2
        E_k = spacings**2
        
        # Sort by wavenumber
        sort_idx = np.argsort(k)
        k_sorted = k[sort_idx]
        E_sorted = E_k[sort_idx]
        
        # Fit power law in log space: log(E) = α*log(k) + c
        # Kolmogorov predicts α = -5/3 ≈ -1.67
        log_k = np.log10(k_sorted)
        log_E = np.log10(E_sorted)
        
        # Fit only in "inertial range" (middle portion)
        n = len(log_k)
        inertial_start = n // 4
        inertial_end = 3 * n // 4
        
        coeffs = np.polyfit(log_k[inertial_start:inertial_end], 
                           log_E[inertial_start:inertial_end], 1)
        spectral_slope = coeffs[0]
        
        print(f"Depth {depth} (N={len(pac_mesh)}): E(k) ~ k^{spectral_slope:.3f}")
        print(f"    Kolmogorov: k^-1.67, Actual: k^{spectral_slope:.2f}")
        print(f"    Deviation from -5/3: {abs(spectral_slope + 5/3):.3f}")
    
    # Check if XI itself encodes turbulence
    print("\n" + "-"*50)
    print("XI RELATIONSHIP TO TURBULENCE:")
    print("-"*50)
    
    # The -5/3 law comes from dimensional analysis
    # E(k) = C * ε^(2/3) * k^(-5/3)
    # 
    # Our XI = 1.0571 gives split ratio 1/(1+XI) = 0.486
    # 
    # In turbulence: energy ratio between scales = (k₁/k₂)^(-5/3)
    # In our mesh: spacing ratio = XI
    
    # If we subdivide by XI factor, the energy ratio is:
    energy_ratio = XI**2  # Because E ~ spacing² ~ 1/k²
    
    # In Kolmogorov: if wavenumber doubles, E changes by 2^(-5/3) = 0.315
    kolmogorov_ratio = 2**(-5/3)
    
    print(f"XI-based energy ratio: {energy_ratio:.4f}")
    print(f"Kolmogorov (k→2k): {kolmogorov_ratio:.4f}")
    print(f"Ratio: {energy_ratio/kolmogorov_ratio:.3f}")
    
    # What XI would give Kolmogorov scaling?
    xi_kolmogorov = np.sqrt(kolmogorov_ratio)
    print(f"\nXI that would match Kolmogorov: {xi_kolmogorov:.4f}")
    print(f"Actual XI: {XI:.4f}")
    print(f"Difference: {abs(XI - xi_kolmogorov):.4f}")


def probe_cascade_dynamics():
    """
    Does the PAC structure show cascade-like dynamics when evolved?
    
    If we treat the mesh spacing as a fluid property and "flow" it,
    does it maintain structure or develop turbulence?
    """
    print("\n" + "="*70)
    print("CASCADE DYNAMICS PROBE")
    print("="*70)
    
    H = 1.0
    mesh_gen = BiFractalPACMesh(H=H)
    
    # Generate initial mesh
    depth = 6
    pac_mesh = mesh_gen.generate_bifractal_mesh(depth)
    spacings = np.diff(pac_mesh)
    y_mid = (pac_mesh[:-1] + pac_mesh[1:]) / 2
    
    print(f"\nInitial mesh: {len(pac_mesh)} points")
    
    # Treat spacings as a conserved "density" field
    # Apply diffusion: ∂ρ/∂t = D ∂²ρ/∂y²
    # Check if it stays PAC-conserved
    
    rho = spacings.copy()
    total_initial = np.sum(rho)
    
    D = 0.01  # Diffusion coefficient
    dt = 0.001
    n_steps = 100
    
    print(f"\nEvolving spacing field with diffusion (D={D}, dt={dt}, steps={n_steps})")
    
    conservation_errors = []
    
    for step in range(n_steps):
        # Compute Laplacian (second derivative)
        d2rho = np.zeros_like(rho)
        for i in range(1, len(rho)-1):
            dy = y_mid[i+1] - y_mid[i-1]
            d2rho[i] = (rho[i+1] - 2*rho[i] + rho[i-1]) / (dy/2)**2
        
        # Update
        rho = rho + dt * D * d2rho
        
        # Check conservation
        total_now = np.sum(rho)
        conservation_errors.append(abs(total_now - total_initial) / total_initial)
    
    print(f"Conservation error after evolution: {conservation_errors[-1]:.2e}")
    print(f"Max conservation error: {max(conservation_errors):.2e}")
    
    # Did it maintain PAC-like structure?
    final_ratio = np.max(rho) / np.min(rho)
    initial_ratio = np.max(spacings) / np.min(spacings)
    
    print(f"\nStructure preservation:")
    print(f"  Initial max/min ratio: {initial_ratio:.2f}")
    print(f"  Final max/min ratio: {final_ratio:.2f}")
    print(f"  Ratio change: {final_ratio/initial_ratio:.2f}x")
    
    if final_ratio < initial_ratio * 0.5:
        print("  -> Structure DISSIPATED (became more uniform)")
    elif final_ratio > initial_ratio * 1.5:
        print("  -> Structure AMPLIFIED (became more extreme)")
    else:
        print("  -> Structure PRESERVED (fluid-like stability)")


if __name__ == "__main__":
    results = probe_fluid_behavior()
    probe_turbulence_compliance()
    probe_cascade_dynamics()
