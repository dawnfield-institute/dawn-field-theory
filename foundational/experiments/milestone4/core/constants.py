"""
Milestone 4: Shared constants for all experiments.

Extends milestone3 constants with relativity and nuclear physics values.
"""

import math

# ============================================================
# Fundamental mathematical constants
# ============================================================
PHI = (1 + math.sqrt(5)) / 2         # Golden ratio: 1.6180339887...
INV_PHI = 1 / PHI                     # 1/φ: 0.6180339887...
LN_PHI = math.log(PHI)                # ln(φ): 0.4812118250...
GAMMA_EM = 0.5772156649015329         # Euler-Mascheroni constant

# Framework constants
XI_BALANCE = GAMMA_EM + LN_PHI        # Ξ = γ + ln(φ) ≈ 1.0584
PI_OVER_55 = math.pi / 55             # π/55 ≈ 0.05712

# ============================================================
# Thermodynamic constants
# ============================================================
KT_DEFAULT = 1.0                       # Default thermal energy (natural units)
LANDAUER_MIN = KT_DEFAULT * math.log(2)  # kT·ln(2) ≈ 0.6931
K_BOLTZMANN = 1.380649e-23             # J/K (exact, SI 2019)

# ============================================================
# Physical constants
# ============================================================
C_LIGHT = 299792458.0                  # m/s (exact)
C_LIGHT_SQ = C_LIGHT ** 2             # c² 
HBAR = 1.054571817e-34                 # J·s (reduced Planck)
G_NEWTON = 6.67430e-11                 # m³/(kg·s²)
M_ELECTRON = 9.1093837015e-31         # kg
M_PROTON = 1.67262192369e-27          # kg
M_NEUTRON = 1.67492749804e-27         # kg
AMU = 1.66053906660e-27               # kg (atomic mass unit)
MEV_TO_JOULE = 1.602176634e-13        # J per MeV
KEV_TO_JOULE = 1.602176634e-16        # J per keV

# ============================================================
# Nuclear physics data (NIST/NNDC)
# ============================================================

# Binding energy per nucleon (MeV) for key nuclides
# Source: AME2020 (Atomic Mass Evaluation)
BINDING_ENERGY_PER_NUCLEON = {
    # (Z, A): BE/A in MeV
    (1, 1): 0.0,         # H-1 (free proton)
    (1, 2): 1.112,       # H-2 (deuterium)
    (1, 3): 2.827,       # H-3 (tritium)
    (2, 3): 2.573,       # He-3
    (2, 4): 7.074,       # He-4 (alpha)
    (3, 6): 5.332,       # Li-6
    (3, 7): 5.606,       # Li-7
    (6, 12): 7.680,      # C-12
    (7, 14): 7.476,      # N-14
    (8, 16): 7.976,      # O-16
    (12, 24): 8.261,     # Mg-24
    (14, 28): 8.448,     # Si-28
    (20, 40): 8.551,     # Ca-40
    (26, 56): 8.790,     # Fe-56 (PEAK)
    (28, 58): 8.732,     # Ni-58
    (28, 62): 8.795,     # Ni-62 (true peak by total BE)
    (36, 84): 8.717,     # Kr-84
    (38, 88): 8.733,     # Sr-88
    (50, 120): 8.505,    # Sn-120
    (54, 131): 8.424,    # Xe-131
    (56, 138): 8.394,    # Ba-138
    (82, 208): 7.867,    # Pb-208
    (90, 232): 7.615,    # Th-232
    (92, 235): 7.591,    # U-235
    (92, 238): 7.570,    # U-238
    (94, 239): 7.560,    # Pu-239
    (94, 244): 7.523,    # Pu-244
}

# Nuclear magic numbers (closed shells)
MAGIC_NUMBERS = [2, 8, 20, 28, 50, 82, 126]

# U-235 fission data
U235_FISSION = {
    'energy_mev': 200.0,           # Average total energy per fission (MeV)
    'kinetic_mev': 170.0,          # Kinetic energy of products
    'gamma_mev': 7.0,              # Prompt gamma rays
    'beta_mev': 8.0,               # Beta decay energy
    'neutrino_mev': 12.0,          # Neutrinos (lost)
    'delayed_gamma_mev': 3.0,      # Delayed gammas
    'mass_defect_fraction': 0.001, # Δm/m
    'neutrons_per_fission': 2.43,  # Average prompt neutrons
    'primary_channels': 60,         # Approximate distinct fission channels
    'daughter_count': 800,          # Approximate distinct fission products
}

# Known nuclide data for configuration space analysis
# decay_modes: number of energetically accessible decay channels
NUCLIDE_DECAY_DATA = {
    # (Z, A): {'half_life_s': t, 'decay_modes': n, 'name': str}
    (1, 3):   {'half_life_s': 3.888e8, 'decay_modes': 1, 'name': 'H-3'},
    (6, 14):  {'half_life_s': 1.808e11, 'decay_modes': 1, 'name': 'C-14'},
    (11, 22): {'half_life_s': 8.211e7, 'decay_modes': 2, 'name': 'Na-22'},
    (19, 40): {'half_life_s': 3.938e16, 'decay_modes': 3, 'name': 'K-40'},
    (27, 60): {'half_life_s': 1.663e8, 'decay_modes': 2, 'name': 'Co-60'},
    (38, 90): {'half_life_s': 9.08e8, 'decay_modes': 1, 'name': 'Sr-90'},
    (53, 131): {'half_life_s': 6.95e5, 'decay_modes': 2, 'name': 'I-131'},
    (55, 137): {'half_life_s': 9.49e8, 'decay_modes': 2, 'name': 'Cs-137'},
    (84, 210): {'half_life_s': 1.196e7, 'decay_modes': 2, 'name': 'Po-210'},
    (86, 222): {'half_life_s': 3.304e5, 'decay_modes': 1, 'name': 'Rn-222'},
    (88, 226): {'half_life_s': 5.049e10, 'decay_modes': 2, 'name': 'Ra-226'},
    (90, 232): {'half_life_s': 4.42e17, 'decay_modes': 2, 'name': 'Th-232'},
    (92, 235): {'half_life_s': 2.22e16, 'decay_modes': 3, 'name': 'U-235'},
    (92, 238): {'half_life_s': 1.41e17, 'decay_modes': 3, 'name': 'U-238'},
    (94, 239): {'half_life_s': 7.61e11, 'decay_modes': 2, 'name': 'Pu-239'},
}

# Comprehensive excited state / level density data
# Cumulative number of known levels below given energy
# Source: RIPL-3 (Reference Input Parameter Library)
NUCLEAR_LEVEL_DENSITY = {
    # (Z, A): {'levels_below_5MeV': n, 'levels_below_10MeV': n}
    (26, 56): {'levels_below_5MeV': 25, 'levels_below_10MeV': 180},
    (28, 62): {'levels_below_5MeV': 20, 'levels_below_10MeV': 150},
    (50, 120): {'levels_below_5MeV': 60, 'levels_below_10MeV': 800},
    (82, 208): {'levels_below_5MeV': 35, 'levels_below_10MeV': 400},
    (92, 235): {'levels_below_5MeV': 200, 'levels_below_10MeV': 5000},
    (92, 238): {'levels_below_5MeV': 150, 'levels_below_10MeV': 4000},
}

# ============================================================
# Fibonacci sequence
# ============================================================
def fibonacci(n):
    """Return the nth Fibonacci number (0-indexed: F(0)=0, F(1)=1, ...)."""
    if n < 0:
        raise ValueError(f"Negative index: {n}")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

FIB = [fibonacci(i) for i in range(21)]

# ============================================================
# Turbulence reference values
# ============================================================
KOLMOGOROV_EXPONENT = -5/3            # Kolmogorov -5/3 law
SHE_LEVEQUE_BETA = 2/3               # She-Lévêque β = F₃/F₄
ORGANIZED_FRACTION_TARGET = 1 - 2**(-5/3)  # ≈ 0.685 for exact -5/3
