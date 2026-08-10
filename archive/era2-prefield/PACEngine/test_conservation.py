from core.lattice_substrate import MultiScaleLatticeSubstrate, ScaleType

lattice = MultiScaleLatticeSubstrate((4,4,4), active_scales=[ScaleType.QUANTUM])
print('Before evolution:')
conservation = lattice.pac_kernel.check_global_conservation()
print(f'  Conservation quality: {conservation["conservation_quality"]:.6f}')
print(f'  Mean residual: {conservation["mean_residual"]:.6f}')
print(f'  Violations: {conservation["violation_count"]}')

lattice.evolve_step()
print('\nAfter 1 evolution step:')
conservation = lattice.pac_kernel.check_global_conservation()
print(f'  Conservation quality: {conservation["conservation_quality"]:.6f}')
print(f'  Mean residual: {conservation["mean_residual"]:.6f}')  
print(f'  Violations: {conservation["violation_count"]}')
