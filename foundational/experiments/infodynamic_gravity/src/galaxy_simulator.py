"""
Galaxy-scale Dark Matter Simulation using Infodynamic Gravity + SEC

Combines InfoGravityField and SECDynamics to simulate galaxy formation
and test dark matter emergence through quantum coherence floors.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
import time

from infodynamic_gravity import InfoGravityField, InfoGravityConfig
from sec_dynamics import SECDynamics, SECConfig

# Astronomical constants
SOLAR_MASS = 1.989e30  # kg
KPC_TO_METERS = 3.086e19  # meters per kiloparsec
MYR_TO_SECONDS = 3.15e13  # seconds per million years
KM_S_TO_M_S = 1000  # km/s to m/s conversion

@dataclass
class GalaxyConfig:
    """Configuration for galaxy simulation"""
    N_particles: int = 1000           # Number of simulation particles
    disk_scale_length: float = 10.0   # Scale length in kpc
    disk_height: float = 1.0          # Disk height in kpc  
    velocity_dispersion: float = 50.0 # Velocity dispersion in km/s
    total_mass: float = 1e12          # Total galaxy mass in solar masses
    dark_matter_fraction: float = 0.85 # Fraction of mass in dark matter
    bulge_fraction: float = 0.2       # Fraction in central bulge
    
class GalaxySimulator:
    """
    Galaxy-scale simulation combining infodynamic gravity and SEC dynamics
    
    Tests whether infodynamic principles can reproduce:
    - Flat rotation curves (dark matter effect)
    - Galaxy structure formation
    - Correct scaling laws
    """
    
    def __init__(self, 
                 galaxy_config: GalaxyConfig,
                 gravity_config: Optional[InfoGravityConfig] = None,
                 sec_config: Optional[SECConfig] = None):
        
        self.galaxy_config = galaxy_config
        
        # Initialize physics modules
        if gravity_config is None:
            gravity_config = InfoGravityConfig(
                lambda_c=galaxy_config.disk_scale_length * KPC_TO_METERS,
                T_info=2.7,  # CMB temperature
                alpha_info=1e-6,
                quantum_floor=0.15  # 15% quantum coherence floor
            )
        
        if sec_config is None:
            sec_config = SECConfig(
                collapse_threshold=0.8,
                stabilization_factor=0.9,
                force_amplification=1e5
            )
        
        self.gravity_field = InfoGravityField(gravity_config)
        self.sec_dynamics = SECDynamics(sec_config)
        
        # Simulation state
        self.state = self._initialize_galaxy()
        self.time = 0.0
        self.trajectory_data = []
        
        # Analysis data
        self.rotation_curves = []
        self.structure_evolution = []
        
        logging.info(f"Galaxy simulator initialized: {galaxy_config.N_particles} particles, "
                    f"λ_c={gravity_config.lambda_c/KPC_TO_METERS:.1f} kpc")
    
    def _initialize_galaxy(self) -> Dict[str, Any]:
        """
        Initialize realistic galaxy structure
        
        Creates:
        - Exponential disk profile
        - Central bulge component  
        - Approximate circular velocities
        - Mixed baryonic/dark matter particles
        """
        N = self.galaxy_config.N_particles
        
        # Disk component (80% of particles)
        N_disk = int(0.8 * N)
        
        # Exponential disk radial profile
        scale_length = self.galaxy_config.disk_scale_length * KPC_TO_METERS
        r_disk = np.random.exponential(scale_length, N_disk)
        theta_disk = np.random.uniform(0, 2*np.pi, N_disk)
        
        # Gaussian vertical distribution
        z_height = self.galaxy_config.disk_height * KPC_TO_METERS
        z_disk = np.random.normal(0, z_height, N_disk)
        
        disk_positions = np.column_stack([
            r_disk * np.cos(theta_disk),
            r_disk * np.sin(theta_disk),
            z_disk
        ])
        
        # Bulge component (20% of particles)
        N_bulge = N - N_disk
        
        # Spherical bulge with smaller scale
        r_bulge = np.random.exponential(scale_length * 0.3, N_bulge)
        theta_bulge = np.random.uniform(0, 2*np.pi, N_bulge)
        phi_bulge = np.random.uniform(0, np.pi, N_bulge)
        
        bulge_positions = np.column_stack([
            r_bulge * np.sin(phi_bulge) * np.cos(theta_bulge),
            r_bulge * np.sin(phi_bulge) * np.sin(theta_bulge),
            r_bulge * np.cos(phi_bulge)
        ])
        
        # Combine positions
        positions = np.vstack([disk_positions, bulge_positions])
        
        # Initial velocities - approximate circular motion for disk
        velocities = np.zeros((N, 3))
        
        # Disk velocities (circular + dispersion)
        total_mass = self.galaxy_config.total_mass * SOLAR_MASS
        for i in range(N_disk):
            r = r_disk[i]
            
            # Rough circular velocity (Keplerian inside scale length)
            if r > 0:
                # Simplified mass profile - should give reasonable rotation curve
                enclosed_mass = total_mass * (1 - np.exp(-r/scale_length))
                v_circular = np.sqrt(6.67e-11 * enclosed_mass / r)
            else:
                v_circular = 0
            
            # Add circular component
            velocities[i, 0] = -v_circular * np.sin(theta_disk[i])
            velocities[i, 1] = v_circular * np.cos(theta_disk[i])
            
            # Add velocity dispersion
            v_disp = self.galaxy_config.velocity_dispersion * KM_S_TO_M_S
            velocities[i] += np.random.normal(0, v_disp, 3)
        
        # Bulge velocities (just random dispersion)
        bulge_dispersion = self.galaxy_config.velocity_dispersion * 1.5 * KM_S_TO_M_S
        for i in range(N_disk, N):
            velocities[i] = np.random.normal(0, bulge_dispersion, 3)
        
        # Mass distribution
        # Mix of baryonic and dark matter particles
        masses = np.zeros(N)
        
        # Disk masses (lighter, mostly baryonic)
        disk_mass_per_particle = (total_mass * (1 - self.galaxy_config.dark_matter_fraction) * 
                                 (1 - self.galaxy_config.bulge_fraction)) / N_disk
        masses[:N_disk] = np.random.lognormal(
            np.log(disk_mass_per_particle), 0.5, N_disk
        )
        
        # Bulge masses (heavier, mixed)
        bulge_mass_per_particle = (total_mass * 
                                  (self.galaxy_config.dark_matter_fraction + 
                                   self.galaxy_config.bulge_fraction)) / N_bulge
        masses[N_disk:] = np.random.lognormal(
            np.log(bulge_mass_per_particle), 0.8, N_bulge
        )
        
        return {
            'positions': positions,
            'velocities': velocities,
            'masses': masses,
            'time': 0.0,
            'dt': 10 * MYR_TO_SECONDS,  # 10 Myr timestep
            'redshift': 3.0,  # Start at high redshift for temporal gradient evolution
            'N_disk': N_disk,
            'N_bulge': N_bulge
        }
    
    def evolve_step(self):
        """Single evolution step combining infodynamic gravity + SEC + temporal gradient"""
        
        # Temporal gradient evolution: redshift decreases with time
        # Simulate cosmic evolution from z=3 to z=0 over simulation time
        total_cosmic_time = 5000 * MYR_TO_SECONDS  # 5 Gyr simulation
        z_initial = 3.0
        z_final = 0.0
        
        # Calculate current redshift based on elapsed time
        time_fraction = self.time / total_cosmic_time
        current_redshift = z_initial * (1 - time_fraction) + z_final * time_fraction
        current_redshift = max(current_redshift, 0.0)  # Don't go negative
        self.state['redshift'] = current_redshift
        
        # Infodynamic gravity evolution (now includes tidal forces via redshift)
        dt = self.state.get('dt', 10 * MYR_TO_SECONDS)
        self.state = self.gravity_field.recursive_evolution_step(self.state, dt)
        
        # SEC collapse dynamics
        self.state = self.sec_dynamics.execute_collapse_step(self.state)
        
        # Update time
        self.time += dt
        self.state['time'] = self.time
    
    def run_simulation(self, n_steps: int, save_interval: int = 10) -> List[Dict[str, Any]]:
        """
        Run full galaxy evolution simulation
        
        Args:
            n_steps: Number of evolution steps
            save_interval: Save data every N steps
            
        Returns:
            List of trajectory snapshots
        """
        
        print(f"Starting galaxy simulation: {n_steps} steps, {n_steps * self.state.get('dt', 10*MYR_TO_SECONDS) / MYR_TO_SECONDS:.1f} Myr total")
        
        start_time = time.time()
        
        for step in range(n_steps):
            self.evolve_step()
            
            if step % save_interval == 0:
                # Calculate current rotation curve
                rotation_curve = self.analyze_rotation_curve()
                
                # Get structure metrics
                structure_metrics = self.sec_dynamics.get_current_structure_metrics(self.state)
                
                # Conservation check
                conservation = self.gravity_field.validate_conservation_laws(self.state)
                
                # Save snapshot
                snapshot = {
                    'step': step,
                    'time_myr': self.time / MYR_TO_SECONDS,
                    'positions': self.state['positions'].copy(),
                    'velocities': self.state['velocities'].copy(),
                    'total_information': self.state['total_information'],
                    'dark_matter_fraction': self.state['dark_matter_fraction'],
                    'collapse_occurred': self.state.get('collapse_occurred', False),
                    'rotation_curve': rotation_curve,
                    'structure_metrics': structure_metrics,
                    'conservation': conservation
                }
                
                self.trajectory_data.append(snapshot)
                
                # Progress reporting
                if step % (save_interval * 10) == 0:
                    elapsed = time.time() - start_time
                    progress = step / n_steps
                    eta = elapsed / (progress + 1e-10) * (1 - progress)
                    
                    print(f"Step {step}/{n_steps} ({progress:.1%}), "
                          f"t={self.time/MYR_TO_SECONDS:.1f} Myr, "
                          f"dark_matter={self.state['dark_matter_fraction']:.1%}, "
                          f"structures={structure_metrics.get('total_structure_events', 0)}, "
                          f"ETA={eta/60:.1f}min")
        
        print(f"Simulation completed in {(time.time() - start_time)/60:.1f} minutes")
        return self.trajectory_data
    
    def analyze_rotation_curve(self) -> Dict[str, np.ndarray]:
        """
        Extract rotation curve from current state
        
        Returns:
            Dictionary with radius and velocity arrays
        """
        positions = self.state['positions']
        velocities = self.state['velocities']
        
        # Calculate cylindrical coordinates
        r = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        v_tangential = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
        
        # Filter disk particles (exclude bulge for cleaner curve)
        disk_mask = np.abs(positions[:, 2]) < 2 * self.galaxy_config.disk_height * KPC_TO_METERS
        r_disk = r[disk_mask]
        v_disk = v_tangential[disk_mask]
        
        # Bin by radius
        r_min, r_max = np.percentile(r_disk, [5, 95])  # Exclude outliers
        r_bins = np.logspace(np.log10(r_min), np.log10(r_max), 15)
        
        r_binned = []
        v_binned = []
        v_error = []
        
        for i in range(len(r_bins)-1):
            mask = (r_disk >= r_bins[i]) & (r_disk < r_bins[i+1])
            if np.sum(mask) > 5:  # Require minimum particles per bin
                r_binned.append(np.mean(r_disk[mask]))
                v_binned.append(np.mean(v_disk[mask]))
                v_error.append(np.std(v_disk[mask]) / np.sqrt(np.sum(mask)))
        
        return {
            'radius_kpc': np.array(r_binned) / KPC_TO_METERS,
            'velocity_km_s': np.array(v_binned) / KM_S_TO_M_S,
            'velocity_error': np.array(v_error) / KM_S_TO_M_S
        }
    
    def plot_evolution_summary(self, save_path: Optional[str] = None):
        """Plot comprehensive evolution summary"""
        
        if not self.trajectory_data:
            print("No trajectory data available")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Extract time series data
        times = [d['time_myr'] for d in self.trajectory_data]
        total_info = [d['total_information'] for d in self.trajectory_data]
        dark_matter_frac = [d['dark_matter_fraction'] for d in self.trajectory_data]
        structure_events = [d['structure_metrics'].get('total_structure_events', 0) for d in self.trajectory_data]
        
        # Information evolution
        axes[0, 0].plot(times, total_info)
        axes[0, 0].set_xlabel('Time (Myr)')
        axes[0, 0].set_ylabel('Total Information')
        axes[0, 0].set_title('Information Evolution')
        axes[0, 0].set_yscale('log')
        
        # Dark matter fraction
        axes[0, 1].plot(times, [100*f for f in dark_matter_frac])
        axes[0, 1].set_xlabel('Time (Myr)')
        axes[0, 1].set_ylabel('Dark Matter Fraction (%)')
        axes[0, 1].set_title('Dark Matter Evolution')
        
        # Structure formation events
        axes[0, 2].plot(times, structure_events)
        axes[0, 2].set_xlabel('Time (Myr)')
        axes[0, 2].set_ylabel('Structure Events')
        axes[0, 2].set_title('Structure Formation')
        
        # Final galaxy structure (top view)
        final_pos = self.trajectory_data[-1]['positions']
        axes[1, 0].scatter(final_pos[:, 0]/KPC_TO_METERS, final_pos[:, 1]/KPC_TO_METERS, 
                          alpha=0.6, s=1)
        axes[1, 0].set_xlabel('X (kpc)')
        axes[1, 0].set_ylabel('Y (kpc)')
        axes[1, 0].set_title('Final Galaxy Structure')
        axes[1, 0].set_aspect('equal')
        
        # Rotation curve evolution
        for i, data in enumerate(self.trajectory_data[::len(self.trajectory_data)//5]):
            curve = data['rotation_curve']
            if len(curve['radius_kpc']) > 0:
                alpha = 0.3 + 0.7 * i / 5
                axes[1, 1].plot(curve['radius_kpc'], curve['velocity_km_s'], 
                               alpha=alpha, label=f"t={data['time_myr']:.0f} Myr")
        
        axes[1, 1].set_xlabel('Radius (kpc)')
        axes[1, 1].set_ylabel('Velocity (km/s)')
        axes[1, 1].set_title('Rotation Curve Evolution')
        axes[1, 1].legend()
        
        # Final rotation curve with error bars
        final_curve = self.trajectory_data[-1]['rotation_curve']
        if len(final_curve['radius_kpc']) > 0:
            axes[1, 2].errorbar(final_curve['radius_kpc'], final_curve['velocity_km_s'],
                               yerr=final_curve['velocity_error'], 
                               marker='o', capsize=3)
            axes[1, 2].set_xlabel('Radius (kpc)')
            axes[1, 2].set_ylabel('Velocity (km/s)')
            axes[1, 2].set_title('Final Rotation Curve')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Evolution summary saved to {save_path}")
        
        plt.show()
    
    def compare_with_newtonian(self) -> Dict[str, Any]:
        """
        Compare infodynamic results with expected Newtonian behavior
        
        Returns:
            Comparison metrics
        """
        if not self.trajectory_data:
            return {}
        
        final_curve = self.trajectory_data[-1]['rotation_curve']
        
        if len(final_curve['radius_kpc']) < 3:
            return {'insufficient_data': True}
        
        r = final_curve['radius_kpc']
        v = final_curve['velocity_km_s']
        
        # Expected Newtonian decline (v ∝ 1/√r for large r)
        # Fit power law to outer region
        outer_mask = r > np.median(r)
        if np.sum(outer_mask) > 2:
            log_r = np.log(r[outer_mask])
            log_v = np.log(v[outer_mask])
            
            # Linear fit in log space
            poly = np.polyfit(log_r, log_v, 1)
            power_law_slope = poly[0]
            
            # Newtonian expectation: slope = -0.5
            newtonian_deviation = abs(power_law_slope + 0.5)
            
            # Flatness measure (std deviation of outer velocities)
            velocity_flatness = np.std(v[outer_mask]) / np.mean(v[outer_mask])
            
            return {
                'power_law_slope': power_law_slope,
                'newtonian_deviation': newtonian_deviation,
                'velocity_flatness': velocity_flatness,
                'dark_matter_signature': newtonian_deviation > 0.3  # Significant deviation
            }
        
        return {'insufficient_outer_data': True}

def run_dark_matter_test():
    """Run a dark matter test simulation"""
    
    # Configuration for dark matter test
    galaxy_config = GalaxyConfig(
        N_particles=2000,
        disk_scale_length=15.0,  # 15 kpc disk
        total_mass=5e11,         # 500 billion solar masses
        dark_matter_fraction=0.85
    )
    
    gravity_config = InfoGravityConfig(
        lambda_c=15 * KPC_TO_METERS,  # Coherence length = disk scale
        quantum_floor=0.2,            # 20% quantum floor for strong dark matter effect
        alpha_info=5e-7               # Tune for realistic forces
    )
    
    sec_config = SECConfig(
        collapse_threshold=0.7,       # Lower threshold for more structure
        force_amplification=5e4       # Moderate amplification
    )
    
    # Create and run simulation
    sim = GalaxySimulator(galaxy_config, gravity_config, sec_config)
    
    print("Running dark matter test simulation...")
    results = sim.run_simulation(n_steps=100, save_interval=5)
    
    # Analysis
    print("\nAnalyzing results...")
    newtonian_comparison = sim.compare_with_newtonian()
    print(f"Newtonian comparison: {newtonian_comparison}")
    
    final_state = results[-1]
    print(f"Final dark matter fraction: {final_state['dark_matter_fraction']:.1%}")
    print(f"Structure formation events: {final_state['structure_metrics'].get('total_structure_events', 0)}")
    
    # Plot results
    sim.plot_evolution_summary('dark_matter_test_results.png')
    
    return sim, results

if __name__ == "__main__":
    # Run test simulation
    sim, results = run_dark_matter_test()
