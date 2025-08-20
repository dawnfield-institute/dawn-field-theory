"""
PATTERN DISCOVERY ENGINE - True Emergent Pattern Learning

Addresses Priority 1: Prove Pattern Convergence by discovering patterns from data
rather than predefining them. This implements bottom-up pattern learning to
test whether finite pattern libraries naturally emerge from SEC dynamics.

Key Innovation: Starts with EMPTY pattern library and builds it incrementally
by processing diverse flow conditions and extracting unique patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class PatternDiscoveryEngine:
    """Discovers patterns from flow data without predefined libraries."""
    
    def __init__(self, grid_size: int = 32, max_patterns: int = 20):
        self.grid_size = grid_size
        self.max_patterns = max_patterns
        self.pattern_library = []  # Start EMPTY - this is the key innovation
        self.pattern_signatures = []  # For duplicate detection
        self.discovery_history = []  # Track discovery process
        
        # Pattern quality thresholds
        self.min_energy = 1e-6  # Minimum pattern energy to consider
        self.similarity_threshold = 0.95  # Pattern similarity for merging
        self.reconstruction_threshold = 0.1  # Error threshold for new patterns
        
        # Grid setup
        x = np.linspace(-2, 2, grid_size)
        y = np.linspace(-2, 2, grid_size)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        
        print(f"✓ Pattern Discovery Engine: {grid_size}x{grid_size} grid")
        print(f"  Empty library - patterns will be discovered from data")
        print(f"  Max patterns: {max_patterns}")
    
    def generate_diverse_flow_conditions(self, n_flows: int = 100) -> List[np.ndarray]:
        """Generate diverse initial flow conditions for pattern discovery."""
        flows = []
        
        for i in range(n_flows):
            flow_type = i % 6  # Cycle through flow types
            
            if flow_type == 0:  # Taylor-Green variations
                kx = 1 + 0.5 * np.random.randn()
                ky = 1 + 0.5 * np.random.randn()
                amp = 0.5 + 0.5 * np.random.random()
                u = amp * np.sin(kx * np.pi * self.X) * np.cos(ky * np.pi * self.Y)
                v = -amp * np.cos(kx * np.pi * self.X) * np.sin(ky * np.pi * self.Y)
                
            elif flow_type == 1:  # Random vortices
                n_vortices = np.random.randint(1, 4)
                u = np.zeros_like(self.X)
                v = np.zeros_like(self.Y)
                
                for _ in range(n_vortices):
                    cx = -1.5 + 3 * np.random.random()
                    cy = -1.5 + 3 * np.random.random()
                    strength = 0.5 + np.random.random()
                    
                    r_sq = (self.X - cx)**2 + (self.Y - cy)**2 + 0.1
                    u += strength * (self.Y - cy) / r_sq
                    v += -strength * (self.X - cx) / r_sq
                    
            elif flow_type == 2:  # Shear layers
                amplitude = 0.5 + 0.5 * np.random.random()
                shear_rate = 0.5 + np.random.random()
                u = amplitude * shear_rate * self.Y
                v = amplitude * 0.1 * np.sin(2 * np.pi * self.X) * np.random.random()
                
            elif flow_type == 3:  # Dipole flows
                separation = 0.5 + np.random.random()
                strength = 0.5 + np.random.random()
                
                r1_sq = (self.X - separation/2)**2 + self.Y**2 + 0.1
                r2_sq = (self.X + separation/2)**2 + self.Y**2 + 0.1
                
                u = strength * (self.X - separation/2) / r1_sq - strength * (self.X + separation/2) / r2_sq
                v = strength * self.Y / r1_sq - strength * self.Y / r2_sq
                
            elif flow_type == 4:  # Wave patterns
                k1 = 1 + np.random.randint(0, 3)
                k2 = 1 + np.random.randint(0, 3)
                phase = 2 * np.pi * np.random.random()
                amp = 0.3 + 0.4 * np.random.random()
                
                u = amp * np.sin(k1 * np.pi * self.X + phase) * np.cos(k2 * np.pi * self.Y)
                v = amp * np.cos(k1 * np.pi * self.X + phase) * np.sin(k2 * np.pi * self.Y)
                
            else:  # flow_type == 5: Random turbulent-like
                # Superposition of random modes
                u = np.zeros_like(self.X)
                v = np.zeros_like(self.Y)
                
                for k in range(1, 4):
                    for l in range(1, 4):
                        amp = np.random.normal(0, 1.0 / (k*l))
                        phase1 = 2 * np.pi * np.random.random()
                        phase2 = 2 * np.pi * np.random.random()
                        
                        u += amp * np.sin(k * np.pi * self.X + phase1) * np.cos(l * np.pi * self.Y + phase2)
                        v += amp * np.cos(k * np.pi * self.X + phase1) * np.sin(l * np.pi * self.Y + phase2)
            
            # Stack as vector field
            flow = np.stack([u, v], axis=-1)
            flows.append(flow)
        
        print(f"✓ Generated {len(flows)} diverse flow conditions")
        return flows
    
    def extract_pattern_from_residual(self, residual: np.ndarray) -> Optional[np.ndarray]:
        """Extract a new pattern from reconstruction residual."""
        if residual.ndim != 3 or residual.shape[2] != 2:
            return None
            
        # Check if residual has sufficient energy
        energy = np.mean(residual**2)
        if energy < self.min_energy:
            return None
        
        # Find dominant structure using PCA or coherent structure detection
        u_res, v_res = residual[:, :, 0], residual[:, :, 1]
        
        # Simple approach: Use the residual itself as pattern if it's coherent
        # More sophisticated: Extract dominant mode via SVD
        U, s, Vt = np.linalg.svd(residual.reshape(-1, 2).T)
        dominant_mode = s[0] * np.outer(U[:, 0], Vt[0, :]).reshape(self.grid_size, self.grid_size, 2)
        
        # Normalize pattern
        pattern_energy = np.mean(dominant_mode**2)
        if pattern_energy > self.min_energy:
            pattern = dominant_mode / np.sqrt(pattern_energy)
            return pattern
        
        return None
    
    def compute_pattern_signature(self, pattern: np.ndarray) -> np.ndarray:
        """Compute a signature for pattern similarity comparison."""
        # Use low-dimensional PCA representation as signature
        flat_pattern = pattern.reshape(-1)
        
        # Normalize for scale invariance
        norm = np.linalg.norm(flat_pattern)
        if norm > 1e-12:
            normalized_pattern = flat_pattern / norm
        else:
            normalized_pattern = flat_pattern
            
        # Use first few PCA components as signature
        # For simplicity, use pattern energy distribution
        u, v = pattern[:, :, 0], pattern[:, :, 1]
        
        signature = np.array([
            np.mean(u), np.std(u), np.mean(v), np.std(v),
            np.mean(u*v), np.mean(u**2), np.mean(v**2),
            np.mean(np.gradient(u)[0]), np.mean(np.gradient(v)[1])
        ])
        
        return signature
    
    def is_pattern_similar(self, pattern: np.ndarray, existing_pattern: np.ndarray) -> bool:
        """Check if pattern is similar to existing pattern."""
        sig1 = self.compute_pattern_signature(pattern)
        sig2 = self.compute_pattern_signature(existing_pattern)
        
        # Compute normalized correlation
        correlation = np.dot(sig1, sig2) / (np.linalg.norm(sig1) * np.linalg.norm(sig2) + 1e-12)
        
        return correlation > self.similarity_threshold
    
    def reconstruct_with_library(self, flow_field: np.ndarray) -> Tuple[np.ndarray, float]:
        """Reconstruct flow using current pattern library."""
        if len(self.pattern_library) == 0:
            # No patterns yet, return zeros with high error
            reconstruction = np.zeros_like(flow_field)
            error = np.linalg.norm(flow_field) / (np.linalg.norm(flow_field) + 1e-12)
            return reconstruction, error
        
        # Solve least squares for pattern coefficients
        A = np.array([p.reshape(-1) for p in self.pattern_library]).T
        b = flow_field.reshape(-1)
        
        try:
            coeffs, residual, rank, s = np.linalg.lstsq(A, b, rcond=None)
            reconstruction = (A @ coeffs).reshape(flow_field.shape)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            coeffs = np.linalg.pinv(A) @ b
            reconstruction = (A @ coeffs).reshape(flow_field.shape)
        
        # Compute reconstruction error
        error = np.linalg.norm(flow_field - reconstruction) / (np.linalg.norm(flow_field) + 1e-12)
        
        return reconstruction, error
    
    def process_new_flow(self, flow_field: np.ndarray) -> Dict:
        """Process a new flow field and potentially discover new patterns."""
        # Try to reconstruct with existing patterns
        reconstruction, error = self.reconstruct_with_library(flow_field)
        
        result = {
            'initial_library_size': len(self.pattern_library),
            'reconstruction_error': error,
            'new_pattern_discovered': False,
            'final_library_size': len(self.pattern_library)
        }
        
        # If error is high and we haven't reached max patterns, try to discover new pattern
        if error > self.reconstruction_threshold and len(self.pattern_library) < self.max_patterns:
            residual = flow_field - reconstruction
            new_pattern = self.extract_pattern_from_residual(residual)
            
            if new_pattern is not None:
                # Check if this pattern is genuinely new
                is_new = True
                for existing_pattern in self.pattern_library:
                    if self.is_pattern_similar(new_pattern, existing_pattern):
                        is_new = False
                        break
                
                if is_new:
                    self.pattern_library.append(new_pattern)
                    signature = self.compute_pattern_signature(new_pattern)
                    self.pattern_signatures.append(signature)
                    
                    result['new_pattern_discovered'] = True
                    result['final_library_size'] = len(self.pattern_library)
                    
                    # Record discovery
                    discovery_record = {
                        'pattern_id': len(self.pattern_library) - 1,
                        'discovery_error': error,
                        'pattern_energy': np.mean(new_pattern**2),
                        'signature': signature.tolist()
                    }
                    self.discovery_history.append(discovery_record)
        
        return result
    
    def run_pattern_discovery_experiment(self, n_flows: int = 100) -> Dict:
        """Run the main pattern discovery experiment."""
        print(f"\n🔬 Starting Pattern Discovery Experiment")
        print(f"   Processing {n_flows} diverse flow conditions")
        print(f"   Starting with empty pattern library")
        
        # Generate test flows
        test_flows = self.generate_diverse_flow_conditions(n_flows)
        
        # Process each flow
        results = []
        library_sizes = []
        errors = []
        
        for i, flow in enumerate(test_flows):
            result = self.process_new_flow(flow)
            results.append(result)
            library_sizes.append(result['final_library_size'])
            errors.append(result['reconstruction_error'])
            
            if i % 20 == 0 or result['new_pattern_discovered']:
                print(f"   Flow {i+1:3d}: Library size = {result['final_library_size']:2d}, "
                      f"Error = {result['reconstruction_error']:.3f}, "
                      f"New pattern: {result['new_pattern_discovered']}")
        
        # Analyze convergence
        final_library_size = len(self.pattern_library)
        mean_final_error = np.mean(errors[-20:])  # Average over last 20 flows
        
        # Check for plateau in library size
        if len(library_sizes) >= 20:
            recent_changes = np.diff(library_sizes[-20:])
            plateau_reached = np.sum(recent_changes) == 0
        else:
            plateau_reached = False
        
        summary = {
            'total_flows_processed': n_flows,
            'final_library_size': final_library_size,
            'mean_final_error': mean_final_error,
            'plateau_reached': plateau_reached,
            'discovery_history': self.discovery_history,
            'library_size_evolution': library_sizes,
            'error_evolution': errors,
            'unique_patterns_discovered': final_library_size
        }
        
        print(f"\n📊 Discovery Results:")
        print(f"   Final library size: {final_library_size} patterns")
        print(f"   Mean reconstruction error: {mean_final_error:.3f}")
        print(f"   Library plateau reached: {plateau_reached}")
        
        return summary
    
    def analyze_discovered_patterns(self) -> Dict:
        """Analyze the patterns that were discovered."""
        if len(self.pattern_library) == 0:
            return {'analysis': 'No patterns discovered'}
        
        analysis = {
            'num_patterns': len(self.pattern_library),
            'pattern_energies': [],
            'pattern_types': [],
            'spatial_characteristics': []
        }
        
        for i, pattern in enumerate(self.pattern_library):
            energy = np.mean(pattern**2)
            analysis['pattern_energies'].append(energy)
            
            # Analyze spatial structure
            u, v = pattern[:, :, 0], pattern[:, :, 1]
            
            # Compute spatial derivatives for pattern classification
            du_dx = np.gradient(u, axis=1)
            du_dy = np.gradient(u, axis=0)
            dv_dx = np.gradient(v, axis=1)
            dv_dy = np.gradient(v, axis=0)
            
            # Vorticity and divergence
            vorticity = dv_dx - du_dy
            divergence = du_dx + dv_dy
            
            spatial_char = {
                'mean_vorticity': np.mean(np.abs(vorticity)),
                'mean_divergence': np.mean(np.abs(divergence)),
                'max_velocity': np.max(np.sqrt(u**2 + v**2)),
                'spatial_scale': self.estimate_spatial_scale(pattern)
            }
            analysis['spatial_characteristics'].append(spatial_char)
            
            # Simple pattern type classification
            if spatial_char['mean_vorticity'] > 2 * spatial_char['mean_divergence']:
                pattern_type = 'vortical'
            elif spatial_char['mean_divergence'] > 2 * spatial_char['mean_vorticity']:
                pattern_type = 'source/sink'
            else:
                pattern_type = 'mixed'
            
            analysis['pattern_types'].append(pattern_type)
        
        return analysis
    
    def estimate_spatial_scale(self, pattern: np.ndarray) -> float:
        """Estimate the characteristic spatial scale of a pattern."""
        u, v = pattern[:, :, 0], pattern[:, :, 1]
        velocity_magnitude = np.sqrt(u**2 + v**2)
        
        # Find autocorrelation length scale
        center = self.grid_size // 2
        if velocity_magnitude[center, center] > 1e-6:
            # Simple estimate: distance to half-maximum
            center_val = velocity_magnitude[center, center]
            half_max = center_val / 2
            
            # Search along x-direction
            for i in range(center, self.grid_size):
                if velocity_magnitude[center, i] < half_max:
                    scale = (i - center) * 4.0 / self.grid_size  # Convert to physical units
                    return scale
        
        return 1.0  # Default scale
    
    def save_results(self, experiment_results: Dict, output_dir: str = "results"):
        """Save discovery results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save experiment summary
        summary_file = output_path / f"pattern_discovery_{self.grid_size}x{self.grid_size}_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(experiment_results, f, indent=2)
        
        # Save discovered patterns
        patterns_file = output_path / f"discovered_patterns_{self.grid_size}x{self.grid_size}_{timestamp}.npy"
        if len(self.pattern_library) > 0:
            patterns_array = np.array(self.pattern_library)
            np.save(patterns_file, patterns_array)
        
        print(f"✓ Results saved to {output_path}")
        
        return str(summary_file), str(patterns_file)


def main():
    """Run the pattern discovery experiment."""
    print("🧪 PATTERN DISCOVERY ENGINE - Emergent Pattern Learning")
    print("=" * 60)
    
    # Test on multiple grid sizes
    grid_sizes = [32, 64]
    
    for grid_size in grid_sizes:
        print(f"\n📐 Testing grid size: {grid_size}x{grid_size}")
        
        engine = PatternDiscoveryEngine(grid_size=grid_size, max_patterns=20)
        
        # Run discovery experiment
        results = engine.run_pattern_discovery_experiment(n_flows=100)
        
        # Analyze discovered patterns
        analysis = engine.analyze_discovered_patterns()
        results['pattern_analysis'] = analysis
        
        # Save results
        engine.save_results(results)
        
        # Print key insights
        print(f"\n🔍 Key Insights for {grid_size}x{grid_size}:")
        if results['unique_patterns_discovered'] > 0:
            print(f"   ✓ Discovered {results['unique_patterns_discovered']} unique patterns")
            print(f"   ✓ Final reconstruction error: {results['mean_final_error']:.3f}")
            
            if 'pattern_types' in analysis:
                pattern_counts = {}
                for ptype in analysis['pattern_types']:
                    pattern_counts[ptype] = pattern_counts.get(ptype, 0) + 1
                print(f"   ✓ Pattern types: {pattern_counts}")
        else:
            print("   ⚠ No patterns discovered - may need parameter adjustment")


if __name__ == "__main__":
    main()
