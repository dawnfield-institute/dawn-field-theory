"""
ENHANCED PATTERN EXTRACTION - Multi-Scale Vortex Detection

Addresses Priority 3: Fix the Reconstruction Error by implementing proper
vortex detection and multi-scale pattern extraction. Uses established CFD
methods (Q-criterion, λ₂ method) to identify coherent structures.

Target: Reduce error from 53% to <10% through better pattern extraction.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class EnhancedPatternExtractor:
    """Multi-scale coherent structure detection for fluid flows."""
    
    def __init__(self, grid_size: int = 32):
        self.grid_size = grid_size
        self.dx = 4.0 / grid_size  # Physical grid spacing
        
        # Grid setup
        x = np.linspace(-2, 2, grid_size)
        y = np.linspace(-2, 2, grid_size)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        
        # Detection thresholds
        self.q_threshold = 0.01  # Q-criterion threshold
        self.lambda2_threshold = -0.01  # λ₂ threshold (negative for vortices)
        self.min_structure_size = 4  # Minimum pixels for coherent structure
        
        print(f"✓ Enhanced Pattern Extractor: {grid_size}x{grid_size} grid")
        print(f"   Physical spacing: dx = {self.dx:.4f}")
    
    def compute_velocity_gradients(self, u: np.ndarray, v: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute velocity gradient tensor components."""
        # Use central differences with proper boundary handling
        du_dx = np.gradient(u, self.dx, axis=1)
        du_dy = np.gradient(u, self.dx, axis=0)
        dv_dx = np.gradient(v, self.dx, axis=1)
        dv_dy = np.gradient(v, self.dx, axis=0)
        
        return {
            'du_dx': du_dx,
            'du_dy': du_dy,
            'dv_dx': dv_dx,
            'dv_dy': dv_dy
        }
    
    def compute_q_criterion(self, gradients: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute Q-criterion for vortex identification."""
        # Q = 0.5 * (Ω² - S²) where Ω is vorticity magnitude and S is strain rate magnitude
        
        # Vorticity magnitude squared
        vorticity = gradients['dv_dx'] - gradients['du_dy']
        omega_squared = vorticity**2
        
        # Strain rate magnitude squared
        S11 = gradients['du_dx']
        S12 = 0.5 * (gradients['du_dy'] + gradients['dv_dx'])
        S22 = gradients['dv_dy']
        
        strain_squared = 2 * (S11**2 + 2*S12**2 + S22**2)
        
        Q = 0.5 * (omega_squared - strain_squared)
        return Q
    
    def compute_lambda2_criterion(self, gradients: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute λ₂ criterion for vortex identification."""
        # λ₂ is the second eigenvalue of S² + Ω²
        
        # Strain rate tensor
        S11 = gradients['du_dx']
        S12 = 0.5 * (gradients['du_dy'] + gradients['dv_dx'])
        S22 = gradients['dv_dy']
        
        # Vorticity tensor (antisymmetric part)
        O12 = 0.5 * (gradients['dv_dx'] - gradients['du_dy'])
        
        # S² + Ω²
        A11 = S11**2 + S12**2 - O12**2
        A12 = S11*S12 + S12*S22
        A22 = S12**2 + S22**2 - O12**2
        
        # Eigenvalues of 2x2 symmetric matrix
        trace = A11 + A22
        det = A11*A22 - A12**2
        discriminant = trace**2 - 4*det
        discriminant = np.maximum(discriminant, 0)  # Ensure non-negative
        
        lambda1 = 0.5 * (trace + np.sqrt(discriminant))
        lambda2 = 0.5 * (trace - np.sqrt(discriminant))
        
        return lambda2
    
    def detect_coherent_structures(self, u: np.ndarray, v: np.ndarray) -> List[Dict]:
        """Detect coherent structures using Q-criterion and λ₂ method."""
        gradients = self.compute_velocity_gradients(u, v)
        
        # Compute detection criteria
        Q = self.compute_q_criterion(gradients)
        lambda2 = self.compute_lambda2_criterion(gradients)
        
        # Identify vortex regions
        vortex_mask = (Q > self.q_threshold) & (lambda2 < self.lambda2_threshold)
        
        # Label connected components
        labeled_regions, num_regions = ndimage.label(vortex_mask)
        
        structures = []
        for region_id in range(1, num_regions + 1):
            region_mask = labeled_regions == region_id
            
            # Filter by size
            if np.sum(region_mask) < self.min_structure_size:
                continue
            
            # Extract region properties
            structure = self.analyze_structure_region(u, v, region_mask)
            if structure is not None:
                structures.append(structure)
        
        return structures
    
    def analyze_structure_region(self, u: np.ndarray, v: np.ndarray, mask: np.ndarray) -> Optional[Dict]:
        """Analyze a detected coherent structure region."""
        if np.sum(mask) == 0:
            return None
        
        # Find region center of mass
        y_indices, x_indices = np.where(mask)
        center_x = np.mean(x_indices) * self.dx - 2.0  # Convert to physical coordinates
        center_y = np.mean(y_indices) * self.dx - 2.0
        
        # Extract velocities in region
        u_region = u[mask]
        v_region = v[mask]
        
        # Compute region properties
        max_velocity = np.max(np.sqrt(u_region**2 + v_region**2))
        mean_velocity = np.mean(np.sqrt(u_region**2 + v_region**2))
        
        # Estimate circulation (for vortices)
        vorticity = np.gradient(v, self.dx, axis=1) - np.gradient(u, self.dx, axis=0)
        circulation = np.sum(vorticity[mask]) * self.dx**2
        
        # Estimate characteristic radius
        distances = np.sqrt((x_indices * self.dx - 2.0 - center_x)**2 + 
                           (y_indices * self.dx - 2.0 - center_y)**2)
        char_radius = np.std(distances)
        
        # Classify structure type
        if abs(circulation) > 0.1 * char_radius * max_velocity:
            structure_type = 'vortex'
        elif np.mean(np.gradient(u[mask])) > 0.1:
            structure_type = 'shear'
        else:
            structure_type = 'other'
        
        return {
            'type': structure_type,
            'center': (center_x, center_y),
            'radius': char_radius,
            'circulation': circulation,
            'max_velocity': max_velocity,
            'mean_velocity': mean_velocity,
            'size': np.sum(mask),
            'mask': mask
        }
    
    def fit_lamb_oseen_vortex(self, structure: Dict, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Fit Lamb-Oseen vortex model to detected vortex structure."""
        if structure['type'] != 'vortex':
            return None
        
        center_x, center_y = structure['center']
        initial_circulation = structure['circulation']
        initial_radius = structure['radius']
        
        def lamb_oseen_model(params):
            """Lamb-Oseen vortex model."""
            circulation, core_radius = params
            
            # Distance from vortex center
            r = np.sqrt((self.X - center_x)**2 + (self.Y - center_y)**2)
            r = np.maximum(r, 1e-6)  # Avoid division by zero
            
            # Lamb-Oseen velocity profile
            velocity_magnitude = (circulation / (2 * np.pi * r)) * (1 - np.exp(-r**2 / core_radius**2))
            
            # Convert to Cartesian components
            u_model = -velocity_magnitude * (self.Y - center_y) / r
            v_model = velocity_magnitude * (self.X - center_x) / r
            
            return u_model, v_model
        
        def objective(params):
            """Objective function for fitting."""
            if params[1] <= 0:  # Core radius must be positive
                return 1e6
            
            u_model, v_model = lamb_oseen_model(params)
            
            # Compute error only in the vortex region
            mask = structure['mask']
            error = np.sum((u[mask] - u_model[mask])**2 + (v[mask] - v_model[mask])**2)
            return error
        
        # Fit parameters
        initial_guess = [initial_circulation, initial_radius]
        bounds = [(-10*abs(initial_circulation), 10*abs(initial_circulation)), (0.1, 2.0)]
        
        try:
            result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            if result.success:
                u_fit, v_fit = lamb_oseen_model(result.x)
                return np.stack([u_fit, v_fit], axis=-1)
        except:
            pass
        
        return None
    
    def fit_shear_layer(self, structure: Dict, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Fit hyperbolic tangent shear layer model."""
        if structure['type'] != 'shear':
            return None
        
        center_x, center_y = structure['center']
        
        def shear_model(params):
            """Hyperbolic tangent shear layer."""
            U_max, delta, angle = params
            
            # Rotate coordinates
            x_rot = (self.X - center_x) * np.cos(angle) + (self.Y - center_y) * np.sin(angle)
            y_rot = -(self.X - center_x) * np.sin(angle) + (self.Y - center_y) * np.cos(angle)
            
            # Shear profile
            u_profile = U_max * np.tanh(y_rot / delta)
            v_profile = np.zeros_like(u_profile)
            
            # Rotate back
            u_model = u_profile * np.cos(angle) - v_profile * np.sin(angle)
            v_model = u_profile * np.sin(angle) + v_profile * np.cos(angle)
            
            return u_model, v_model
        
        def objective(params):
            """Objective function for shear fitting."""
            if params[1] <= 0:  # Delta must be positive
                return 1e6
            
            u_model, v_model = shear_model(params)
            
            mask = structure['mask']
            error = np.sum((u[mask] - u_model[mask])**2 + (v[mask] - v_model[mask])**2)
            return error
        
        # Initial guess
        initial_guess = [structure['max_velocity'], structure['radius'], 0.0]
        bounds = [(-2*structure['max_velocity'], 2*structure['max_velocity']), 
                 (0.1, 2.0), (-np.pi, np.pi)]
        
        try:
            result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            if result.success:
                u_fit, v_fit = shear_model(result.x)
                return np.stack([u_fit, v_fit], axis=-1)
        except:
            pass
        
        return None
    
    def fit_gaussian_blob(self, structure: Dict, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Fit Gaussian blob for unclassified structures."""
        center_x, center_y = structure['center']
        
        def gaussian_model(params):
            """Gaussian velocity blob."""
            u_amp, v_amp, sigma_x, sigma_y, angle = params
            
            # Rotate coordinates
            x_rot = (self.X - center_x) * np.cos(angle) + (self.Y - center_y) * np.sin(angle)
            y_rot = -(self.X - center_x) * np.sin(angle) + (self.Y - center_y) * np.cos(angle)
            
            # Gaussian profile
            gaussian = np.exp(-0.5 * (x_rot**2/sigma_x**2 + y_rot**2/sigma_y**2))
            
            u_model = u_amp * gaussian
            v_model = v_amp * gaussian
            
            return u_model, v_model
        
        def objective(params):
            """Objective function for Gaussian fitting."""
            if params[2] <= 0 or params[3] <= 0:  # Sigmas must be positive
                return 1e6
                
            u_model, v_model = gaussian_model(params)
            
            mask = structure['mask']
            error = np.sum((u[mask] - u_model[mask])**2 + (v[mask] - v_model[mask])**2)
            return error
        
        # Initial guess
        mask = structure['mask']
        u_mean = np.mean(u[mask])
        v_mean = np.mean(v[mask])
        
        initial_guess = [u_mean, v_mean, structure['radius'], structure['radius'], 0.0]
        bounds = [(-5*abs(u_mean), 5*abs(u_mean)), (-5*abs(v_mean), 5*abs(v_mean)),
                 (0.1, 2.0), (0.1, 2.0), (-np.pi, np.pi)]
        
        try:
            result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            if result.success:
                u_fit, v_fit = gaussian_model(result.x)
                return np.stack([u_fit, v_fit], axis=-1)
        except:
            pass
        
        return None
    
    def enhanced_pattern_extraction(self, flow_field: np.ndarray) -> List[np.ndarray]:
        """Extract patterns using multi-scale coherent structure detection."""
        u, v = flow_field[:, :, 0], flow_field[:, :, 1]
        patterns = []
        
        # Detect coherent structures
        structures = self.detect_coherent_structures(u, v)
        
        print(f"   Detected {len(structures)} coherent structures")
        
        # Fit analytical models to each structure
        for i, structure in enumerate(structures):
            if structure['type'] == 'vortex':
                pattern = self.fit_lamb_oseen_vortex(structure, u, v)
                if pattern is not None:
                    patterns.append(pattern)
                    print(f"     ✓ Fitted Lamb-Oseen vortex {i+1}")
                    
            elif structure['type'] == 'shear':
                pattern = self.fit_shear_layer(structure, u, v)
                if pattern is not None:
                    patterns.append(pattern)
                    print(f"     ✓ Fitted shear layer {i+1}")
                    
            else:
                pattern = self.fit_gaussian_blob(structure, u, v)
                if pattern is not None:
                    patterns.append(pattern)
                    print(f"     ✓ Fitted Gaussian blob {i+1}")
        
        # If no structures detected, try global fitting
        if len(patterns) == 0:
            print("   No structures detected, attempting global pattern extraction")
            
            # Try fitting simple global patterns
            global_patterns = self.extract_global_patterns(flow_field)
            patterns.extend(global_patterns)
        
        return patterns
    
    def extract_global_patterns(self, flow_field: np.ndarray) -> List[np.ndarray]:
        """Extract global patterns when local structures aren't detected."""
        u, v = flow_field[:, :, 0], flow_field[:, :, 1]
        patterns = []
        
        # Try Taylor-Green pattern
        tg_pattern = self.fit_taylor_green_pattern(u, v)
        if tg_pattern is not None:
            patterns.append(tg_pattern)
            print("     ✓ Global Taylor-Green pattern fitted")
        
        # Try uniform flow
        mean_u = np.mean(u)
        mean_v = np.mean(v)
        if abs(mean_u) > 0.01 or abs(mean_v) > 0.01:
            uniform_u = np.full_like(u, mean_u)
            uniform_v = np.full_like(v, mean_v)
            uniform_pattern = np.stack([uniform_u, uniform_v], axis=-1)
            patterns.append(uniform_pattern)
            print("     ✓ Uniform flow pattern extracted")
        
        return patterns
    
    def fit_taylor_green_pattern(self, u: np.ndarray, v: np.ndarray) -> Optional[np.ndarray]:
        """Fit Taylor-Green vortex pattern."""
        def taylor_green_model(params):
            """Taylor-Green vortex."""
            amp, kx, ky, phase_x, phase_y = params
            
            u_model = amp * np.sin(kx * np.pi * self.X + phase_x) * np.cos(ky * np.pi * self.Y + phase_y)
            v_model = -amp * np.cos(kx * np.pi * self.X + phase_x) * np.sin(ky * np.pi * self.Y + phase_y)
            
            return u_model, v_model
        
        def objective(params):
            """Objective function for Taylor-Green fitting."""
            u_model, v_model = taylor_green_model(params)
            error = np.sum((u - u_model)**2 + (v - v_model)**2)
            return error
        
        # Initial guess
        max_vel = np.max(np.sqrt(u**2 + v**2))
        initial_guess = [max_vel, 1.0, 1.0, 0.0, 0.0]
        bounds = [(-2*max_vel, 2*max_vel), (0.5, 3.0), (0.5, 3.0), 
                 (-np.pi, np.pi), (-np.pi, np.pi)]
        
        try:
            result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            if result.success and result.fun < 0.1 * np.sum(u**2 + v**2):
                u_fit, v_fit = taylor_green_model(result.x)
                return np.stack([u_fit, v_fit], axis=-1)
        except:
            pass
        
        return None
    
    def reconstruct_from_patterns(self, patterns: List[np.ndarray], flow_field: np.ndarray) -> Tuple[np.ndarray, float]:
        """Reconstruct flow field from extracted patterns."""
        if len(patterns) == 0:
            reconstruction = np.zeros_like(flow_field)
            error = 1.0
            return reconstruction, error
        
        # Solve least squares for pattern coefficients
        A = np.array([p.reshape(-1) for p in patterns]).T
        b = flow_field.reshape(-1)
        
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            reconstruction = (A @ coeffs).reshape(flow_field.shape)
        except:
            # Fallback to pseudo-inverse
            coeffs = np.linalg.pinv(A) @ b
            reconstruction = (A @ coeffs).reshape(flow_field.shape)
        
        # Compute reconstruction error
        error = np.linalg.norm(flow_field - reconstruction) / np.linalg.norm(flow_field)
        
        return reconstruction, error


def test_enhanced_extraction():
    """Test the enhanced pattern extraction on various flow types."""
    print("🔬 Testing Enhanced Pattern Extraction")
    print("=" * 50)
    
    extractor = EnhancedPatternExtractor(grid_size=64)
    
    # Test cases
    test_cases = {
        'Taylor-Green': create_taylor_green_vortex(extractor),
        'Double Vortex': create_double_vortex(extractor),
        'Shear Layer': create_shear_layer(extractor),
        'Mixed Pattern': create_mixed_pattern(extractor)
    }
    
    results = {}
    
    for name, flow_field in test_cases.items():
        print(f"\n📊 Testing: {name}")
        
        # Extract patterns
        patterns = extractor.enhanced_pattern_extraction(flow_field)
        
        # Reconstruct
        reconstruction, error = extractor.reconstruct_from_patterns(patterns, flow_field)
        
        results[name] = {
            'num_patterns': len(patterns),
            'reconstruction_error': error,
            'success': error < 0.1  # Target: <10% error
        }
        
        print(f"   Patterns extracted: {len(patterns)}")
        print(f"   Reconstruction error: {error:.4f}")
        print(f"   Success (error < 10%): {error < 0.1}")
    
    # Summary
    print(f"\n📈 Summary:")
    success_count = sum(1 for r in results.values() if r['success'])
    print(f"   Successful reconstructions: {success_count}/{len(test_cases)}")
    
    mean_error = np.mean([r['reconstruction_error'] for r in results.values()])
    print(f"   Mean reconstruction error: {mean_error:.4f}")
    
    return results


def create_taylor_green_vortex(extractor):
    """Create Taylor-Green vortex test case."""
    u = np.sin(np.pi * extractor.X) * np.cos(np.pi * extractor.Y)
    v = -np.cos(np.pi * extractor.X) * np.sin(np.pi * extractor.Y)
    return np.stack([u, v], axis=-1)


def create_double_vortex(extractor):
    """Create double vortex test case."""
    # Two counter-rotating vortices
    r1 = np.sqrt((extractor.X + 0.8)**2 + extractor.Y**2)
    r2 = np.sqrt((extractor.X - 0.8)**2 + extractor.Y**2)
    
    r1 = np.maximum(r1, 0.2)
    r2 = np.maximum(r2, 0.2)
    
    # Vortex 1 (clockwise)
    u1 = -(extractor.Y) / r1**2
    v1 = (extractor.X + 0.8) / r1**2
    
    # Vortex 2 (counter-clockwise)
    u2 = (extractor.Y) / r2**2
    v2 = -(extractor.X - 0.8) / r2**2
    
    u = u1 + u2
    v = v1 + v2
    
    return np.stack([u, v], axis=-1)


def create_shear_layer(extractor):
    """Create shear layer test case."""
    u = np.tanh((extractor.Y) / 0.3)
    v = 0.1 * np.sin(2 * np.pi * extractor.X)
    return np.stack([u, v], axis=-1)


def create_mixed_pattern(extractor):
    """Create mixed pattern test case."""
    # Combination of Taylor-Green and shear
    u_tg = 0.7 * np.sin(np.pi * extractor.X) * np.cos(np.pi * extractor.Y)
    v_tg = -0.7 * np.cos(np.pi * extractor.X) * np.sin(np.pi * extractor.Y)
    
    u_shear = 0.3 * extractor.Y
    v_shear = 0.1 * np.sin(3 * np.pi * extractor.X)
    
    u = u_tg + u_shear
    v = v_tg + v_shear
    
    return np.stack([u, v], axis=-1)


if __name__ == "__main__":
    test_enhanced_extraction()
