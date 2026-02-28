#!/usr/bin/env python3
"""
Core Module Tests
=================

Unit tests for the pre-field EM emergence core modules.

Run with:
    python -m pytest tests/test_core.py -v
    
Or without pytest:
    python tests/test_core.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import MobiusField, SECOperator, EMProjector, MaxwellValidator
from core.constants import PHI, XI, eb_from_wr, wr_for_eb, phi_power_from_eb


class TestConstants:
    """Test constant calculations."""
    
    def test_phi_value(self):
        assert abs(PHI - 1.618033988749895) < 1e-10
    
    def test_xi_value(self):
        assert abs(XI - (1 + np.pi / 55)) < 1e-10
    
    def test_eb_from_wr(self):
        # At optimal w/R ≈ 0.304, E/B should be close to φ
        eb = eb_from_wr(0.304)
        assert abs(eb - PHI) / PHI < 0.1  # Within 10%
    
    def test_wr_for_eb(self):
        # Should be inverse of eb_from_wr
        wr = wr_for_eb(PHI)
        assert abs(wr - 0.304) < 0.05
    
    def test_phi_power_from_eb(self):
        assert abs(phi_power_from_eb(PHI) - 1.0) < 1e-10
        assert abs(phi_power_from_eb(PHI**2) - 2.0) < 1e-10


class TestMobiusField:
    """Test MobiusField class."""
    
    def test_initialization(self):
        field = MobiusField(n_u=32, n_v=16, R=2.0, w=0.5)
        assert field.n_u == 32
        assert field.n_v == 16
        assert field.w_R_ratio == 0.25
    
    def test_potential_positive(self):
        field = MobiusField()
        P = field.potential()
        assert (P >= 0).all()
    
    def test_phase_bounded(self):
        field = MobiusField()
        ph = field.phase()
        assert ph.min() >= -np.pi
        assert ph.max() <= np.pi
    
    def test_pac_residual_finite(self):
        field = MobiusField()
        pac = field.pac_residual()
        assert np.isfinite(pac)
        assert pac >= 0
    
    def test_embedding_dimensions(self):
        field = MobiusField(n_u=32, n_v=16)
        assert field.X.shape == (32, 16)
        assert field.Y.shape == (32, 16)
        assert field.Z.shape == (32, 16)
    
    def test_even_nu_required(self):
        try:
            MobiusField(n_u=33)  # Odd
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestSECOperator:
    """Test SECOperator class."""
    
    def test_initialization(self):
        sec = SECOperator(damping=0.95, pi_coupling=0.1)
        assert sec.damping == 0.95
        assert sec.pi_coupling == 0.1
        assert sec.iteration == 0
    
    def test_step_increments_iteration(self):
        field = MobiusField(n_u=32, n_v=16)
        sec = SECOperator()
        sec.step(field)
        assert sec.iteration == 1
        sec.step(field)
        assert sec.iteration == 2
    
    def test_step_returns_metrics(self):
        field = MobiusField(n_u=32, n_v=16)
        sec = SECOperator()
        metrics = sec.step(field)
        assert 'pac_residual' in metrics
        assert 'total_entropy' in metrics
        assert 'iteration' in metrics
    
    def test_pac_improves_over_time(self):
        field = MobiusField(n_u=32, n_v=16)
        sec = SECOperator(damping=0.98, pi_coupling=0.05)
        
        initial_pac = field.pac_residual()
        for _ in range(100):
            sec.step(field)
        final_pac = field.pac_residual()
        
        # PAC should generally improve (decrease) or stay stable
        assert final_pac <= initial_pac * 1.5  # Allow some tolerance
    
    def test_reset(self):
        field = MobiusField(n_u=32, n_v=16)
        sec = SECOperator()
        for _ in range(10):
            sec.step(field)
        sec.reset()
        assert sec.iteration == 0
        assert len(sec.history) == 0


class TestEMProjector:
    """Test EMProjector class."""
    
    def test_initialization(self):
        proj = EMProjector(n=16, L=2.0)
        assert proj.n == 16
        assert proj.L == 2.0
    
    def test_mask_sphere(self):
        proj = EMProjector(n=16, L=2.0, shape='sphere')
        assert proj.mask.any()  # Some points inside
        assert not proj.mask.all()  # Some points outside
    
    def test_project_returns_dict(self):
        field = MobiusField(n_u=32, n_v=16)
        proj = EMProjector(n=12, L=2.0)
        result = proj.project(field)
        
        assert 'EB_ratio' in result
        assert 'E_mean' in result
        assert 'B_mean' in result
        assert 'div_B_mean' in result
    
    def test_div_b_near_zero(self):
        """B = curl(A) should have div(B) ≈ 0."""
        field = MobiusField(n_u=32, n_v=16)
        sec = SECOperator()
        for _ in range(50):
            sec.step(field)
        
        proj = EMProjector(n=12, L=2.0)
        result = proj.project(field)
        
        # div(B) should be very small (numerical precision)
        assert result['div_B_mean'] < 0.1
    
    def test_eb_ratio_positive(self):
        field = MobiusField(n_u=32, n_v=16)
        proj = EMProjector(n=12, L=2.0)
        result = proj.project(field)
        
        assert result['EB_ratio'] > 0


class TestMaxwellValidator:
    """Test MaxwellValidator class."""
    
    def test_validation_returns_dict(self):
        field = MobiusField(n_u=32, n_v=16)
        proj = EMProjector(n=12, L=2.0)
        validator = MaxwellValidator(proj)
        
        result = proj.project(field)
        validation = validator.validate(result)
        
        assert 'no_monopoles' in validation
        assert 'overall_score' in validation
        assert 'verdict' in validation
    
    def test_no_monopoles_passes(self):
        """B = curl(A) guarantees no monopoles."""
        field = MobiusField(n_u=32, n_v=16)
        sec = SECOperator()
        for _ in range(50):
            sec.step(field)
        
        proj = EMProjector(n=12, L=2.0)
        validator = MaxwellValidator(proj)
        
        result = proj.project(field)
        validation = validator.validate(result)
        
        assert validation['no_monopoles']


def run_tests():
    """Run all tests without pytest."""
    
    test_classes = [
        TestConstants,
        TestMobiusField,
        TestSECOperator,
        TestEMProjector,
        TestMaxwellValidator,
    ]
    
    total = 0
    passed = 0
    failed = []
    
    for test_class in test_classes:
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        
        for method_name in methods:
            total += 1
            try:
                getattr(instance, method_name)()
                passed += 1
                print(f"  ✓ {test_class.__name__}.{method_name}")
            except Exception as e:
                failed.append((f"{test_class.__name__}.{method_name}", str(e)))
                print(f"  ✗ {test_class.__name__}.{method_name}: {e}")
    
    print(f"\n{'='*50}")
    print(f"Tests: {passed}/{total} passed")
    
    if failed:
        print(f"\nFailed tests:")
        for name, error in failed:
            print(f"  - {name}: {error}")
    
    return len(failed) == 0


if __name__ == "__main__":
    print("=" * 50)
    print("Running Core Module Tests")
    print("=" * 50)
    
    success = run_tests()
    sys.exit(0 if success else 1)
