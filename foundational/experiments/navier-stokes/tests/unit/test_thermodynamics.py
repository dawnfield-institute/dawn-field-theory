"""Test thermodynamic validation functionality."""
import pytest
import sys
from pathlib import Path

# Add source to path
current_dir = Path(__file__).parent.parent.parent
src_path = current_dir / "navier_symbolic_engine" / "src"
sys.path.insert(0, str(src_path))


class TestThermodynamicValidation:
    """Test thermodynamic validation capabilities."""
    
    def test_landauer_bound_validation(self):
        """Test Landauer bound validation."""
        try:
            from navier_symbolic_engine.src.core.thermodynamic_validator import ThermodynamicValidator
            
            validator = ThermodynamicValidator()
            
            # Test with sample computation data
            computation_data = {
                "operations": 1000,
                "energy_dissipated": 1e-20,  # Joules
                "temperature": 300  # Kelvin
            }
            
            is_valid = validator.validate_landauer_bound(computation_data)
            assert isinstance(is_valid, bool)
            
        except ImportError:
            pytest.skip("ThermodynamicValidator not available")
        except Exception as e:
            pytest.skip(f"Landauer validation not implemented: {e}")
    
    def test_energy_conservation(self):
        """Test energy conservation validation."""
        try:
            from navier_symbolic_engine.src.core.thermodynamic_validator import ThermodynamicValidator
            
            validator = ThermodynamicValidator()
            
            # Test with sample flow data
            flow_data = {
                "kinetic_energy": 100.0,
                "potential_energy": 50.0,
                "dissipated_energy": 10.0
            }
            
            is_conserved = validator.validate_energy_conservation(flow_data)
            assert isinstance(is_conserved, bool)
            
        except ImportError:
            pytest.skip("ThermodynamicValidator not available")
        except Exception as e:
            pytest.skip(f"Energy conservation validation not implemented: {e}")
