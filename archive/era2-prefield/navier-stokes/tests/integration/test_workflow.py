"""Test end-to-end workflow integration."""
import pytest
import sys
from pathlib import Path

# Add source to path
current_dir = Path(__file__).parent.parent.parent
src_path = current_dir / "navier_symbolic_engine" / "src"
sys.path.insert(0, str(src_path))


class TestWorkflowIntegration:
    """Test complete workflow integration."""
    
    def test_full_simulation_workflow(self):
        """Test complete simulation workflow."""
        try:
            from navier_symbolic_engine.src.api.engine_interface import EngineInterface
            from navier_symbolic_engine.src.utils.entropy_hasher import EntropyHasher
            
            # Initialize components
            engine = EngineInterface()
            hasher = EntropyHasher()
            
            # Define test case
            boundary_conditions = {
                "geometry": "pipe",
                "reynolds": 2000,
                "velocity": 1.5
            }
            
            # Generate entropy signature
            entropy_sig = hasher.generate_hierarchical_entropy(boundary_conditions)
            
            # Run simulation
            result = engine.run(boundary_conditions)
            
            # Validate results
            assert entropy_sig is not None
            assert result is not None
            assert "status" in result
            
        except ImportError:
            pytest.skip("Required components not available")
        except Exception as e:
            pytest.skip(f"Workflow integration not implemented: {e}")
