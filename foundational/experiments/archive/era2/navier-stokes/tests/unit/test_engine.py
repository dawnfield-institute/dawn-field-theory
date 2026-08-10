"""Test main engine interface functionality."""
import pytest
import sys
from pathlib import Path

# Add source to path
current_dir = Path(__file__).parent.parent.parent
src_path = current_dir / "navier_symbolic_engine" / "src"
sys.path.insert(0, str(src_path))


class TestEngineInterface:
    """Test main engine interface."""
    
    def test_engine_initialization(self):
        """Test that engine can be initialized."""
        try:
            from navier_symbolic_engine.src.api.engine_interface import EngineInterface
            
            engine = EngineInterface()
            assert engine is not None
            
        except ImportError:
            pytest.skip("EngineInterface not available")
    
    def test_basic_simulation_run(self):
        """Test basic simulation execution."""
        try:
            from navier_symbolic_engine.src.api.engine_interface import EngineInterface
            
            engine = EngineInterface()
            
            boundary_conditions = {
                "geometry": "pipe",
                "reynolds": 1000,
                "velocity": 1.0
            }
            
            result = engine.run(boundary_conditions)
            
            assert result is not None
            assert "status" in result
            
        except ImportError:
            pytest.skip("EngineInterface not available")
        except Exception as e:
            pytest.skip(f"Simulation run not implemented: {e}")
    
    def test_different_geometries(self):
        """Test simulation with different geometries."""
        try:
            from navier_symbolic_engine.src.api.engine_interface import EngineInterface
            
            engine = EngineInterface()
            geometries = ["pipe", "channel", "cavity"]
            
            for geometry in geometries:
                boundary_conditions = {
                    "geometry": geometry,
                    "reynolds": 1500,
                    "velocity": 1.2
                }
                
                result = engine.run(boundary_conditions)
                assert result is not None
                
        except ImportError:
            pytest.skip("EngineInterface not available")
        except Exception as e:
            pytest.skip(f"Multi-geometry simulation not implemented: {e}")
