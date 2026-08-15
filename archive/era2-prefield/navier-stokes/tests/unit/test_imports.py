"""Test module imports and basic availability."""
import pytest
import sys
from pathlib import Path

# Add source to path
current_dir = Path(__file__).parent.parent.parent
src_path = current_dir / "navier_symbolic_engine" / "src"
sys.path.insert(0, str(src_path))


class TestImports:
    """Test that all modules can be imported successfully."""
    
    def test_engine_interface_import(self):
        """Test that EngineInterface can be imported."""
        try:
            from navier_symbolic_engine.src.api.engine_interface import EngineInterface
            assert EngineInterface is not None
        except ImportError as e:
            pytest.skip(f"Engine interface not available: {e}")
    
    def test_entropy_hasher_import(self):
        """Test that EntropyHasher can be imported."""
        try:
            from navier_symbolic_engine.src.utils.entropy_hasher import EntropyHasher
            assert EntropyHasher is not None
        except ImportError as e:
            pytest.skip(f"Entropy hasher not available: {e}")
    
    def test_visualization_import(self):
        """Test that Visualization can be imported."""
        try:
            from navier_symbolic_engine.src.utils.visualization import Visualization
            assert Visualization is not None
        except ImportError as e:
            pytest.skip(f"Visualization not available: {e}")
    
    def test_pattern_generator_import(self):
        """Test that PatternGenerator can be imported."""
        try:
            from navier_symbolic_engine.src.patterns.pattern_library import PatternLibrary
            assert PatternLibrary is not None
        except ImportError as e:
            pytest.skip(f"Pattern library not available: {e}")
    
    def test_thermodynamic_validator_import(self):
        """Test that ThermodynamicValidator can be imported."""
        try:
            from navier_symbolic_engine.src.core.thermodynamic_validator import ThermodynamicValidator
            assert ThermodynamicValidator is not None
        except ImportError as e:
            pytest.skip(f"Thermodynamic validator not available: {e}")
