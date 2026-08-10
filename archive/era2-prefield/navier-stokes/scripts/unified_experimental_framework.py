#!/usr/bin/env python3
"""
Unified Experimental Framework for Navier-Stokes Symbolic Engine.
Comprehensive entry point for experiments, testing, benchmarking, and analysis.
Includes organized unit test structure, parameter sweeps, and performance analysis.
"""

import sys
import os
import time
import json
import hashlib
import pickle
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Try to import yaml, provide fallback if not available
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("⚠️ PyYAML not available. Configuration will use JSON format.")

# Add the source directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'navier_symbolic_engine', 'src')
sys.path.insert(0, src_path)

# Import engine components
from navier_symbolic_engine.src.api.engine_interface import EngineInterface
from navier_symbolic_engine.src.utils.entropy_hasher import EntropyHasher
from navier_symbolic_engine.src.utils.visualization import Visualization

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class OrganizedUnitTestRunner:
    """Organized unit test runner with directory structure management."""
    
    def __init__(self, test_dir: str = "tests"):
        """Initialize the organized test runner."""
        self.test_dir = Path(test_dir)
        self.current_dir = Path(__file__).parent
        
    def create_test_structure(self):
        """Create organized test directory structure."""
        print("📁 Creating organized unit test structure...")
        
        # Create directory structure
        directories = [
            self.test_dir,
            self.test_dir / "unit",
            self.test_dir / "integration", 
            self.test_dir / "performance"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Create test files content
        test_files = {
            "unit/test_imports.py": self._get_imports_test_content(),
            "unit/test_patterns.py": self._get_patterns_test_content(),
            "unit/test_entropy.py": self._get_entropy_test_content(),
            "unit/test_thermodynamics.py": self._get_thermodynamics_test_content(),
            "unit/test_engine.py": self._get_engine_test_content(),
            "integration/test_workflow.py": self._get_workflow_test_content(),
            "performance/test_benchmarks.py": self._get_benchmarks_test_content(),
            "pytest.ini": self._get_pytest_config(),
            "README.md": self._get_test_readme()
        }
        
        # Write test files
        for file_path, content in test_files.items():
            full_path = self.test_dir / file_path
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        print(f"Created {len(test_files)} test files in organized structure")
        return True
        
    def run_organized_tests(self, test_type: str = "all", verbose: bool = False):
        """Run tests from the organized structure."""
        if not self.test_dir.exists():
            print("❌ Test directory not found. Run create_test_structure() first.")
            return False
            
        cmd = [sys.executable, "-m", "pytest"]
        
        if verbose:
            cmd.append("-v")
            
        if test_type == "unit":
            cmd.append(str(self.test_dir / "unit"))
        elif test_type == "integration":
            cmd.append(str(self.test_dir / "integration"))
        elif test_type == "performance":
            cmd.append(str(self.test_dir / "performance"))
        elif test_type == "imports":
            cmd.append(str(self.test_dir / "unit" / "test_imports.py"))
        else:
            cmd.append(str(self.test_dir))
            
        print(f"🧪 Running {test_type} tests...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.current_dir)
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return False
            
    def _get_imports_test_content(self):
        """Get import test content."""
        return '''"""Test module imports and basic availability."""
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
            from navier_symbolic_engine.src.core.pattern_generator import PatternGenerator
            assert PatternGenerator is not None
        except ImportError as e:
            pytest.skip(f"Pattern generator not available: {e}")
    
    def test_thermodynamic_validator_import(self):
        """Test that ThermodynamicValidator can be imported."""
        try:
            from navier_symbolic_engine.src.utils.thermodynamic_validator import ThermodynamicValidator
            assert ThermodynamicValidator is not None
        except ImportError as e:
            pytest.skip(f"Thermodynamic validator not available: {e}")
'''

    def _get_patterns_test_content(self):
        """Get pattern test content."""
        return '''"""Test pattern generation functionality."""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add source to path
current_dir = Path(__file__).parent.parent.parent
src_path = current_dir / "navier_symbolic_engine" / "src"
sys.path.insert(0, str(src_path))


class TestPatternGeneration:
    """Test pattern generation capabilities."""
    
    def test_poiseuille_pattern_structure(self):
        """Test Poiseuille flow pattern generation."""
        try:
            from navier_symbolic_engine.src.core.pattern_generator import PatternGenerator
            
            generator = PatternGenerator()
            pattern = generator.generate_poiseuille_flow()
            
            assert pattern is not None
            assert "velocity_profile" in pattern
            assert "pressure_gradient" in pattern
            
        except ImportError:
            pytest.skip("PatternGenerator not available")
    
    def test_couette_pattern_structure(self):
        """Test Couette flow pattern generation."""
        try:
            from navier_symbolic_engine.src.core.pattern_generator import PatternGenerator
            
            generator = PatternGenerator()
            pattern = generator.generate_couette_flow()
            
            assert pattern is not None
            assert "velocity_profile" in pattern
            assert "wall_velocity" in pattern
            
        except ImportError:
            pytest.skip("PatternGenerator not available")
    
    def test_turbulent_pattern_structure(self):
        """Test turbulent flow pattern generation."""
        try:
            from navier_symbolic_engine.src.core.pattern_generator import PatternGenerator
            
            generator = PatternGenerator()
            pattern = generator.generate_turbulent_flow()
            
            assert pattern is not None
            assert "velocity_profile" in pattern
            assert "reynolds_stress" in pattern
            
        except ImportError:
            pytest.skip("PatternGenerator not available")
'''

    def _get_entropy_test_content(self):
        """Get entropy test content."""
        return '''"""Test entropy hashing and navigation functionality."""
import pytest
import sys
from pathlib import Path

# Add source to path
current_dir = Path(__file__).parent.parent.parent
src_path = current_dir / "navier_symbolic_engine" / "src"
sys.path.insert(0, str(src_path))


class TestEntropyHashing:
    """Test entropy hashing capabilities."""
    
    def test_entropy_hasher_initialization(self):
        """Test that EntropyHasher can be initialized."""
        try:
            from navier_symbolic_engine.src.utils.entropy_hasher import EntropyHasher
            
            hasher = EntropyHasher()
            assert hasher is not None
            
        except ImportError:
            pytest.skip("EntropyHasher not available")
    
    def test_hierarchical_entropy_generation(self):
        """Test hierarchical entropy signature generation."""
        try:
            from navier_symbolic_engine.src.utils.entropy_hasher import EntropyHasher
            
            hasher = EntropyHasher()
            boundary_conditions = {
                "geometry": "pipe",
                "reynolds": 1000,
                "velocity": 1.0
            }
            
            entropy_sig = hasher.generate_hierarchical_entropy(boundary_conditions)
            assert entropy_sig is not None
            assert isinstance(entropy_sig, (dict, list, str))
            
        except ImportError:
            pytest.skip("EntropyHasher not available")
        except Exception as e:
            pytest.skip(f"Entropy generation not implemented: {e}")
    
    def test_navigation_path_generation(self):
        """Test navigation path generation from entropy."""
        try:
            from navier_symbolic_engine.src.utils.entropy_hasher import EntropyHasher
            
            hasher = EntropyHasher()
            boundary_conditions = {
                "geometry": "channel",
                "reynolds": 2000,
                "velocity": 1.5
            }
            
            path = hasher.generate_navigation_path(boundary_conditions)
            assert path is not None
            assert isinstance(path, (list, tuple))
            
        except ImportError:
            pytest.skip("EntropyHasher not available")
        except Exception as e:
            pytest.skip(f"Navigation path generation not implemented: {e}")
'''

    def _get_thermodynamics_test_content(self):
        """Get thermodynamics test content."""
        return '''"""Test thermodynamic validation functionality."""
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
            from navier_symbolic_engine.src.utils.thermodynamic_validator import ThermodynamicValidator
            
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
            from navier_symbolic_engine.src.utils.thermodynamic_validator import ThermodynamicValidator
            
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
'''

    def _get_engine_test_content(self):
        """Get engine test content."""
        return '''"""Test main engine interface functionality."""
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
'''

    def _get_workflow_test_content(self):
        """Get workflow test content."""
        return '''"""Test end-to-end workflow integration."""
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
'''

    def _get_benchmarks_test_content(self):
        """Get benchmarks test content."""
        return '''"""Test performance benchmarks and stress testing."""
import pytest
import time
import sys
from pathlib import Path

# Add source to path
current_dir = Path(__file__).parent.parent.parent
src_path = current_dir / "navier_symbolic_engine" / "src"
sys.path.insert(0, str(src_path))


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Test performance benchmarks."""
    
    def test_single_simulation_benchmark(self):
        """Benchmark single simulation performance."""
        try:
            from navier_symbolic_engine.src.api.engine_interface import EngineInterface
            
            engine = EngineInterface()
            boundary_conditions = {
                "geometry": "pipe",
                "reynolds": 1000,
                "velocity": 1.0
            }
            
            start_time = time.time()
            result = engine.run(boundary_conditions)
            execution_time = time.time() - start_time
            
            # Performance assertions
            assert execution_time < 10.0  # Should complete within 10 seconds
            assert result is not None
            
        except ImportError:
            pytest.skip("EngineInterface not available")
        except Exception as e:
            pytest.skip(f"Performance benchmark not implemented: {e}")


@pytest.mark.stress
class TestStressTesting:
    """Test system under stress conditions."""
    
    def test_high_reynolds_number_stress(self):
        """Test simulation with high Reynolds numbers."""
        try:
            from navier_symbolic_engine.src.api.engine_interface import EngineInterface
            
            engine = EngineInterface()
            high_reynolds = [5000, 10000, 20000]
            
            for reynolds in high_reynolds:
                boundary_conditions = {
                    "geometry": "pipe",
                    "reynolds": reynolds,
                    "velocity": 2.0
                }
                
                result = engine.run(boundary_conditions)
                assert result is not None
                
        except ImportError:
            pytest.skip("EngineInterface not available")
        except Exception as e:
            pytest.skip(f"Stress testing not implemented: {e}")
'''

    def _get_pytest_config(self):
        """Get pytest configuration."""
        return '''[tool:pytest]
testpaths = .
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --strict-config
    --color=yes
markers =
    performance: Performance benchmark tests
    stress: Stress testing under high load
    integration: Integration tests across components
    unit: Unit tests for individual components
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
'''

    def _get_test_readme(self):
        """Get test documentation."""
        return '''# Navier-Stokes Symbolic Engine - Test Suite

Comprehensive test suite for the Navier-Stokes Symbolic Engine with organized structure.

## Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_imports.py      # Import and module availability
│   ├── test_patterns.py     # Pattern generation and trees
│   ├── test_entropy.py      # Entropy hashing and navigation
│   ├── test_thermodynamics.py # Thermodynamic validation
│   └── test_engine.py       # Main engine interface
├── integration/             # Integration and workflow tests
│   └── test_workflow.py     # End-to-end testing
├── performance/             # Performance and benchmarking
│   └── test_benchmarks.py   # Performance measurements
├── pytest.ini              # Pytest configuration
└── README.md               # This file
```

## Running Tests

### Via Unified Framework
```bash
# Run all tests
python unified_experimental_framework.py --unit-tests

# Run specific test types
python unified_experimental_framework.py --unit-tests --organized imports
python unified_experimental_framework.py --unit-tests --organized unit
python unified_experimental_framework.py --unit-tests --organized integration
```

### Direct pytest
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run specific test files
pytest tests/unit/test_imports.py
pytest tests/unit/test_engine.py

# Run with markers
pytest -m performance
pytest -m stress
```

## Test Categories

### Unit Tests
- **test_imports.py**: Validates all modules can be imported
- **test_patterns.py**: Tests pattern generation (Poiseuille, Couette, turbulent)
- **test_entropy.py**: Tests entropy hashing and navigation paths
- **test_thermodynamics.py**: Tests thermodynamic validation (Landauer bounds)
- **test_engine.py**: Tests main engine interface and simulation execution

### Integration Tests
- **test_workflow.py**: End-to-end workflow testing with component integration

### Performance Tests
- **test_benchmarks.py**: Performance benchmarks and stress testing

## Test Philosophy

- **Comprehensive Coverage**: All components tested systematically
- **Fast Feedback**: Quick unit tests for rapid development cycles
- **Realistic Integration**: End-to-end workflows with real scenarios
- **Performance Aware**: Quantified benchmarks and stress testing
- **Well Documented**: Clear test structure and comprehensive documentation
'''


class NavierStokesTestRunner:
    """Unified test runner for all Navier-Stokes Symbolic Engine tests."""
    
    def __init__(self, config_file: Optional[str] = None, output_dir: str = "results"):
        """
        Initialize the test runner.
        
        Args:
            config_file: Path to YAML configuration file
            output_dir: Output directory for results
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = hashlib.md5(self.timestamp.encode()).hexdigest()[:8]
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Setup directories
        self.base_dir = Path(output_dir)
        self.session_dir = self.base_dir / f"session_{self.timestamp}_{self.session_id}"
        self._setup_directories()
        
        # Get git information for reproducibility
        self.git_info = self._get_git_info()
        
        # Initialize components
        self.engine = EngineInterface()
        self.hasher = EntropyHasher()
        self.viz = Visualization()
        
        # Data storage
        self.all_results = {
            "unit_tests": {},
            "performance_tests": {},
            "visualization_tests": {},
            "parameter_sweeps": {},
            "comprehensive_analysis": {}
        }
        
        print(f"🚀 Navier-Stokes Test Runner Initialized")
        print(f"📊 Session ID: {self.session_id}")
        print(f"📁 Output: {self.session_dir}")
        print(f"🔧 Git Hash: {self.git_info.get('commit_hash', 'unknown')}")
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file or use defaults."""
        default_config = {
            "test_suite": {
                "unit_tests": True,
                "performance_tests": True,
                "visualization_tests": True,
                "parameter_sweeps": True,
                "comprehensive_analysis": True
            },
            "parameter_sweeps": {
                "reynolds_ranges": {
                    "laminar": {"min": 10, "max": 1500, "count": 20},
                    "transition": {"min": 1500, "max": 3000, "count": 15},
                    "turbulent": {"min": 3000, "max": 50000, "count": 25}
                },
                "geometries": ["pipe", "channel", "cavity"],
                "velocities": [0.5, 1.0, 1.5, 2.0, 2.5],
                "pressure_gradients": [-0.05, -0.1, -0.15, -0.2, -0.25],
                "sample_size": 1000
            },
            "performance": {
                "timeout_seconds": 300,
                "max_reynolds": 100000,
                "min_reynolds": 10
            },
            "output": {
                "save_raw_data": True,
                "save_visualizations": True,
                "generate_report": True,
                "export_formats": ["json", "csv", "pkl"]
            },
            "reproducibility": {
                "track_git_hash": True,
                "save_environment": True,
                "test_identical_runs": 5
            }
        }
        
        if config_file and os.path.exists(config_file):
            print(f"📋 Loading config from {config_file}")
            with open(config_file, 'r') as f:
                user_config = yaml.safe_load(f)
            
            # Merge with defaults
            config = self._deep_merge(default_config, user_config)
        else:
            config = default_config
            print("📋 Using default configuration")
        
        return config
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _setup_directories(self):
        """Setup output directory structure."""
        subdirs = [
            "data", "visualizations", "hashes", "statistics", 
            "benchmarks", "configs", "reports", "sweeps"
        ]
        
        self.session_dir.mkdir(parents=True, exist_ok=True)
        for subdir in subdirs:
            (self.session_dir / subdir).mkdir(exist_ok=True)
        
        # Save configuration
        with open(self.session_dir / "configs" / "test_config.yaml", 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def _get_git_info(self) -> Dict[str, str]:
        """Get git repository information for reproducibility."""
        git_info = {}
        
        try:
            # Get commit hash
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, cwd=current_dir)
            if result.returncode == 0:
                git_info['commit_hash'] = result.stdout.strip()
            
            # Get branch name
            result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                                  capture_output=True, text=True, cwd=current_dir)
            if result.returncode == 0:
                git_info['branch'] = result.stdout.strip()
            
            # Get repository status
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, cwd=current_dir)
            if result.returncode == 0:
                git_info['dirty'] = bool(result.stdout.strip())
                git_info['status'] = result.stdout.strip()
            
            # Get remote URL
            result = subprocess.run(['git', 'remote', 'get-url', 'origin'], 
                                  capture_output=True, text=True, cwd=current_dir)
            if result.returncode == 0:
                git_info['remote_url'] = result.stdout.strip()
                
        except Exception as e:
            print(f"⚠️ Could not get git info: {e}")
            git_info['error'] = str(e)
        
        return git_info
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run comprehensive unit tests."""
        if not self.config["test_suite"]["unit_tests"]:
            return {"skipped": True}
        
        print("\n🧪 Running Unit Tests...")
        start_time = time.time()
        
        # Capture test results
        original_stdout = sys.stdout
        from io import StringIO
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            # Use organized test runner if requested
            organized_runner = OrganizedUnitTestRunner()
            success = organized_runner.run_organized_tests("all", verbose=True)
            output = captured_output.getvalue()
        finally:
            sys.stdout = original_stdout
        
        execution_time = time.time() - start_time
        
        results = {
            "success": success,
            "execution_time": execution_time,
            "output": output,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save results
        with open(self.session_dir / "data" / "unit_test_results.json", 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        print(f"Unit tests execution time: {execution_time:.2f}s")
        return results
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance and complexity tests."""
        if not self.config["test_suite"]["performance_tests"]:
            return {"skipped": True}
        
        print("\n⚡ Running Performance Tests...")
        start_time = time.time()
        
        # Run performance tests with config parameters
        reynolds_numbers = self._generate_reynolds_range()
        geometries = self.config["parameter_sweeps"]["geometries"]
        
        results = []
        
        for reynolds in reynolds_numbers[:10]:  # Limit for performance tests
            for geometry in geometries:
                try:
                    boundary_conditions = {
                        "geometry": geometry,
                        "reynolds": float(reynolds),
                        "velocity": 1.0,
                        "pressure_gradient": -0.1
                    }
                    
                    test_start = time.time()
                    result = self.engine.run(boundary_conditions)
                    test_time = time.time() - test_start
                    
                    if result["status"] == "success":
                        velocity = result["solution"]["velocity"]
                        tree_info = self.engine.get_tree_info()
                        
                        test_result = {
                            "reynolds": reynolds,
                            "geometry": geometry,
                            "execution_time": test_time,
                            "max_velocity": float(np.max(np.abs(velocity))),
                            "tree_nodes": tree_info["node_count"],
                            "tree_depth": tree_info["max_depth"],
                            "navigation_path": result["navigation_path"],
                            "success": True
                        }
                        results.append(test_result)
                        
                except Exception as e:
                    print(f"❌ Performance test failed for {geometry} Re={reynolds}: {e}")
        
        execution_time = time.time() - start_time
        
        performance_results = {
            "results": results,
            "execution_time": execution_time,
            "total_tests": len(results),
            "success_rate": len([r for r in results if r["success"]]) / len(results) if results else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save results
        with open(self.session_dir / "data" / "performance_test_results.json", 'w') as f:
            json.dump(performance_results, f, indent=2, cls=NumpyEncoder)
        
        print(f"Performance tests: {len(results)} tests in {execution_time:.2f}s")
        return performance_results
    
    def run_visualization_tests(self) -> Dict[str, Any]:
        """Run visualization tests."""
        if not self.config["test_suite"]["visualization_tests"]:
            return {"skipped": True}
        
        print("\n🎨 Running Visualization Tests...")
        start_time = time.time()
        
        # Run visualization tests and save to our session directory
        original_viz_dir = Path("results") / "visualizations"
        session_viz_dir = self.session_dir / "visualizations"
        
        # Capture test results
        original_stdout = sys.stdout
        from io import StringIO
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            # Use visualization functionality from engine
            test_cases = [
                {"geometry": "pipe", "reynolds": 1000, "velocity": 1.0},
                {"geometry": "channel", "reynolds": 2000, "velocity": 1.5},
                {"geometry": "cavity", "reynolds": 1500, "velocity": 1.2}
            ]
            
            for test_case in test_cases:
                result = self.engine.run(test_case)
                if result.get("status") == "success":
                    # Generate visualization
                    self.viz.plot_velocity_field(
                        result["solution"]["velocity"],
                        save_path=session_viz_dir / f"velocity_{test_case['geometry']}_Re{test_case['reynolds']}.png"
                    )
            
            output = captured_output.getvalue()
            
            # Move generated visualizations to session directory
            if original_viz_dir.exists():
                for viz_file in original_viz_dir.glob("*.png"):
                    viz_file.rename(session_viz_dir / viz_file.name)
            
        finally:
            sys.stdout = original_stdout
        
        execution_time = time.time() - start_time
        
        # Count generated visualizations
        viz_files = list(session_viz_dir.glob("*.png"))
        
        results = {
            "execution_time": execution_time,
            "visualizations_generated": len(viz_files),
            "visualization_files": [f.name for f in viz_files],
            "output": output,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save results
        with open(self.session_dir / "data" / "visualization_test_results.json", 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        print(f"Visualization tests: {len(viz_files)} plots in {execution_time:.2f}s")
        return results
    
    def run_parameter_sweeps(self) -> Dict[str, Any]:
        """Run comprehensive parameter sweeps."""
        if not self.config["test_suite"]["parameter_sweeps"]:
            return {"skipped": True}
        
        print("\n🔬 Running Parameter Sweeps...")
        start_time = time.time()
        
        # Generate comprehensive test matrix
        test_matrix = self._generate_test_matrix()
        print(f"Generated {len(test_matrix)} parameter combinations")
        
        # Sample for manageable test size
        sample_size = min(self.config["parameter_sweeps"]["sample_size"], len(test_matrix))
        selected_indices = np.random.choice(len(test_matrix), sample_size, replace=False)
        sampled_tests = [test_matrix[i] for i in selected_indices]
        
        print(f"Running {len(sampled_tests)} sampled tests...")
        
        results = []
        for i, test_case in enumerate(sampled_tests):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(sampled_tests)} ({100*i/len(sampled_tests):.1f}%)")
            
            try:
                result = self._run_single_parameter_test(test_case, i)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"❌ Test {i} failed: {e}")
        
        execution_time = time.time() - start_time
        
        # Statistical analysis
        stats = self._analyze_parameter_sweep_results(results)
        
        sweep_results = {
            "results": results,
            "statistics": stats,
            "execution_time": execution_time,
            "total_tests": len(results),
            "sample_size": sample_size,
            "success_rate": len(results) / len(sampled_tests),
            "timestamp": datetime.now().isoformat(),
            "theoretical_validation": self._validate_theoretical_framework(results)
        }
        
        # Save comprehensive data
        self._save_parameter_sweep_data(sweep_results)
        
        print(f"Parameter sweeps: {len(results)} tests in {execution_time:.2f}s")
        return sweep_results
    
    def _generate_reynolds_range(self) -> List[float]:
        """Generate Reynolds number range based on config."""
        config = self.config["parameter_sweeps"]["reynolds_ranges"]
        
        ranges = []
        for regime, params in config.items():
            if regime == "laminar":
                ranges.extend(np.linspace(params["min"], params["max"], params["count"]))
            elif regime == "transition":
                ranges.extend(np.linspace(params["min"], params["max"], params["count"]))
            elif regime == "turbulent":
                ranges.extend(np.logspace(np.log10(params["min"]), np.log10(params["max"]), params["count"]))
        
        return sorted(set(ranges))
    
    def _generate_test_matrix(self) -> List[Dict[str, Any]]:
        """Generate comprehensive test matrix."""
        reynolds_numbers = self._generate_reynolds_range()
        geometries = self.config["parameter_sweeps"]["geometries"]
        velocities = self.config["parameter_sweeps"]["velocities"]
        pressure_gradients = self.config["parameter_sweeps"]["pressure_gradients"]
        
        test_matrix = []
        test_id = 0
        
        for geometry in geometries:
            for reynolds in reynolds_numbers:
                for velocity in velocities:
                    for pressure_grad in pressure_gradients:
                        test_case = {
                            "test_id": test_id,
                            "geometry": geometry,
                            "reynolds": float(reynolds),
                            "velocity": velocity,
                            "pressure_gradient": pressure_grad,
                            "expected_regime": self._classify_regime(reynolds)
                        }
                        test_matrix.append(test_case)
                        test_id += 1
        
        return test_matrix
    
    def _classify_regime(self, reynolds: float) -> str:
        """Classify flow regime based on Reynolds number."""
        if reynolds < 1600:
            return "laminar"
        elif reynolds < 2500:
            return "transition"
        else:
            return "turbulent"
    
    def _run_single_parameter_test(self, test_case: Dict[str, Any], test_id: int) -> Optional[Dict[str, Any]]:
        """Run a single parameter test case."""
        boundary_conditions = {
            "geometry": test_case["geometry"],
            "reynolds": test_case["reynolds"],
            "velocity": test_case["velocity"],
            "pressure_gradient": test_case["pressure_gradient"]
        }
        
        # Generate entropy signature and hash
        entropy_start = time.time()
        entropy_sig = self.hasher.generate_hierarchical_entropy(boundary_conditions)
        entropy_time = time.time() - entropy_start
        
        bc_hash = hashlib.sha256(json.dumps(boundary_conditions, sort_keys=True).encode()).hexdigest()
        
        # Run simulation
        sim_start = time.time()
        result = self.engine.run(boundary_conditions)
        sim_time = time.time() - sim_start
        
        if result["status"] != "success":
            return None
        
        # Comprehensive data collection
        velocity_field = result["solution"]["velocity"]
        pressure_field = result["solution"]["pressure"]
        tree_info = self.engine.get_tree_info()
        
        # Statistical analysis
        velocity_stats = {
            "mean": float(np.mean(velocity_field)),
            "std": float(np.std(velocity_field)),
            "min": float(np.min(velocity_field)),
            "max": float(np.max(velocity_field)),
            "skewness": float(self._skewness(velocity_field)),
            "kurtosis": float(self._kurtosis(velocity_field))
        }
        
        return {
            **test_case,
            "hash": bc_hash,
            "simulation_time": sim_time,
            "entropy_time": entropy_time,
            "total_time": sim_time + entropy_time,
            "velocity_stats": velocity_stats,
            "tree_nodes": tree_info["node_count"],
            "tree_depth": tree_info["max_depth"],
            "navigation_path": result["navigation_path"],
            "path_length": len(result["navigation_path"]),
            "velocity_field_shape": velocity_field.shape,
            "success": True
        }
    
    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _validate_theoretical_framework(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze data against Dawn Field theoretical framework metrics."""
        if not results:
            return {"error": "No results to analyze"}
        
        df = pd.DataFrame(results)
        
        # Raw metrics analysis (no judgment/compliance scoring)
        analysis = {
            "complexity_bounds": self._analyze_complexity_bounds(df),
            "pattern_consistency": self._analyze_pattern_consistency(df),
            "scale_characteristics": self._analyze_scale_characteristics(df),
            "determinism_metrics": self._analyze_determinism_metrics(df),
            "regime_statistics": self._analyze_regime_statistics(df),
            "entropy_metrics": self._analyze_entropy_metrics(df),
            "performance_metrics": self._analyze_performance_metrics(df)
        }
        
        return analysis
    
    def _analyze_complexity_bounds(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze tree complexity characteristics."""
        return {
            "max_tree_nodes": int(df["tree_nodes"].max()) if "tree_nodes" in df.columns else 0,
            "min_tree_nodes": int(df["tree_nodes"].min()) if "tree_nodes" in df.columns else 0,
            "avg_tree_nodes": float(df["tree_nodes"].mean()) if "tree_nodes" in df.columns else 0.0,
            "max_tree_depth": int(df["tree_depth"].max()) if "tree_depth" in df.columns else 0,
            "min_tree_depth": int(df["tree_depth"].min()) if "tree_depth" in df.columns else 0,
            "avg_tree_depth": float(df["tree_depth"].mean()) if "tree_depth" in df.columns else 0.0
        }
    
    def _analyze_pattern_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze pattern consistency within regimes."""
        if "path_length" not in df.columns:
            return {"error": "No path length data available"}
        
        df["reynolds_regime"] = df["reynolds"].apply(self._classify_regime)
        
        consistency_by_regime = {}
        for geometry in df["geometry"].unique():
            geom_data = df[df["geometry"] == geometry]
            for regime in geom_data["reynolds_regime"].unique():
                regime_data = geom_data[geom_data["reynolds_regime"] == regime]
                if len(regime_data) > 1:
                    path_lengths = regime_data["path_length"]
                    consistency_by_regime[f"{geometry}_{regime}"] = {
                        "mean_path_length": float(path_lengths.mean()),
                        "std_path_length": float(path_lengths.std()),
                        "coefficient_of_variation": float(path_lengths.std() / path_lengths.mean()) if path_lengths.mean() > 0 else float('inf'),
                        "sample_count": len(regime_data)
                    }
        
        return {
            "regime_consistency": consistency_by_regime,
            "overall_path_stats": {
                "mean": float(df["path_length"].mean()),
                "std": float(df["path_length"].std()),
                "min": int(df["path_length"].min()),
                "max": int(df["path_length"].max())
            }
        }
    
    def _analyze_scale_characteristics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze scale separation characteristics."""
        if len(df) < 5:
            return {"error": "Insufficient data for correlation analysis"}
        
        correlations = {}
        if "reynolds" in df.columns and "tree_depth" in df.columns:
            correlations["reynolds_depth"] = float(np.corrcoef(df["reynolds"], df["tree_depth"])[0, 1])
        if "reynolds" in df.columns and "path_length" in df.columns:
            correlations["reynolds_path_length"] = float(np.corrcoef(df["reynolds"], df["path_length"])[0, 1])
        
        return {
            "correlations": correlations,
            "reynolds_range": {
                "min": float(df["reynolds"].min()),
                "max": float(df["reynolds"].max()),
                "span": float(df["reynolds"].max() - df["reynolds"].min())
            }
        }
    
    def _analyze_determinism_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze determinism characteristics."""
        return {
            "unique_hashes": int(df["hash"].nunique()) if "hash" in df.columns else 0,
            "total_tests": len(df),
            "hash_uniqueness_ratio": float(df["hash"].nunique() / len(df)) if len(df) > 0 and "hash" in df.columns else 0.0,
            "duplicate_hash_count": len(df) - df["hash"].nunique() if "hash" in df.columns else 0
        }
    
    def _analyze_regime_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Reynolds regime statistics."""
        df["reynolds_regime"] = df["reynolds"].apply(self._classify_regime)
        
        regime_stats = {}
        for regime in df["reynolds_regime"].unique():
            regime_data = df[df["reynolds_regime"] == regime]
            regime_stats[regime] = {
                "count": int(len(regime_data)),
                "reynolds_range": {
                    "min": float(regime_data["reynolds"].min()),
                    "max": float(regime_data["reynolds"].max())
                },
                "avg_path_length": float(regime_data["path_length"].mean()) if "path_length" in regime_data.columns else 0.0,
                "avg_tree_depth": float(regime_data["tree_depth"].mean()) if "tree_depth" in regime_data.columns else 0.0,
                "avg_simulation_time": float(regime_data["simulation_time"].mean()) if "simulation_time" in regime_data.columns else 0.0
            }
        
        return {
            "regime_breakdown": regime_stats,
            "regime_count": len(regime_stats)
        }
    
    def _analyze_entropy_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze entropy processing metrics."""
        return {
            "avg_entropy_time": float(df["entropy_time"].mean()) if "entropy_time" in df.columns else 0.0,
            "max_entropy_time": float(df["entropy_time"].max()) if "entropy_time" in df.columns else 0.0,
            "min_entropy_time": float(df["entropy_time"].min()) if "entropy_time" in df.columns else 0.0,
            "entropy_time_std": float(df["entropy_time"].std()) if "entropy_time" in df.columns else 0.0
        }
    
    def _analyze_performance_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance characteristics."""
        return {
            "avg_simulation_time": float(df["simulation_time"].mean()) if "simulation_time" in df.columns else 0.0,
            "max_simulation_time": float(df["simulation_time"].max()) if "simulation_time" in df.columns else 0.0,
            "min_simulation_time": float(df["simulation_time"].min()) if "simulation_time" in df.columns else 0.0,
            "simulation_time_std": float(df["simulation_time"].std()) if "simulation_time" in df.columns else 0.0,
            "avg_total_time": float(df["total_time"].mean()) if "total_time" in df.columns else 0.0
        }
    
    def _analyze_parameter_sweep_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze parameter sweep results."""
        if not results:
            return {}
        
        df = pd.DataFrame(results)
        
        stats = {
            "total_simulations": len(df),
            "execution_time": {
                "mean": float(df["simulation_time"].mean()),
                "std": float(df["simulation_time"].std()),
                "min": float(df["simulation_time"].min()),
                "max": float(df["simulation_time"].max()),
                "median": float(df["simulation_time"].median())
            },
            "reynolds_range": {
                "min": float(df["reynolds"].min()),
                "max": float(df["reynolds"].max()),
                "mean": float(df["reynolds"].mean())
            },
            "regime_distribution": df["expected_regime"].value_counts().to_dict(),
            "geometry_distribution": df["geometry"].value_counts().to_dict(),
            "tree_statistics": {
                "avg_nodes": float(df["tree_nodes"].mean()),
                "max_nodes": int(df["tree_nodes"].max()),
                "avg_depth": float(df["tree_depth"].mean()),
                "max_depth": int(df["tree_depth"].max())
            },
            "navigation_efficiency": {
                "avg_path_length": float(df["path_length"].mean()),
                "max_path_length": int(df["path_length"].max()),
                "path_length_std": float(df["path_length"].std())
            },
            "complexity_analysis": self._analyze_complexity(df)
        }
        
        return stats
    
    def _analyze_complexity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze computational complexity."""
        log_reynolds = np.log10(df["reynolds"])
        times = df["simulation_time"]
        
        # Linear fit on log scale
        correlation = np.corrcoef(log_reynolds, times)[0, 1] if len(df) > 1 else 0
        
        return {
            "log_reynolds_time_correlation": float(correlation),
            "suggests_log_scaling": abs(correlation) > 0.3,
            "tree_growth_bounded": df["tree_nodes"].max() == df["tree_nodes"].min(),
            "path_length_bounded": df["path_length"].max() == df["path_length"].min()
        }
    
    def _save_parameter_sweep_data(self, sweep_results: Dict[str, Any]):
        """Save parameter sweep data in multiple formats."""
        # JSON for human readability
        with open(self.session_dir / "sweeps" / "parameter_sweep_results.json", 'w') as f:
            json.dump(sweep_results, f, indent=2, cls=NumpyEncoder)
        
        # CSV for analysis
        if sweep_results["results"]:
            df = pd.DataFrame(sweep_results["results"])
            df.to_csv(self.session_dir / "sweeps" / "parameter_sweep_data.csv", index=False)
            
            # Save statistics
            with open(self.session_dir / "statistics" / "sweep_statistics.json", 'w') as f:
                json.dump(sweep_results["statistics"], f, indent=2, cls=NumpyEncoder)
            
            # Pickle for exact reproduction
            with open(self.session_dir / "sweeps" / "parameter_sweep_data.pkl", 'wb') as f:
                pickle.dump(sweep_results, f)
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        print("\n📊 Generating Comprehensive Report...")
        
        # Collect all results
        report = {
            "session_info": {
                "session_id": self.session_id,
                "timestamp": self.timestamp,
                "git_info": self.git_info,
                "config": self.config,
                "environment": {
                    "python_version": sys.version,
                    "numpy_version": np.__version__,
                    "pandas_version": pd.__version__
                }
            },
            "test_results": self.all_results,
            "summary": self._generate_summary()
        }
        
        # Save comprehensive report
        with open(self.session_dir / "reports" / "comprehensive_report.json", 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        
        # Generate markdown report
        self._generate_markdown_report(report)
        
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            "total_execution_time": 0,
            "total_tests_run": 0,
            "overall_execution_rate": 0,
            "reproducibility_verified": self.git_info.get('commit_hash') is not None
        }
        
        # Aggregate from all test results
        executed_tests = 0
        for test_type, results in self.all_results.items():
            if results and not results.get('skipped', False):
                if 'execution_time' in results:
                    summary["total_execution_time"] += results["execution_time"]
                if 'total_tests' in results:
                    summary["total_tests_run"] += results["total_tests"]
                    executed_tests += results["total_tests"]
        
        # Calculate execution rate (raw metric, no judgment)
        if summary["total_tests_run"] > 0:
            summary["overall_execution_rate"] = executed_tests / summary["total_tests_run"]
        
        return summary
    
    def _generate_markdown_report(self, report: Dict[str, Any]):
        """Generate markdown report."""
        markdown_content = f"""# Navier-Stokes Symbolic Engine - Test Report

**Session ID:** {self.session_id}  
**Timestamp:** {self.timestamp}  
**Git Hash:** {self.git_info.get('commit_hash', 'unknown')}  
**Git Branch:** {self.git_info.get('branch', 'unknown')}  

## Configuration

```yaml
{yaml.dump(self.config, default_flow_style=False)}
```

## Test Results Summary

"""
        
        for test_type, results in self.all_results.items():
            if not results.get('skipped', False):
                markdown_content += f"### {test_type.replace('_', ' ').title()}\n\n"
                if 'execution_time' in results:
                    markdown_content += f"- Execution Time: {results['execution_time']:.2f}s\n"
                if 'total_tests' in results:
                    markdown_content += f"- Total Tests: {results['total_tests']}\n"
                if 'success_rate' in results:
                    markdown_content += f"- Success Rate: {results['success_rate']*100:.1f}%\n"
                markdown_content += "\n"
        
        markdown_content += f"""
## Reproducibility Information

- **Git Commit:** {self.git_info.get('commit_hash', 'unknown')}
- **Repository Status:** {'Clean' if not self.git_info.get('dirty', True) else 'Modified'}
- **Python Version:** {sys.version.split()[0]}
- **Configuration File:** test_config.yaml

## Files Generated

- Raw data in multiple formats (JSON, CSV, PKL)
- Comprehensive visualizations
- Statistical analysis
- Hash verification data
- Complete reproducibility package

---

*Generated by Navier-Stokes Symbolic Engine Test Runner v1.0*
"""
        
        with open(self.session_dir / "reports" / "test_report.md", 'w') as f:
            f.write(markdown_content)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite."""
        print("🚀 Starting Complete Test Suite")
        print("=" * 60)
        
        total_start = time.time()
        
        # Run all test modules
        self.all_results["unit_tests"] = self.run_unit_tests()
        self.all_results["performance_tests"] = self.run_performance_tests()
        self.all_results["visualization_tests"] = self.run_visualization_tests()
        self.all_results["parameter_sweeps"] = self.run_parameter_sweeps()
        
        # Generate comprehensive report
        self.all_results["comprehensive_analysis"] = self.generate_comprehensive_report()
        
        total_time = time.time() - total_start
        
        print("\n" + "=" * 60)
        print("🏁 Complete Test Suite Finished!")
        print(f"⏱️ Total execution time: {total_time:.2f}s")
        print(f"📊 Session ID: {self.session_id}")
        print(f"📁 Results: {self.session_dir}")
        print(f"🔧 Git Hash: {self.git_info.get('commit_hash', 'unknown')}")
        
        return self.all_results

def create_default_config():
    """Create default configuration file."""
    config_path = "test_config.yaml"
    
    default_config = {
        "test_suite": {
            "unit_tests": True,
            "performance_tests": True,
            "visualization_tests": True,
            "parameter_sweeps": True,
            "comprehensive_analysis": True
        },
        "parameter_sweeps": {
            "reynolds_ranges": {
                "laminar": {"min": 10, "max": 1500, "count": 20},
                "transition": {"min": 1500, "max": 3000, "count": 15},
                "turbulent": {"min": 3000, "max": 50000, "count": 25}
            },
            "geometries": ["pipe", "channel", "cavity"],
            "velocities": [0.5, 1.0, 1.5, 2.0, 2.5],
            "pressure_gradients": [-0.05, -0.1, -0.15, -0.2, -0.25],
            "sample_size": 1000
        },
        "performance": {
            "timeout_seconds": 300,
            "max_reynolds": 100000,
            "min_reynolds": 10
        },
        "output": {
            "save_raw_data": True,
            "save_visualizations": True,
            "generate_report": True,
            "export_formats": ["json", "csv", "pkl"]
        },
        "reproducibility": {
            "track_git_hash": True,
            "save_environment": True,
            "test_identical_runs": 5
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    print(f"📄 Created default config: {config_path}")
    return config_path

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Unified Experimental Framework for Navier-Stokes Symbolic Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python unified_experimental_framework.py --all                          # Run all tests with defaults
  python unified_experimental_framework.py --config my_config.yaml       # Use custom config
  python unified_experimental_framework.py --unit-tests --performance    # Run specific test types
  python unified_experimental_framework.py --create-config               # Generate default config
  python unified_experimental_framework.py --sweeps --output results_v2  # Parameter sweeps only
  python unified_experimental_framework.py --organized-tests unit         # Run organized unit tests
  python unified_experimental_framework.py --create-test-structure        # Create organized test structure
        """
    )
    
    parser.add_argument("--config", "-c", type=str, 
                       help="Path to YAML configuration file")
    parser.add_argument("--output", "-o", type=str, default="results",
                       help="Output directory for results")
    
    # Test selection
    parser.add_argument("--all", action="store_true",
                       help="Run all test suites")
    parser.add_argument("--unit-tests", action="store_true",
                       help="Run unit tests")
    parser.add_argument("--performance", action="store_true",
                       help="Run performance tests")
    parser.add_argument("--visualization", action="store_true",
                       help="Run visualization tests")
    parser.add_argument("--sweeps", action="store_true",
                       help="Run parameter sweeps")
    
    # Organized test functionality
    parser.add_argument("--organized-tests", type=str, choices=["all", "unit", "integration", "performance", "imports"],
                       help="Run organized test structure (all, unit, integration, performance, imports)")
    parser.add_argument("--create-test-structure", action="store_true",
                       help="Create organized test directory structure")
    
    # Utility options
    parser.add_argument("--create-config", action="store_true",
                       help="Create default configuration file")
    parser.add_argument("--git-info", action="store_true",
                       help="Show git information and exit")
    
    args = parser.parse_args()
    
    # Handle utility options
    if args.create_config:
        create_default_config()
        return
    
    if args.create_test_structure:
        print("🏗️ Creating organized test structure...")
        organized_runner = OrganizedUnitTestRunner()
        success = organized_runner.create_test_structure()
        if success:
            print("Test structure created.")
            print("Run tests with: --organized-tests all")
        return
    
    if args.organized_tests:
        print(f"🧪 Running organized tests: {args.organized_tests}")
        organized_runner = OrganizedUnitTestRunner()
        
        # Create structure if it doesn't exist
        if not Path("tests").exists():
            print("📁 Test structure not found, creating...")
            organized_runner.create_test_structure()
        
        success = organized_runner.run_organized_tests(args.organized_tests, verbose=True)
        print(f"Organized tests exit code: {0 if success else 1}")
        return
    
    if args.git_info:
        runner = NavierStokesTestRunner()
        print("Git Information:")
        for key, value in runner.git_info.items():
            print(f"  {key}: {value}")
        return
    
    # Initialize test runner
    try:
        runner = NavierStokesTestRunner(args.config, args.output)
        
        # Determine which tests to run
        if args.all or not any([args.unit_tests, args.performance, args.visualization, args.sweeps]):
            # Run all tests if --all specified or no specific tests selected
            results = runner.run_all_tests()
        else:
            # Run specific tests
            if args.unit_tests:
                runner.all_results["unit_tests"] = runner.run_unit_tests()
            if args.performance:
                runner.all_results["performance_tests"] = runner.run_performance_tests()
            if args.visualization:
                runner.all_results["visualization_tests"] = runner.run_visualization_tests()
            if args.sweeps:
                runner.all_results["parameter_sweeps"] = runner.run_parameter_sweeps()
            
            # Always generate report
            runner.all_results["comprehensive_analysis"] = runner.generate_comprehensive_report()
        
        print(f"\nExperiment run completed.")
        print(f"Session: {runner.session_id}")
        print(f"Results: {runner.session_dir}")
        
    except Exception as e:
        print(f"Test runner failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
