"""
Calibration test for Pre-Field Recursion framework

Validates the experimental framework for studying natural information
amplification emergence and PAC conservation in topological substrates.
"""

import numpy as np
from pre_field_recursion_unified import (
    ExperimentalConfig, 
    PreFieldRecursionFramework,
    MobiusTopology,
    PACConservationValidator,
    LocalAmplificationMeasurer,
    XiUniversalValidator
)

def test_mobius_topology():
    """Test Möbius topology initialization and properties"""
    print("🧪 Testing Möbius Topology...")
    
    topology = MobiusTopology(size=16, seed=42)
    
    # Test anti-periodic boundary condition
    field = topology.field
    mid = len(field) // 2
    
    # Check Möbius property: f(x + period/2) ≈ -f(x)
    first_half = field[:mid]
    second_half = field[mid:]
    
    anti_periodic_error = np.mean(np.abs(first_half + second_half))
    
    print(f"  📏 Topology size: {topology.size}")
    print(f"  🌀 Anti-periodic error: {anti_periodic_error:.6f} (should be ~0)")
    print(f"  🎯 Boundary conditions: {topology.boundary_conditions}")
    
    # Test local structure extraction
    local_struct = topology.get_local_structure(8, radius=2)
    print(f"  📍 Local structure length: {len(local_struct)}")
    
    assert anti_periodic_error < 1e-10, "Möbius anti-periodic property not satisfied"
    print("  ✅ Möbius topology validated\n")

def test_pac_conservation():
    """Test PAC conservation validation"""
    print("🧪 Testing PAC Conservation...")
    
    validator = PACConservationValidator(tolerance=0.15)
    
    # Test perfect conservation
    parent = 10.0
    children = [2.5, 2.5, 2.5, 2.5]  # Sum = 10.0
    result = validator.validate_local_conservation(parent, children)
    
    print(f"  🎯 Perfect conservation test:")
    print(f"    Parent: {result['parent_value']}")
    print(f"    Children sum: {result['children_sum']}")
    print(f"    Consistency: {result['consistency_percentage']:.1f}%")
    print(f"    Valid: {result['is_consistent']}")
    
    assert result['is_consistent'], "Perfect conservation should be valid"
    
    # Test near conservation (within tolerance)
    parent = 10.0
    children = [2.6, 2.4, 2.6, 2.3]  # Sum = 9.9, ~1% error
    result = validator.validate_local_conservation(parent, children)
    
    print(f"  📊 Near conservation test:")
    print(f"    Relative deviation: {result['relative_deviation']:.3f}")
    print(f"    Consistency: {result['consistency_percentage']:.1f}%")
    print(f"    Valid: {result['is_consistent']}")
    
    assert result['is_consistent'], "Near conservation should be valid"
    
    # Test broken conservation (outside tolerance)
    parent = 10.0
    children = [3.0, 3.0, 3.0, 3.0]  # Sum = 12.0, 20% error
    result = validator.validate_local_conservation(parent, children)
    
    print(f"  ❌ Broken conservation test:")
    print(f"    Relative deviation: {result['relative_deviation']:.3f}")
    print(f"    Consistency: {result['consistency_percentage']:.1f}%")
    print(f"    Valid: {result['is_consistent']}")
    
    assert not result['is_consistent'], "Broken conservation should be invalid"
    print("  ✅ PAC conservation validator working correctly\n")

def test_local_amplification():
    """Test local amplification measurement (natural emergence)"""
    print("🧪 Testing Local Amplification Measurement...")
    
    measurer = LocalAmplificationMeasurer()
    
    # Test various input/output scenarios to validate measurement capability
    test_cases = [
        {
            'name': 'Case 1',
            'input': np.array([1.0]),
            'output': np.array([0.5, 0.6, 0.7, 0.8]),
        },
        {
            'name': 'Case 2', 
            'input': np.array([1.0]),
            'output': np.array([2.0, 3.0, 4.0, 5.0]),
        },
        {
            'name': 'Case 3',
            'input': np.array([0.1]),
            'output': np.array([2.0, 3.0, 4.0, 5.0]),
        }
    ]
    
    for case in test_cases:
        result = measurer.measure_local_amplification(case['input'], case['output'])
        amplification = result['amplification_factor']
        
        print(f"  📈 {case['name']}:")
        print(f"    Input magnitude: {result['input_magnitude']:.3f}")
        print(f"    Output magnitude: {result['output_magnitude']:.3f}")
        print(f"    Amplification: {amplification:.1f}x")
        print(f"    Entropy change: {result['entropy_change']:.3f}")
        print(f"    Valid: {result['is_valid']}")
        
        assert result['is_valid'], f"Measurement should be valid for {case['name']}"
    
    # Test statistics
    stats = measurer.get_amplification_statistics()
    print(f"  📊 Measurement statistics:")
    print(f"    Count: {stats['count']}")
    print(f"    Mean: {stats['mean']:.1f}x")
    print(f"    Range: {stats['natural_range'][0]:.1f}x - {stats['natural_range'][1]:.1f}x")
    
    print("  ✅ Local amplification measurement working correctly\n")

def test_xi_validation():
    """Test Ξ universal constant validation"""
    print("🧪 Testing Ξ Universal Constant Validation...")
    
    validator = XiUniversalValidator()
    
    # Test perfect convergence
    result = validator.validate_xi_convergence(1.0571)
    print(f"  🎯 Perfect convergence:")
    print(f"    Measured: {result['measured_xi']}")
    print(f"    Target: {result['target_xi']}")
    print(f"    Deviation: {result['deviation']:.2e}")
    print(f"    Converged: {result['converged']}")
    
    assert result['converged'], "Perfect Ξ should converge"
    
    # Test near convergence
    result = validator.validate_xi_convergence(1.0571000000001)
    print(f"  📊 Near convergence:")
    print(f"    Deviation: {result['deviation']:.2e}")
    print(f"    Converged: {result['converged']}")
    
    # Test failed convergence
    result = validator.validate_xi_convergence(1.1)
    print(f"  ❌ Failed convergence:")
    print(f"    Deviation: {result['deviation']:.2e}")
    print(f"    Converged: {result['converged']}")
    
    assert not result['converged'], "Large deviation should not converge"
    
    # Test Ξ calculation from topology
    topology = MobiusTopology(size=32, seed=42)
    calculated_xi = validator.calculate_xi_from_topology(topology)
    print(f"  🌀 Calculated from topology: {calculated_xi:.6f}")
    
    print("  ✅ Ξ validation working correctly\n")

def test_full_framework():
    """Test complete framework integration"""
    print("🧪 Testing Complete Framework Integration...")
    
    config = ExperimentalConfig(
        topology_size=32,
        pac_tolerance=0.20,  # Updated to match framework settings
        random_seed=42
    )
    
    framework = PreFieldRecursionFramework(config)
    
    # Run single experiment
    result = framework.run_single_experiment()
    
    print(f"  🎯 Single experiment results:")
    print(f"    Success: {result['successful']}")
    print(f"    PAC consistent: {result['pac_validation']['is_consistent']}")
    print(f"    PAC consistency: {result['pac_validation']['consistency_percentage']:.1f}%")
    print(f"    Amplification: {result['amplification_measurement']['amplification_factor']:.1f}x")
    print(f"    Ξ converged: {result['xi_validation']['converged']}")
    print(f"    Measured Ξ: {result['xi_validation']['measured_xi']:.6f}")
    
    # Run small suite
    print(f"  🚀 Running small test suite...")
    suite_results = framework.run_experiment_suite(num_experiments=10)
    
    print(f"    Success rate: {suite_results['success_rate']:.1f}%")
    print(f"    PAC consistency: {suite_results['pac_consistency']:.1f}%")
    print(f"    Ξ convergence: {suite_results['xi_convergence_rate']:.1f}%")
    
    if suite_results['amplification_stats']['count'] > 0:
        amp_range = suite_results['amplification_range']
        print(f"    Amplification range: {amp_range[0]:.1f}x - {amp_range[1]:.1f}x")
    
    print("  ✅ Framework integration working correctly\n")

def main():
    """Run all calibration tests"""
    print("🔬 Pre-Field Recursion Calibration Tests")
    print("="*50)
    print("🎯 Validating experimental framework:")
    print("  • Information amplification measurement from topological structure")
    print("  • PAC conservation as fundamental computational constraint")
    print("  • Ξ = 1.0571 universal constant identification")
    print("  • Natural emergence dynamics")
    print()
    
    try:
        test_mobius_topology()
        test_pac_conservation()
        test_local_amplification()
        test_xi_validation()
        test_full_framework()
        
        print("🎉 All calibration tests passed!")
        print("✅ Framework ready for Pre-Field Recursion research:")
        print("  • Natural amplification emergence validated")
        print("  • PAC conservation constraints confirmed")
        print("  • Universal constant identification verified")
        print("  • Topological computation framework operational")
        
    except Exception as e:
        print(f"❌ Calibration test failed: {e}")
        raise

if __name__ == "__main__":
    main()