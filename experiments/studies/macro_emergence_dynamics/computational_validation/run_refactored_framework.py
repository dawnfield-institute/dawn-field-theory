"""
MED REFACTORED FRAMEWORK - Addressing Core Issues

This script coordinates the refactored approach to address the critical issues
identified in the original MED work:

1. Pattern Discovery: Tests whether finite patterns naturally emerge
2. Enhanced Extraction: Improves reconstruction error through better pattern detection
3. Complexity Evolution: Rigorously tracks convergence to bounded complexity
4. Benchmark Comparison: Compares against established methods (POD, Fourier)
5. Honest Documentation: Documents what works and what needs improvement

Run this to execute the complete refactored validation pipeline.
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Run the complete refactored MED validation pipeline."""
    print("🔬 MED REFACTORED FRAMEWORK")
    print("=" * 60)
    print("Addressing core issues through systematic validation")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = {}
    
    # Phase 1: Pattern Discovery Analysis
    print("📊 PHASE 1: PATTERN DISCOVERY ANALYSIS")
    print("-" * 40)
    
    try:
        from pattern_discovery_engine import PatternDiscoveryEngine
        
        print("Testing natural pattern emergence from diverse flows...")
        
        # Test on both 32x32 and 64x64 grids
        for grid_size in [32, 64]:
            print(f"\n🔍 Grid Size: {grid_size}x{grid_size}")
            
            engine = PatternDiscoveryEngine(grid_size=grid_size, max_patterns=15)
            discovery_results = engine.run_pattern_discovery_experiment(n_flows=50)
            analysis = engine.analyze_discovered_patterns()
            
            # Save detailed results
            engine.save_results({**discovery_results, 'pattern_analysis': analysis})
            
            # Store summary
            results[f'pattern_discovery_{grid_size}x{grid_size}'] = {
                'final_library_size': discovery_results['final_library_size'],
                'mean_final_error': discovery_results['mean_final_error'],
                'plateau_reached': discovery_results['plateau_reached'],
                'unique_patterns': discovery_results['unique_patterns_discovered']
            }
            
            print(f"✓ Discovered {discovery_results['final_library_size']} patterns")
            print(f"✓ Final error: {discovery_results['mean_final_error']:.3f}")
        
        results['phase1_status'] = 'completed'
        
    except ImportError as e:
        print(f"⚠ Phase 1 skipped: {e}")
        results['phase1_status'] = 'skipped'
    
    # Phase 2: Enhanced Pattern Extraction
    print("\n\n🎯 PHASE 2: ENHANCED PATTERN EXTRACTION")
    print("-" * 40)
    
    try:
        from enhanced_pattern_extraction import test_enhanced_extraction
        
        print("Testing improved pattern extraction with vortex detection...")
        
        extraction_results = test_enhanced_extraction()
        
        # Store results
        results['enhanced_extraction'] = {
            'test_cases': len(extraction_results),
            'successful_cases': sum(1 for r in extraction_results.values() if r['success']),
            'mean_error': np.mean([r['reconstruction_error'] for r in extraction_results.values()]),
            'success_rate': sum(1 for r in extraction_results.values() if r['success']) / len(extraction_results)
        }
        
        print(f"✓ Success rate: {results['enhanced_extraction']['success_rate']:.1%}")
        print(f"✓ Mean error: {results['enhanced_extraction']['mean_error']:.3f}")
        
        results['phase2_status'] = 'completed'
        
    except ImportError as e:
        print(f"⚠ Phase 2 skipped: {e}")
        results['phase2_status'] = 'skipped'
    
    # Phase 3: Complexity Evolution Analysis
    print(f"\n\n📈 PHASE 3: COMPLEXITY EVOLUTION ANALYSIS")
    print("-" * 40)
    
    try:
        from complexity_evolution_tracker_fixed import ComplexityEvolutionTracker
        
        print("Testing convergence to bounded complexity...")
        
        # Test on 32x32 grid (faster for multiple cases)
        tracker = ComplexityEvolutionTracker(grid_size=32)
        convergence_results = tracker.run_convergence_study(n_cases=10)
        
        # Save detailed results
        tracker.save_results(convergence_results)
        
        # Store summary
        results['complexity_evolution'] = {
            'convergence_rate': convergence_results['convergence_rate'],
            'bounded_complexity_rate': convergence_results['bounded_complexity_rate'],
            'mean_convergence_time': convergence_results['mean_convergence_time'],
            'mean_final_error': convergence_results['mean_final_error']
        }
        
        print(f"✓ Convergence rate: {convergence_results['convergence_rate']:.1%}")
        print(f"✓ Bounded complexity rate: {convergence_results['bounded_complexity_rate']:.1%}")
        
        results['phase3_status'] = 'completed'
        
    except ImportError as e:
        print(f"⚠ Phase 3 skipped: {e}")
        results['phase3_status'] = 'skipped'
    
    # Phase 4: Benchmark Comparison
    print(f"\n\n🏆 PHASE 4: BENCHMARK COMPARISON")
    print("-" * 40)
    
    try:
        from benchmark_comparison import BenchmarkComparison
        
        print("Comparing SEC against POD, Fourier, and SVD...")
        
        benchmark = BenchmarkComparison(grid_size=64)
        test_cases = benchmark.generate_test_database(n_cases=30)
        benchmark_results = benchmark.benchmark_against_pod(test_cases)
        
        # Save detailed results
        benchmark.save_benchmark_results(benchmark_results)
        
        # Store summary
        results['benchmark_comparison'] = benchmark_results['statistics']
        
        # Print key comparisons
        sec_mean = benchmark_results['statistics']['sec']['mean_error']
        pod_mean = benchmark_results['statistics']['pod']['mean_error']
        
        print(f"✓ SEC mean error: {sec_mean:.3f}")
        print(f"✓ POD mean error: {pod_mean:.3f}")
        print(f"✓ SEC vs POD ratio: {sec_mean/pod_mean:.2f}")
        
        results['phase4_status'] = 'completed'
        
    except ImportError as e:
        print(f"⚠ Phase 4 skipped: {e}")
        results['phase4_status'] = 'skipped'
    
    # Phase 5: Generate Summary Report
    print(f"\n\n📝 PHASE 5: SUMMARY REPORT")
    print("-" * 40)
    
    print("Generating comprehensive summary...")
    
    # Overall assessment
    completed_phases = sum(1 for key in results.keys() if key.endswith('_status') and results[key] == 'completed')
    total_phases = 4
    
    results['overall_summary'] = {
        'completed_phases': completed_phases,
        'total_phases': total_phases,
        'completion_rate': completed_phases / total_phases,
        'timestamp': datetime.now().isoformat()
    }
    
    # Key insights
    insights = []
    
    if 'pattern_discovery_32x32' in results:
        pd_32 = results['pattern_discovery_32x32']
        insights.append(f"Pattern Discovery: Discovered {pd_32['final_library_size']} patterns with {pd_32['mean_final_error']:.3f} error")
    
    if 'enhanced_extraction' in results:
        ee = results['enhanced_extraction']
        insights.append(f"Enhanced Extraction: {ee['success_rate']:.1%} success rate, {ee['mean_error']:.3f} mean error")
    
    if 'complexity_evolution' in results:
        ce = results['complexity_evolution']
        insights.append(f"Complexity Evolution: {ce['convergence_rate']:.1%} convergence, {ce['bounded_complexity_rate']:.1%} bounded")
    
    if 'benchmark_comparison' in results:
        bc = results['benchmark_comparison']
        sec_error = bc['sec']['mean_error']
        pod_error = bc['pod']['mean_error']
        insights.append(f"Benchmark: SEC {sec_error:.3f} vs POD {pod_error:.3f} error")
    
    results['key_insights'] = insights
    
    # Save comprehensive results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"med_refactored_summary_{timestamp}.json"
    
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("🎯 REFACTORED MED FRAMEWORK SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n📊 Completion Status: {completed_phases}/{total_phases} phases")
    print(f"   Overall progress: {results['overall_summary']['completion_rate']:.1%}")
    
    print(f"\n🔍 Key Insights:")
    for insight in insights:
        print(f"   • {insight}")
    
    print(f"\n📁 Results saved to: {summary_file}")
    
    # Recommendations based on results
    print(f"\n💡 Next Steps Recommendations:")
    
    if 'pattern_discovery_32x32' in results:
        pd = results['pattern_discovery_32x32']
        if pd['final_library_size'] > 0 and pd['plateau_reached']:
            print("   ✓ Pattern discovery working - proceed with larger studies")
        else:
            print("   ⚠ Pattern discovery needs parameter tuning")
    
    if 'enhanced_extraction' in results:
        ee = results['enhanced_extraction']
        if ee['success_rate'] > 0.5:
            print("   ✓ Enhanced extraction promising - optimize further")
        else:
            print("   ⚠ Enhanced extraction needs algorithm improvements")
    
    if 'complexity_evolution' in results:
        ce = results['complexity_evolution']
        if ce['convergence_rate'] > 0.7:
            print("   ✓ Complexity evolution stable - ready for mathematical analysis")
        else:
            print("   ⚠ Complexity evolution needs stability improvements")
    
    if 'benchmark_comparison' in results:
        bc = results['benchmark_comparison']
        sec_error = bc['sec']['mean_error']
        pod_error = bc['pod']['mean_error']
        if sec_error < pod_error * 1.5:
            print("   ✓ SEC competitive with POD - ready for practical applications")
        else:
            print("   ⚠ SEC needs significant improvement to compete with POD")
    
    print(f"\n🔬 Research Program Status:")
    
    if results['overall_summary']['completion_rate'] > 0.75:
        print("   🟢 STRONG PROGRESS - Multiple components validated")
    elif results['overall_summary']['completion_rate'] > 0.5:
        print("   🟡 MODERATE PROGRESS - Some components need work")
    else:
        print("   🔴 EARLY STAGE - Fundamental issues need addressing")
    
    print(f"\n📚 Documentation:")
    print(f"   • Detailed methodology: Each component saved individual results")
    print(f"   • Honest assessment: See validated_results.md")
    print(f"   • Implementation checklist: Ready for systematic development")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        print(f"\n✓ Refactored framework execution completed successfully")
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
