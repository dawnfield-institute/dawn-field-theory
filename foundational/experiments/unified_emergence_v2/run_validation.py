#!/usr/bin/env python3
"""
Command-line interface for the Unified Emergence Framework v2.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from unified_emergence_v2 import UnifiedEmergenceFramework


def main():
    parser = argparse.ArgumentParser(
        description='Unified Emergence Framework v2 - Clean architecture emergence analysis'
    )
    
    parser.add_argument(
        '--domains',
        nargs='+',
        default=['gravity', 'med'],
        choices=['gravity', 'med', 'navier', 'tinycimm', 'hodge'],
        help='Domains to analyze (default: gravity med)'
    )
    
    parser.add_argument(
        '--field-sizes',
        nargs='+',
        type=int,
        default=[32],
        help='Field sizes to test (default: 32)'
    )
    
    parser.add_argument(
        '--runs',
        type=int,
        default=1,
        help='Number of runs per domain (default: 1)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        default=True,
        help='Run domains in parallel (default: True)'
    )
    
    parser.add_argument(
        '--sequential',
        dest='parallel',
        action='store_false',
        help='Run domains sequentially'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Timeout in seconds (default: 300)'
    )
    
    parser.add_argument(
        '--session-id',
        type=str,
        help='Custom session ID'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test with minimal configuration'
    )
    
    parser.add_argument(
        '--config-file',
        type=str,
        help='Load configuration from JSON file'
    )
    
    parser.add_argument(
        '--save-raw',
        action='store_true',
        help='Save raw domain results'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output'
    )
    
    # Parameter sweep arguments
    parser.add_argument(
        '--param-sweep',
        action='store_true',
        help='Run comprehensive parameter sweep analysis'
    )
    
    parser.add_argument(
        '--sweep-field-sizes',
        nargs='+',
        type=int,
        default=[16, 32, 64, 128],
        help='Field sizes for parameter sweep (default: 16 32 64 128)'
    )
    
    parser.add_argument(
        '--sweep-runs',
        type=int,
        default=5,
        help='Number of runs per parameter configuration (default: 5)'
    )
    
    parser.add_argument(
        '--statistical-analysis',
        action='store_true',
        help='Enable comprehensive statistical analysis'
    )
    
    parser.add_argument(
        '--convergence-analysis',
        action='store_true',
        help='Enable convergence analysis'
    )
    
    parser.add_argument(
        '--confidence-level',
        type=float,
        default=0.95,
        help='Statistical confidence level (default: 0.95)'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    import logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    # Build configuration
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override with command line arguments
    if args.quick_test:
        config.update({
            'domains': ['gravity', 'med'],
            'field_sizes': [32],
            'runs_per_domain': 1,
            'timeout_seconds': 120
        })
    elif args.param_sweep:
        config.update({
            'domains': args.domains,
            'field_sizes': args.field_sizes,  # Base field sizes
            'runs_per_domain': 1,  # Individual runs are handled by sweep
            'parallel_execution': args.parallel,
            'timeout_seconds': args.timeout,
            'save_raw_domain_results': args.save_raw,
            'enable_parameter_sweep': True,
            'sweep_field_sizes': args.sweep_field_sizes,
            'sweep_runs_per_config': args.sweep_runs,
            'enable_statistical_analysis': args.statistical_analysis or True,  # Default to True for sweeps
            'enable_convergence_analysis': args.convergence_analysis,
            'statistical_confidence_level': args.confidence_level,
            'outlier_threshold': 2.0,
            'bootstrap_samples': 1000
        })
    else:
        config.update({
            'domains': args.domains,
            'field_sizes': args.field_sizes,
            'runs_per_domain': args.runs,
            'parallel_execution': args.parallel,
            'timeout_seconds': args.timeout,
            'save_raw_domain_results': args.save_raw,
            'enable_statistical_analysis': args.statistical_analysis,
            'enable_convergence_analysis': args.convergence_analysis,
            'statistical_confidence_level': args.confidence_level
        })
    
    if args.session_id:
        config['session_id'] = args.session_id
    
    if args.output_dir:
        config['output_directory'] = args.output_dir
    
    # Validate configuration
    framework = UnifiedEmergenceFramework()
    
    config_errors = framework.validate_config(config)
    if config_errors:
        print("Configuration errors:")
        for error in config_errors:
            print(f"  - {error}")
        return 1
    
    # Run validation
    if args.param_sweep:
        print("Starting Unified Emergence Framework v2 PARAMETER SWEEP...")
        print(f"Domains: {config['domains']}")
        print(f"Sweep field sizes: {config['sweep_field_sizes']}")
        print(f"Runs per configuration: {config['sweep_runs_per_config']}")
        print(f"Total configurations: {len(config['domains']) * len(config['sweep_field_sizes']) * config['sweep_runs_per_config']}")
        print(f"Statistical analysis: {config['enable_statistical_analysis']}")
        print(f"Confidence level: {config['statistical_confidence_level']}")
        print()
        
        try:
            sweep_analysis = framework.run_parameter_sweep(config)
            
            # Print sweep results summary
            print("=" * 80)
            print("PARAMETER SWEEP ANALYSIS RESULTS")
            print("=" * 80)
            print(f"Total Runs: {sweep_analysis.total_runs}")
            print(f"Successful Runs: {sweep_analysis.successful_runs}")
            print(f"Phase 1 Success Rate: {sweep_analysis.phase1_success_rate:.1%}")
            print()
            
            print("STATISTICAL SUMMARY:")
            print(f"  Overall Score: {sweep_analysis.overall_score_stats.mean:.3f} ± {sweep_analysis.overall_score_stats.std_dev:.3f}")
            print(f"  SEC Classification: {sweep_analysis.sec_classification_stats.mean:.3f} ± {sweep_analysis.sec_classification_stats.std_dev:.3f}")
            print(f"  Pattern Assembly: {sweep_analysis.pattern_assembly_stats.mean:.3f} ± {sweep_analysis.pattern_assembly_stats.std_dev:.3f}")
            print(f"  Emergence Consistency: {sweep_analysis.emergence_consistency_stats.mean:.3f} ± {sweep_analysis.emergence_consistency_stats.std_dev:.3f}")
            print(f"  Phase 1 Readiness: {sweep_analysis.phase1_readiness_stats.mean:.3f} ± {sweep_analysis.phase1_readiness_stats.std_dev:.3f}")
            print()
            
            print("OPTIMAL PARAMETERS:")
            opt = sweep_analysis.optimal_parameters
            print(f"  Best Configuration: Field Size {opt.get('best_field_size')}, Domain {opt.get('best_domain')}")
            print(f"  Best Overall Score: {opt.get('best_overall_score', 0):.3f}")
            print(f"  Most Reliable Domain: {opt.get('most_reliable_domain')}")
            print(f"  Most Reliable Field Size: {opt.get('most_reliable_field_size')}")
            print()
            
            if sweep_analysis.parameter_correlations:
                print("PARAMETER CORRELATIONS:")
                for param, correlation in sweep_analysis.parameter_correlations.items():
                    print(f"  {param}: {correlation:.3f}")
                print()
            
            if sweep_analysis.phase1_success_rate >= 0.8:
                print("🎉 EXCELLENT: High Phase 1 success rate! Framework is performing very well.")
            elif sweep_analysis.phase1_success_rate >= 0.5:
                print("✅ GOOD: Moderate Phase 1 success rate. Consider optimization.")
            else:
                print("⚠️  NEEDS WORK: Low Phase 1 success rate. Requires significant improvement.")
            
        except Exception as e:
            print(f"Parameter sweep failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
            
    else:
        print("Starting Unified Emergence Framework v2 validation...")
        print(f"Domains: {config['domains']}")
        print(f"Field sizes: {config['field_sizes']}")
        print(f"Runs per domain: {config['runs_per_domain']}")
        print(f"Parallel execution: {config.get('parallel_execution', True)}")
        print()
        
        try:
            results = framework.run_phase1_validation(config)
            
            # Print results summary
            print("=" * 60)
            print("VALIDATION RESULTS")
            print("=" * 60)
            print(f"Session ID: {results.session_id}")
            print(f"Success: {results.success}")
            print(f"Execution time: {results.execution_time_seconds:.2f}s")
            print(f"Peak memory: {results.peak_memory_usage_mb:.1f} MB")
            print()
            
            if results.success:
                metrics = results.metrics
                print("CORE METRICS:")
                print(f"  SEC Classification Accuracy: {metrics.sec_classification_accuracy:.3f}")
                print(f"  Pattern Assembly Success Rate: {metrics.pattern_assembly_success_rate:.3f}")
                print(f"  Emergence Consistency Score: {metrics.emergence_consistency_score:.3f}")
                print(f"  Phase 1 Readiness Score: {metrics.phase1_readiness_score:.3f}")
                print(f"  Overall Score: {metrics.get_overall_score():.3f}")
                print()
                
                print("PATTERN STATISTICS:")
                print(f"  Total patterns extracted: {metrics.total_patterns_extracted}")
                print(f"  Average confidence: {metrics.average_pattern_confidence:.3f}")
                print(f"  Average emergence strength: {metrics.average_emergence_strength:.3f}")
                print(f"  Pattern diversity: {metrics.pattern_diversity_score:.3f}")
                print()
                
                print("CROSS-DOMAIN ANALYSIS:")
                print(f"  Cross-domain correlations: {metrics.cross_domain_correlations:.3f}")
                print(f"  Correlation consistency: {metrics.correlation_consistency:.3f}")
                print()
                
                print("PATTERNS PER DOMAIN:")
                for domain, count in metrics.patterns_per_domain.items():
                    print(f"  {domain}: {count}")
                print()
                
                print(f"PHASE 1 READY: {results.is_phase1_ready()}")
                
                if results.is_phase1_ready():
                    print("🎉 Framework is ready for Phase 1 deployment!")
                else:
                    print("⚠️  Framework needs improvement before Phase 1 deployment")
                    
            else:
                print("ERRORS:")
                for error in results.error_messages:
                    print(f"  - {error}")
            
            if results.warnings:
                print("\nWARNINGS:")
                for warning in results.warnings:
                    print(f"  - {warning}")
            
            return 0 if results.success else 1
            
        except KeyboardInterrupt:
            print("\nValidation interrupted by user")
            return 1
        except Exception as e:
            print(f"Validation failed with error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1


if __name__ == '__main__':
    sys.exit(main())
