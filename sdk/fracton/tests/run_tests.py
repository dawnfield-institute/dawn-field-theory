"""
Fracton Test Runner

Comprehensive test execution for the Fracton computational modeling language.
Includes foundational theory compliance validation and performance benchmarking.
"""

import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import argparse


class FractonTestRunner:
    """Test runner for Fracton with foundational theory validation."""
    
    def __init__(self, fracton_root: Path):
        self.fracton_root = fracton_root
        self.test_dir = fracton_root / "tests"
        self.results = {}
        
    def run_all_tests(self, verbose: bool = True, coverage: bool = False) -> Dict[str, Any]:
        """Run complete test suite."""
        print("🧬 Fracton Test Suite - Foundational Theory Validation")
        print("=" * 60)
        
        start_time = time.time()
        
        # Core component tests
        self.results["core"] = self._run_core_tests(verbose, coverage)
        
        # Language tests
        self.results["language"] = self._run_language_tests(verbose, coverage)
        
        # Foundational theory compliance
        self.results["theory"] = self._run_theory_compliance_tests(verbose, coverage)
        
        # Integration tests
        self.results["integration"] = self._run_integration_tests(verbose, coverage)
        
        # Performance tests
        self.results["performance"] = self._run_performance_tests(verbose, coverage)
        
        total_time = time.time() - start_time
        
        # Generate summary
        self._print_summary(total_time)
        
        return self.results
    
    def run_theory_only(self, verbose: bool = True) -> Dict[str, Any]:
        """Run only foundational theory compliance tests."""
        print("🌌 Fracton Foundational Theory Compliance Tests")
        print("=" * 50)
        
        theory_results = self._run_theory_compliance_tests(verbose, False)
        self._print_theory_summary(theory_results)
        
        return theory_results
    
    def run_performance_only(self, verbose: bool = True) -> Dict[str, Any]:
        """Run only performance benchmarks."""
        print("⚡ Fracton Performance Benchmarks")
        print("=" * 35)
        
        perf_results = self._run_performance_tests(verbose, False)
        self._print_performance_summary(perf_results)
        
        return perf_results
    
    def _run_core_tests(self, verbose: bool, coverage: bool) -> Dict[str, Any]:
        """Run core component tests."""
        print("\n🔧 Core Component Tests")
        print("-" * 25)
        
        core_tests = [
            "test_recursive_engine.py",
            "test_memory_field.py", 
            "test_entropy_dispatch.py",
            "test_bifractal_trace.py"
        ]
        
        results = {}
        for test_file in core_tests:
            print(f"  Running {test_file}...")
            result = self._run_pytest(test_file, verbose, coverage)
            results[test_file] = result
            
            if result["passed"]:
                print(f"    ✅ {result['test_count']} tests passed")
            else:
                print(f"    ❌ {result['failures']} failures, {result['errors']} errors")
        
        return results
    
    def _run_language_tests(self, verbose: bool, coverage: bool) -> Dict[str, Any]:
        """Run language construct tests."""
        print("\n📝 Language Construct Tests")
        print("-" * 28)
        
        language_tests = [
            "test_decorators.py",
            "test_primitives.py",
            "test_context.py",
            "test_compiler.py"
        ]
        
        results = {}
        for test_file in language_tests:
            test_path = self.test_dir / test_file
            if test_path.exists():
                print(f"  Running {test_file}...")
                result = self._run_pytest(test_file, verbose, coverage)
                results[test_file] = result
                
                if result["passed"]:
                    print(f"    ✅ {result['test_count']} tests passed")
                else:
                    print(f"    ❌ {result['failures']} failures, {result['errors']} errors")
            else:
                print(f"  ⏭️  {test_file} (not implemented yet)")
                results[test_file] = {"skipped": True}
        
        return results
    
    def _run_theory_compliance_tests(self, verbose: bool, coverage: bool) -> Dict[str, Any]:
        """Run foundational theory compliance tests."""
        print("\n🌌 Foundational Theory Compliance")
        print("-" * 35)
        
        theory_tests = [
            ("test_sec_compliance.py", "SEC (Symbolic Entropy Collapse)"),
            ("test_med_compliance.py", "MED (Macro Emergence Dynamics)")
        ]
        
        results = {}
        for test_file, description in theory_tests:
            print(f"  Testing {description}...")
            result = self._run_pytest(test_file, verbose, coverage, markers=["foundational_theory"])
            results[test_file] = result
            
            if result["passed"]:
                print(f"    ✅ {description} compliance verified")
                print(f"       {result['test_count']} theoretical validations passed")
            else:
                print(f"    ❌ {description} compliance FAILED")
                print(f"       {result['failures']} failures, {result['errors']} errors")
        
        return results
    
    def _run_integration_tests(self, verbose: bool, coverage: bool) -> Dict[str, Any]:
        """Run integration tests."""
        print("\n🔗 Integration Tests")
        print("-" * 20)
        
        integration_tests = [
            "test_api.py",
            "test_examples.py",
            "test_field_dynamics.py"
        ]
        
        results = {}
        for test_file in integration_tests:
            test_path = self.test_dir / test_file
            if test_path.exists():
                print(f"  Running {test_file}...")
                result = self._run_pytest(test_file, verbose, coverage, markers=["integration"])
                results[test_file] = result
                
                if result["passed"]:
                    print(f"    ✅ {result['test_count']} integration tests passed")
                else:
                    print(f"    ❌ {result['failures']} failures, {result['errors']} errors")
            else:
                print(f"  ⏭️  {test_file} (not implemented yet)")
                results[test_file] = {"skipped": True}
        
        return results
    
    def _run_performance_tests(self, verbose: bool, coverage: bool) -> Dict[str, Any]:
        """Run performance benchmarks."""
        print("\n⚡ Performance Benchmarks")
        print("-" * 25)
        
        performance_tests = [
            "test_performance.py",
            "test_memory_limits.py",
            "test_concurrency.py"
        ]
        
        results = {}
        for test_file in performance_tests:
            test_path = self.test_dir / test_file
            if test_path.exists():
                print(f"  Running {test_file}...")
                result = self._run_pytest(test_file, verbose, coverage, markers=["performance"])
                results[test_file] = result
                
                if result["passed"]:
                    print(f"    ✅ {result['test_count']} benchmarks passed")
                else:
                    print(f"    ❌ {result['failures']} performance issues detected")
            else:
                print(f"  ⏭️  {test_file} (not implemented yet)")
                results[test_file] = {"skipped": True}
        
        return results
    
    def _run_pytest(self, test_file: str, verbose: bool, coverage: bool, 
                   markers: List[str] = None) -> Dict[str, Any]:
        """Run pytest for a specific test file."""
        
        cmd = ["python", "-m", "pytest", str(self.test_dir / test_file)]
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend(["--cov=fracton", "--cov-report=term-missing"])
        
        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.fracton_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Parse pytest output
            output_lines = result.stdout.split('\n')
            
            test_count = 0
            failures = 0
            errors = 0
            passed = result.returncode == 0
            
            # Simple parsing of pytest output
            for line in output_lines:
                if " passed" in line and " failed" not in line:
                    try:
                        test_count = int(line.split()[0])
                    except (ValueError, IndexError):
                        pass
                elif " failed" in line:
                    try:
                        failures = int(line.split()[0])
                    except (ValueError, IndexError):
                        pass
                elif " error" in line:
                    try:
                        errors = int(line.split()[0])
                    except (ValueError, IndexError):
                        pass
            
            return {
                "passed": passed,
                "test_count": test_count,
                "failures": failures,
                "errors": errors,
                "output": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "test_count": 0,
                "failures": 0,
                "errors": 1,
                "output": "",
                "stderr": "Test timed out after 5 minutes"
            }
        except Exception as e:
            return {
                "passed": False,
                "test_count": 0,
                "failures": 0,
                "errors": 1,
                "output": "",
                "stderr": str(e)
            }
    
    def _print_summary(self, total_time: float):
        """Print comprehensive test summary."""
        print("\n" + "=" * 60)
        print("📊 Fracton Test Suite Summary")
        print("=" * 60)
        
        total_tests = 0
        total_failures = 0
        total_errors = 0
        
        categories = ["core", "language", "theory", "integration", "performance"]
        
        for category in categories:
            if category in self.results:
                print(f"\n{category.upper()} TESTS:")
                category_tests = 0
                category_failures = 0
                category_errors = 0
                
                for test_file, result in self.results[category].items():
                    if "skipped" in result:
                        print(f"  ⏭️  {test_file} (skipped)")
                    elif result["passed"]:
                        print(f"  ✅ {test_file} - {result['test_count']} tests")
                        category_tests += result['test_count']
                    else:
                        print(f"  ❌ {test_file} - {result['failures']} failures, {result['errors']} errors")
                        category_tests += result['test_count']
                        category_failures += result['failures']
                        category_errors += result['errors']
                
                print(f"     Subtotal: {category_tests} tests, {category_failures} failures, {category_errors} errors")
                total_tests += category_tests
                total_failures += category_failures
                total_errors += category_errors
        
        print(f"\n🏁 FINAL RESULTS:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {total_tests - total_failures - total_errors}")
        print(f"   Failed: {total_failures}")
        print(f"   Errors: {total_errors}")
        print(f"   Success Rate: {((total_tests - total_failures - total_errors) / max(total_tests, 1)) * 100:.1f}%")
        print(f"   Execution Time: {total_time:.2f} seconds")
        
        # Foundational theory compliance status
        if "theory" in self.results:
            theory_passed = all(
                result.get("passed", False) 
                for result in self.results["theory"].values() 
                if "skipped" not in result
            )
            
            if theory_passed:
                print(f"\n🌌 FOUNDATIONAL THEORY: ✅ COMPLIANT")
                print("   SEC and MED dynamics validated")
            else:
                print(f"\n🌌 FOUNDATIONAL THEORY: ❌ NON-COMPLIANT")
                print("   Theoretical validation failures detected")
    
    def _print_theory_summary(self, theory_results: Dict[str, Any]):
        """Print foundational theory compliance summary."""
        print("\n🌌 Foundational Theory Compliance Summary")
        print("=" * 45)
        
        for test_file, result in theory_results.items():
            theory_name = "SEC" if "sec" in test_file else "MED"
            
            if result["passed"]:
                print(f"✅ {theory_name} Compliance: VERIFIED")
                print(f"   {result['test_count']} theoretical validations passed")
            else:
                print(f"❌ {theory_name} Compliance: FAILED")
                print(f"   {result['failures']} failures, {result['errors']} errors")
        
        print("\nFoundational theory validation complete.")
    
    def _print_performance_summary(self, perf_results: Dict[str, Any]):
        """Print performance benchmark summary."""
        print("\n⚡ Performance Benchmark Summary")
        print("=" * 35)
        
        for test_file, result in perf_results.items():
            if "skipped" in result:
                print(f"⏭️  {test_file} (not implemented)")
            elif result["passed"]:
                print(f"✅ {test_file}: ALL BENCHMARKS PASSED")
                print(f"   {result['test_count']} performance tests within limits")
            else:
                print(f"❌ {test_file}: PERFORMANCE ISSUES DETECTED")
                print(f"   {result['failures']} benchmarks failed")
        
        print("\nPerformance analysis complete.")


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description="Fracton Test Runner")
    parser.add_argument("--theory-only", action="store_true", 
                       help="Run only foundational theory compliance tests")
    parser.add_argument("--performance-only", action="store_true",
                       help="Run only performance benchmarks")
    parser.add_argument("--coverage", action="store_true",
                       help="Generate coverage report")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    parser.add_argument("--fracton-root", type=Path, default=Path(__file__).parent.parent,
                       help="Path to Fracton root directory")
    
    args = parser.parse_args()
    
    runner = FractonTestRunner(args.fracton_root)
    
    verbose = not args.quiet
    
    if args.theory_only:
        results = runner.run_theory_only(verbose)
    elif args.performance_only:
        results = runner.run_performance_only(verbose)
    else:
        results = runner.run_all_tests(verbose, args.coverage)
    
    # Exit with appropriate code
    if args.theory_only:
        # For theory tests, exit non-zero if any theory test failed
        theory_failed = any(
            not result.get("passed", False) 
            for result in results.values() 
            if "skipped" not in result
        )
        sys.exit(1 if theory_failed else 0)
    else:
        # For full test suite, exit non-zero if any test failed
        any_failed = False
        for category in results.values():
            for result in category.values():
                if "skipped" not in result and not result.get("passed", False):
                    any_failed = True
                    break
            if any_failed:
                break
        
        sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
