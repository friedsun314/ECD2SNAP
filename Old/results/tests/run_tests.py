#!/usr/bin/env python3
"""
Test suite runner for ECD-to-SNAP optimization.

Usage:
    python tests/run_tests.py              # Run all tests
    python tests/run_tests.py unit         # Run only unit tests
    python tests/run_tests.py integration  # Run only integration tests
    python tests/run_tests.py benchmarks   # Run only benchmarks
    python tests/run_tests.py quick        # Run quick validation tests
"""

import sys
import time
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_unit_tests():
    """Run all unit tests."""
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60)
    
    test_modules = [
        'tests.unit.test_gates',
        'tests.unit.test_gradients',
        'tests.unit.test_core_logic',
        'tests.unit.test_fidelity',
        'tests.unit.test_displacement'
    ]
    
    results = {}
    for module_name in test_modules:
        print(f"\nTesting {module_name}...")
        try:
            module = __import__(module_name, fromlist=[''])
            # Run all test functions
            test_count = 0
            for name in dir(module):
                if name.startswith('test_'):
                    test_func = getattr(module, name)
                    try:
                        test_func()
                        test_count += 1
                    except Exception as e:
                        results[f"{module_name}.{name}"] = f"FAILED: {e}"
                        print(f"  ✗ {name}: {e}")
                    else:
                        results[f"{module_name}.{name}"] = "PASSED"
                        print(f"  ✓ {name}")
            
            if test_count == 0:
                print(f"  No tests found in {module_name}")
                
        except ImportError as e:
            print(f"  Could not import {module_name}: {e}")
            results[module_name] = f"IMPORT FAILED: {e}"
    
    return results


def run_integration_tests():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("RUNNING INTEGRATION TESTS")
    print("="*60)
    
    test_modules = [
        'tests.integration.test_basic_optimization',
        'tests.integration.test_optimization_strategies',
        'tests.integration.test_initialization',
        'tests.integration.test_quick_validation'
    ]
    
    results = {}
    for module_name in test_modules:
        print(f"\nTesting {module_name}...")
        try:
            module = __import__(module_name, fromlist=[''])
            # Run test functions
            for name in dir(module):
                if name.startswith('test_'):
                    test_func = getattr(module, name)
                    try:
                        start_time = time.time()
                        test_func()
                        elapsed = time.time() - start_time
                        results[f"{module_name}.{name}"] = f"PASSED ({elapsed:.2f}s)"
                        print(f"  ✓ {name} ({elapsed:.2f}s)")
                    except Exception as e:
                        results[f"{module_name}.{name}"] = f"FAILED: {e}"
                        print(f"  ✗ {name}: {e}")
                        
        except ImportError as e:
            print(f"  Could not import {module_name}: {e}")
            results[module_name] = f"IMPORT FAILED: {e}"
    
    return results


def run_benchmarks():
    """Run benchmark tests."""
    print("\n" + "="*60)
    print("RUNNING BENCHMARKS")
    print("="*60)
    print("Note: Benchmarks may take several minutes to complete.")
    
    test_modules = [
        'tests.benchmarks.test_comprehensive',
        'tests.benchmarks.test_optimizer_comparison'
    ]
    
    results = {}
    for module_name in test_modules:
        print(f"\nRunning {module_name}...")
        try:
            module = __import__(module_name, fromlist=[''])
            # Run benchmark functions
            for name in dir(module):
                if name.startswith('test_'):
                    test_func = getattr(module, name)
                    try:
                        print(f"  Running {name}...")
                        start_time = time.time()
                        test_func()
                        elapsed = time.time() - start_time
                        results[f"{module_name}.{name}"] = f"COMPLETED ({elapsed:.2f}s)"
                        print(f"  ✓ {name} completed in {elapsed:.2f}s")
                    except Exception as e:
                        results[f"{module_name}.{name}"] = f"FAILED: {e}"
                        print(f"  ✗ {name}: {e}")
                        
        except ImportError as e:
            print(f"  Could not import {module_name}: {e}")
            results[module_name] = f"IMPORT FAILED: {e}"
    
    return results


def run_quick_tests():
    """Run a quick subset of tests for rapid validation."""
    print("\n" + "="*60)
    print("RUNNING QUICK VALIDATION TESTS")
    print("="*60)
    
    quick_tests = [
        ('tests.unit.test_gates', 'test_displacement_properties'),
        ('tests.unit.test_gradients', 'test_displacement_gradient'),
        ('tests.unit.test_fidelity', 'test_fidelity_calculation'),
        ('tests.integration.test_quick_validation', 'test_basic_targets')
    ]
    
    results = {}
    for module_name, test_name in quick_tests:
        full_name = f"{module_name}.{test_name}"
        print(f"\nTesting {full_name}...")
        try:
            module = __import__(module_name, fromlist=[''])
            test_func = getattr(module, test_name)
            test_func()
            results[full_name] = "PASSED"
            print(f"  ✓ {test_name}")
        except Exception as e:
            results[full_name] = f"FAILED: {e}"
            print(f"  ✗ {test_name}: {e}")
    
    return results


def print_summary(all_results):
    """Print test summary."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in all_results.values() if 'PASSED' in str(r) or 'COMPLETED' in str(r))
    failed = sum(1 for r in all_results.values() if 'FAILED' in str(r))
    total = len(all_results)
    
    print(f"\nTotal: {total} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed tests:")
        for test_name, result in all_results.items():
            if 'FAILED' in str(result):
                print(f"  - {test_name}: {result}")
    
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description='Run ECD-to-SNAP test suite')
    parser.add_argument('suite', nargs='?', default='all',
                      choices=['all', 'unit', 'integration', 'benchmarks', 'quick'],
                      help='Test suite to run (default: all)')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Verbose output')
    
    args = parser.parse_args()
    
    print("ECD-to-SNAP Test Suite")
    print(f"Running: {args.suite} tests")
    
    all_results = {}
    
    if args.suite == 'all':
        all_results.update(run_unit_tests())
        all_results.update(run_integration_tests())
        # Benchmarks are optional for 'all'
        response = input("\nRun benchmarks? (may take several minutes) [y/N]: ")
        if response.lower() == 'y':
            all_results.update(run_benchmarks())
    elif args.suite == 'unit':
        all_results = run_unit_tests()
    elif args.suite == 'integration':
        all_results = run_integration_tests()
    elif args.suite == 'benchmarks':
        all_results = run_benchmarks()
    elif args.suite == 'quick':
        all_results = run_quick_tests()
    
    success = print_summary(all_results)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()