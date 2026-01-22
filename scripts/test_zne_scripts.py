#!/usr/bin/env python3
"""
Quick test script for ZNE benchmarking components.

Runs dry runs and basic tests without requiring IBM credentials.

Usage:
    python scripts/test_zne_scripts.py
"""

import subprocess
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent


def run_command(cmd: str, description: str) -> bool:
    """Run a command and report result."""
    print(f"\n{'='*60}")
    print(f"TEST: {description}")
    print(f"CMD:  {cmd}")
    print("-" * 60)

    result = subprocess.run(cmd, shell=True, cwd=ROOT)
    success = result.returncode == 0

    status = "PASS" if success else "FAIL"
    print(f"\nResult: {status}")
    return success


def main():
    print("ZNE Benchmarking Scripts Test Suite")
    print("=" * 60)

    tests = []

    # Test 1: Dry run of benchmark script
    tests.append(run_command(
        "python scripts/run_zne_benchmark.py --dry-run",
        "ZNE benchmark dry run"
    ))

    # Test 2: Analysis of sample results
    tests.append(run_command(
        "python scripts/analyze_zne_results.py examples/sample_zne_results.json --quiet",
        "Analysis of sample results"
    ))

    # Test 3: LaTeX table generation
    tests.append(run_command(
        "python scripts/analyze_zne_results.py examples/sample_zne_results.json --latex --quiet",
        "LaTeX table generation"
    ))

    # Test 4: Quick demo (runs actual VQE)
    tests.append(run_command(
        "timeout 120 python examples/quick_zne_demo.py || true",
        "Quick ZNE demo (limited to 2 min)"
    ))

    # Test 5: ZNE visualization tests
    tests.append(run_command(
        "python -m pytest tests/test_zne_visualization.py -v -x",
        "Visualization tests"
    ))

    # Test 6: ZNE analysis tests
    tests.append(run_command(
        "python -m pytest tests/test_zne_analysis.py -v -x",
        "Analysis tests"
    ))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(tests)
    total = len(tests)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\nAll tests passed!")
        return 0
    else:
        print(f"\n{total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
