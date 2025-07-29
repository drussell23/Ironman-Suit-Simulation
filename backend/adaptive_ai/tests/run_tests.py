#!/usr/bin/env python3
"""
Test runner for the Adaptive AI test suite

This script provides a convenient way to run all tests with various options.
"""

import sys
import os
import subprocess
import argparse


def run_command(cmd):
    """Run a command and print output"""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run Adaptive AI tests')
    parser.add_argument('--coverage', action='store_true', 
                       help='Run tests with coverage report')
    parser.add_argument('--unit-only', action='store_true',
                       help='Run only unit tests')
    parser.add_argument('--integration-only', action='store_true',
                       help='Run only integration tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--failfast', '-x', action='store_true',
                       help='Stop on first failure')
    parser.add_argument('--parallel', '-n', type=int,
                       help='Run tests in parallel with N workers')
    parser.add_argument('tests', nargs='*',
                       help='Specific test files or patterns to run')
    
    args = parser.parse_args()
    
    # Base command
    cmd = [sys.executable, '-m', 'pytest']
    
    # Add test directory or specific tests
    if args.tests:
        cmd.extend(args.tests)
    else:
        cmd.append('.')
    
    # Add options
    if args.verbose:
        cmd.append('-v')
    
    if args.failfast:
        cmd.append('-x')
    
    if args.unit_only:
        cmd.extend(['-m', 'unit'])
    elif args.integration_only:
        cmd.extend(['-m', 'integration'])
    
    if args.coverage:
        cmd.extend(['--cov=../adaptive_ai', '--cov-report=html', '--cov-report=term'])
    
    if args.parallel:
        cmd.extend(['-n', str(args.parallel)])
    
    # Change to tests directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run tests
    return_code = run_command(cmd)
    
    if args.coverage and return_code == 0:
        print("\nCoverage report generated in htmlcov/")
        print("Open htmlcov/index.html to view detailed coverage")
    
    return return_code


if __name__ == '__main__':
    sys.exit(main())