#!/usr/bin/env python3
"""
Compare FLINT baseline against GPU implementation.
Generates a combined results table for the paper.
"""

import csv
import os

def read_flint_results(filepath):
    """Read FLINT benchmark results."""
    results = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row['n'])
            results[n] = {
                'flint_solve_ms': float(row['flint_solve_ms']),
                'flint_det_ms': float(row['flint_det_ms']),
            }
    return results

def read_gpu_results(filepath):
    """Read GPU benchmark results."""
    results = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row['n'])
            results[n] = {
                'gpu_total_ms': float(row['total_ms']),
                'gpu_compute_ms': float(row['compute_ms']),
                'gpu_htod_ms': float(row['htod_ms']),
                'gpu_dtoh_ms': float(row['dtoh_ms']),
                'primes': int(row['primes']),
            }
    return results

def main():
    # Read results
    flint_results = read_flint_results('../paper_results/flint_baseline.csv')
    gpu_results = read_gpu_results('../paper_results/gpu_multi_rhs_baseline.csv')

    print("=" * 80)
    print("FLINT vs GPU Comparison - Multi-RHS Linear Solve (k=16)")
    print("=" * 80)
    print()

    # Print header
    print(f"{'n':>6} | {'Primes':>7} | {'FLINT (ms)':>12} | {'GPU (ms)':>12} | {'Speedup':>8} | {'GPU Compute':>12}")
    print("-" * 80)

    combined = []
    for n in sorted(set(flint_results.keys()) & set(gpu_results.keys())):
        flint = flint_results[n]
        gpu = gpu_results[n]

        speedup = flint['flint_solve_ms'] / gpu['gpu_total_ms']

        print(f"{n:>6} | {gpu['primes']:>7} | {flint['flint_solve_ms']:>12.2f} | {gpu['gpu_total_ms']:>12.2f} | {speedup:>7.2f}x | {gpu['gpu_compute_ms']:>12.2f}")

        combined.append({
            'n': n,
            'primes': gpu['primes'],
            'flint_ms': flint['flint_solve_ms'],
            'gpu_total_ms': gpu['gpu_total_ms'],
            'gpu_compute_ms': gpu['gpu_compute_ms'],
            'speedup': speedup,
        })

    print("-" * 80)
    print()

    # Summary statistics
    avg_speedup = sum(r['speedup'] for r in combined) / len(combined)
    max_speedup = max(r['speedup'] for r in combined)
    max_speedup_n = [r['n'] for r in combined if r['speedup'] == max_speedup][0]

    print("Summary:")
    print(f"  Average speedup over FLINT: {avg_speedup:.2f}x")
    print(f"  Maximum speedup: {max_speedup:.2f}x at n={max_speedup_n}")
    print()

    # Export combined results
    output_file = '../paper_results/flint_gpu_comparison.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['n', 'primes', 'flint_ms', 'gpu_total_ms', 'gpu_compute_ms', 'speedup'])
        writer.writeheader()
        for row in combined:
            writer.writerow(row)

    print(f"Results exported to: {output_file}")
    print()

    # LaTeX table
    print("LaTeX Table for Paper:")
    print("-" * 60)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Comparison with FLINT: Multi-RHS Linear Solve ($k=16$)}")
    print(r"\label{tab:flint-comparison}")
    print(r"\begin{tabular}{rrrrrr}")
    print(r"\toprule")
    print(r"$n$ & Primes & FLINT (ms) & GPU (ms) & Speedup \\")
    print(r"\midrule")
    for r in combined:
        print(f"{r['n']} & {r['primes']} & {r['flint_ms']:.2f} & {r['gpu_total_ms']:.2f} & {r['speedup']:.2f}$\\times$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
