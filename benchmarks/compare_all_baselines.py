#!/usr/bin/env python3
"""
Compare FLINT, IML, and GPU implementations.
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
            results[n] = float(row['flint_solve_ms'])
    return results

def read_iml_results(filepath):
    """Read IML benchmark results."""
    results = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row['n'])
            results[n] = float(row['iml_solve_ms'])
    return results

def read_gpu_results(filepath):
    """Read GPU benchmark results."""
    results = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row['n'])
            results[n] = {
                'total_ms': float(row['total_ms']),
                'compute_ms': float(row['compute_ms']),
                'primes': int(row['primes']),
            }
    return results

def main():
    # Read results
    flint_results = read_flint_results('../paper_results/flint_baseline.csv')
    iml_results = read_iml_results('../paper_results/iml_baseline.csv')
    gpu_results = read_gpu_results('../paper_results/gpu_multi_rhs_baseline.csv')

    print("=" * 100)
    print("CPU Libraries vs GPU Comparison - Multi-RHS Linear Solve (k=16)")
    print("=" * 100)
    print()

    # Print header
    header = f"{'n':>6} | {'Primes':>7} | {'FLINT (ms)':>12} | {'IML (ms)':>12} | {'GPU (ms)':>12} | {'vs FLINT':>10} | {'vs IML':>10} | {'Best CPU':>10}"
    print(header)
    print("-" * 100)

    combined = []
    all_sizes = sorted(set(flint_results.keys()) & set(iml_results.keys()) & set(gpu_results.keys()))

    for n in all_sizes:
        flint_ms = flint_results[n]
        iml_ms = iml_results[n]
        gpu = gpu_results[n]

        speedup_flint = flint_ms / gpu['total_ms']
        speedup_iml = iml_ms / gpu['total_ms']
        best_cpu = min(flint_ms, iml_ms)
        best_cpu_name = "FLINT" if flint_ms <= iml_ms else "IML"
        speedup_best = best_cpu / gpu['total_ms']

        print(f"{n:>6} | {gpu['primes']:>7} | {flint_ms:>12.2f} | {iml_ms:>12.2f} | {gpu['total_ms']:>12.2f} | {speedup_flint:>9.2f}x | {speedup_iml:>9.2f}x | {speedup_best:>9.2f}x")

        combined.append({
            'n': n,
            'primes': gpu['primes'],
            'flint_ms': flint_ms,
            'iml_ms': iml_ms,
            'gpu_total_ms': gpu['total_ms'],
            'gpu_compute_ms': gpu['compute_ms'],
            'speedup_flint': speedup_flint,
            'speedup_iml': speedup_iml,
            'best_cpu_ms': best_cpu,
            'best_cpu_name': best_cpu_name,
            'speedup_best': speedup_best,
        })

    print("-" * 100)
    print()

    # Summary statistics
    avg_speedup_flint = sum(r['speedup_flint'] for r in combined) / len(combined)
    avg_speedup_iml = sum(r['speedup_iml'] for r in combined) / len(combined)
    avg_speedup_best = sum(r['speedup_best'] for r in combined) / len(combined)

    print("Summary:")
    print(f"  Average speedup over FLINT: {avg_speedup_flint:.2f}x")
    print(f"  Average speedup over IML:   {avg_speedup_iml:.2f}x")
    print(f"  Average speedup over best CPU: {avg_speedup_best:.2f}x")
    print()

    # Analysis
    print("Analysis:")
    for r in combined:
        if r['speedup_best'] < 1.0:
            print(f"  WARNING: At n={r['n']}, {r['best_cpu_name']} ({r['best_cpu_ms']:.2f}ms) beats GPU ({r['gpu_total_ms']:.2f}ms)")
        else:
            print(f"  n={r['n']}: GPU is {r['speedup_best']:.2f}x faster than {r['best_cpu_name']}")
    print()

    # Export combined results
    output_file = '../paper_results/all_baselines_comparison.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'n', 'primes', 'flint_ms', 'iml_ms', 'gpu_total_ms', 'gpu_compute_ms',
            'speedup_flint', 'speedup_iml', 'best_cpu_ms', 'best_cpu_name', 'speedup_best'
        ])
        writer.writeheader()
        for row in combined:
            writer.writerow(row)

    print(f"Results exported to: {output_file}")
    print()

    # LaTeX table
    print("LaTeX Table for Paper:")
    print("-" * 80)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Comparison with CPU Libraries: Multi-RHS Linear Solve ($k=16$)}")
    print(r"\label{tab:cpu-comparison}")
    print(r"\begin{tabular}{rrrrrr}")
    print(r"\toprule")
    print(r"$n$ & Primes & FLINT (ms) & IML (ms) & GPU (ms) & Speedup$^*$ \\")
    print(r"\midrule")
    for r in combined:
        speedup_str = f"{r['speedup_best']:.2f}$\\times$" if r['speedup_best'] >= 1.0 else f"\\textbf{{{r['speedup_best']:.2f}$\\times$}}"
        print(f"{r['n']} & {r['primes']} & {r['flint_ms']:.2f} & {r['iml_ms']:.2f} & {r['gpu_total_ms']:.2f} & {speedup_str} \\\\")
    print(r"\bottomrule")
    print(r"\multicolumn{6}{l}{\footnotesize $^*$Speedup over faster CPU library (FLINT or IML)} \\")
    print(r"\end{tabular}")
    print(r"\end{table}")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
