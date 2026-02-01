#!/usr/bin/env python3
"""
Final V2 Comparison: GPU V1 vs GPU V2 vs IML vs FLINT

Key insight: V2's 64-bit kernels are slower than V1's 32-bit kernels,
but CRT reconstruction with half the primes is much faster.
"""

import csv
import os

def read_v2_results(filepath):
    """Read V2 benchmark results."""
    results = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row['n'])
            results[n] = {
                'v1_gpu_ms': float(row['v1_gpu_ms']),
                'v2_gpu_ms': float(row['v2_gpu_ms']),
                'v1_total_ms': float(row['v1_total_ms']),
                'v2_total_ms': float(row['v2_total_ms']),
                'v1_primes': int(row['v1_primes']),
                'v2_primes': int(row['v2_primes']),
            }
    return results

def read_gpu_baseline(filepath):
    """Read GPU baseline results (matches V1)."""
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

def read_iml_results(filepath):
    """Read IML benchmark results."""
    results = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row['n'])
            results[n] = float(row['iml_solve_ms'])
    return results

def read_flint_results(filepath):
    """Read FLINT benchmark results."""
    results = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row['n'])
            results[n] = float(row['flint_solve_ms'])
    return results

def main():
    # Read results
    v2_results = read_v2_results('../paper_results/v2_benchmark_final.csv')
    gpu_baseline = read_gpu_baseline('../paper_results/gpu_multi_rhs_baseline.csv')
    iml_results = read_iml_results('../paper_results/iml_baseline.csv')
    flint_results = read_flint_results('../paper_results/flint_baseline.csv')

    print("=" * 120)
    print("GPU Performance Analysis: V1 (32-bit) vs V2 (62-bit) vs CPU Libraries")
    print("=" * 120)
    print()

    print("Key Insight: V2's 64-bit kernels are SLOWER than V1's 32-bit kernels,")
    print("but V2 wins overall because CRT reconstruction with half the primes is much faster.")
    print()

    # Table 1: GPU-only comparison
    print("GPU Compute Time Only (excluding CRT):")
    print("-" * 100)
    print(f"{'n':>6} | {'V1 Primes':>10} | {'V2 Primes':>10} | {'V1 GPU':>10} | {'V2 GPU':>10} | {'IML':>10} | {'FLINT':>10}")
    print("-" * 100)

    all_sizes = sorted(set(v2_results.keys()) & set(iml_results.keys()))
    for n in all_sizes:
        v1_gpu = v2_results[n]['v1_gpu_ms']
        v2_gpu = v2_results[n]['v2_gpu_ms']
        iml = iml_results[n]
        flint = flint_results.get(n, float('nan'))
        v1_primes = v2_results[n]['v1_primes']
        v2_primes = v2_results[n]['v2_primes']

        print(f"{n:>6} | {v1_primes:>10} | {v2_primes:>10} | {v1_gpu:>10.2f} | {v2_gpu:>10.2f} | {iml:>10.2f} | {flint:>10.2f}")

    print("-" * 100)
    print()

    # Table 2: Speedup comparison (GPU-only time vs IML)
    print("Speedup vs IML (GPU-only time, no CRT):")
    print("-" * 80)
    print(f"{'n':>6} | {'V1 vs IML':>12} | {'V2 vs IML':>12} | {'Best GPU':>12} | {'Winner':>10}")
    print("-" * 80)

    combined = []
    for n in all_sizes:
        v1_gpu = v2_results[n]['v1_gpu_ms']
        v2_gpu = v2_results[n]['v2_gpu_ms']
        iml = iml_results[n]

        v1_vs_iml = iml / v1_gpu
        v2_vs_iml = iml / v2_gpu

        # Best GPU option
        if v1_gpu < v2_gpu:
            best_gpu_ms = v1_gpu
            best_vs_iml = v1_vs_iml
            best_name = "V1 (32-bit)"
        else:
            best_gpu_ms = v2_gpu
            best_vs_iml = v2_vs_iml
            best_name = "V2 (64-bit)"

        winner = best_name if best_vs_iml >= 1.0 else "IML"

        print(f"{n:>6} | {v1_vs_iml:>10.2f}x | {v2_vs_iml:>10.2f}x | {best_vs_iml:>10.2f}x | {winner:>10}")

        combined.append({
            'n': n,
            'v1_gpu_ms': v1_gpu,
            'v2_gpu_ms': v2_gpu,
            'iml_ms': iml,
            'v1_vs_iml': v1_vs_iml,
            'v2_vs_iml': v2_vs_iml,
            'best_gpu_ms': best_gpu_ms,
            'best_vs_iml': best_vs_iml,
            'winner': winner,
        })

    print("-" * 80)
    print()

    # Summary
    print("Summary:")
    v1_wins = sum(1 for r in combined if r['v1_vs_iml'] >= 1.0)
    v2_wins = sum(1 for r in combined if r['v2_vs_iml'] >= 1.0)
    gpu_wins = sum(1 for r in combined if r['winner'] != "IML")

    print(f"  V1 beats IML: {v1_wins}/{len(combined)} sizes")
    print(f"  V2 beats IML: {v2_wins}/{len(combined)} sizes")
    print(f"  Best GPU beats IML: {gpu_wins}/{len(combined)} sizes")
    print()

    # Recommendations
    print("Recommendations:")
    print("  1. For n <= 128: Use V1 (32-bit primes) - faster GPU compute")
    print("  2. For n >= 256: Both V1 and V2 roughly tie on GPU; V2 wins with CRT included")
    print("  3. IML is competitive at all sizes due to optimized BLAS + Hensel lifting")
    print()

    # GPU baseline comparison
    print("GPU Baseline (original benchmark) vs V2 Benchmark:")
    print("-" * 60)
    for n in all_sizes:
        if n in gpu_baseline:
            baseline = gpu_baseline[n]['total_ms']
            v1_current = v2_results[n]['v1_gpu_ms']
            diff_pct = ((v1_current - baseline) / baseline) * 100
            print(f"  n={n}: Baseline {baseline:.2f}ms, V2 bench V1 {v1_current:.2f}ms ({diff_pct:+.1f}%)")

    print()

    # Export
    output_file = '../paper_results/v2_final_analysis.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'n', 'v1_gpu_ms', 'v2_gpu_ms', 'iml_ms',
            'v1_vs_iml', 'v2_vs_iml', 'best_gpu_ms', 'best_vs_iml', 'winner'
        ])
        writer.writeheader()
        for row in combined:
            writer.writerow(row)

    print(f"Results exported to: {output_file}")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
