#!/usr/bin/env python3
"""
Compare V2 GPU implementation against IML baseline.
Shows whether 62-bit primes close the performance gap.
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
                'v1_ms': float(row['v1_ms']),
                'v2_ms': float(row['v2_ms']),
                'v1_primes': int(row['v1_primes']),
                'v2_primes': int(row['v2_primes']),
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
    v2_results = read_v2_results('../paper_results/v2_benchmark.csv')
    iml_results = read_iml_results('../paper_results/iml_baseline.csv')
    flint_results = read_flint_results('../paper_results/flint_baseline.csv')

    print("=" * 110)
    print("V1 vs V2 vs IML vs FLINT - Multi-RHS Linear Solve (k=16)")
    print("=" * 110)
    print()

    # Print header
    header = (
        f"{'n':>6} | {'V1 (ms)':>10} | {'V2 (ms)':>10} | {'IML (ms)':>10} | "
        f"{'FLINT (ms)':>10} | {'V2 vs IML':>10} | {'V2 vs FLINT':>12} | {'V2 vs V1':>10}"
    )
    print(header)
    print("-" * 110)

    combined = []
    all_sizes = sorted(set(v2_results.keys()) & set(iml_results.keys()))

    for n in all_sizes:
        v1_ms = v2_results[n]['v1_ms']
        v2_ms = v2_results[n]['v2_ms']
        iml_ms = iml_results[n]
        flint_ms = flint_results.get(n, float('nan'))

        speedup_vs_iml = iml_ms / v2_ms
        speedup_vs_flint = flint_ms / v2_ms if flint_ms == flint_ms else float('nan')
        speedup_vs_v1 = v1_ms / v2_ms

        status_iml = "✓" if speedup_vs_iml >= 1.0 else "✗"
        status_flint = "✓" if speedup_vs_flint >= 1.0 else "✗"

        print(
            f"{n:>6} | {v1_ms:>10.2f} | {v2_ms:>10.2f} | {iml_ms:>10.2f} | "
            f"{flint_ms:>10.2f} | {speedup_vs_iml:>8.2f}x {status_iml} | "
            f"{speedup_vs_flint:>10.2f}x {status_flint} | {speedup_vs_v1:>9.2f}x"
        )

        combined.append({
            'n': n,
            'v1_ms': v1_ms,
            'v2_ms': v2_ms,
            'iml_ms': iml_ms,
            'flint_ms': flint_ms,
            'speedup_vs_iml': speedup_vs_iml,
            'speedup_vs_flint': speedup_vs_flint,
            'speedup_vs_v1': speedup_vs_v1,
        })

    print("-" * 110)
    print()

    # Summary
    print("Summary:")
    avg_vs_iml = sum(r['speedup_vs_iml'] for r in combined) / len(combined)
    avg_vs_flint = sum(r['speedup_vs_flint'] for r in combined if r['speedup_vs_flint'] == r['speedup_vs_flint']) / len([r for r in combined if r['speedup_vs_flint'] == r['speedup_vs_flint']])
    avg_vs_v1 = sum(r['speedup_vs_v1'] for r in combined) / len(combined)

    print(f"  Average V2 speedup over IML: {avg_vs_iml:.2f}x")
    print(f"  Average V2 speedup over FLINT: {avg_vs_flint:.2f}x")
    print(f"  Average V2 speedup over V1: {avg_vs_v1:.2f}x")
    print()

    # Analysis
    print("Analysis:")
    beats_iml = [r for r in combined if r['speedup_vs_iml'] >= 1.0]
    loses_iml = [r for r in combined if r['speedup_vs_iml'] < 1.0]

    if beats_iml:
        print(f"  V2 beats IML at: n = {', '.join(str(r['n']) for r in beats_iml)}")
    if loses_iml:
        print(f"  V2 loses to IML at: n = {', '.join(str(r['n']) for r in loses_iml)}")
        for r in loses_iml:
            gap = (r['iml_ms'] / r['v2_ms'] - 1) * 100
            print(f"    n={r['n']}: V2 is {-gap:.1f}% slower than IML ({r['v2_ms']:.2f}ms vs {r['iml_ms']:.2f}ms)")

    print()

    # Export combined results
    output_file = '../paper_results/v2_vs_iml_comparison.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'n', 'v1_ms', 'v2_ms', 'iml_ms', 'flint_ms',
            'speedup_vs_iml', 'speedup_vs_flint', 'speedup_vs_v1'
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
    print(r"\caption{V2 (62-bit primes) vs CPU Libraries: Multi-RHS Linear Solve ($k=16$)}")
    print(r"\label{tab:v2-comparison}")
    print(r"\begin{tabular}{rrrrrr}")
    print(r"\toprule")
    print(r"$n$ & V1 (ms) & V2 (ms) & IML (ms) & V2 vs IML & V2 vs V1 \\")
    print(r"\midrule")
    for r in combined:
        iml_str = f"{r['speedup_vs_iml']:.2f}$\\times$" if r['speedup_vs_iml'] >= 1.0 else f"\\textbf{{{r['speedup_vs_iml']:.2f}$\\times$}}"
        print(f"{r['n']} & {r['v1_ms']:.2f} & {r['v2_ms']:.2f} & {r['iml_ms']:.2f} & {iml_str} & {r['speedup_vs_v1']:.2f}$\\times$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
