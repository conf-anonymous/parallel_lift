#!/usr/bin/env python3
"""
Compare CPU CRT vs GPU CRT performance for V1 and V2 solvers.

Key insights:
- GPU CRT dramatically speeds up CRT reconstruction (5-7x)
- With GPU CRT, V1 (32-bit) is the clear winner as 64-bit operations are slower on GPU
- For maximum performance, use V1 with GPU CRT
"""

import csv
import os

def read_v2_cpu_crt(filepath):
    """Read V2 benchmark results with CPU CRT."""
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
                # Compute CRT time as total - GPU
                'v1_crt_ms': float(row['v1_total_ms']) - float(row['v1_gpu_ms']),
                'v2_crt_ms': float(row['v2_total_ms']) - float(row['v2_gpu_ms']),
            }
    return results

def read_v2_gpu_crt(filepath):
    """Read V2 benchmark results with GPU CRT."""
    results = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row['n'])
            results[n] = {
                'v1_gpu_ms': float(row['v1_gpu_ms']),
                'v2_gpu_ms': float(row['v2_gpu_ms']),
                'v1_crt_ms': float(row['v1_crt_ms']),
                'v2_crt_ms': float(row['v2_crt_ms']),
                'v1_total_ms': float(row['v1_total_ms']),
                'v2_total_ms': float(row['v2_total_ms']),
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

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Read results
    cpu_crt = read_v2_cpu_crt('../paper_results/v2_benchmark_final.csv')
    gpu_crt = read_v2_gpu_crt('../paper_results/v2_gpu_crt_benchmark.csv')
    iml = read_iml_results('../paper_results/iml_baseline.csv')

    print("=" * 140)
    print("GPU CRT vs CPU CRT Performance Comparison")
    print("=" * 140)
    print()

    print("Key Finding: GPU CRT provides massive speedup for CRT reconstruction!")
    print("With GPU CRT, V1 (32-bit) becomes the clear winner as 64-bit GPU operations are slower.")
    print()

    # Table 1: CRT Speedup
    print("CRT Reconstruction Time (ms): CPU vs GPU")
    print("-" * 120)
    print(f"{'n':>6} | {'V1 CPU CRT':>12} | {'V1 GPU CRT':>12} | {'V1 Speedup':>12} | {'V2 CPU CRT':>12} | {'V2 GPU CRT':>12} | {'V2 Speedup':>12}")
    print("-" * 120)

    all_sizes = sorted(set(cpu_crt.keys()) & set(gpu_crt.keys()))
    for n in all_sizes:
        v1_cpu_crt = cpu_crt[n]['v1_crt_ms']
        v1_gpu_crt = gpu_crt[n]['v1_crt_ms']
        v1_speedup = v1_cpu_crt / v1_gpu_crt

        v2_cpu_crt = cpu_crt[n]['v2_crt_ms']
        v2_gpu_crt = gpu_crt[n]['v2_crt_ms']
        v2_speedup = v2_cpu_crt / v2_gpu_crt

        print(f"{n:>6} | {v1_cpu_crt:>12.2f} | {v1_gpu_crt:>12.2f} | {v1_speedup:>11.2f}x | {v2_cpu_crt:>12.2f} | {v2_gpu_crt:>12.2f} | {v2_speedup:>11.2f}x")

    print("-" * 120)
    print()

    # Table 2: Total Time Comparison
    print("Total Solve Time (ms): CPU CRT vs GPU CRT vs IML")
    print("-" * 140)
    print(f"{'n':>6} | {'V1 CPU CRT':>12} | {'V1 GPU CRT':>12} | {'V1 vs IML':>12} | {'V2 CPU CRT':>12} | {'V2 GPU CRT':>12} | {'V2 vs IML':>12} | {'IML':>10}")
    print("-" * 140)

    for n in all_sizes:
        v1_cpu_total = cpu_crt[n]['v1_total_ms']
        v1_gpu_total = gpu_crt[n]['v1_total_ms']

        v2_cpu_total = cpu_crt[n]['v2_total_ms']
        v2_gpu_total = gpu_crt[n]['v2_total_ms']

        iml_time = iml.get(n, float('nan'))
        v1_vs_iml = iml_time / v1_gpu_total if v1_gpu_total > 0 else float('nan')
        v2_vs_iml = iml_time / v2_gpu_total if v2_gpu_total > 0 else float('nan')

        print(f"{n:>6} | {v1_cpu_total:>12.2f} | {v1_gpu_total:>12.2f} | {v1_vs_iml:>11.2f}x | {v2_cpu_total:>12.2f} | {v2_gpu_total:>12.2f} | {v2_vs_iml:>11.2f}x | {iml_time:>10.2f}")

    print("-" * 140)
    print()

    # Summary
    print("RECOMMENDATIONS FOR PAPER:")
    print()
    print("1. Use V1 (32-bit) with GPU CRT for best performance")
    print("   - 32-bit GPU kernels are faster than 64-bit")
    print("   - GPU CRT gives ~5-12x speedup over CPU CRT")
    print()
    print("2. GPU CRT dramatically improves our competitiveness with IML:")

    for n in all_sizes:
        v1_gpu_total = gpu_crt[n]['v1_total_ms']
        iml_time = iml.get(n, float('nan'))
        if iml_time > 0:
            speedup = iml_time / v1_gpu_total
            status = "faster" if speedup > 1 else "slower"
            print(f"   n={n}: V1+GPU-CRT is {abs(speedup):.2f}x {status} than IML")

    print()
    print("3. V2 (64-bit) is best for extreme precision needs")
    print("   - Uses half as many primes")
    print("   - Suitable when solution bounds exceed 31-bit prime capacity")
    print()

    # Export summary
    output_file = '../paper_results/gpu_crt_comparison.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n', 'v1_cpu_crt_ms', 'v1_gpu_crt_ms', 'v1_crt_speedup',
                        'v2_cpu_crt_ms', 'v2_gpu_crt_ms', 'v2_crt_speedup',
                        'v1_total_cpu_ms', 'v1_total_gpu_ms', 'v2_total_cpu_ms', 'v2_total_gpu_ms',
                        'iml_ms', 'v1_gpu_vs_iml'])
        for n in all_sizes:
            v1_cpu_crt = cpu_crt[n]['v1_crt_ms']
            v1_gpu_crt = gpu_crt[n]['v1_crt_ms']
            v1_crt_speedup = v1_cpu_crt / v1_gpu_crt

            v2_cpu_crt = cpu_crt[n]['v2_crt_ms']
            v2_gpu_crt = gpu_crt[n]['v2_crt_ms']
            v2_crt_speedup = v2_cpu_crt / v2_gpu_crt

            v1_total_cpu = cpu_crt[n]['v1_total_ms']
            v1_total_gpu = gpu_crt[n]['v1_total_ms']
            v2_total_cpu = cpu_crt[n]['v2_total_ms']
            v2_total_gpu = gpu_crt[n]['v2_total_ms']

            iml_time = iml.get(n, float('nan'))
            v1_vs_iml = iml_time / v1_total_gpu if v1_total_gpu > 0 else float('nan')

            writer.writerow([n, v1_cpu_crt, v1_gpu_crt, v1_crt_speedup,
                           v2_cpu_crt, v2_gpu_crt, v2_crt_speedup,
                           v1_total_cpu, v1_total_gpu, v2_total_cpu, v2_total_gpu,
                           iml_time, v1_vs_iml])

    print(f"Results exported to: {output_file}")

if __name__ == '__main__':
    main()
