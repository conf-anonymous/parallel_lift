#!/usr/bin/env python3
"""
V2 vs V1 Analysis: Understanding why IML beats our GPU implementation
and how V2 improvements could close the gap.
"""

import math

def estimate_hadamard_bound(n, max_entry=100):
    """Estimate Hadamard bound for n×n matrix with entries in [-max_entry, max_entry]."""
    # Hadamard bound: |det(A)| <= prod(||row_i||) <= n^(n/2) * max_entry^n
    # For the solution bound, we also need to account for A^(-1) entries
    # Upper bound: n! * max_entry^n (very conservative)
    log2_bound = n * math.log2(n * max_entry) + n * math.log2(max_entry)
    return log2_bound

def primes_needed(bit_width, prime_bits):
    """Calculate number of primes needed to cover a given bit width."""
    return (bit_width + prime_bits - 1) // prime_bits + 1

def hensel_iterations(bit_width, prime_bits):
    """Calculate number of Hensel lifting iterations needed."""
    # Hensel lifting doubles precision each iteration
    return math.ceil(math.log2(bit_width / prime_bits)) + 1

def analyze_comparison(sizes=[32, 64, 128, 256, 512]):
    print("=" * 100)
    print("CRT vs Hensel Lifting: Algorithmic Comparison")
    print("=" * 100)
    print()

    print("Legend:")
    print("  - V1: Our current CRT approach with 31-bit primes")
    print("  - V2: Proposed CRT approach with 62-bit primes (halves prime count)")
    print("  - IML: Hensel lifting with single prime (BLAS-optimized)")
    print()

    print(f"{'n':>6} | {'Bound':>8} | {'V1 Primes':>10} | {'V2 Primes':>10} | {'Hensel Iters':>12} | {'V1/Hensel':>10} | {'V2/Hensel':>10}")
    print("-" * 100)

    results = []
    for n in sizes:
        # Estimate bit width needed for solution entries
        bit_width = int(estimate_hadamard_bound(n))

        v1_primes = primes_needed(bit_width, 31)
        v2_primes = primes_needed(bit_width, 62)
        hensel_iters = hensel_iterations(bit_width, 62)

        v1_ratio = v1_primes / hensel_iters
        v2_ratio = v2_primes / hensel_iters

        print(f"{n:>6} | {bit_width:>8} | {v1_primes:>10} | {v2_primes:>10} | {hensel_iters:>12} | {v1_ratio:>9.1f}x | {v2_ratio:>9.1f}x")

        results.append({
            'n': n,
            'bit_width': bit_width,
            'v1_primes': v1_primes,
            'v2_primes': v2_primes,
            'hensel_iters': hensel_iters,
            'v1_ratio': v1_ratio,
            'v2_ratio': v2_ratio,
        })

    print("-" * 100)
    print()

    # Analysis
    print("Analysis:")
    print()
    print("1. Work Ratio (# of n³ operations):")
    print("   - V1: Each prime requires one LU factorization = n³/3 operations")
    print("   - V2: Same as V1, but half the primes")
    print("   - Hensel: Each iteration is one matrix-matrix multiply = n³ operations")
    print()
    print("   Total work (excluding constants):")
    for r in results:
        v1_work = r['v1_primes'] / 3
        v2_work = r['v2_primes'] / 3
        hensel_work = r['hensel_iters']
        print(f"   n={r['n']}: V1={v1_work:.0f}, V2={v2_work:.0f}, Hensel={hensel_work:.0f} (relative n³ units)")
    print()

    print("2. Memory Bandwidth:")
    print("   - V1/V2: Must load matrix for each prime (poor cache reuse)")
    print("   - Hensel: Loads matrix once, reuses in L1/L2 cache")
    print()

    print("3. BLAS Optimization:")
    print("   - V1/V2: Custom CUDA kernels for modular Gaussian elimination")
    print("   - Hensel: Uses cuBLAS/MKL with optimized GEMM (tensor cores, etc.)")
    print()

    print("4. Why IML wins at n≥256:")
    print("   - Our ratio of operations (V1/Hensel) is ~15-20x")
    print("   - GPU parallelism only compensates ~5-10x")
    print("   - IML's BLAS is highly optimized (near-peak FLOPS)")
    print()

    print("5. V2 Improvement Strategy:")
    print("   a) 62-bit primes: Halves prime count (2x improvement)")
    print("   b) cuBLAS batched LU: Replace custom kernels (~3-5x improvement)")
    print("   c) Better memory layout: Coalesced access patterns")
    print("   d) Fused kernels: Reduce memory bandwidth")
    print()
    print("   Combined potential: 6-10x improvement over V1")
    print()

    print("6. Alternative: GPU Hensel Lifting")
    print("   - Implement p-adic lifting on GPU")
    print("   - Use cuBLAS for matrix operations")
    print("   - Custom kernels only for modular reduction")
    print("   - Would match IML's algorithmic efficiency + GPU parallelism")
    print()

def main():
    analyze_comparison()

    print("=" * 100)
    print("Concrete Recommendations for V2")
    print("=" * 100)
    print()
    print("Phase 1: Quick wins (1-2 days)")
    print("  1. Add 64-bit modular arithmetic kernels")
    print("  2. Switch to 62-bit primes")
    print("  3. Expected improvement: ~2x")
    print()
    print("Phase 2: cuBLAS integration (1 week)")
    print("  1. Use cuBLAS batched LU (cublasSgetrfBatched)")
    print("  2. Convert modular matrices to float, then back")
    print("  3. Expected improvement: ~3-5x")
    print()
    print("Phase 3: Hensel on GPU (2-3 weeks)")
    print("  1. Implement full p-adic lifting on GPU")
    print("  2. Use cuBLAS for GEMM operations")
    print("  3. Expected: Match or beat IML at all sizes")
    print()

if __name__ == '__main__':
    main()
