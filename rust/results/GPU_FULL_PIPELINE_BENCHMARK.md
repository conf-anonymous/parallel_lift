# GPU Full Pipeline Benchmark Results

**Date:** January 2026
**Platform:** Linux (Ubuntu 22.04)
**GPU:** CUDA 12.4
**Rust Version:** 1.92.0

## Summary

The full GPU pipeline combines:
1. **GPU Modular Solve** - Batched Gaussian elimination across all primes
2. **GPU CRT Reconstruction** - Garner's algorithm to recover exact BigInt results

This is the core operation for ZK preprocessing with multi-RHS linear systems.

## Configuration

- **k (RHS vectors):** 16
- **Operation:** Solve AX = B where A is n×n and B has k columns

## Results

| Size (n) | Primes | CPU Total | GPU+CPU CRT | GPU+GPU CRT | Speedup |
|----------|--------|-----------|-------------|-------------|---------|
| 32 | 68 | 148.7 ms | 29.0 ms | **2.8 ms** | **53.4×** |
| 64 | 136 | 2163.4 ms | 81.9 ms | **12.2 ms** | **177.1×** |
| 128 | 273 | 33722.6 ms | 479.5 ms | **61.1 ms** | **552.1×** |

## Scaling Analysis

```
Speedup vs Matrix Size:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
n=32:   ████████████████ 53×
n=64:   ████████████████████████████████████████████████████████ 177×
n=128:  ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 552×
```

## Key Observations

1. **Superlinear Speedup:** The speedup increases faster than linearly with problem size
   - 32×32: 53× speedup
   - 64×64: 177× speedup (3.3× better than n=32)
   - 128×128: 552× speedup (3.1× better than n=64)

2. **GPU CRT Critical:** The GPU CRT reconstruction is essential for achieving high speedups
   - Without GPU CRT: 70× speedup at n=128
   - With GPU CRT: **552×** speedup at n=128

3. **Compute Bound:** At larger sizes, the GPU solve becomes the bottleneck rather than CRT

## Comparison: CPU vs GPU Components

### n=128 Breakdown

| Component | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Modular Solve | ~33,000 ms | ~36 ms | ~900× |
| CRT Reconstruction | ~700 ms | ~25 ms | ~28× |
| **Total** | **33,722 ms** | **61 ms** | **552×** |

## Applicability

This benchmark demonstrates the viability of CRT-lifted GPU acceleration for:
- **ZK Preprocessing:** Multi-witness generation
- **Constraint Solving:** R1CS, Plonk, Groth16 setup
- **Polynomial Commitment:** KZG/Kate commitment preprocessing
- **Matrix Inverse:** Computing A⁻¹ via AX = I

## Hardware Utilization

The high speedups are achieved through:
1. **Massive Parallelism:** 273 primes × 128² matrix elements = 4.5M independent operations
2. **Fixed-Width Arithmetic:** All operations use native 32-bit integers
3. **Memory Coalescing:** Contiguous memory access patterns for matrices
4. **Minimal Synchronization:** Each prime's computation is independent
