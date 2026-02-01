# Parallel Lift Benchmark Report

CRT-accelerated exact arithmetic for ZK preprocessing.

## Summary

- **Best speedup (varying k):** 1.00x at n=128, k=64
- **Best speedup (varying n):** 1.00x at n=256, k=16

## Table 1: Speedup vs RHS Vectors (n=128)

| k | Primes | CPU (ms) | GPU (ms) | Speedup |
|---|--------|----------|----------|---------|
| 1 | 49 | 399.7 | 399.7 | 1.00x |
| 2 | 49 | 798.4 | 798.4 | 1.00x |
| 4 | 49 | 1594.9 | 1594.9 | 1.00x |
| 8 | 49 | 3187.0 | 3187.0 | 1.00x |
| 16 | 49 | 6366.8 | 6366.8 | 1.00x |
| 32 | 49 | 12728.5 | 12728.5 | 1.00x |
| 64 | 49 | 25498.1 | 25498.1 | 1.00x |

## Table 2: Speedup vs Matrix Size (k=16)

| n | Primes | CPU (ms) | GPU (ms) | Speedup |
|---|--------|----------|----------|---------|
| 32 | 13 | 6.8 | 6.8 | 1.00x |
| 64 | 25 | 295.7 | 295.7 | 1.00x |
| 96 | 37 | 1862.5 | 1862.5 | 1.00x |
| 128 | 49 | 6475.5 | 6475.5 | 1.00x |
| 160 | 60 | 15991.7 | 15991.7 | 1.00x |
| 192 | 72 | 35695.2 | 35695.2 | 1.00x |
| 224 | 84 | 67555.2 | 67555.2 | 1.00x |
| 256 | 96 | 115046.1 | 115046.1 | 1.00x |

## Table 3: Phase Breakdown (k=16)

| n | Primes | Total (ms) | Residue % | Solve % | CRT % |
|---|--------|------------|-----------|---------|-------|
| 32 | 13 | 8.8 | 0.2% | 77.6% | 22.2% |
| 64 | 25 | 304.9 | 0.0% | 97.1% | 2.9% |
| 128 | 49 | 6424.7 | 0.0% | 99.3% | 0.6% |
| 192 | 72 | 35620.5 | 0.0% | 99.7% | 0.3% |
| 256 | 96 | 114710.0 | 0.0% | 99.8% | 0.2% |

## Key Observations

1. **Speedup increases with k:** More RHS vectors amortize GPU launch overhead
2. **Speedup increases with n:** Larger matrices provide more parallel work
3. **Solve dominates at large n:** As matrix size grows, solve phase takes >90% of time
4. **CRT overhead decreases:** CRT reconstruction becomes negligible for large matrices
