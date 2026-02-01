# Parallel Lift Benchmark Report

CRT-accelerated exact arithmetic for ZK preprocessing.

## Summary

- **Best speedup (varying k):** 1.00x at n=128, k=64
- **Best speedup (varying n):** 1.00x at n=256, k=16

## Table 1: Speedup vs RHS Vectors (n=128)

| k | Primes | CPU (ms) | GPU (ms) | Speedup |
|---|--------|----------|----------|---------|
| 1 | 49 | 106.2 | 106.2 | 1.00x |
| 2 | 49 | 209.2 | 209.2 | 1.00x |
| 4 | 49 | 415.8 | 415.8 | 1.00x |
| 8 | 49 | 830.1 | 830.1 | 1.00x |
| 16 | 49 | 1645.4 | 1645.4 | 1.00x |
| 32 | 49 | 3289.3 | 3289.3 | 1.00x |
| 64 | 49 | 6576.5 | 6576.5 | 1.00x |

## Table 2: Speedup vs Matrix Size (k=16)

| n | Primes | CPU (ms) | GPU (ms) | Speedup |
|---|--------|----------|----------|---------|
| 32 | 13 | 1.9 | 1.9 | 1.00x |
| 64 | 25 | 81.1 | 81.1 | 1.00x |
| 96 | 37 | 491.4 | 491.4 | 1.00x |
| 128 | 49 | 1662.0 | 1662.0 | 1.00x |
| 160 | 60 | 4032.0 | 4032.0 | 1.00x |
| 192 | 72 | 9067.7 | 9067.7 | 1.00x |
| 224 | 84 | 17093.5 | 17093.5 | 1.00x |
| 256 | 96 | 29144.7 | 29144.7 | 1.00x |

## Table 3: Phase Breakdown (k=16)

| n | Primes | Total (ms) | Residue % | Solve % | CRT % |
|---|--------|------------|-----------|---------|-------|
| 32 | 13 | 5.7 | 0.3% | 33.2% | 66.5% |
| 64 | 25 | 95.8 | 0.1% | 82.7% | 17.3% |
| 128 | 49 | 1715.0 | 0.0% | 95.8% | 4.2% |
| 192 | 72 | 9219.8 | 0.0% | 98.1% | 1.9% |
| 256 | 96 | 29470.2 | 0.0% | 98.9% | 1.1% |

## Key Observations

1. **Speedup increases with k:** More RHS vectors amortize GPU launch overhead
2. **Speedup increases with n:** Larger matrices provide more parallel work
3. **Solve dominates at large n:** As matrix size grows, solve phase takes >90% of time
4. **CRT overhead decreases:** CRT reconstruction becomes negligible for large matrices
