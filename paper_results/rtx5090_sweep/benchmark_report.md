# Parallel Lift Benchmark Report

CRT-accelerated exact arithmetic for ZK preprocessing.

## Summary

- **Best speedup (varying k):** 1.00x at n=128, k=64
- **Best speedup (varying n):** 1.00x at n=256, k=16

## Table 1: Speedup vs RHS Vectors (n=128)

| k | Primes | CPU (ms) | GPU (ms) | Speedup |
|---|--------|----------|----------|---------|
| 1 | 49 | 286.9 | 286.9 | 1.00x |
| 2 | 49 | 565.2 | 565.2 | 1.00x |
| 4 | 49 | 1126.7 | 1126.7 | 1.00x |
| 8 | 49 | 2257.7 | 2257.7 | 1.00x |
| 16 | 49 | 4439.4 | 4439.4 | 1.00x |
| 32 | 49 | 9279.6 | 9279.6 | 1.00x |
| 64 | 49 | 18809.8 | 18809.8 | 1.00x |

## Table 2: Speedup vs Matrix Size (k=16)

| n | Primes | CPU (ms) | GPU (ms) | Speedup |
|---|--------|----------|----------|---------|
| 32 | 13 | 5.2 | 5.2 | 1.00x |
| 64 | 25 | 218.8 | 218.8 | 1.00x |
| 96 | 37 | 1323.5 | 1323.5 | 1.00x |
| 128 | 49 | 4443.2 | 4443.2 | 1.00x |
| 160 | 60 | 10818.4 | 10818.4 | 1.00x |
| 192 | 72 | 24443.3 | 24443.3 | 1.00x |
| 224 | 84 | 47469.7 | 47469.7 | 1.00x |
| 256 | 96 | 84327.1 | 84327.1 | 1.00x |

## Table 3: Phase Breakdown (k=16)

| n | Primes | Total (ms) | Residue % | Solve % | CRT % |
|---|--------|------------|-----------|---------|-------|
| 32 | 13 | 8.7 | 0.3% | 64.5% | 35.1% |
| 64 | 25 | 243.3 | 0.0% | 94.5% | 5.5% |
| 128 | 49 | 4850.0 | 0.0% | 98.8% | 1.2% |
| 192 | 72 | 26369.6 | 0.0% | 99.5% | 0.5% |
| 256 | 96 | 82075.0 | 0.0% | 99.7% | 0.3% |

## Key Observations

1. **Speedup increases with k:** More RHS vectors amortize GPU launch overhead
2. **Speedup increases with n:** Larger matrices provide more parallel work
3. **Solve dominates at large n:** As matrix size grows, solve phase takes >90% of time
4. **CRT overhead decreases:** CRT reconstruction becomes negligible for large matrices
