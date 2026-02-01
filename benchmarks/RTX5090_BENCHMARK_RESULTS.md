# Parallel Lift - RTX 5090 Benchmark Results

**Date:** January 31, 2026
**Machine:** RTX 5090 Development System

## System Configuration

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce RTX 5090, 32607 MiB VRAM |
| Driver | 570.195.03 |
| CUDA | 12.8 |
| CPU | AMD EPYC 7543 32-Core Processor |
| RAM | 944 GB |
| OS | Linux 6.14.0-35-generic (Ubuntu 24.04) |

---

## 1. Main Performance Results: GPU Full Pipeline

**Best headline result: 1565.9x speedup at n=256**

| Size | Primes | CPU Total | GPU+CPU CRT | GPU+GPU CRT | Speedup |
|------|--------|-----------|-------------|-------------|---------|
| 32 | 68 | 165.1ms | 33.6ms | 2.7ms | **61.5x** |
| 64 | 136 | 2317.1ms | 111.4ms | 11.7ms | **198.9x** |
| 128 | 273 | 34245.3ms | 476.6ms | 56.4ms | **607.6x** |
| 256 | 546 | 546467.4ms | 3035.1ms | 349.0ms | **1565.9x** |

---

## 2. Hensel (Dixon) p-adic Lifting vs CRT vs IML

**Key Finding: Hensel beats IML at ALL sizes, beats CRT at n >= 128**

| n | Hensel (ms) | CRT (ms) | IML (ms) | vs CRT | vs IML | Iters |
|---|-------------|----------|----------|--------|--------|-------|
| 32 | 2.33 | 1.09 | 6.55 | 0.47x | **2.81x** | 5 |
| 64 | 9.68 | 7.05 | 15.09 | 0.73x | **1.56x** | 6 |
| 128 | 43.49 | 54.52 | 63.40 | **1.25x** | **1.46x** | 7 |
| 256 | 257.34 | 338.27 | 318.92 | **1.31x** | **1.24x** | 8 |

### Hensel Timing Breakdown (n=256)
- Inverse computation: 126.4ms (49%)
- GPU lifting (8 iterations): 127.3ms (49%)
- BigInt reconstruction: 3.3ms (1%)

---

## 3. GPU CRT Reconstruction

| Primes | Values | CPU (ms) | GPU (ms) | Speedup | Verified |
|--------|--------|----------|----------|---------|----------|
| 32 | 256 | 3.84 | 8.64 | 0.4x | Yes |
| 64 | 256 | 8.33 | 1.72 | 4.9x | Yes |
| 64 | 1024 | 38.87 | 2.06 | **18.9x** | Yes |
| 128 | 1024 | 86.62 | 5.74 | 15.1x | Yes |
| 128 | 4096 | 278.33 | 8.68 | **32.1x** | Yes |
| 256 | 1024 | 183.60 | 20.37 | 9.0x | Yes |
| 256 | 4096 | 742.85 | 27.19 | **27.3x** | Yes |
| 512 | 4096 | 2512.42 | 89.04 | **28.2x** | Yes |

### CRT Breakdown (512 primes x 4096 values)
- Precompute: 1.99ms
- Upload: 8.87ms
- Kernel: 70.94ms
- Download: 8.87ms

---

## 4. ZK Preprocessing Scenarios

### Merkle Ledger Witness (n=128, k=16)
| Backend | Total Time | Speedup |
|---------|------------|---------|
| CPU | 4788.5ms | 1x |
| CUDA | 81.4ms | **58.8x** |

### Sparse R1CS Constraints (n=128, k=16)
| Backend | Total Time | Speedup |
|---------|------------|---------|
| CPU | 552.5ms | 1x |
| CUDA | 46.8ms | **11.8x** |

---

## 5. Pipeline Optimization

| n | k | Sequential (ms) | Async Pipeline (ms) | Speedup | Overlap |
|---|---|-----------------|---------------------|---------|---------|
| 16 | 8 | 5.99 | 2.31 | 2.59x | 41.7% |
| 32 | 8 | 35.76 | 6.58 | 5.44x | 58.8% |
| 64 | 8 | 260.65 | 37.85 | 6.89x | 57.0% |
| 128 | 8 | 2028.10 | 270.96 | **7.48x** | 54.2% |

---

## 6. PCIe Transfer Analysis

| n | Primes | Prepare | H->D | Compute | D->H | Total | Transfer% |
|---|--------|---------|------|---------|------|-------|-----------|
| 64 | 136 | 2.8ms | 1.8ms | 2.1ms | 0.1ms | 6.7ms | 28.0% |
| 128 | 273 | 19.8ms | 22.5ms | 8.8ms | 1.6ms | 52.7ms | 45.6% |
| 256 | 546 | 154.6ms | 117.2ms | 62.8ms | 4.0ms | 338.6ms | 35.8% |
| 512 | 1092 | 1192.1ms | 942.9ms | 809.7ms | 6.7ms | 2951.4ms | 32.2% |

---

## 7. Sparse vs Dense Solvers

| n | NNZ | Sparsity | Dense (ms) | Sparse (ms) | Speedup |
|---|-----|----------|------------|-------------|---------|
| 32 | 152 | 85.2% | 1.34 | 0.54 | 2.50x |
| 64 | 309 | 92.5% | 9.65 | 8.70 | 1.11x |
| 128 | 624 | 96.2% | 78.26 | 46.66 | 1.68x |
| 256 | 1256 | 98.1% | 550.09 | 252.88 | 2.18x |
| 512 | 2518 | 99.0% | 4454.73 | 1747.78 | **2.55x** |

---

## 8. V2 Benchmark: 31-bit vs 62-bit Primes

| n | V1 Primes | V2 Primes | V1 GPU | V2 GPU | Prime Reduction |
|---|-----------|-----------|--------|--------|-----------------|
| 32 | 68 | 34 | 1.04ms | 6.16ms | 2.00x |
| 64 | 136 | 68 | 5.43ms | 15.62ms | 2.00x |
| 128 | 273 | 137 | 40.34ms | 49.40ms | 1.99x |
| 256 | 546 | 273 | 341.48ms | 318.97ms | 2.00x |

Note: 62-bit primes reduce prime count by 2x but don't provide speedup on current GPU due to 64-bit arithmetic overhead.

---

## 9. LLL Lattice Reduction

LLL reduction shows CPU outperforming GPU on current implementation:

| n | bits | CPU (ms) | GPU (ms) | Speedup |
|---|------|----------|----------|---------|
| 20 | 16 | 1063.78 | 2994.81 | 0.36x |
| 30 | 16 | 4608.54 | 11798.25 | 0.39x |
| 40 | 20 | 22043.82 | 69363.59 | 0.32x |

Note: LLL is inherently sequential (swap-based), limiting GPU parallelism.

---

## 10. Multi-RHS Solve Scaling (CPU vs GPU)

| Size | CPU Total | GPU Total | GPU Solve | GPU CRT | Speedup |
|------|-----------|-----------|-----------|---------|---------|
| 8 | 1.30ms | 8.94ms | 8.25ms | 0.66ms | 0.15x |
| 16 | 10.50ms | 5.36ms | 2.38ms | 2.92ms | 1.96x |
| 32 | 143.46ms | 15.51ms | 1.16ms | 14.23ms | 9.25x |
| 64 | 2115.81ms | 76.57ms | 5.28ms | 70.89ms | **27.6x** |
| 128 | 33485.76ms | 464.36ms | 31.20ms | 431.10ms | **72.1x** |
| 256 | 539061.72ms | 3129.12ms | 233.11ms | 2887.06ms | **172.3x** |

Note: GPU CRT in this benchmark uses CPU reconstruction. With GPU CRT reconstruction (see Section 1), speedups reach 1565x.

---

## Summary of Key Results

### Headline Numbers for Paper

| Metric | Value |
|--------|-------|
| **Max GPU Speedup (full pipeline)** | 1565.9x at n=256 |
| **Max GPU CRT Speedup** | 32.1x (128 primes, 4096 values) |
| **Max Async Pipeline Speedup** | 7.48x at n=128 |
| **ZK Merkle Witness Speedup** | 58.8x at n=128 |
| **Hensel vs IML (all sizes)** | 1.24x - 2.81x faster |
| **Hensel vs CRT (n>=128)** | 1.25x - 1.31x faster |

### Algorithm Comparison

| Approach | Best Use Case | Limitation |
|----------|---------------|------------|
| CRT (multi-prime) | Small n, embarrassingly parallel | Prime count scales with n |
| Hensel (p-adic) | Large n (>=128) | Sequential lifting iterations |
| Sparse Wiedemann | High sparsity (>95%) | Dense matrices |

---

## Files Generated

- `paper_results/rtx5090_gpu_full.csv` - Full pipeline results
- `paper_results/rtx5090_gpu_crt.csv` - GPU CRT results
- `paper_results/rtx5090_hensel.csv` - Hensel benchmark results
- `paper_results/rtx5090_sweep/` - Comprehensive sweep data
