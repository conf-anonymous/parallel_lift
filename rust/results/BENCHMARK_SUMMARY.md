# Parallel Lift - GPU Benchmark Summary

**Date:** January 2026
**Platform:** Linux (Ubuntu 22.04), CUDA 12.4
**Rust Version:** 1.92.0

## Executive Summary

Parallel Lift achieves **up to 552× speedup** for exact integer linear algebra using CRT (Chinese Remainder Theorem) to convert precision into parallelism.

## Benchmark Results Overview

### 1. Full Pipeline (Multi-RHS Solve + CRT)

The core ZK preprocessing operation: solve AX = B with 16 RHS vectors.

| Matrix Size | Primes | CPU Time | GPU Time | **Speedup** |
|-------------|--------|----------|----------|-------------|
| 32×32 | 68 | 149 ms | 2.8 ms | **53×** |
| 64×64 | 136 | 2.2 s | 12 ms | **177×** |
| 128×128 | 273 | 33.7 s | 61 ms | **552×** |

### 2. CRT Reconstruction

Garner's algorithm for recovering BigInt from residues.

| Configuration | CPU Time | GPU Time | Speedup |
|---------------|----------|----------|---------|
| 128 primes × 4096 values | 277 ms | 11.5 ms | **24×** |
| 256 primes × 4096 values | 782 ms | 31.5 ms | **25×** |
| 512 primes × 4096 values | 2.6 s | 104 ms | **25×** |

### 3. Determinant (CPU Baseline)

| Matrix Size | Primes | Time (ms) |
|-------------|--------|-----------|
| 8×8 | 17 | 0.9 |
| 32×32 | 68 | 7.8 |
| 64×64 | 136 | 84 |
| 128×128 | 273 | 1059 |

### 4. LLL Lattice Reduction

**Status:** Working implementation using CPU exact rational arithmetic

After investigation, pure GPU LLL with CRT doesn't work because modular division gives different results than exact division. The hybrid implementation correctly converges but doesn't provide GPU speedup.

| n | bits | CPU LLL | Verified |
|---|------|---------|----------|
| 10 | 12 | 29 ms | ✓ |
| 20 | 16 | 813 ms | ✓ |
| 30 | 20 | 5.9 s | ✓ |

**Recommendation:** Use CPU `LLL::reduce()` for lattice reduction. See `GPU_LLL_STATUS.md` for details.

## Key Insights

### Why CRT Works for GPU

| Property | Benefit |
|----------|---------|
| **Fixed-width arithmetic** | Native u32 operations, no BigInt |
| **No data dependencies** | Each prime computed independently |
| **Predictable memory** | O(n²) per prime, no dynamic allocation |
| **Exact reconstruction** | Garner's algorithm is deterministic |

### Scaling Behavior

```
Speedup increases with problem size:

32×32:   ████████████████ 53×
64×64:   ████████████████████████████████████████████████████████ 177×
128×128: ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 552×
```

### GPU CRT is Critical

Without GPU CRT reconstruction:
- 128×128: ~70× speedup (CPU CRT bottleneck)

With GPU CRT reconstruction:
- 128×128: **552×** speedup

## Applicability

### Strong Use Cases

| Application | Why It Benefits |
|-------------|-----------------|
| **ZK Preprocessing** | Batched witness generation, k RHS vectors |
| **Multi-RHS Systems** | Factor A once, solve for k columns |
| **Matrix Inverse** | Compute A⁻¹ via AX = I (n RHS) |
| **Determinant** | Direct batched elimination |

### Limited Benefit

| Application | Why |
|-------------|-----|
| **Small matrices (n < 32)** | GPU dispatch overhead dominates |
| **Single RHS** | Less work to amortize |
| **RREF/Nullspace** | Row dependencies limit parallelism |
| **LLL Lattice Reduction** | Decision logic requires exact rational comparisons |

## Hardware Requirements

- **CUDA:** 12.0+ (tested with 12.4)
- **GPU Memory:** ~4MB per prime for 128×128 matrix
- **Recommended:** NVIDIA RTX 3090/4090 or datacenter GPU

## Files in This Directory

| File | Description |
|------|-------------|
| `GPU_CRT_BENCHMARK.md` | CRT reconstruction benchmark details |
| `GPU_FULL_PIPELINE_BENCHMARK.md` | Full pipeline (solve + CRT) results |
| `GPU_LLL_STATUS.md` | LLL implementation status and issues |
| `gpu_crt_benchmark.txt` | Raw CRT benchmark output |
| `gpu_full_pipeline_benchmark.txt` | Raw pipeline benchmark output |
| `determinant_benchmark.txt` | Raw determinant benchmark output |
| `multi_rhs_cuda_benchmark.txt` | Raw multi-RHS benchmark output |

## Running Benchmarks

```bash
# Build with CUDA support
cd rust
cargo build --release --features cuda

# Run benchmarks
./target/release/parallel-lift gpu-crt-bench --max-primes 512 --max-values 4096
./target/release/parallel-lift gpu-full-bench --max-size 128 --k 16
./target/release/parallel-lift benchmark --bench det --max-size 128
./target/release/parallel-lift benchmark --bench multi-rhs --max-size 128 --backend cuda

# LLL benchmarks (note: convergence issues)
./target/release/parallel-lift lll-bench --max-dim 30 --max-bits 20
```

## Conclusion

The CRT-lifted approach to GPU-accelerated exact arithmetic achieves remarkable speedups (up to **552×**) for the core operations in ZK preprocessing. The technique converts the precision problem (coefficient explosion) into a parallelism opportunity (independent modular computations).

For LLL lattice reduction, a hybrid CPU/GPU approach is recommended to handle the decision logic that requires actual value comparisons while still leveraging GPU parallelism for the expensive Gram-Schmidt computation.
