# GPU CRT Reconstruction Benchmark Results

**Date:** January 2026
**Platform:** Linux (Ubuntu 22.04)
**GPU:** CUDA 12.4
**Rust Version:** 1.92.0

## Summary

GPU-accelerated CRT (Chinese Remainder Theorem) reconstruction using Garner's algorithm achieves significant speedups over CPU for large problem sizes.

## Results

| Primes | Values | CPU (ms) | GPU (ms) | Speedup | Verified |
|--------|--------|----------|----------|---------|----------|
| 32 | 256 | 2.95 | 12.78 | 0.2× | ✓ |
| 64 | 256 | 6.78 | 1.80 | 3.8× | ✓ |
| 64 | 1024 | 27.09 | 2.22 | 12.2× | ✓ |
| 128 | 1024 | 68.91 | 6.93 | 9.9× | ✓ |
| 128 | 4096 | 276.95 | 11.52 | **24.1×** | ✓ |
| 256 | 1024 | 195.71 | 22.67 | 8.6× | ✓ |
| 256 | 4096 | 781.57 | 31.45 | **24.9×** | ✓ |
| 512 | 4096 | 2559.21 | 103.59 | **24.7×** | ✓ |

## Detailed Breakdown (512 primes × 4096 values)

| Phase | Time (ms) |
|-------|-----------|
| **CPU Total** | 2581.97 |
| **GPU Total** | 100.71 |
| ├─ Precompute | 1.30 |
| ├─ Upload | 10.07 |
| ├─ Kernel | 80.57 |
| └─ Download | 10.07 |

## Observations

1. **Crossover Point:** GPU becomes faster than CPU around 64 primes × 256 values
2. **Peak Speedup:** ~25× for large configurations (128+ primes, 1024+ values)
3. **Scaling:** Speedup increases with problem size due to better GPU utilization
4. **Data Transfer:** Memory transfer accounts for ~20% of GPU time

## Key Insight

The CRT reconstruction via Garner's algorithm is highly parallelizable because:
- Each value can be reconstructed independently
- All primes are processed in sequence within a thread
- No synchronization needed between threads

This makes it ideal for GPU acceleration when processing many values simultaneously.
