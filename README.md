# Parallel Lift

**GPU-Accelerated Exact Integer Arithmetic via CRT-Based Representation Transformation**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) [![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/) [![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit) [![Metal](https://img.shields.io/badge/Metal-Apple%20Silicon-silver.svg)](https://developer.apple.com/metal/)

## Abstract

Exact integer linear algebra is fundamental to cryptographic systems, symbolic computation, and zero-knowledge proof preprocessing, yet has long been considered poorly suited for GPU acceleration due to intermediate coefficient explosion. **Parallel Lift** challenges this assumption by demonstrating that the Chinese Remainder Theorem (CRT) transforms exact arithmetic from a precision problem into a parallelism opportunity.

Our approach achieves **up to 1,566× speedup** over optimized CPU baselines for multi-RHS linear system solving by executing thousands of independent modular operations in parallel across GPU threads, then reconstructing exact results via Garner's algorithm.

## Key Results

### Performance Summary (RTX 5090)

| Operation | Configuration | Speedup | Notes |
|-----------|---------------|---------|-------|
| **Multi-RHS Solve** | 256×256, k=16 | **1,566×** | Full GPU pipeline with GPU CRT |
| **Multi-RHS Solve** | 128×128, k=16 | **608×** | Full GPU pipeline with GPU CRT |
| **ZK Merkle Witness** | 128×128, k=16 | **59×** | Realistic ZK preprocessing |
| **GPU CRT Reconstruction** | 128×4096 values | **32×** | Isolated CRT speedup |
| **Hensel vs IML** | All sizes | **1.2–2.8×** | GPU Hensel lifting vs CPU IML |

### Scaling with Matrix Size (k=16 RHS, 48-bit entries)

| Matrix Size (n) | Primes | CPU Time | GPU Time | Speedup |
|-----------------|--------|----------|----------|---------|
| 32 | 68 | 165 ms | 2.7 ms | 61× |
| 64 | 136 | 2.3 s | 11.7 ms | 199× |
| 128 | 273 | 34.2 s | 56.4 ms | 608× |
| 256 | 546 | 546 s | 349 ms | **1,566×** |

### Metal Backend (Apple M3 Max)

| Matrix Size (n) | Primes | CPU Time | GPU Time | Speedup |
|-----------------|--------|----------|----------|---------|
| 64 | 59 | 0.23 s | 0.14 s | 1.7× |
| 128 | 108 | 4.81 s | 1.13 s | 4.3× |
| 192 | 159 | 31.2 s | 4.51 s | 6.9× |

## Core Insight

Traditional exact arithmetic suffers from *intermediate coefficient explosion*: solving an n×n integer system can produce results with O(n²) bits even when inputs have O(1) bits. Previous approaches attack this problem with better big-integer libraries or fraction-free algorithms.

**Parallel Lift takes a different approach**: rather than fighting coefficient growth, we *change the representation* to convert precision into parallelism.

### The Algorithm

1. **Decompose**: Map the problem into residue space using k coprime 31-bit primes, where k is determined by Hadamard bounds on result magnitude

2. **Solve**: Execute k independent modular Gaussian eliminations in parallel on GPU, each thread handles one prime with fixed 32-bit arithmetic

3. **Reconstruct**: Apply Garner's algorithm on GPU to recover the exact integer result from its residues

This decomposition exposes massive parallelism: for a 256×256 matrix requiring 546 primes, we dispatch 546 independent O(n³) computations simultaneously.

### Why CRT Enables GPU Acceleration

| Property | Benefit |
|----------|---------|
| **Fixed-width arithmetic** | 31-bit primes enable native u32 operations |
| **No data dependencies** | Each prime's computation is independent |
| **Predictable memory** | O(n²) per prime, no dynamic allocation |
| **Exact reconstruction** | Garner's algorithm is deterministic |

## Technical Contributions

1. **Representation over Algorithm**: Changing numerical representation (CRT) outperforms algorithmic optimization (fraction-free methods, better BigInt) for GPU targets

2. **Full GPU Pipeline**: Both modular solve and CRT reconstruction execute on GPU, eliminating CPU bottlenecks

3. **GPU Hensel Lifting**: For larger matrices, p-adic lifting reduces complexity from O(k·n³) to O(n³ + iter·n²), outperforming IML by 1.2–2.8×

4. **Multi-RHS Amortization**: Factoring A once and solving for k right-hand sides achieves k× work amplification

5. **Cross-Platform Implementation**: CUDA (NVIDIA) and Metal (Apple Silicon) backends sharing a common `Backend` trait

## Applicability

### Strong Use Cases

| Application | Why It Benefits |
|-------------|-----------------|
| **ZK Preprocessing** | Batched witness generation, constraint solving |
| **Multi-RHS Systems** | Factor once, solve k times |
| **Matrix Inverse** | Compute A⁻¹ via AX = I |
| **Determinant** | Direct application of batched elimination |

### Limited Benefit

| Application | Why |
|-------------|-----|
| **RREF/Nullspace** | Data-dependent control flow limits parallelism |
| **Small matrices (n < 32)** | GPU dispatch overhead dominates |
| **Single RHS** | Less work to amortize factorization |
| **LLL Lattice Reduction** | Decision logic requires actual values, not residues |

## Architecture

```
parallel_lift/
├── rust/
│   └── crates/
│       ├── parallel_lift_core/    # Backend trait, CRT, primes, reconstruction
│       ├── parallel_lift_metal/   # Metal GPU backend (Apple Silicon)
│       ├── parallel_lift_cuda/    # CUDA GPU backend (NVIDIA)
│       └── parallel_lift_cli/     # Benchmarks, ZK scenarios
├── swift/                         # Swift/Metal reference implementation
├── paper_results/                 # Benchmark data
├── benchmarks/                    # Benchmark scripts
└── paper/                         # Paper submission materials
```

### Backend Abstraction

```rust
pub trait Backend: Send + Sync {
    fn batch_determinant_mod(&self, matrix: &[u32], n: usize, primes: &[u32]) -> Vec<u32>;
    fn batch_solve_mod(&self, matrix: &[u32], b: &[u32], n: usize, primes: &[u32]) -> Option<Vec<Vec<u32>>>;
    fn batch_multi_rhs_solve_mod(&self, matrix: &[u32], b_cols: &[Vec<u32>], n: usize, k: usize, primes: &[u32]) -> Option<Vec<Vec<Vec<u32>>>>;
}
```

Implementations: `CpuBackend`, `MetalBackend`, `CudaBackend`

## Quick Start

### Prerequisites

- **Rust**: 1.75+ (`rustup` recommended)
- **CUDA**: 12.0+ with `nvcc` (for NVIDIA GPUs)
- **Metal**: Xcode Command Line Tools (for Apple Silicon)

### Build and Run

```bash
# Clone repository
git clone https://github.com/conf-anonymous/parallel_lift.git
cd parallel_lift/rust

# Build all crates
cargo build --release

# Run benchmark suite
cargo run --release --bin parallel-lift -- benchmark --bench multi-rhs --size 128 --rhs 16

# Compare backends
cargo run --release --bin parallel-lift -- benchmark --bench multi-rhs --size 128 --backend cpu
cargo run --release --bin parallel-lift -- benchmark --bench multi-rhs --size 128 --backend cuda

# Full GPU pipeline with GPU CRT
cargo run --release --bin parallel-lift -- gpu-full-bench --size 128 --rhs 16
```

## Benchmark Reproduction

All results in this repository are reproducible. The benchmark suite includes:

| Command | Description |
|---------|-------------|
| `benchmark --bench det` | Determinant scaling |
| `benchmark --bench solve` | Single-RHS solve |
| `benchmark --bench multi-rhs` | Multi-RHS solve (primary) |
| `benchmark --bench inverse` | Matrix inverse via multi-RHS |
| `gpu-crt-bench` | Isolated CRT reconstruction |
| `gpu-full-bench` | Full GPU pipeline |
| `zk-preprocess` | ZK-specific scenarios |

Results are exported to `paper_results/` in CSV format.

## Implementation Details

### GPU Kernels (CUDA)

**Modular Solve** (`modular_solve_multi_rhs`):
- One thread per prime handles entire elimination
- 31-bit primes with 64-bit intermediate products
- Partial pivoting for numerical stability

**CRT Reconstruction** (`crt_reconstruct_full`):
- Full Garner's algorithm on GPU
- One thread per output value
- Precomputed partial products and inverses

### Hensel (p-adic) Lifting

For larger matrices where multi-prime CRT requires many primes:
- Compute A⁻¹ mod p using GPU-accelerated modular inverse
- Iteratively lift solution with O(log B) iterations
- Complexity: O(n³ + iter·n²) vs O(k·n³) for multi-prime CRT

## Related Work

- **Fraction-free methods** (Bareiss): Avoid fractions but still O(n) bit growth
- **Modular algorithms** (Dixon, Pan): CRT for determinant, less explored for solve
- **GPU linear algebra** (cuBLAS, cuSOLVER): Floating-point focus
- **RNS in cryptography**: FHE, lattice-based schemes
- **IML library**: BLAS-based exact linear algebra (our Hensel lifting comparison baseline)

Parallel Lift extends modular techniques to full linear system solving with GPU acceleration, demonstrating practical speedups for ZK preprocessing workloads.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
