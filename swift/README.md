# Parallel Lift - Swift/Metal Implementation

Production Swift implementation of CRT-accelerated exact arithmetic on Apple Silicon.

## Building

```bash
swift build -c release
```

## Running Benchmarks

```bash
# Default determinant scaling
.build/release/parallel-lift

# Specific benchmarks
.build/release/parallel-lift --solve           # Single-RHS solve
.build/release/parallel-lift --multi-rhs       # Multi-RHS solve (AX = B)
.build/release/parallel-lift --inverse         # Matrix inverse
.build/release/parallel-lift --batched-inverse # Batched inverse via solve
.build/release/parallel-lift --nullspace       # Nullspace computation
.build/release/parallel-lift --overdetermined  # Overdetermined systems
.build/release/parallel-lift --fhe             # FHE primitives
```

## Key Files

- `Sources/ParallelLift/main.swift` - Benchmark entry point
- `Sources/ParallelLift/GPUDeterminant.swift` - GPU determinant via CRT
- `Sources/ParallelLift/GPUSolve.swift` - GPU linear system solver
- `Sources/ParallelLift/GPUMultiRHSSolve.swift` - Multi-RHS solver
- `Sources/ParallelLift/GPUInverse.swift` - Batched inverse
- `Sources/ParallelLift/CRTReconstruction.swift` - Garner's algorithm

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Swift 5.9+
- Metal framework
