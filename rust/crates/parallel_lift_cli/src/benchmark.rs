//! Scaling benchmarks
//!
//! Runs benchmarks across a range of sizes.

use crate::{BackendChoice, BenchType};
use num_bigint::BigInt;
use parallel_lift_core::{CRTBasis, CpuBackend, Determinant, PrimeGenerator, Solver};
use rand::Rng;
use std::path::PathBuf;

#[cfg(target_os = "macos")]
use parallel_lift_metal::MetalBackend;

#[cfg(feature = "cuda")]
use parallel_lift_cuda::CudaBackend;

/// Run scaling benchmark
pub fn run_scaling_benchmark(
    bench: BenchType,
    max_size: usize,
    backend: BackendChoice,
    export: Option<PathBuf>,
) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║             Parallel Lift - Scaling Benchmark                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let bench_name = match bench {
        BenchType::Det => "Determinant",
        BenchType::Solve => "Single-RHS Solve",
        BenchType::MultiRhs => "Multi-RHS Solve",
        BenchType::Inverse => "Matrix Inverse",
    };

    let backend_name = match backend {
        BackendChoice::Cpu => "CPU",
        #[cfg(target_os = "macos")]
        BackendChoice::Metal => "Metal GPU",
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => "CUDA GPU",
    };

    println!("Benchmark: {}", bench_name);
    println!("Backend:   {}", backend_name);
    println!("Max size:  {}", max_size);
    println!();

    // Size progression
    let sizes: Vec<usize> = [8, 16, 32, 64, 128, 256, 512]
        .into_iter()
        .filter(|&s| s <= max_size)
        .collect();

    let mut results = Vec::new();

    println!("┌─────────┬────────────┬────────────┬────────────┬──────────┐");
    println!("│  Size   │  Total(ms) │  Solve(ms) │   CRT(ms)  │  Primes  │");
    println!("├─────────┼────────────┼────────────┼────────────┼──────────┤");

    for &n in &sizes {
        let result = run_single_benchmark(bench, n, backend);
        println!(
            "│ {:>7} │ {:>10.3} │ {:>10.3} │ {:>10.3} │ {:>8} │",
            n, result.total_ms, result.solve_ms, result.crt_ms, result.num_primes
        );
        results.push((n, result));
    }

    println!("└─────────┴────────────┴────────────┴────────────┴──────────┘");

    // Export if requested
    if let Some(path) = export {
        export_benchmark_results(&path, &results);
        println!("\nResults exported to: {}", path.display());
    }
}

#[derive(Debug, Clone)]
struct BenchResult {
    num_primes: usize,
    total_ms: f64,
    solve_ms: f64,
    crt_ms: f64,
}

fn run_single_benchmark(bench: BenchType, n: usize, backend: BackendChoice) -> BenchResult {
    let mut rng = rand::thread_rng();

    // Generate random matrix
    let matrix: Vec<BigInt> = (0..n * n)
        .map(|i| {
            let val = if i / n == i % n {
                rng.gen_range(100..1000) // Diagonal dominance
            } else {
                rng.gen_range(-50..50i32)
            };
            BigInt::from(val)
        })
        .collect();

    // Ensure non-singular
    let mut matrix = matrix;
    for i in 0..n {
        matrix[i * n + i] = &matrix[i * n + i] + BigInt::from(n as i64 * 2);
    }

    // Generate RHS
    let b: Vec<BigInt> = (0..n)
        .map(|_| BigInt::from(rng.gen_range(-100..100i32)))
        .collect();

    // Estimate primes needed
    let num_primes = (n * 64 / 30).max(4);
    let primes = PrimeGenerator::generate_31bit_primes(num_primes);
    let basis = CRTBasis::new(primes);

    match bench {
        BenchType::Det => run_det_benchmark(&matrix, n, &basis, backend),
        BenchType::Solve => run_solve_benchmark(&matrix, &b, n, &basis, backend),
        BenchType::MultiRhs => {
            let k = 16;
            let b_cols: Vec<Vec<BigInt>> = (0..k)
                .map(|_| {
                    (0..n)
                        .map(|_| BigInt::from(rng.gen_range(-100..100i32)))
                        .collect()
                })
                .collect();
            run_multi_rhs_benchmark(&matrix, &b_cols, n, k, &basis, backend)
        }
        BenchType::Inverse => run_inverse_benchmark(&matrix, n, &basis, backend),
    }
}

fn run_det_benchmark(
    matrix: &[BigInt],
    n: usize,
    basis: &CRTBasis,
    backend: BackendChoice,
) -> BenchResult {
    let (_, timings) = match backend {
        BackendChoice::Cpu => {
            let det = Determinant::new(CpuBackend::new());
            det.compute(matrix, n, basis)
        }
        #[cfg(target_os = "macos")]
        BackendChoice::Metal => {
            if let Some(metal) = MetalBackend::new() {
                let det = Determinant::new(metal);
                det.compute(matrix, n, basis)
            } else {
                let det = Determinant::new(CpuBackend::new());
                det.compute(matrix, n, basis)
            }
        }
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => {
            if let Some(cuda) = CudaBackend::new() {
                let det = Determinant::new(cuda);
                det.compute(matrix, n, basis)
            } else {
                let det = Determinant::new(CpuBackend::new());
                det.compute(matrix, n, basis)
            }
        }
    };

    BenchResult {
        num_primes: timings.num_primes,
        total_ms: timings.total_time * 1000.0,
        solve_ms: timings.det_time * 1000.0,
        crt_ms: timings.crt_time * 1000.0,
    }
}

fn run_solve_benchmark(
    matrix: &[BigInt],
    b: &[BigInt],
    n: usize,
    basis: &CRTBasis,
    backend: BackendChoice,
) -> BenchResult {
    let (_, timings) = match backend {
        BackendChoice::Cpu => {
            let solver = Solver::new(CpuBackend::new());
            solver.solve(matrix, b, basis)
        }
        #[cfg(target_os = "macos")]
        BackendChoice::Metal => {
            if let Some(metal) = MetalBackend::new() {
                let solver = Solver::new(metal);
                solver.solve(matrix, b, basis)
            } else {
                let solver = Solver::new(CpuBackend::new());
                solver.solve(matrix, b, basis)
            }
        }
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => {
            if let Some(cuda) = CudaBackend::new() {
                let solver = Solver::new(cuda);
                solver.solve(matrix, b, basis)
            } else {
                let solver = Solver::new(CpuBackend::new());
                solver.solve(matrix, b, basis)
            }
        }
    };

    BenchResult {
        num_primes: timings.num_primes,
        total_ms: timings.total_time * 1000.0,
        solve_ms: timings.solve_time * 1000.0,
        crt_ms: timings.crt_time * 1000.0,
    }
}

fn run_multi_rhs_benchmark(
    matrix: &[BigInt],
    b_cols: &[Vec<BigInt>],
    n: usize,
    k: usize,
    basis: &CRTBasis,
    backend: BackendChoice,
) -> BenchResult {
    let (_, timings) = match backend {
        BackendChoice::Cpu => {
            let solver = Solver::new(CpuBackend::new());
            solver.solve_multi_rhs(matrix, b_cols, basis)
        }
        #[cfg(target_os = "macos")]
        BackendChoice::Metal => {
            if let Some(metal) = MetalBackend::new() {
                let solver = Solver::new(metal);
                solver.solve_multi_rhs(matrix, b_cols, basis)
            } else {
                let solver = Solver::new(CpuBackend::new());
                solver.solve_multi_rhs(matrix, b_cols, basis)
            }
        }
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => {
            if let Some(cuda) = CudaBackend::new() {
                let solver = Solver::new(cuda);
                solver.solve_multi_rhs(matrix, b_cols, basis)
            } else {
                let solver = Solver::new(CpuBackend::new());
                solver.solve_multi_rhs(matrix, b_cols, basis)
            }
        }
    };

    BenchResult {
        num_primes: timings.num_primes,
        total_ms: timings.total_time * 1000.0,
        solve_ms: timings.solve_time * 1000.0,
        crt_ms: timings.crt_time * 1000.0,
    }
}

fn run_inverse_benchmark(
    matrix: &[BigInt],
    n: usize,
    basis: &CRTBasis,
    backend: BackendChoice,
) -> BenchResult {
    // Matrix inverse via solving AX = I
    let mut rng = rand::thread_rng();

    // Create identity columns
    let identity_cols: Vec<Vec<BigInt>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if i == j { BigInt::from(1) } else { BigInt::from(0) })
                .collect()
        })
        .collect();

    let (_, timings) = match backend {
        BackendChoice::Cpu => {
            let solver = Solver::new(CpuBackend::new());
            solver.solve_multi_rhs(matrix, &identity_cols, basis)
        }
        #[cfg(target_os = "macos")]
        BackendChoice::Metal => {
            if let Some(metal) = MetalBackend::new() {
                let solver = Solver::new(metal);
                solver.solve_multi_rhs(matrix, &identity_cols, basis)
            } else {
                let solver = Solver::new(CpuBackend::new());
                solver.solve_multi_rhs(matrix, &identity_cols, basis)
            }
        }
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => {
            if let Some(cuda) = CudaBackend::new() {
                let solver = Solver::new(cuda);
                solver.solve_multi_rhs(matrix, &identity_cols, basis)
            } else {
                let solver = Solver::new(CpuBackend::new());
                solver.solve_multi_rhs(matrix, &identity_cols, basis)
            }
        }
    };

    BenchResult {
        num_primes: timings.num_primes,
        total_ms: timings.total_time * 1000.0,
        solve_ms: timings.solve_time * 1000.0,
        crt_ms: timings.crt_time * 1000.0,
    }
}

fn export_benchmark_results(path: &PathBuf, results: &[(usize, BenchResult)]) {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path).expect("Failed to create export file");
    writeln!(file, "size,num_primes,total_ms,solve_ms,crt_ms").unwrap();
    for (size, result) in results {
        writeln!(
            file,
            "{},{},{:.6},{:.6},{:.6}",
            size, result.num_primes, result.total_ms, result.solve_ms, result.crt_ms
        )
        .unwrap();
    }
}
