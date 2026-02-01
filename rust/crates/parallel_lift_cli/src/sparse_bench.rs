//! Sparse vs Dense Benchmark
//!
//! Compares Wiedemann sparse solver against dense Gaussian elimination
//! on R1CS-like constraint matrices.

use num_bigint::BigInt;
use parallel_lift_core::{
    SparseMatrix, WiedemannSolver, CRTBasis, PrimeGenerator, CpuBackend, Backend,
};
#[cfg(target_os = "macos")]
use parallel_lift_metal::{MetalBackend, GpuWiedemannSolver};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::path::PathBuf;
use std::time::Instant;
use std::fs::File;
use std::io::Write;

/// Result from sparse benchmark
#[derive(Debug, Clone)]
pub struct SparseBenchResult {
    pub n: usize,
    pub nnz: usize,
    pub sparsity: f64,
    pub k: usize,
    pub dense_ms: f64,
    pub sparse_ms: f64,
    pub speedup: f64,
    pub dense_correct: bool,
    pub sparse_correct: bool,
}

/// Run comprehensive sparse vs dense benchmark
pub fn run_sparse_benchmark(
    max_size: usize,
    nnz_per_row: usize,
    k: usize,
    export: Option<PathBuf>,
) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         Sparse vs Dense Solver Benchmark                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Configuration:");
    println!("  Max size:        {}", max_size);
    println!("  NNZ per row:     {} (R1CS-like)", nnz_per_row);
    println!("  RHS vectors (k): {}", k);
    println!();

    println!("┌────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Why Sparse Matters for ZK                                                             │");
    println!("├────────────────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Real R1CS matrices are ~99% sparse (3-5 nonzeros per row).                           │");
    println!("│ Dense: O(n³) operations, Sparse Wiedemann: O(n × nnz × iterations)                   │");
    println!("│ For n=1000, nnz≈5000: Dense = 1B ops, Sparse ≈ 10M ops (100× fewer)                  │");
    println!("└────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Test sizes - geometric progression
    let sizes: Vec<usize> = (0..)
        .map(|i| 32 * (1 << i))
        .take_while(|&s| s <= max_size)
        .collect();

    let mut results = Vec::new();

    println!("┌──────┬────────┬──────────┬─────┬────────────┬────────────┬─────────┬───────┬───────┐");
    println!("│   n  │   nnz  │ sparsity │  k  │  Dense(ms) │ Sparse(ms) │ Speedup │ D✓    │ S✓    │");
    println!("├──────┼────────┼──────────┼─────┼────────────┼────────────┼─────────┼───────┼───────┤");

    for n in sizes {
        let result = benchmark_size(n, nnz_per_row, k);
        println!(
            "│ {:>4} │ {:>6} │ {:>7.1}% │ {:>3} │ {:>10.2} │ {:>10.2} │ {:>6.2}x │ {:>5} │ {:>5} │",
            result.n,
            result.nnz,
            result.sparsity * 100.0,
            result.k,
            result.dense_ms,
            result.sparse_ms,
            result.speedup,
            if result.dense_correct { "✓" } else { "✗" },
            if result.sparse_correct { "✓" } else { "✗" },
        );
        results.push(result);
    }

    println!("└──────┴────────┴──────────┴─────┴────────────┴────────────┴─────────┴───────┴───────┘");

    // Summary
    println!();
    println!("Key Observations:");

    let best_speedup = results.iter().max_by(|a, b| a.speedup.partial_cmp(&b.speedup).unwrap());
    if let Some(best) = best_speedup {
        println!("  • Best sparse speedup: {:.2}x at n={}", best.speedup, best.n);
    }

    // Find crossover point
    let crossover = results.iter().find(|r| r.speedup > 1.0);
    if let Some(c) = crossover {
        println!("  • Sparse wins for n ≥ {}", c.n);
    }

    // Complexity analysis
    if results.len() >= 2 {
        let r1 = &results[0];
        let r2 = &results[results.len() - 1];
        let n_ratio = (r2.n as f64) / (r1.n as f64);
        let dense_ratio = r2.dense_ms / r1.dense_ms;
        let sparse_ratio = r2.sparse_ms / r1.sparse_ms;

        println!();
        println!("  Empirical Scaling (n: {} → {}):", r1.n, r2.n);
        println!("    Dense:  {:.1}x time for {:.1}x size (expected n³ = {:.1}x)",
                 dense_ratio, n_ratio, n_ratio.powi(3));
        println!("    Sparse: {:.1}x time for {:.1}x size (expected n×nnz ≈ {:.1}x)",
                 sparse_ratio, n_ratio, n_ratio.powi(2));
    }

    // Export if requested
    if let Some(path) = export {
        export_results(&path, &results);
        println!("\n✓ Results exported to: {}", path.display());
    }
}

fn benchmark_size(n: usize, nnz_per_row: usize, k: usize) -> SparseBenchResult {
    let seed = 42u64;
    let mut rng = StdRng::seed_from_u64(seed);

    // Generate R1CS-like sparse matrix
    let sparse = SparseMatrix::generate_r1cs_like(n, nnz_per_row, seed);

    // Convert to dense for comparison
    let dense = sparse_to_dense(&sparse);

    // Generate RHS vectors
    let b_cols: Vec<Vec<BigInt>> = (0..k)
        .map(|_| (0..n).map(|_| BigInt::from(rng.gen_range(-100..100i32))).collect())
        .collect();

    // Setup primes
    let entry_bits = 10;
    let output_bits = entry_bits * n + 64;
    let num_primes = (output_bits / 30).max(4);
    let primes = PrimeGenerator::generate_31bit_primes(num_primes);

    // Choose a single prime for this benchmark (comparing solver algorithms, not CRT)
    let p = primes[0];

    // Prepare modular data
    let sparse_mod = sparse.to_mod(p);
    let dense_mod: Vec<u32> = dense.iter()
        .map(|v| {
            let p_big = BigInt::from(p);
            let r = v % &p_big;
            if r < num_bigint::BigInt::from(0) {
                ((&r + &p_big) % &p_big).try_into().unwrap_or(0u32)
            } else {
                r.try_into().unwrap_or(0u32)
            }
        })
        .collect();

    let b_mod: Vec<Vec<u32>> = b_cols.iter()
        .map(|col| col.iter()
            .map(|v| {
                let p_big = BigInt::from(p);
                let r = v % &p_big;
                if r < num_bigint::BigInt::from(0) {
                    ((&r + &p_big) % &p_big).try_into().unwrap_or(0u32)
                } else {
                    r.try_into().unwrap_or(0u32)
                }
            })
            .collect())
        .collect();

    // Benchmark dense solver
    let dense_start = Instant::now();
    let dense_solutions: Vec<Option<Vec<u32>>> = b_mod.iter()
        .map(|b| {
            let cpu = CpuBackend::new();
            cpu.solve_mod(&dense_mod, b, n, p)
        })
        .collect();
    let dense_ms = dense_start.elapsed().as_secs_f64() * 1000.0;

    // Benchmark sparse Wiedemann solver
    let wiedemann = WiedemannSolver::new();
    let sparse_start = Instant::now();
    let sparse_solutions: Vec<Option<Vec<u32>>> = b_mod.iter()
        .map(|b| wiedemann.solve(&sparse_mod, b, p))
        .collect();
    let sparse_ms = sparse_start.elapsed().as_secs_f64() * 1000.0;

    // Verify correctness
    let dense_correct = dense_solutions.iter().all(|s| s.is_some());
    let sparse_correct = sparse_solutions.iter().all(|s| s.is_some());

    // Check if solutions match (where both succeeded)
    let solutions_match = dense_solutions.iter().zip(sparse_solutions.iter())
        .all(|(d, s)| {
            match (d, s) {
                (Some(dv), Some(sv)) => dv == sv,
                (None, None) => true,
                _ => false,
            }
        });

    if !solutions_match && dense_correct && sparse_correct {
        eprintln!("Warning: Solutions differ at n={}", n);
    }

    let speedup = if sparse_ms > 0.0 { dense_ms / sparse_ms } else { 1.0 };

    SparseBenchResult {
        n,
        nnz: sparse.nnz(),
        sparsity: sparse.sparsity(),
        k,
        dense_ms,
        sparse_ms,
        speedup,
        dense_correct,
        sparse_correct,
    }
}

fn sparse_to_dense(sparse: &SparseMatrix) -> Vec<BigInt> {
    let n = sparse.nrows;
    let mut dense = vec![BigInt::from(0); n * n];

    for row in 0..n {
        let start = sparse.row_ptrs[row];
        let end = sparse.row_ptrs[row + 1];
        for idx in start..end {
            let col = sparse.col_indices[idx];
            dense[row * n + col] = sparse.values[idx].clone();
        }
    }

    dense
}

fn export_results(path: &PathBuf, results: &[SparseBenchResult]) {
    let mut file = File::create(path).expect("Failed to create file");
    writeln!(file, "n,nnz,sparsity,k,dense_ms,sparse_ms,speedup,dense_ok,sparse_ok").unwrap();
    for r in results {
        writeln!(
            file,
            "{},{},{:.4},{},{:.6},{:.6},{:.4},{},{}",
            r.n, r.nnz, r.sparsity, r.k, r.dense_ms, r.sparse_ms, r.speedup,
            r.dense_correct, r.sparse_correct
        ).unwrap();
    }
}

/// Sparsity benchmark result
#[derive(Debug, Clone)]
pub struct SparsityBenchResult {
    pub sparsity: f64,
    pub n: usize,
    pub nnz: usize,
    pub k: usize,
    pub dense_ms: f64,
    pub sparse_ms: f64,
    pub speedup: f64,
}

/// Run benchmark with varying sparsity levels
pub fn run_sparsity_benchmark(
    n: usize,
    sparsities_str: &str,
    k: usize,
    trials: usize,
    export: Option<PathBuf>,
) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Sparse Solver - Varying Sparsity Benchmark             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Parse sparsity levels
    let sparsities: Vec<f64> = sparsities_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    println!("Configuration:");
    println!("  Matrix size:     {} × {}", n, n);
    println!("  Sparsity levels: {:?}", sparsities);
    println!("  RHS vectors (k): {}", k);
    println!("  Trials:          {}", trials);
    println!();

    println!("┌──────────┬────────┬────────┬─────┬────────────┬────────────┬─────────┐");
    println!("│ Sparsity │   n    │   nnz  │  k  │ Dense(ms)  │ Sparse(ms) │ Speedup │");
    println!("├──────────┼────────┼────────┼─────┼────────────┼────────────┼─────────┤");

    let mut results = Vec::new();

    for &sparsity in &sparsities {
        let result = benchmark_sparsity(n, sparsity, k, trials);
        println!(
            "│   {:>5.1}% │ {:>6} │ {:>6} │ {:>3} │ {:>10.2} │ {:>10.2} │ {:>6.2}x │",
            sparsity * 100.0,
            result.n,
            result.nnz,
            result.k,
            result.dense_ms,
            result.sparse_ms,
            result.speedup
        );
        results.push(result);
    }

    println!("└──────────┴────────┴────────┴─────┴────────────┴────────────┴─────────┘");

    // Summary
    println!();
    println!("Key Observations:");

    // Find crossover point where sparse becomes faster
    let crossover = results.iter().find(|r| r.speedup > 1.0);
    if let Some(c) = crossover {
        println!("  • Sparse solver wins at sparsity ≥ {:.0}%", c.sparsity * 100.0);
    } else {
        println!("  • Dense solver faster for all tested sparsities");
    }

    let best = results.iter().max_by(|a, b| a.speedup.partial_cmp(&b.speedup).unwrap());
    if let Some(b) = best {
        println!("  • Best sparse speedup: {:.2}x at {:.0}% sparsity", b.speedup, b.sparsity * 100.0);
    }

    // Export if requested
    if let Some(path) = export {
        export_sparsity_results(&path, &results);
        println!("\n✓ Results exported to: {}", path.display());
    }
}

fn benchmark_sparsity(n: usize, target_sparsity: f64, k: usize, trials: usize) -> SparsityBenchResult {
    let seed = 42u64;
    let mut rng = StdRng::seed_from_u64(seed);

    // Generate sparse matrix with target sparsity
    let sparse = SparseMatrix::generate_with_sparsity(n, target_sparsity, seed);
    let actual_sparsity = sparse.sparsity();

    // Convert to dense for comparison
    let dense = sparse_to_dense(&sparse);

    // Generate RHS vectors
    let b_cols: Vec<Vec<BigInt>> = (0..k)
        .map(|_| (0..n).map(|_| BigInt::from(rng.gen_range(-100..100i32))).collect())
        .collect();

    // Setup primes - use a single prime for algorithm comparison
    let primes = PrimeGenerator::generate_31bit_primes(4);
    let p = primes[0];

    // Prepare modular data
    let sparse_mod = sparse.to_mod(p);
    let dense_mod: Vec<u32> = dense.iter()
        .map(|v| {
            let p_big = BigInt::from(p);
            let r = v % &p_big;
            if r < num_bigint::BigInt::from(0) {
                ((&r + &p_big) % &p_big).try_into().unwrap_or(0u32)
            } else {
                r.try_into().unwrap_or(0u32)
            }
        })
        .collect();

    let b_mod: Vec<Vec<u32>> = b_cols.iter()
        .map(|col| col.iter()
            .map(|v| {
                let p_big = BigInt::from(p);
                let r = v % &p_big;
                if r < num_bigint::BigInt::from(0) {
                    ((&r + &p_big) % &p_big).try_into().unwrap_or(0u32)
                } else {
                    r.try_into().unwrap_or(0u32)
                }
            })
            .collect())
        .collect();

    // Run multiple trials and take median
    let mut dense_times = Vec::new();
    let mut sparse_times = Vec::new();

    for _ in 0..trials {
        // Benchmark dense solver
        let dense_start = Instant::now();
        let _: Vec<Option<Vec<u32>>> = b_mod.iter()
            .map(|b| {
                let cpu = CpuBackend::new();
                cpu.solve_mod(&dense_mod, b, n, p)
            })
            .collect();
        dense_times.push(dense_start.elapsed().as_secs_f64() * 1000.0);

        // Benchmark sparse Wiedemann solver
        let wiedemann = WiedemannSolver::new();
        let sparse_start = Instant::now();
        let _: Vec<Option<Vec<u32>>> = b_mod.iter()
            .map(|b| wiedemann.solve(&sparse_mod, b, p))
            .collect();
        sparse_times.push(sparse_start.elapsed().as_secs_f64() * 1000.0);
    }

    // Compute median
    dense_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sparse_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let dense_ms = dense_times[trials / 2];
    let sparse_ms = sparse_times[trials / 2];

    let speedup = if sparse_ms > 0.0 { dense_ms / sparse_ms } else { 1.0 };

    SparsityBenchResult {
        sparsity: actual_sparsity,
        n,
        nnz: sparse.nnz(),
        k,
        dense_ms,
        sparse_ms,
        speedup,
    }
}

fn export_sparsity_results(path: &PathBuf, results: &[SparsityBenchResult]) {
    let mut file = File::create(path).expect("Failed to create file");
    writeln!(file, "sparsity,n,nnz,k,dense_ms,sparse_ms,speedup").unwrap();
    for r in results {
        writeln!(
            file,
            "{:.4},{},{},{},{:.6},{:.6},{:.4}",
            r.sparsity, r.n, r.nnz, r.k, r.dense_ms, r.sparse_ms, r.speedup
        ).unwrap();
    }
}

/// GPU Wiedemann benchmark result
#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
pub struct GpuWiedemannResult {
    pub n: usize,
    pub nnz: usize,
    pub sparsity: f64,
    pub num_primes: usize,
    pub cpu_wiedemann_ms: f64,
    pub gpu_wiedemann_ms: f64,
    pub gpu_speedup: f64,
    pub gpu_matvec_ms: f64,
    pub cpu_minpoly_ms: f64,
    pub crt_ms: f64,
}

/// Run GPU vs CPU Wiedemann benchmark
#[cfg(not(target_os = "macos"))]
pub fn run_gpu_wiedemann_benchmark(_max_size: usize, _nnz_per_row: usize, _export: Option<PathBuf>) {
    println!("GPU Wiedemann benchmark requires Metal (macOS only).");
    println!("This benchmark is not available on Linux/Windows.");
}

/// Run GPU vs CPU Wiedemann benchmark
#[cfg(target_os = "macos")]
pub fn run_gpu_wiedemann_benchmark(max_size: usize, nnz_per_row: usize, export: Option<PathBuf>) {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║           GPU vs CPU Wiedemann Benchmark                                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let backend = match MetalBackend::new() {
        Some(b) => b,
        None => {
            println!("Metal not available - GPU benchmark cannot run");
            return;
        }
    };

    println!("Configuration:");
    println!("  Max size:      {}", max_size);
    println!("  NNZ per row:   {} (R1CS-like)", nnz_per_row);
    println!("  Backend:       Metal GPU");
    println!();

    println!("┌──────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ GPU Wiedemann Optimization                                                                  │");
    println!("├──────────────────────────────────────────────────────────────────────────────────────────────┤");
    println!("│ CPU: Solve each prime sequentially, O(n × nnz × iters) per prime                           │");
    println!("│ GPU: Batch all primes, GPU computes A*v for all primes in parallel                         │");
    println!("│ For 32 primes: potential 32× improvement in matvec phase                                   │");
    println!("└──────────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    let sizes: Vec<usize> = (0..)
        .map(|i| 128 * (1 << i))
        .take_while(|&s| s <= max_size)
        .collect();

    let num_primes = 16;
    let mut results = Vec::new();

    println!("┌──────┬────────┬──────────┬────────┬────────────┬────────────┬─────────┬─────────┬─────────┬─────────┐");
    println!("│   n  │   nnz  │ sparsity │ primes │  CPU(ms)   │  GPU(ms)   │ Speedup │ MatVec  │ MinPoly │   CRT   │");
    println!("├──────┼────────┼──────────┼────────┼────────────┼────────────┼─────────┼─────────┼─────────┼─────────┤");

    for n in sizes {
        let result = benchmark_gpu_wiedemann(&backend, n, nnz_per_row, num_primes);
        println!(
            "│ {:>4} │ {:>6} │ {:>7.1}% │ {:>6} │ {:>10.2} │ {:>10.2} │ {:>6.2}x │ {:>6.1}% │ {:>6.1}% │ {:>6.1}% │",
            result.n,
            result.nnz,
            result.sparsity * 100.0,
            result.num_primes,
            result.cpu_wiedemann_ms,
            result.gpu_wiedemann_ms,
            result.gpu_speedup,
            (result.gpu_matvec_ms / result.gpu_wiedemann_ms) * 100.0,
            (result.cpu_minpoly_ms / result.gpu_wiedemann_ms) * 100.0,
            (result.crt_ms / result.gpu_wiedemann_ms) * 100.0,
        );
        results.push(result);
    }

    println!("└──────┴────────┴──────────┴────────┴────────────┴────────────┴─────────┴─────────┴─────────┴─────────┘");

    // Summary
    println!();
    println!("Key Observations:");

    let avg_speedup: f64 = results.iter().map(|r| r.gpu_speedup).sum::<f64>() / results.len() as f64;
    println!("  • Average GPU speedup: {:.2}x", avg_speedup);

    let best = results.iter().max_by(|a, b| a.gpu_speedup.partial_cmp(&b.gpu_speedup).unwrap());
    if let Some(b) = best {
        println!("  • Best speedup: {:.2}x at n={}", b.gpu_speedup, b.n);
    }

    if let Some(path) = export {
        export_gpu_results(&path, &results);
        println!("\n✓ Results exported to: {}", path.display());
    }
}

#[cfg(target_os = "macos")]
fn benchmark_gpu_wiedemann(
    backend: &MetalBackend,
    n: usize,
    nnz_per_row: usize,
    num_primes: usize,
) -> GpuWiedemannResult {
    let seed = 42u64;
    let mut rng = StdRng::seed_from_u64(seed);

    // Generate R1CS-like sparse matrix
    let sparse = SparseMatrix::generate_r1cs_like(n, nnz_per_row, seed);

    // Generate known solution x, then compute b = A*x
    // This guarantees the system has a solution
    let x_known: Vec<BigInt> = (0..n)
        .map(|_| BigInt::from(rng.gen_range(1..50i32)))
        .collect();

    // Compute b = A * x_known (sparse matvec)
    let b: Vec<BigInt> = (0..n)
        .map(|row| {
            let start = sparse.row_ptrs[row];
            let end = sparse.row_ptrs[row + 1];
            let mut sum = BigInt::from(0);
            for idx in start..end {
                let col = sparse.col_indices[idx];
                sum += &sparse.values[idx] * &x_known[col];
            }
            sum
        })
        .collect();

    // Setup CRT basis
    let primes = PrimeGenerator::generate_31bit_primes(num_primes);
    let basis = CRTBasis::new(primes.clone());

    // CPU Wiedemann (per-prime sequential)
    let cpu_wiedemann = WiedemannSolver::new();
    let cpu_start = Instant::now();
    for &p in &primes {
        let sparse_mod = sparse.to_mod(p);
        let b_mod: Vec<u32> = b.iter()
            .map(|v| {
                let p_big = BigInt::from(p);
                let r = v % &p_big;
                if r < BigInt::from(0) {
                    ((&r + &p_big) % &p_big).try_into().unwrap_or(0)
                } else {
                    r.try_into().unwrap_or(0)
                }
            })
            .collect();
        let _ = cpu_wiedemann.solve(&sparse_mod, &b_mod, p);
    }
    let cpu_wiedemann_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

    // GPU Wiedemann (batched)
    let gpu_solver = GpuWiedemannSolver::new(backend);
    let gpu_start = Instant::now();
    let gpu_result = gpu_solver.solve(&sparse, &b, &basis);
    let gpu_wiedemann_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;

    let (gpu_matvec_ms, cpu_minpoly_ms, crt_ms) = if let Some((_, stats)) = gpu_result {
        (
            stats.gpu_matvec_time * 1000.0,
            stats.cpu_minpoly_time * 1000.0,
            stats.crt_time * 1000.0,
        )
    } else {
        (0.0, 0.0, 0.0)
    };

    let gpu_speedup = if gpu_wiedemann_ms > 0.0 { cpu_wiedemann_ms / gpu_wiedemann_ms } else { 1.0 };

    GpuWiedemannResult {
        n,
        nnz: sparse.nnz(),
        sparsity: sparse.sparsity(),
        num_primes,
        cpu_wiedemann_ms,
        gpu_wiedemann_ms,
        gpu_speedup,
        gpu_matvec_ms,
        cpu_minpoly_ms,
        crt_ms,
    }
}

#[cfg(target_os = "macos")]
fn export_gpu_results(path: &PathBuf, results: &[GpuWiedemannResult]) {
    let mut file = File::create(path).expect("Failed to create file");
    writeln!(file, "n,nnz,sparsity,num_primes,cpu_ms,gpu_ms,speedup,matvec_ms,minpoly_ms,crt_ms").unwrap();
    for r in results {
        writeln!(
            file,
            "{},{},{:.4},{},{:.6},{:.6},{:.4},{:.6},{:.6},{:.6}",
            r.n, r.nnz, r.sparsity, r.num_primes, r.cpu_wiedemann_ms, r.gpu_wiedemann_ms,
            r.gpu_speedup, r.gpu_matvec_ms, r.cpu_minpoly_ms, r.crt_ms
        ).unwrap();
    }
}
