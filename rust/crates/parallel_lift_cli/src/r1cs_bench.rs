//! R1CS Benchmarking
//!
//! Benchmark solving R1CS constraint systems using GPU-accelerated sparse solvers.
//! Note: This benchmark requires Metal (macOS) for GPU-accelerated sparse solving.

#[cfg(target_os = "macos")]
use num_bigint::BigInt;
#[cfg(target_os = "macos")]
use parallel_lift_core::{
    generate_synthetic_r1cs, R1csFile, R1csStats, SparseMatrix, WiedemannSolver,
    CRTBasis, PrimeGenerator, CpuBackend, Backend,
};
#[cfg(target_os = "macos")]
use parallel_lift_metal::{MetalBackend, GpuWiedemannSolver, GpuWiedemannStats};
use std::path::PathBuf;
#[cfg(target_os = "macos")]
use std::time::Instant;
#[cfg(target_os = "macos")]
use std::fs::File;
#[cfg(target_os = "macos")]
use std::io::Write;

/// Result from R1CS benchmark
#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
pub struct R1csBenchResult {
    pub n_constraints: usize,
    pub n_wires: usize,
    pub total_nnz: usize,
    pub sparsity: f64,
    pub cpu_ms: f64,
    pub gpu_ms: f64,
    pub speedup: f64,
    pub verified: bool,
    // Phase timing breakdown (GPU only)
    pub gpu_matvec_ms: f64,
    pub gpu_minpoly_ms: f64,
    pub gpu_precond_ms: f64,
    pub gpu_block_ms: f64,
    pub gpu_dense_fallback_ms: f64,
    pub gpu_crt_ms: f64,
    pub gpu_iterations: usize,
    // Success ladder stats
    pub primes_plain_success: usize,
    pub primes_multitry_success: usize,
    pub primes_precond_success: usize,
    pub primes_block_success: usize,
    pub primes_fallback_count: usize,
    pub total_wiedemann_attempts: usize,
}

/// Stub for R1CS benchmark on non-macOS platforms
#[cfg(not(target_os = "macos"))]
pub fn run_r1cs_benchmark(
    _r1cs_path: Option<PathBuf>,
    _max_size: usize,
    _export: Option<PathBuf>,
) {
    println!("R1CS benchmark requires Metal (macOS only).");
    println!("This benchmark uses GPU-accelerated sparse solvers that are not available on Linux/Windows.");
}

/// Run R1CS benchmark on a file or synthetic data
#[cfg(target_os = "macos")]
pub fn run_r1cs_benchmark(
    r1cs_path: Option<PathBuf>,
    max_size: usize,
    export: Option<PathBuf>,
) {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║              R1CS Constraint System Benchmark                            ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let backend = match MetalBackend::new() {
        Some(b) => b,
        None => {
            println!("Metal not available - GPU benchmark cannot run");
            return;
        }
    };

    if let Some(path) = r1cs_path {
        // Benchmark a specific R1CS file
        run_file_benchmark(&backend, &path);
    } else {
        // Benchmark synthetic R1CS systems
        run_synthetic_benchmark(&backend, max_size, export);
    }
}

#[cfg(target_os = "macos")]
fn run_file_benchmark(backend: &MetalBackend, path: &PathBuf) {
    println!("Loading R1CS from: {}", path.display());
    println!();

    match R1csFile::load(path) {
        Ok(r1cs) => {
            let stats = r1cs.stats();
            println!("{}", stats);

            // R1CS matrices are typically m x n (constraints x wires)
            // For Wiedemann solver, we need a square matrix
            // Create a square R1CS-like sparse matrix with similar characteristics
            let n = stats.n_constraints.max(stats.n_wires);
            let nnz_per_row = if stats.n_constraints > 0 {
                (stats.total_nnz as f64 / stats.n_constraints as f64).ceil() as usize
            } else {
                5
            };

            println!("Creating square {}x{} R1CS-like matrix for benchmark", n, n);
            println!("  (Original R1CS: {} x {})", stats.n_constraints, stats.n_wires);
            println!("  NNZ per row: ~{}", nnz_per_row);
            println!();

            // Generate synthetic square matrix with similar sparsity
            let sparse = SparseMatrix::generate_r1cs_like(n, nnz_per_row.max(3).min(10), 42);
            println!("Synthetic matrix sparsity: {:.2}%", sparse.sparsity() * 100.0);

            let result = benchmark_r1cs_solve(backend, &sparse, 16);
            print_result(&result);
        }
        Err(e) => {
            println!("Error loading R1CS: {:?}", e);
        }
    }
}

#[cfg(target_os = "macos")]
fn run_synthetic_benchmark(backend: &MetalBackend, max_size: usize, export: Option<PathBuf>) {
    println!("Configuration:");
    println!("  Max constraints: {}", max_size);
    println!("  Backend:         Metal GPU");
    println!();

    println!("┌────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ R1CS-style Sparse Benchmark                                                               │");
    println!("├────────────────────────────────────────────────────────────────────────────────────────────┤");
    println!("│ 1. Generate R1CS-like sparse matrix (~5 NNZ per row, similar to ZK constraints)          │");
    println!("│ 2. Solve Ax = b using CPU Wiedemann and GPU Wiedemann across 16 primes                   │");
    println!("│ 3. Compare CPU sequential vs GPU parallel performance                                    │");
    println!("└────────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Sizes to test (geometric progression)
    let sizes: Vec<usize> = (0..)
        .map(|i| 256 * (1 << i))
        .take_while(|&s| s <= max_size)
        .collect();

    let num_primes = 16;
    let mut results = Vec::new();

    println!("┌────────┬────────┬────────┬──────────┬────────────┬────────────┬─────────┬──────────┐");
    println!("│      n │    NNZ │ NNZ/row│ Sparsity │   CPU(ms)  │   GPU(ms)  │ Speedup │ Verified │");
    println!("├────────┼────────┼────────┼──────────┼────────────┼────────────┼─────────┼──────────┤");

    for n in sizes {
        // Generate R1CS-like sparse square matrix
        let sparse = SparseMatrix::generate_r1cs_like(n, 5, 42);

        let result = benchmark_r1cs_solve(backend, &sparse, num_primes);
        let nnz_per_row = if result.n_constraints > 0 {
            result.total_nnz as f64 / result.n_constraints as f64
        } else { 0.0 };

        println!(
            "│ {:>6} │ {:>6} │ {:>6.1} │ {:>7.2}% │ {:>10.2} │ {:>10.2} │ {:>6.2}x │ {:>8} │",
            result.n_constraints,
            result.total_nnz,
            nnz_per_row,
            result.sparsity * 100.0,
            result.cpu_ms,
            result.gpu_ms,
            result.speedup,
            if result.verified { "✓" } else { "✗" },
        );

        results.push(result);
    }

    println!("└────────┴────────┴────────┴──────────┴────────────┴────────────┴─────────┴──────────┘");

    // Summary
    println!();
    println!("Key Observations:");

    if !results.is_empty() {
        let avg_speedup: f64 = results.iter().map(|r| r.speedup).sum::<f64>() / results.len() as f64;
        println!("  • Average GPU speedup: {:.2}x", avg_speedup);

        let best = results.iter().max_by(|a, b| a.speedup.partial_cmp(&b.speedup).unwrap());
        if let Some(b) = best {
            println!("  • Best speedup: {:.2}x at {} constraints", b.speedup, b.n_constraints);
        }

        // Scaling analysis
        if results.len() >= 2 {
            let r1 = &results[0];
            let r2 = &results[results.len() - 1];
            let size_ratio = r2.n_constraints as f64 / r1.n_constraints as f64;
            let cpu_ratio = r2.cpu_ms / r1.cpu_ms.max(0.001);
            let gpu_ratio = r2.gpu_ms / r1.gpu_ms.max(0.001);

            println!();
            println!("  Scaling Analysis ({} → {} constraints):", r1.n_constraints, r2.n_constraints);
            println!("    Size: {:.1}x, CPU time: {:.1}x, GPU time: {:.1}x",
                     size_ratio, cpu_ratio, gpu_ratio);
        }

        // Success ladder and timing for largest result
        let largest = results.last().unwrap();
        let sparse_success = largest.primes_plain_success + largest.primes_multitry_success
            + largest.primes_precond_success + largest.primes_block_success;
        let sparse_rate = (sparse_success as f64 / 16.0) * 100.0;

        println!();
        println!("  Success Ladder (n={}):", largest.n_constraints);
        println!("    Plain: {}/16, Multi-try: +{}/16, Precond: +{}/16, Block: +{}/16 = {}/16 sparse ({:.0}%)",
                 largest.primes_plain_success,
                 largest.primes_multitry_success,
                 largest.primes_precond_success,
                 largest.primes_block_success,
                 sparse_success,
                 sparse_rate);
        println!("    Dense fallback: {}/16", largest.primes_fallback_count);

        println!();
        println!("  GPU Phase Timing (n={}):", largest.n_constraints);
        println!("    Matvec:   {:>8.1} ms ({:>4.1}%)", largest.gpu_matvec_ms,
                 if largest.gpu_ms > 0.0 { largest.gpu_matvec_ms / largest.gpu_ms * 100.0 } else { 0.0 });
        println!("    Minpoly:  {:>8.1} ms ({:>4.1}%)", largest.gpu_minpoly_ms,
                 if largest.gpu_ms > 0.0 { largest.gpu_minpoly_ms / largest.gpu_ms * 100.0 } else { 0.0 });
        println!("    Precond:  {:>8.1} ms ({:>4.1}%)", largest.gpu_precond_ms,
                 if largest.gpu_ms > 0.0 { largest.gpu_precond_ms / largest.gpu_ms * 100.0 } else { 0.0 });
        println!("    Block:    {:>8.1} ms ({:>4.1}%)", largest.gpu_block_ms,
                 if largest.gpu_ms > 0.0 { largest.gpu_block_ms / largest.gpu_ms * 100.0 } else { 0.0 });
        println!("    Dense GE: {:>8.1} ms ({:>4.1}%)", largest.gpu_dense_fallback_ms,
                 if largest.gpu_ms > 0.0 { largest.gpu_dense_fallback_ms / largest.gpu_ms * 100.0 } else { 0.0 });
        println!("    CRT:      {:>8.1} ms ({:>4.1}%)", largest.gpu_crt_ms,
                 if largest.gpu_ms > 0.0 { largest.gpu_crt_ms / largest.gpu_ms * 100.0 } else { 0.0 });
        let other_ms = largest.gpu_ms - largest.gpu_matvec_ms - largest.gpu_minpoly_ms
            - largest.gpu_precond_ms - largest.gpu_block_ms - largest.gpu_dense_fallback_ms - largest.gpu_crt_ms;
        println!("    Other:    {:>8.1} ms ({:>4.1}%)", other_ms,
                 if largest.gpu_ms > 0.0 { other_ms / largest.gpu_ms * 100.0 } else { 0.0 });
    }

    // Export if requested
    if let Some(path) = export {
        export_results(&path, &results);
        println!("\n✓ Results exported to: {}", path.display());
    }
}

#[cfg(target_os = "macos")]
fn benchmark_r1cs_solve(
    backend: &MetalBackend,
    matrix: &SparseMatrix,
    num_primes: usize,
) -> R1csBenchResult {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    let n = matrix.nrows;
    let mut rng = StdRng::seed_from_u64(42);

    // Generate known solution x, then compute b = A*x
    // This guarantees the system has a solution
    let x_known: Vec<BigInt> = (0..n)
        .map(|_| BigInt::from(rng.gen_range(1..50i32)))
        .collect();

    // Compute b = A * x_known
    let b: Vec<BigInt> = (0..n)
        .map(|row| {
            let start = matrix.row_ptrs[row];
            let end = matrix.row_ptrs[row + 1];
            let mut sum = BigInt::from(0);
            for idx in start..end {
                let col = matrix.col_indices[idx];
                sum += &matrix.values[idx] * &x_known[col];
            }
            sum
        })
        .collect();

    // Setup CRT basis
    let primes = PrimeGenerator::generate_31bit_primes(num_primes);
    let basis = CRTBasis::new(primes.clone());

    // CPU Wiedemann benchmark
    let cpu_wiedemann = WiedemannSolver::new();
    let cpu_start = Instant::now();
    let mut cpu_verified = true;

    for &p in &primes {
        let matrix_mod = matrix.to_mod(p);
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

        if cpu_wiedemann.solve(&matrix_mod, &b_mod, p).is_none() {
            cpu_verified = false;
        }
    }
    let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

    // GPU Wiedemann benchmark
    let gpu_solver = GpuWiedemannSolver::new(backend);
    let gpu_start = Instant::now();
    let gpu_result = gpu_solver.solve(matrix, &b, &basis);
    let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;

    let (gpu_verified, gpu_stats) = match gpu_result {
        Some((_solution, stats)) => (true, stats),
        None => (false, GpuWiedemannStats::default()),
    };
    let verified = cpu_verified && gpu_verified;

    let speedup = if gpu_ms > 0.0 { cpu_ms / gpu_ms } else { 1.0 };

    R1csBenchResult {
        n_constraints: matrix.nrows,
        n_wires: matrix.ncols,
        total_nnz: matrix.nnz(),
        sparsity: matrix.sparsity(),
        cpu_ms,
        gpu_ms,
        speedup,
        verified,
        gpu_matvec_ms: gpu_stats.gpu_matvec_time * 1000.0,
        gpu_minpoly_ms: gpu_stats.cpu_minpoly_time * 1000.0,
        gpu_precond_ms: gpu_stats.preconditioning_time * 1000.0,
        gpu_block_ms: gpu_stats.block_wiedemann_time * 1000.0,
        gpu_dense_fallback_ms: gpu_stats.dense_fallback_time * 1000.0,
        gpu_crt_ms: gpu_stats.crt_time * 1000.0,
        gpu_iterations: gpu_stats.iterations,
        primes_plain_success: gpu_stats.primes_plain_success,
        primes_multitry_success: gpu_stats.primes_multitry_success,
        primes_precond_success: gpu_stats.primes_precond_success,
        primes_block_success: gpu_stats.primes_block_success,
        primes_fallback_count: gpu_stats.primes_fallback_count,
        total_wiedemann_attempts: gpu_stats.total_wiedemann_attempts,
    }
}

#[cfg(target_os = "macos")]
fn print_result(result: &R1csBenchResult) {
    println!();
    println!("Benchmark Results:");
    println!("  Constraints: {}", result.n_constraints);
    println!("  Wires:       {}", result.n_wires);
    println!("  NNZ:         {}", result.total_nnz);
    println!("  Sparsity:    {:.2}%", result.sparsity * 100.0);
    println!("  CPU time:    {:.2} ms", result.cpu_ms);
    println!("  GPU time:    {:.2} ms", result.gpu_ms);
    println!("  Speedup:     {:.2}x", result.speedup);
    println!("  Verified:    {}", if result.verified { "✓" } else { "✗" });

    // Success ladder breakdown
    println!();
    println!("Success Ladder (16 primes):");
    let sparse_success = result.primes_plain_success + result.primes_multitry_success
        + result.primes_precond_success + result.primes_block_success;
    let sparse_rate = (sparse_success as f64 / 16.0) * 100.0;
    println!("  Plain Wiedemann:        {:>2}/16 solved", result.primes_plain_success);
    println!("  + Multi-try (8 seeds):  {:>2}/16 solved", result.primes_multitry_success);
    println!("  + Preconditioned:       {:>2}/16 solved", result.primes_precond_success);
    println!("  + Block Wiedemann:      {:>2}/16 solved", result.primes_block_success);
    println!("  = Sparse total:         {:>2}/16 ({:.0}%)", sparse_success, sparse_rate);
    println!("  Dense GE fallback:      {:>2}/16", result.primes_fallback_count);
    println!("  Total attempts: {}", result.total_wiedemann_attempts);

    // Phase timing breakdown
    println!();
    println!("GPU Phase Timing:");
    println!("  Wiedemann iters: {}", result.gpu_iterations);
    println!("  Matvec:    {:>8.2} ms ({:>5.1}%)", result.gpu_matvec_ms,
             if result.gpu_ms > 0.0 { result.gpu_matvec_ms / result.gpu_ms * 100.0 } else { 0.0 });
    println!("  Minpoly:   {:>8.2} ms ({:>5.1}%)", result.gpu_minpoly_ms,
             if result.gpu_ms > 0.0 { result.gpu_minpoly_ms / result.gpu_ms * 100.0 } else { 0.0 });
    println!("  Precond:   {:>8.2} ms ({:>5.1}%)", result.gpu_precond_ms,
             if result.gpu_ms > 0.0 { result.gpu_precond_ms / result.gpu_ms * 100.0 } else { 0.0 });
    println!("  Block:     {:>8.2} ms ({:>5.1}%)", result.gpu_block_ms,
             if result.gpu_ms > 0.0 { result.gpu_block_ms / result.gpu_ms * 100.0 } else { 0.0 });
    println!("  Dense GE:  {:>8.2} ms ({:>5.1}%)", result.gpu_dense_fallback_ms,
             if result.gpu_ms > 0.0 { result.gpu_dense_fallback_ms / result.gpu_ms * 100.0 } else { 0.0 });
    println!("  CRT:       {:>8.2} ms ({:>5.1}%)", result.gpu_crt_ms,
             if result.gpu_ms > 0.0 { result.gpu_crt_ms / result.gpu_ms * 100.0 } else { 0.0 });
    let other_ms = result.gpu_ms - result.gpu_matvec_ms - result.gpu_minpoly_ms
        - result.gpu_precond_ms - result.gpu_block_ms - result.gpu_dense_fallback_ms - result.gpu_crt_ms;
    println!("  Other:     {:>8.2} ms ({:>5.1}%)", other_ms,
             if result.gpu_ms > 0.0 { other_ms / result.gpu_ms * 100.0 } else { 0.0 });
}

#[cfg(target_os = "macos")]
fn export_results(path: &PathBuf, results: &[R1csBenchResult]) {
    let mut file = File::create(path).expect("Failed to create file");
    writeln!(file, "n_constraints,n_wires,nnz,sparsity,cpu_ms,gpu_ms,speedup,verified,gpu_matvec_ms,gpu_minpoly_ms,gpu_dense_fallback_ms,gpu_crt_ms,gpu_iterations,primes_fallback_count").unwrap();
    for r in results {
        writeln!(
            file,
            "{},{},{},{:.6},{:.6},{:.6},{:.4},{},{:.6},{:.6},{:.6},{:.6},{},{}",
            r.n_constraints, r.n_wires, r.total_nnz, r.sparsity,
            r.cpu_ms, r.gpu_ms, r.speedup, r.verified,
            r.gpu_matvec_ms, r.gpu_minpoly_ms, r.gpu_dense_fallback_ms, r.gpu_crt_ms,
            r.gpu_iterations, r.primes_fallback_count
        ).unwrap();
    }
}
