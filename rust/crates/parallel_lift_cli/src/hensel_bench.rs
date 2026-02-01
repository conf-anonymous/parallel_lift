//! Hensel Lifting Benchmark
//!
//! Compares GPU Hensel (Dixon) lifting against CRT-based approach and IML.

use std::path::PathBuf;
use std::time::Instant;

#[cfg(feature = "cuda")]
pub fn run_hensel_benchmark(
    sizes_str: &str,
    k: usize,
    trials: usize,
    export: Option<PathBuf>,
) {
    use parallel_lift_cuda::CudaBackend;
    use parallel_lift_core::{
        PrimeGenerator, CRTBasis, HenselConfig,
    };
    use num_bigint::BigInt;

    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║            Hensel Lifting Benchmark: Hensel vs CRT vs IML                    ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let backend = match CudaBackend::new() {
        Some(b) => b,
        None => {
            println!("Error: CUDA backend not available");
            return;
        }
    };

    // Parse sizes
    let sizes: Vec<usize> = sizes_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    println!("Configuration:");
    println!("  Sizes: {:?}", sizes);
    println!("  RHS (k): {}", k);
    println!("  Trials: {}", trials);
    println!();

    let mut results = Vec::new();

    // Table header
    println!("┌────────┬────────────┬────────────┬────────────┬──────────────┬──────────────┬────────────┐");
    println!("│   n    │ Hensel(ms) │ CRT(ms)    │ IML(ms)    │ Hensel vs CRT│ Hensel vs IML│ Iterations │");
    println!("├────────┼────────────┼────────────┼────────────┼──────────────┼──────────────┼────────────┤");

    for &n in &sizes {
        // Generate test matrix (diagonally dominant for non-singularity)
        let matrix: Vec<i64> = generate_test_matrix_i64(n, 42);
        let b_cols: Vec<Vec<i64>> = (0..k)
            .map(|seed| generate_test_vector_i64(n, seed as u64 + 100))
            .collect();

        // Convert to BigInt for CRT
        let matrix_bigint: Vec<BigInt> = matrix.iter().map(|&x| BigInt::from(x)).collect();
        let b_cols_bigint: Vec<Vec<BigInt>> = b_cols.iter().map(|col| {
            col.iter().map(|&x| BigInt::from(x)).collect()
        }).collect();

        // Hensel config
        let hensel_config = HenselConfig::for_matrix(n, 10);

        // CRT config (use standard prime counts)
        let num_primes_crt = match n {
            32 => 68,
            64 => 136,
            128 => 273,
            256 => 546,
            512 => 1092,
            _ => {
                let hadamard_bits = (n as f64 * (n as f64).log2() + n as f64 * 10.0) as usize + n * 10;
                (hadamard_bits + 30) / 31 + 1
            }
        };
        let primes_crt = PrimeGenerator::generate_31bit_primes(num_primes_crt);
        let basis_crt = CRTBasis::new(primes_crt.clone());

        // Warmup
        let _ = backend.gpu_hensel_solve(&matrix, &b_cols, n, k, &hensel_config);

        // Hensel benchmark with timing breakdown
        let mut hensel_times: Vec<f64> = Vec::with_capacity(trials);
        let mut hensel_iterations = 0;
        let mut inverse_ms_avg = 0.0;
        let mut lifting_ms_avg = 0.0;
        let mut recon_ms_avg = 0.0;
        for _ in 0..trials {
            if let Some((_, timings)) = backend.gpu_hensel_solve(&matrix, &b_cols, n, k, &hensel_config) {
                hensel_times.push(timings.total_ms);
                hensel_iterations = timings.iterations;
                inverse_ms_avg += timings.inverse_ms;
                lifting_ms_avg += timings.lifting_ms;
                recon_ms_avg += timings.reconstruction_ms;
            }
        }

        if hensel_times.is_empty() {
            println!("│ {:>6} │ FAILED     │            │            │              │              │            │", n);
            continue;
        }
        let hensel_avg = hensel_times.iter().sum::<f64>() / hensel_times.len() as f64;
        inverse_ms_avg /= trials as f64;
        lifting_ms_avg /= trials as f64;
        recon_ms_avg /= trials as f64;

        // CRT benchmark (using existing infrastructure)
        let matrix_u32: Vec<u32> = matrix.iter().map(|&x| {
            if x >= 0 { x as u32 } else { (x + i64::MAX) as u32 % u32::MAX }
        }).collect();
        let b_cols_u32: Vec<Vec<u32>> = b_cols.iter().map(|col| {
            col.iter().map(|&x| {
                if x >= 0 { x as u32 } else { (x + i64::MAX) as u32 % u32::MAX }
            }).collect()
        }).collect();

        let mut crt_times: Vec<f64> = Vec::with_capacity(trials);
        for _ in 0..trials {
            let (result_opt, timing) = backend.gpu_batch_multi_rhs_solve_timed(&matrix_u32, &b_cols_u32, n, k, &primes_crt);
            if result_opt.is_some() {
                // Note: This doesn't include CRT reconstruction time
                // For fair comparison, we'd need to add GPU CRT time
                crt_times.push(timing.total_ms);
            }
        }
        let crt_avg = if crt_times.is_empty() {
            f64::NAN
        } else {
            crt_times.iter().sum::<f64>() / crt_times.len() as f64
        };

        // IML comparison (from baseline)
        let iml_time = match n {
            32 => 6.55,
            64 => 15.09,
            128 => 63.40,
            256 => 318.92,
            _ => f64::NAN,
        };

        let hensel_vs_crt = crt_avg / hensel_avg;
        let hensel_vs_iml = iml_time / hensel_avg;

        println!(
            "│ {:>6} │ {:>10.2} │ {:>10.2} │ {:>10.2} │ {:>11.2}x  │ {:>11.2}x  │ {:>10} │",
            n, hensel_avg, crt_avg, iml_time, hensel_vs_crt, hensel_vs_iml, hensel_iterations
        );
        // Print timing breakdown for each size
        println!(
            "│        │  inv:{:>5.1} lift:{:>5.1} rec:{:>5.1}                                              │",
            inverse_ms_avg, lifting_ms_avg, recon_ms_avg
        );

        results.push(HenselResult {
            n,
            k,
            hensel_ms: hensel_avg,
            crt_ms: crt_avg,
            iml_ms: iml_time,
            hensel_vs_crt,
            hensel_vs_iml,
            iterations: hensel_iterations,
        });
    }

    println!("└────────┴────────────┴────────────┴────────────┴──────────────┴──────────────┴────────────┘");
    println!();

    // Summary
    if !results.is_empty() {
        let avg_vs_crt = results.iter().filter(|r| !r.hensel_vs_crt.is_nan()).map(|r| r.hensel_vs_crt).sum::<f64>()
            / results.iter().filter(|r| !r.hensel_vs_crt.is_nan()).count() as f64;
        let avg_vs_iml = results.iter().filter(|r| !r.hensel_vs_iml.is_nan()).map(|r| r.hensel_vs_iml).sum::<f64>()
            / results.iter().filter(|r| !r.hensel_vs_iml.is_nan()).count() as f64;

        println!("Summary:");
        println!("  Average speedup vs CRT (GPU solve only): {:.2}x", avg_vs_crt);
        println!("  Average speedup vs IML: {:.2}x", avg_vs_iml);
        println!();

        println!("Analysis:");
        println!("  - Hensel does O(n³) inverse + O(iter × n²) lifting");
        println!("  - CRT does O(num_primes × n³) independent solves");
        println!("  - For n=256 with 546 primes, Hensel should be ~{}x faster", 546);
        println!();
    }

    // Export to CSV
    if let Some(path) = export {
        export_hensel_results(&results, &path);
        println!("Results exported to: {}", path.display());
    }
}

#[cfg(feature = "cuda")]
fn generate_test_matrix_i64(n: usize, seed: u64) -> Vec<i64> {
    let mut matrix = Vec::with_capacity(n * n);
    let mut state = seed;

    for i in 0..n {
        for j in 0..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = if i == j {
                100 + (state % 900) as i64
            } else {
                ((state % 101) as i64) - 50
            };
            matrix.push(val);
        }
    }
    matrix
}

#[cfg(feature = "cuda")]
fn generate_test_vector_i64(n: usize, seed: u64) -> Vec<i64> {
    let mut vec = Vec::with_capacity(n);
    let mut state = seed;

    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = ((state % 201) as i64) - 100;
        vec.push(val);
    }
    vec
}

#[cfg(feature = "cuda")]
struct HenselResult {
    n: usize,
    k: usize,
    hensel_ms: f64,
    crt_ms: f64,
    iml_ms: f64,
    hensel_vs_crt: f64,
    hensel_vs_iml: f64,
    iterations: usize,
}

#[cfg(feature = "cuda")]
fn export_hensel_results(results: &[HenselResult], path: &PathBuf) {
    use std::fs::File;
    use std::io::Write;

    let mut file = match File::create(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error creating file: {}", e);
            return;
        }
    };

    writeln!(file, "n,k,hensel_ms,crt_ms,iml_ms,hensel_vs_crt,hensel_vs_iml,iterations").unwrap();
    for r in results {
        writeln!(
            file,
            "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
            r.n, r.k, r.hensel_ms, r.crt_ms, r.iml_ms, r.hensel_vs_crt, r.hensel_vs_iml, r.iterations
        ).unwrap();
    }
}

#[cfg(not(feature = "cuda"))]
pub fn run_hensel_benchmark(
    _sizes_str: &str,
    _k: usize,
    _trials: usize,
    _export: Option<PathBuf>,
) {
    println!("CUDA feature not enabled. Compile with --features cuda");
}
