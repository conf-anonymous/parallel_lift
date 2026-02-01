//! V2 Benchmark: Compare 31-bit vs 62-bit prime implementations
//!
//! This benchmark measures the actual speedup from using 62-bit primes
//! which require roughly half as many primes for CRT reconstruction.

use std::path::PathBuf;
use std::time::Instant;

#[cfg(feature = "cuda")]
pub fn run_v2_benchmark(
    sizes_str: &str,
    k: usize,
    trials: usize,
    export: Option<PathBuf>,
) {
    use parallel_lift_cuda::{CudaBackend, GpuCrtPrecomputed, GpuCrtPrecomputed64};
    use parallel_lift_core::{
        PrimeGenerator, CRTBasis,
        PrimeGenerator62, CRTBasis62,
        Rational,
    };
    use num_bigint::BigInt;

    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                V2 Benchmark: 31-bit vs 62-bit Primes                         ║");
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
    println!("┌────────┬───────────┬───────────┬────────────┬────────────┬────────────┬────────────┬──────────────┬────────────┬──────────┐");
    println!("│   n    │ V1 Primes │ V2 Primes │ V1 GPU(ms) │ V2 GPU(ms) │ V1 CRT(ms) │ V2 CRT(ms) │ CRT Speedup  │ Total Spd. │ Prime    │");
    println!("│        │ (31-bit)  │ (62-bit)  │            │            │ (GPU)      │ (GPU)      │ (V2/V1)      │            │ Reduction│");
    println!("├────────┼───────────┼───────────┼────────────┼────────────┼────────────┼────────────┼──────────────┼────────────┼──────────┤");

    for &n in &sizes {
        // Generate test matrix (diagonally dominant for non-singularity)
        let matrix: Vec<BigInt> = generate_test_matrix(n, 42);
        let b_cols: Vec<Vec<BigInt>> = (0..k)
            .map(|seed| generate_test_vector(n, seed as u64 + 100))
            .collect();

        // Use actual prime counts from GPU baseline for fair comparison
        // These are the conservative Hadamard bounds actually used in production
        let num_primes_v1 = match n {
            32 => 68,
            64 => 136,
            128 => 273,
            256 => 546,
            512 => 1092,
            _ => {
                // Fallback to formula for other sizes
                let hadamard_bits = (n as f64 * (n as f64).log2() + n as f64 * 10.0) as usize + n * 10;
                (hadamard_bits + 30) / 31 + 1
            }
        };
        let primes_v1 = PrimeGenerator::generate_31bit_primes(num_primes_v1);
        let basis_v1 = CRTBasis::new(primes_v1.clone());
        let gpu_crt_v1 = GpuCrtPrecomputed::from_basis(&basis_v1);

        // V2: 62-bit primes (roughly half as many)
        let num_primes_v2 = (num_primes_v1 + 1) / 2;
        let primes_v2 = PrimeGenerator62::generate_62bit_primes(num_primes_v2);
        let basis_v2 = CRTBasis62::new(primes_v2.clone());
        let gpu_crt_v2 = GpuCrtPrecomputed64::from_basis(&basis_v2);

        // Prepare matrices as u32 for V1
        let matrix_u32: Vec<u32> = matrix.iter().map(|v| {
            let (_, digits) = v.to_u32_digits();
            *digits.first().unwrap_or(&0)
        }).collect();

        let b_cols_u32: Vec<Vec<u32>> = b_cols.iter().map(|col| {
            col.iter().map(|v| {
                let (_, digits) = v.to_u32_digits();
                *digits.first().unwrap_or(&0)
            }).collect::<Vec<u32>>()
        }).collect();

        // Prepare matrices as u64 for V2
        let matrix_u64: Vec<u64> = matrix.iter().map(|v| {
            let (_, digits) = v.to_u64_digits();
            *digits.first().unwrap_or(&0)
        }).collect();

        let b_cols_u64: Vec<Vec<u64>> = b_cols.iter().map(|col| {
            col.iter().map(|v| {
                let (_, digits) = v.to_u64_digits();
                *digits.first().unwrap_or(&0)
            }).collect::<Vec<u64>>()
        }).collect();

        // Warmup runs
        let _ = backend.gpu_batch_multi_rhs_solve_timed(&matrix_u32, &b_cols_u32, n, k, &primes_v1);
        let _ = backend.gpu_batch_multi_rhs_solve_64_timed(&matrix_u64, &b_cols_u64, n, k, &primes_v2);

        // V1 benchmark (with GPU CRT)
        let mut v1_gpu_times: Vec<f64> = Vec::with_capacity(trials);
        let mut v1_crt_times: Vec<f64> = Vec::with_capacity(trials);
        let mut v1_total_times: Vec<f64> = Vec::with_capacity(trials);
        for _ in 0..trials {
            let (result_opt, timing) = backend
                .gpu_batch_multi_rhs_solve_timed(&matrix_u32, &b_cols_u32, n, k, &primes_v1);

            if let Some(result) = result_opt {
                // GPU CRT reconstruction
                let crt_start = Instant::now();
                let _solutions_v1: Vec<Vec<Rational>> = reconstruct_solutions_v1_gpu(
                    &backend, &result, n, k, &gpu_crt_v1
                );
                let crt_ms = crt_start.elapsed().as_secs_f64() * 1000.0;

                v1_gpu_times.push(timing.total_ms);
                v1_crt_times.push(crt_ms);
                v1_total_times.push(timing.total_ms + crt_ms);
            } else {
                println!("Warning: V1 solve failed for n={}", n);
                continue;
            }
        }
        if v1_gpu_times.is_empty() {
            println!("All V1 solves failed for n={}, skipping", n);
            continue;
        }
        let v1_gpu_avg = v1_gpu_times.iter().sum::<f64>() / v1_gpu_times.len() as f64;
        let v1_crt_avg = v1_crt_times.iter().sum::<f64>() / v1_crt_times.len() as f64;
        let v1_total_avg = v1_total_times.iter().sum::<f64>() / v1_total_times.len() as f64;

        // V2 benchmark (with GPU CRT)
        let mut v2_gpu_times: Vec<f64> = Vec::with_capacity(trials);
        let mut v2_crt_times: Vec<f64> = Vec::with_capacity(trials);
        let mut v2_total_times: Vec<f64> = Vec::with_capacity(trials);
        for _ in 0..trials {
            if let Some((result, timing)) = backend
                .gpu_batch_multi_rhs_solve_64_timed(&matrix_u64, &b_cols_u64, n, k, &primes_v2)
            {
                // GPU CRT reconstruction
                let crt_start = Instant::now();
                let _solutions_v2: Vec<Vec<Rational>> = reconstruct_solutions_v2_gpu(
                    &backend, &result, n, k, &gpu_crt_v2
                );
                let crt_ms = crt_start.elapsed().as_secs_f64() * 1000.0;

                v2_gpu_times.push(timing.total_ms);
                v2_crt_times.push(crt_ms);
                v2_total_times.push(timing.total_ms + crt_ms);
            } else {
                println!("Warning: V2 solve failed for n={}", n);
                continue;
            }
        }
        if v2_gpu_times.is_empty() {
            println!("All V2 solves failed for n={}, skipping", n);
            continue;
        }
        let v2_gpu_avg = v2_gpu_times.iter().sum::<f64>() / v2_gpu_times.len() as f64;
        let v2_crt_avg = v2_crt_times.iter().sum::<f64>() / v2_crt_times.len() as f64;
        let v2_total_avg = v2_total_times.iter().sum::<f64>() / v2_total_times.len() as f64;

        let gpu_speedup = v1_gpu_avg / v2_gpu_avg;
        let crt_speedup = v1_crt_avg / v2_crt_avg;
        let total_speedup = v1_total_avg / v2_total_avg;
        let prime_reduction = num_primes_v1 as f64 / num_primes_v2 as f64;

        println!(
            "│ {:>6} │ {:>9} │ {:>9} │ {:>10.2} │ {:>10.2} │ {:>10.2} │ {:>10.2} │ {:>11.2}x  │ {:>9.2}x  │ {:>7.2}x │",
            n, num_primes_v1, num_primes_v2, v1_gpu_avg, v2_gpu_avg, v1_crt_avg, v2_crt_avg, crt_speedup, total_speedup, prime_reduction
        );

        results.push(V2Result {
            n,
            k,
            v1_primes: num_primes_v1,
            v2_primes: num_primes_v2,
            v1_gpu_ms: v1_gpu_avg,
            v2_gpu_ms: v2_gpu_avg,
            v1_crt_ms: v1_crt_avg,
            v2_crt_ms: v2_crt_avg,
            v1_total_ms: v1_total_avg,
            v2_total_ms: v2_total_avg,
            gpu_speedup,
            crt_speedup,
            total_speedup,
            prime_reduction,
        });
    }

    println!("└────────┴───────────┴───────────┴────────────┴────────────┴────────────┴────────────┴──────────────┴────────────┴──────────┘");
    println!();

    // Summary
    if !results.is_empty() {
        let avg_gpu_speedup = results.iter().map(|r| r.gpu_speedup).sum::<f64>() / results.len() as f64;
        let avg_total_speedup = results.iter().map(|r| r.total_speedup).sum::<f64>() / results.len() as f64;
        let avg_reduction = results.iter().map(|r| r.prime_reduction).sum::<f64>() / results.len() as f64;
        let max_gpu_speedup = results.iter().map(|r| r.gpu_speedup).fold(0.0f64, f64::max);
        let max_n = results.iter().find(|r| r.gpu_speedup == max_gpu_speedup).map(|r| r.n).unwrap_or(0);

        println!("Summary:");
        println!("  Average GPU speedup (V2 over V1): {:.2}x", avg_gpu_speedup);
        println!("  Average total speedup (incl CRT): {:.2}x", avg_total_speedup);
        println!("  Average prime reduction: {:.2}x", avg_reduction);
        println!("  Maximum GPU speedup: {:.2}x at n={}", max_gpu_speedup, max_n);
        println!();
    }

    // Export to CSV
    if let Some(path) = export {
        export_v2_results(&results, &path);
        println!("Results exported to: {}", path.display());
    }
}

#[cfg(feature = "cuda")]
fn generate_test_matrix(n: usize, seed: u64) -> Vec<num_bigint::BigInt> {
    use num_bigint::BigInt;
    let mut matrix: Vec<BigInt> = Vec::with_capacity(n * n);
    let mut state = seed;

    for i in 0..n {
        for j in 0..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = if i == j {
                // Diagonal: 100 to 999 (ensures dominance)
                100 + (state % 900) as i64
            } else {
                // Off-diagonal: -50 to 50
                ((state % 101) as i64) - 50
            };
            matrix.push(BigInt::from(val));
        }
    }
    matrix
}

#[cfg(feature = "cuda")]
fn generate_test_vector(n: usize, seed: u64) -> Vec<num_bigint::BigInt> {
    use num_bigint::BigInt;
    let mut vec: Vec<BigInt> = Vec::with_capacity(n);
    let mut state = seed;

    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = ((state % 201) as i64) - 100;
        vec.push(BigInt::from(val));
    }
    vec
}

#[cfg(feature = "cuda")]
fn reconstruct_solutions_v1_gpu(
    backend: &parallel_lift_cuda::CudaBackend,
    solutions_per_prime: &[Vec<Vec<u32>>],
    n: usize,
    k: usize,
    gpu_crt: &parallel_lift_cuda::GpuCrtPrecomputed,
) -> Vec<Vec<parallel_lift_core::Rational>> {
    use parallel_lift_core::Rational;
    use num_bigint::{BigInt, Sign};

    let num_primes = solutions_per_prime.len();
    let num_values = n * k;

    // Flatten residues: [num_primes][col][row] -> [num_values][num_primes]
    let mut residues_flat = vec![0u32; num_values * num_primes];
    for pi in 0..num_primes {
        for col_idx in 0..k {
            for row_idx in 0..n {
                let v = col_idx * n + row_idx;  // value index
                residues_flat[v * num_primes + pi] = solutions_per_prime[pi][col_idx][row_idx];
            }
        }
    }

    // GPU CRT reconstruction
    let (limbs_vec, signs) = backend.gpu_crt_reconstruct(&residues_flat, num_values, gpu_crt);

    // Convert to Rational (grouped by column)
    let mut result = Vec::with_capacity(k);
    for col_idx in 0..k {
        let mut col = Vec::with_capacity(n);
        for row_idx in 0..n {
            let v = col_idx * n + row_idx;
            let is_negative = signs[v];
            let bigint = if limbs_vec[v].iter().all(|&x| x == 0) {
                BigInt::from(0)
            } else {
                let sign = if is_negative { Sign::Minus } else { Sign::Plus };
                BigInt::from_slice(sign, &limbs_vec[v])
            };
            col.push(Rational::from_bigint(bigint));
        }
        result.push(col);
    }
    result
}

#[cfg(feature = "cuda")]
fn reconstruct_solutions_v2_gpu(
    backend: &parallel_lift_cuda::CudaBackend,
    solutions_per_prime: &[Vec<Vec<u64>>],
    n: usize,
    k: usize,
    gpu_crt: &parallel_lift_cuda::GpuCrtPrecomputed64,
) -> Vec<Vec<parallel_lift_core::Rational>> {
    use parallel_lift_core::Rational;
    use num_bigint::{BigInt, Sign};

    let num_primes = solutions_per_prime.len();
    let num_values = n * k;

    // Flatten residues: [num_primes][col][row] -> [num_values][num_primes]
    let mut residues_flat = vec![0u64; num_values * num_primes];
    for pi in 0..num_primes {
        for col_idx in 0..k {
            for row_idx in 0..n {
                let v = col_idx * n + row_idx;  // value index
                residues_flat[v * num_primes + pi] = solutions_per_prime[pi][col_idx][row_idx];
            }
        }
    }

    // GPU CRT reconstruction (64-bit)
    let (limbs_vec, signs) = backend.gpu_crt_reconstruct_64(&residues_flat, num_values, gpu_crt);

    // Convert to Rational (grouped by column)
    let mut result = Vec::with_capacity(k);
    for col_idx in 0..k {
        let mut col = Vec::with_capacity(n);
        for row_idx in 0..n {
            let v = col_idx * n + row_idx;
            let is_negative = signs[v];
            let bigint = if limbs_vec[v].iter().all(|&x| x == 0) {
                BigInt::from(0)
            } else {
                let sign = if is_negative { Sign::Minus } else { Sign::Plus };
                BigInt::from_slice(sign, &limbs_vec[v])
            };
            col.push(Rational::from_bigint(bigint));
        }
        result.push(col);
    }
    result
}

#[cfg(feature = "cuda")]
struct V2Result {
    n: usize,
    k: usize,
    v1_primes: usize,
    v2_primes: usize,
    v1_gpu_ms: f64,
    v2_gpu_ms: f64,
    v1_crt_ms: f64,
    v2_crt_ms: f64,
    v1_total_ms: f64,
    v2_total_ms: f64,
    gpu_speedup: f64,
    crt_speedup: f64,
    total_speedup: f64,
    prime_reduction: f64,
}

#[cfg(feature = "cuda")]
fn export_v2_results(results: &[V2Result], path: &PathBuf) {
    use std::fs::File;
    use std::io::Write;

    let mut file = match File::create(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error creating file: {}", e);
            return;
        }
    };

    writeln!(file, "n,k,v1_primes,v2_primes,v1_gpu_ms,v2_gpu_ms,v1_crt_ms,v2_crt_ms,v1_total_ms,v2_total_ms,gpu_speedup,crt_speedup,total_speedup,prime_reduction").unwrap();
    for r in results {
        writeln!(
            file,
            "{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            r.n, r.k, r.v1_primes, r.v2_primes, r.v1_gpu_ms, r.v2_gpu_ms, r.v1_crt_ms, r.v2_crt_ms, r.v1_total_ms, r.v2_total_ms, r.gpu_speedup, r.crt_speedup, r.total_speedup, r.prime_reduction
        ).unwrap();
    }
}

#[cfg(not(feature = "cuda"))]
pub fn run_v2_benchmark(
    _sizes_str: &str,
    _k: usize,
    _trials: usize,
    _export: Option<PathBuf>,
) {
    println!("CUDA feature not enabled. Compile with --features cuda");
}
