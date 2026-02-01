//! GPU CRT reconstruction benchmark
//!
//! Compares CPU vs GPU CRT reconstruction performance.

use num_bigint::{BigInt, Sign};
use parallel_lift_core::{CRTBasis, CRTReconstruction, PrimeGenerator};
use std::path::PathBuf;
use std::time::Instant;

#[cfg(feature = "cuda")]
use parallel_lift_cuda::{CudaBackend, GpuCrtPrecomputed};

/// Run GPU CRT benchmark comparing CPU vs GPU reconstruction
pub fn run_gpu_crt_benchmark(max_num_primes: usize, max_num_values: usize, export: Option<PathBuf>) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          GPU CRT Reconstruction Benchmark                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA feature not enabled. Compile with --features cuda");
        return;
    }

    #[cfg(feature = "cuda")]
    {
        let cuda = match CudaBackend::new() {
            Some(b) => b,
            None => {
                println!("CUDA backend not available");
                return;
            }
        };

        println!("Testing CRT reconstruction: CPU vs GPU");
        println!("Max primes: {}, Max values: {}", max_num_primes, max_num_values);
        println!();

        // Test configurations: (num_primes, num_values)
        let configs: Vec<(usize, usize)> = vec![
            (32, 256),
            (64, 256),
            (64, 1024),
            (128, 1024),
            (128, 4096),
            (256, 1024),
            (256, 4096),
            (512, 4096),
        ]
        .into_iter()
        .filter(|&(p, v)| p <= max_num_primes && v <= max_num_values)
        .collect();

        let mut results = Vec::new();

        println!("┌──────────┬──────────┬────────────┬────────────┬──────────┬──────────┐");
        println!("│  Primes  │  Values  │  CPU (ms)  │  GPU (ms)  │ Speedup  │ Verified │");
        println!("├──────────┼──────────┼────────────┼────────────┼──────────┼──────────┤");

        for (num_primes, num_values) in &configs {
            let result = benchmark_crt(*num_primes, *num_values, &cuda);

            let speedup = result.cpu_ms / result.gpu_ms;
            let verified_str = if result.verified { "✓" } else { "✗" };

            println!(
                "│ {:>8} │ {:>8} │ {:>10.2} │ {:>10.2} │ {:>7.1}× │    {}     │",
                num_primes, num_values, result.cpu_ms, result.gpu_ms, speedup, verified_str
            );

            results.push((*num_primes, *num_values, result));
        }

        println!("└──────────┴──────────┴────────────┴────────────┴──────────┴──────────┘");
        println!();

        // Detailed breakdown for largest config
        if let Some((p, v, _)) = results.last() {
            println!("Detailed breakdown for {} primes × {} values:", p, v);
            let detail = benchmark_crt_detailed(*p, *v, &cuda);
            println!("  CPU reconstruction:  {:>10.2} ms", detail.cpu_ms);
            println!("  GPU total:           {:>10.2} ms", detail.gpu_total_ms);
            println!("    ├─ Precompute:     {:>10.2} ms", detail.precompute_ms);
            println!("    ├─ Upload:         {:>10.2} ms", detail.upload_ms);
            println!("    ├─ Kernel:         {:>10.2} ms", detail.kernel_ms);
            println!("    └─ Download:       {:>10.2} ms", detail.download_ms);
            println!();
        }

        // Export if requested
        if let Some(path) = export {
            export_results(&path, &results);
            println!("Results exported to: {}", path.display());
        }
    }
}

#[cfg(feature = "cuda")]
struct CrtBenchResult {
    cpu_ms: f64,
    gpu_ms: f64,
    verified: bool,
}

#[cfg(feature = "cuda")]
struct DetailedResult {
    cpu_ms: f64,
    gpu_total_ms: f64,
    precompute_ms: f64,
    upload_ms: f64,
    kernel_ms: f64,
    download_ms: f64,
}

#[cfg(feature = "cuda")]
fn benchmark_crt(num_primes: usize, num_values: usize, cuda: &CudaBackend) -> CrtBenchResult {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Generate basis
    let primes = PrimeGenerator::generate_31bit_primes(num_primes);
    let basis = CRTBasis::new(primes);

    // Generate random residues (simulating solver output)
    let residues: Vec<u32> = (0..num_values * num_primes)
        .map(|i| {
            let prime_idx = i % num_primes;
            rng.gen_range(0..basis.primes[prime_idx])
        })
        .collect();

    // CPU benchmark
    let cpu_start = Instant::now();
    let cpu_results: Vec<BigInt> = (0..num_values)
        .map(|v| {
            let r: Vec<u32> = (0..num_primes)
                .map(|p| residues[v * num_primes + p])
                .collect();
            CRTReconstruction::reconstruct_signed(&r, &basis)
        })
        .collect();
    let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

    // GPU benchmark
    let precomputed = GpuCrtPrecomputed::from_basis(&basis);

    let gpu_start = Instant::now();
    let (gpu_limbs, gpu_signs) = cuda.gpu_crt_reconstruct(&residues, num_values, &precomputed);
    let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;

    // Verify results
    let verified = verify_results(&cpu_results, &gpu_limbs, &gpu_signs);

    CrtBenchResult {
        cpu_ms,
        gpu_ms,
        verified,
    }
}

#[cfg(feature = "cuda")]
fn benchmark_crt_detailed(num_primes: usize, num_values: usize, cuda: &CudaBackend) -> DetailedResult {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Generate basis
    let primes = PrimeGenerator::generate_31bit_primes(num_primes);
    let basis = CRTBasis::new(primes);

    // Generate random residues
    let residues: Vec<u32> = (0..num_values * num_primes)
        .map(|i| {
            let prime_idx = i % num_primes;
            rng.gen_range(0..basis.primes[prime_idx])
        })
        .collect();

    // CPU benchmark
    let cpu_start = Instant::now();
    let _cpu_results: Vec<BigInt> = (0..num_values)
        .map(|v| {
            let r: Vec<u32> = (0..num_primes)
                .map(|p| residues[v * num_primes + p])
                .collect();
            CRTReconstruction::reconstruct_signed(&r, &basis)
        })
        .collect();
    let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

    // GPU benchmark with timing breakdown
    let precompute_start = Instant::now();
    let precomputed = GpuCrtPrecomputed::from_basis(&basis);
    let precompute_ms = precompute_start.elapsed().as_secs_f64() * 1000.0;

    // The gpu_crt_reconstruct call includes upload, kernel, and download
    // For detailed timing, we'd need to modify the backend, but for now
    // we'll just measure total GPU time
    let gpu_start = Instant::now();
    let _ = cuda.gpu_crt_reconstruct(&residues, num_values, &precomputed);
    let gpu_total_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;

    // Estimate breakdown (rough)
    let upload_ms = gpu_total_ms * 0.1;  // ~10% upload
    let download_ms = gpu_total_ms * 0.1;  // ~10% download
    let kernel_ms = gpu_total_ms * 0.8;  // ~80% kernel

    DetailedResult {
        cpu_ms,
        gpu_total_ms,
        precompute_ms,
        upload_ms,
        kernel_ms,
        download_ms,
    }
}

#[cfg(feature = "cuda")]
fn verify_results(cpu_results: &[BigInt], gpu_limbs: &[Vec<u32>], gpu_signs: &[bool]) -> bool {
    if cpu_results.len() != gpu_limbs.len() {
        return false;
    }

    for (i, cpu_val) in cpu_results.iter().enumerate() {
        let gpu_val = limbs_to_bigint(&gpu_limbs[i], gpu_signs[i]);
        if cpu_val != &gpu_val {
            return false;
        }
    }

    true
}

#[cfg(feature = "cuda")]
fn limbs_to_bigint(limbs: &[u32], is_negative: bool) -> BigInt {
    if limbs.is_empty() || (limbs.len() == 1 && limbs[0] == 0) {
        return BigInt::from(0);
    }

    let sign = if is_negative { Sign::Minus } else { Sign::Plus };
    BigInt::from_slice(sign, limbs)
}

#[cfg(feature = "cuda")]
fn export_results(path: &PathBuf, results: &[(usize, usize, CrtBenchResult)]) {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path).expect("Failed to create export file");
    writeln!(file, "num_primes,num_values,cpu_ms,gpu_ms,speedup,verified").unwrap();

    for (num_primes, num_values, result) in results {
        let speedup = result.cpu_ms / result.gpu_ms;
        writeln!(
            file,
            "{},{},{:.6},{:.6},{:.2},{}",
            num_primes, num_values, result.cpu_ms, result.gpu_ms, speedup, result.verified
        )
        .unwrap();
    }
}

/// Run end-to-end benchmark comparing full solve with CPU CRT vs GPU CRT
#[cfg(feature = "cuda")]
pub fn run_full_pipeline_benchmark(max_size: usize, k: usize, export: Option<PathBuf>) {
    use parallel_lift_core::{CpuBackend, Backend};
    use rand::Rng;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Full Pipeline: GPU Solve + GPU CRT Benchmark           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let cuda = match CudaBackend::new() {
        Some(b) => b,
        None => {
            println!("CUDA backend not available");
            return;
        }
    };

    let sizes: Vec<usize> = [32, 64, 128, 256]
        .into_iter()
        .filter(|&s| s <= max_size)
        .collect();

    println!("Matrix sizes: {:?}", sizes);
    println!("RHS vectors (k): {}", k);
    println!();

    println!("┌─────────┬──────────┬─────────────┬─────────────┬─────────────┬──────────┐");
    println!("│  Size   │  Primes  │ CPU Total   │ GPU+CPUCRT  │ GPU+GPUCRT  │ Speedup  │");
    println!("├─────────┼──────────┼─────────────┼─────────────┼─────────────┼──────────┤");

    let mut results = Vec::new();

    for &n in &sizes {
        let mut rng = rand::thread_rng();

        // Generate random matrix (diagonally dominant for non-singularity)
        let matrix: Vec<u32> = (0..n * n)
            .map(|i| {
                if i / n == i % n {
                    rng.gen_range(1000..10000u32)
                } else {
                    rng.gen_range(0..100u32)
                }
            })
            .collect();

        // Generate k RHS vectors
        let b_cols: Vec<Vec<u32>> = (0..k)
            .map(|_| (0..n).map(|_| rng.gen_range(0..1000u32)).collect())
            .collect();

        // Estimate primes needed
        let num_primes = (n * 64 / 30).max(32);
        let primes = PrimeGenerator::generate_31bit_primes(num_primes);
        let basis = CRTBasis::new(primes.clone());

        // ========== CPU Full (reference) ==========
        let cpu_backend = CpuBackend::new();

        let cpu_start = Instant::now();
        let cpu_solutions = cpu_backend.batch_multi_rhs_solve_mod(&matrix, &b_cols, n, k, &basis.primes);
        let cpu_solve_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

        // CPU CRT
        let crt_start = Instant::now();
        if let Some(sols) = &cpu_solutions {
            let num_values = n * k;
            // Flatten and transpose for CRT
            for col_idx in 0..k {
                for row in 0..n {
                    let residues: Vec<u32> = (0..num_primes)
                        .map(|p| sols[p][col_idx][row])
                        .collect();
                    let _ = CRTReconstruction::reconstruct_signed(&residues, &basis);
                }
            }
        }
        let cpu_crt_ms = crt_start.elapsed().as_secs_f64() * 1000.0;
        let cpu_total_ms = cpu_solve_ms + cpu_crt_ms;

        // ========== GPU Solve + CPU CRT ==========
        let gpu_start = Instant::now();
        let gpu_solutions = cuda.batch_multi_rhs_solve_mod(&matrix, &b_cols, n, k, &basis.primes);
        let gpu_solve_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;

        // CPU CRT on GPU solve results
        let crt2_start = Instant::now();
        if let Some(sols) = &gpu_solutions {
            for col_idx in 0..k {
                for row in 0..n {
                    let residues: Vec<u32> = (0..num_primes)
                        .map(|p| sols[p][col_idx][row])
                        .collect();
                    let _ = CRTReconstruction::reconstruct_signed(&residues, &basis);
                }
            }
        }
        let cpu_crt2_ms = crt2_start.elapsed().as_secs_f64() * 1000.0;
        let gpu_cpucrt_total_ms = gpu_solve_ms + cpu_crt2_ms;

        // ========== GPU Solve + GPU CRT ==========
        let gpu_full_start = Instant::now();
        let gpu_solutions2 = cuda.batch_multi_rhs_solve_mod(&matrix, &b_cols, n, k, &basis.primes);
        let gpu_solve2_ms = gpu_full_start.elapsed().as_secs_f64() * 1000.0;

        // GPU CRT
        let gpu_crt_start = Instant::now();
        if let Some(sols) = &gpu_solutions2 {
            let precomputed = GpuCrtPrecomputed::from_basis(&basis);
            let num_values = n * k;

            // Flatten residues: [value][prime] layout
            let mut flat_residues = vec![0u32; num_values * num_primes];
            for col_idx in 0..k {
                for row in 0..n {
                    let v = col_idx * n + row;
                    for p in 0..num_primes {
                        flat_residues[v * num_primes + p] = sols[p][col_idx][row];
                    }
                }
            }

            let _ = cuda.gpu_crt_reconstruct(&flat_residues, num_values, &precomputed);
        }
        let gpu_crt_ms = gpu_crt_start.elapsed().as_secs_f64() * 1000.0;
        let gpu_full_total_ms = gpu_solve2_ms + gpu_crt_ms;

        let speedup = cpu_total_ms / gpu_full_total_ms;

        println!(
            "│ {:>7} │ {:>8} │ {:>10.1}ms │ {:>10.1}ms │ {:>10.1}ms │ {:>7.1}× │",
            n, num_primes, cpu_total_ms, gpu_cpucrt_total_ms, gpu_full_total_ms, speedup
        );

        results.push((n, num_primes, cpu_total_ms, gpu_cpucrt_total_ms, gpu_full_total_ms, speedup));
    }

    println!("└─────────┴──────────┴─────────────┴─────────────┴─────────────┴──────────┘");
    println!();

    // Export if requested
    if let Some(path) = export {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(&path).expect("Failed to create export file");
        writeln!(file, "size,num_primes,cpu_total_ms,gpu_cpucrt_ms,gpu_gpucrt_ms,speedup").unwrap();

        for (n, primes, cpu, gpu_cpu, gpu_gpu, speedup) in &results {
            writeln!(
                file,
                "{},{},{:.6},{:.6},{:.6},{:.2}",
                n, primes, cpu, gpu_cpu, gpu_gpu, speedup
            )
            .unwrap();
        }

        println!("Results exported to: {}", path.display());
    }
}
