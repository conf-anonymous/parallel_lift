//! Pipeline Benchmark
//!
//! Compares pipelined vs sequential execution to measure overlap efficiency.

use num_bigint::BigInt;
use parallel_lift_core::{
    CRTBasis, CpuBackend, Solver, PipelineSolver, PipelineConfig, AsyncPipelineSolver,
    primes::PrimeGenerator,
};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::path::PathBuf;
use std::time::Instant;
use std::fs::File;
use std::io::Write;

/// Result from pipeline benchmark
#[derive(Debug, Clone)]
pub struct PipelineBenchResult {
    pub n: usize,
    pub k: usize,
    pub num_primes: usize,
    pub batch_size: usize,
    pub sequential_ms: f64,
    pub pipeline_ms: f64,
    pub async_pipeline_ms: f64,
    pub speedup_sync: f64,
    pub speedup_async: f64,
    pub overlap_efficiency: f64,
}

/// Run pipeline benchmark
pub fn run_pipeline_benchmark(
    max_size: usize,
    k: usize,
    export: Option<PathBuf>,
) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         Pipeline vs Sequential Benchmark                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Configuration:");
    println!("  Max size:        {}", max_size);
    println!("  RHS vectors (k): {}", k);
    println!();

    println!("┌────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Why Pipelining Matters                                                                    │");
    println!("├────────────────────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Sequential: CPU residue → GPU solve → CPU CRT (each waits for previous)                  │");
    println!("│ Pipelined:  Overlap residue(i+1) with solve(i) with CRT(i-1)                             │");
    println!("│ With GPU: hides CPU latency behind GPU computation                                        │");
    println!("└────────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Test configurations
    let sizes: Vec<usize> = (0..)
        .map(|i| 16 * (1 << i))
        .take_while(|&s| s <= max_size)
        .collect();

    let batch_sizes = [4, 8, 16];
    let num_primes = 32;

    let mut results = Vec::new();

    println!("┌──────┬─────┬────────┬───────┬────────────┬────────────┬────────────┬─────────┬─────────┬─────────┐");
    println!("│   n  │  k  │ Primes │ Batch │  Seq(ms)   │ Pipe(ms)   │ Async(ms)  │ Sync↑   │ Async↑  │ Overlap │");
    println!("├──────┼─────┼────────┼───────┼────────────┼────────────┼────────────┼─────────┼─────────┼─────────┤");

    for n in &sizes {
        for &batch_size in &batch_sizes {
            let result = benchmark_pipeline(*n, k, num_primes, batch_size);
            println!(
                "│ {:>4} │ {:>3} │ {:>6} │ {:>5} │ {:>10.2} │ {:>10.2} │ {:>10.2} │ {:>6.2}x │ {:>6.2}x │ {:>6.1}% │",
                result.n,
                result.k,
                result.num_primes,
                result.batch_size,
                result.sequential_ms,
                result.pipeline_ms,
                result.async_pipeline_ms,
                result.speedup_sync,
                result.speedup_async,
                result.overlap_efficiency * 100.0,
            );
            results.push(result);
        }
    }

    println!("└──────┴─────┴────────┴───────┴────────────┴────────────┴────────────┴─────────┴─────────┴─────────┘");

    // Summary
    println!();
    println!("Key Observations:");

    let avg_sync_speedup: f64 = results.iter().map(|r| r.speedup_sync).sum::<f64>() / results.len() as f64;
    let avg_async_speedup: f64 = results.iter().map(|r| r.speedup_async).sum::<f64>() / results.len() as f64;
    let avg_overlap: f64 = results.iter().map(|r| r.overlap_efficiency).sum::<f64>() / results.len() as f64;

    println!("  • Average sync pipeline speedup: {:.2}x", avg_sync_speedup);
    println!("  • Average async pipeline speedup: {:.2}x", avg_async_speedup);
    println!("  • Average overlap efficiency: {:.1}%", avg_overlap * 100.0);

    let best = results.iter().max_by(|a, b| a.speedup_async.partial_cmp(&b.speedup_async).unwrap());
    if let Some(b) = best {
        println!("  • Best async speedup: {:.2}x at n={}, batch={}", b.speedup_async, b.n, b.batch_size);
    }

    // Export if requested
    if let Some(path) = export {
        export_results(&path, &results);
        println!("\n✓ Results exported to: {}", path.display());
    }
}

fn benchmark_pipeline(n: usize, k: usize, num_primes: usize, batch_size: usize) -> PipelineBenchResult {
    let seed = 42u64;
    let mut rng = StdRng::seed_from_u64(seed);

    // Generate diagonally dominant matrix (ensures non-singularity)
    let matrix: Vec<BigInt> = (0..n*n)
        .map(|idx| {
            let row = idx / n;
            let col = idx % n;
            if row == col {
                BigInt::from(n as i64 * 2 + rng.gen_range(1..10i64))
            } else {
                BigInt::from(rng.gen_range(-10i64..10))
            }
        })
        .collect();

    let b_cols: Vec<Vec<BigInt>> = (0..k)
        .map(|_| (0..n).map(|_| BigInt::from(rng.gen_range(-100i64..100))).collect())
        .collect();

    // Sequential solver
    let primes = PrimeGenerator::generate_31bit_primes(num_primes);
    let basis = CRTBasis::new(primes);
    let sequential_solver = Solver::new(CpuBackend::new());

    let seq_start = Instant::now();
    let (_seq_results, _seq_timings) = sequential_solver.solve_multi_rhs(&matrix, &b_cols, &basis);
    let sequential_ms = seq_start.elapsed().as_secs_f64() * 1000.0;

    // Sync pipelined solver
    let pipeline_config = PipelineConfig {
        batch_size,
        total_primes: num_primes,
        cpu_threads: 4,
    };
    let pipeline_solver = PipelineSolver::with_config(CpuBackend::new(), pipeline_config.clone());

    let pipe_start = Instant::now();
    let (_pipe_results, _pipe_timings, pipe_stats) = pipeline_solver.solve_multi_rhs(&matrix, &b_cols);
    let pipeline_ms = pipe_start.elapsed().as_secs_f64() * 1000.0;

    // Async pipelined solver
    let async_solver = AsyncPipelineSolver::with_config(CpuBackend::new(), pipeline_config);

    // For async, we need to clone since it takes ownership
    let matrix_clone = matrix.clone();
    let b_clone = b_cols[0].clone(); // Single RHS for async test

    let async_start = Instant::now();
    let (_async_result, _async_timings, async_stats) = async_solver.solve_async(matrix_clone, b_clone);
    let async_pipeline_ms = async_start.elapsed().as_secs_f64() * 1000.0;

    let speedup_sync = if pipeline_ms > 0.0 { sequential_ms / pipeline_ms } else { 1.0 };
    let speedup_async = if async_pipeline_ms > 0.0 { sequential_ms / async_pipeline_ms } else { 1.0 };

    PipelineBenchResult {
        n,
        k,
        num_primes,
        batch_size,
        sequential_ms,
        pipeline_ms,
        async_pipeline_ms,
        speedup_sync,
        speedup_async,
        overlap_efficiency: async_stats.overlap_efficiency,
    }
}

fn export_results(path: &PathBuf, results: &[PipelineBenchResult]) {
    let mut file = File::create(path).expect("Failed to create file");
    writeln!(file, "n,k,num_primes,batch_size,sequential_ms,pipeline_ms,async_ms,speedup_sync,speedup_async,overlap").unwrap();
    for r in results {
        writeln!(
            file,
            "{},{},{},{},{:.6},{:.6},{:.6},{:.4},{:.4},{:.4}",
            r.n, r.k, r.num_primes, r.batch_size, r.sequential_ms, r.pipeline_ms,
            r.async_pipeline_ms, r.speedup_sync, r.speedup_async, r.overlap_efficiency
        ).unwrap();
    }
}
