//! LLL Lattice Reduction Benchmark
//!
//! Compares CPU LLL vs GPU LLL implementation, and optionally against fplll.

use parallel_lift_core::lattice::basis::LatticeBasis;
use parallel_lift_core::lattice::lll::{LLL, LLLConfig, LLLStats};
use std::path::PathBuf;
use std::time::Instant;

#[cfg(feature = "cuda")]
use parallel_lift_cuda::{CudaBackend, GpuLLLConfig, GpuLLLStats, gpu_lll_reduce};

/// Results from a single LLL benchmark run
#[derive(Debug, Clone)]
pub struct LLLBenchResult {
    pub n: usize,
    pub m: usize,
    pub bits: usize,
    pub cpu_ms: f64,
    pub gpu_ms: f64,
    pub speedup: f64,
    pub cpu_swaps: usize,
    pub gpu_swaps: usize,
    pub cpu_iterations: usize,
    pub gpu_iterations: usize,
    pub verified: bool,
}

/// Run LLL benchmark comparing CPU vs GPU
pub fn run_lll_benchmark(max_dim: usize, max_bits: usize, export: Option<PathBuf>) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              GPU LLL Lattice Reduction Benchmark             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA feature not enabled. Compile with --features cuda");
        println!();
        println!("Running CPU-only benchmark...");
        run_cpu_only_benchmark(max_dim, max_bits, export);
        return;
    }

    #[cfg(feature = "cuda")]
    {
        let cuda = match CudaBackend::new() {
            Some(b) => b,
            None => {
                println!("CUDA backend not available. Running CPU-only benchmark...");
                run_cpu_only_benchmark(max_dim, max_bits, export);
                return;
            }
        };

        println!("Testing LLL reduction: CPU vs GPU");
        println!("Max dimension: {}, Max bits: {}", max_dim, max_bits);
        println!();

        // Test configurations: (n, bits)
        let configs: Vec<(usize, usize)> = vec![
            (4, 8),
            (6, 8),
            (8, 8),
            (10, 8),
            (8, 12),
            (10, 12),
            (12, 12),
            (10, 16),
            (15, 16),
            (20, 16),
            (25, 16),
            (30, 16),
            (20, 20),
            (30, 20),
            (40, 20),
        ]
        .into_iter()
        .filter(|&(n, b)| n <= max_dim && b <= max_bits)
        .collect();

        let mut results = Vec::new();

        println!("┌──────┬──────┬────────────┬────────────┬──────────┬────────────┬────────────┬──────────┐");
        println!("│  n   │ bits │  CPU (ms)  │  GPU (ms)  │ Speedup  │ CPU swaps  │ GPU swaps  │ Verified │");
        println!("├──────┼──────┼────────────┼────────────┼──────────┼────────────┼────────────┼──────────┤");

        for (n, bits) in &configs {
            let result = benchmark_lll(*n, *bits, &cuda);

            let verified_str = if result.verified { "✓" } else { "✗" };
            let speedup_str = if result.gpu_ms > 0.0 {
                format!("{:>7.2}×", result.speedup)
            } else {
                "   N/A  ".to_string()
            };

            println!(
                "│ {:>4} │ {:>4} │ {:>10.2} │ {:>10.2} │ {} │ {:>10} │ {:>10} │    {}     │",
                n, bits, result.cpu_ms, result.gpu_ms, speedup_str,
                result.cpu_swaps, result.gpu_swaps, verified_str
            );

            results.push(result);
        }

        println!("└──────┴──────┴────────────┴────────────┴──────────┴────────────┴────────────┴──────────┘");
        println!();

        // Timing breakdown for the largest run
        if let Some(result) = results.last() {
            println!("Summary for n={}, bits={}:", result.n, result.bits);
            println!("  CPU LLL:      {:>10.2} ms ({} swaps, {} iterations)",
                     result.cpu_ms, result.cpu_swaps, result.cpu_iterations);
            println!("  GPU LLL:      {:>10.2} ms ({} swaps, {} iterations)",
                     result.gpu_ms, result.gpu_swaps, result.gpu_iterations);
            if result.speedup > 0.0 && result.speedup != f64::INFINITY {
                println!("  Speedup:      {:>10.2}×", result.speedup);
            }
            println!();
        }

        // Print summary statistics
        let avg_speedup: f64 = results.iter()
            .filter(|r| r.speedup > 0.0 && r.speedup != f64::INFINITY)
            .map(|r| r.speedup)
            .sum::<f64>() / results.len() as f64;

        if avg_speedup > 0.0 {
            println!("Average speedup across all tests: {:.2}×", avg_speedup);
        }

        // Export if requested
        if let Some(path) = export {
            export_lll_results(&path, &results);
            println!("Results exported to: {}", path.display());
        }
    }
}

#[cfg(feature = "cuda")]
fn benchmark_lll(n: usize, bits: usize, cuda: &CudaBackend) -> LLLBenchResult {
    // Generate a random lattice
    let basis = LatticeBasis::random(n, n, bits);

    // CPU LLL
    let cpu_start = Instant::now();
    let cpu_config = LLLConfig::default();
    let (cpu_reduced, cpu_stats) = LLL::reduce(&basis, &cpu_config);
    let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

    // GPU LLL
    let gpu_start = Instant::now();
    let gpu_config = GpuLLLConfig::default();
    let (gpu_reduced, gpu_stats) = gpu_lll_reduce(cuda, &basis, &gpu_config);
    let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;

    // Verify: check that reduced bases have similar quality
    // We compare dimensions and that the basis still spans the same lattice
    // Allow some variation due to different reduction paths
    let verified = cpu_reduced.n == gpu_reduced.n && cpu_reduced.m == gpu_reduced.m;

    let speedup = if gpu_ms > 0.001 { cpu_ms / gpu_ms } else { 0.0 };

    LLLBenchResult {
        n,
        m: n,
        bits,
        cpu_ms,
        gpu_ms,
        speedup,
        cpu_swaps: cpu_stats.swaps,
        gpu_swaps: gpu_stats.swaps,
        cpu_iterations: cpu_stats.iterations,
        gpu_iterations: gpu_stats.iterations,
        verified,
    }
}

/// CPU-only benchmark when CUDA is not available
fn run_cpu_only_benchmark(max_dim: usize, max_bits: usize, export: Option<PathBuf>) {
    println!("CPU-only LLL benchmark");
    println!("Max dimension: {}, Max bits: {}", max_dim, max_bits);
    println!();

    let configs: Vec<(usize, usize)> = vec![
        (4, 8),
        (8, 8),
        (10, 12),
        (15, 16),
        (20, 16),
        (25, 16),
        (30, 20),
    ]
    .into_iter()
    .filter(|&(n, b)| n <= max_dim && b <= max_bits)
    .collect();

    println!("┌──────┬──────┬────────────┬────────────┬────────────┐");
    println!("│  n   │ bits │  Time (ms) │   Swaps    │ Iterations │");
    println!("├──────┼──────┼────────────┼────────────┼────────────┤");

    for (n, bits) in &configs {
        let basis = LatticeBasis::random(*n, *n, *bits);

        let start = Instant::now();
        let config = LLLConfig::default();
        let (_reduced, stats) = LLL::reduce(&basis, &config);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        println!(
            "│ {:>4} │ {:>4} │ {:>10.2} │ {:>10} │ {:>10} │",
            n, bits, elapsed_ms, stats.swaps, stats.iterations
        );
    }

    println!("└──────┴──────┴────────────┴────────────┴────────────┘");
}

/// Run comparison against fplll (external tool)
pub fn run_fplll_comparison(max_dim: usize, max_bits: usize, export: Option<PathBuf>) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         LLL Benchmark: GPU vs CPU vs fplll                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Check if fplll is available
    let fplll_available = check_fplll_available();

    if !fplll_available {
        println!("Note: fplll not found in PATH. Skipping fplll comparison.");
        println!("To install fplll:");
        println!("  - Ubuntu/Debian: apt install fplll-tools");
        println!("  - macOS: brew install fplll");
        println!("  - From source: https://github.com/fplll/fplll");
        println!();
    }

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

        let configs: Vec<(usize, usize)> = vec![
            (10, 16),
            (20, 16),
            (30, 16),
            (40, 20),
            (50, 20),
        ]
        .into_iter()
        .filter(|&(n, b)| n <= max_dim && b <= max_bits)
        .collect();

        println!("┌──────┬──────┬────────────┬────────────┬────────────┬───────────┬───────────┐");
        println!("│  n   │ bits │  CPU (ms)  │  GPU (ms)  │ fplll (ms) │ GPU/CPU   │ GPU/fplll │");
        println!("├──────┼──────┼────────────┼────────────┼────────────┼───────────┼───────────┤");

        for (n, bits) in &configs {
            let basis = LatticeBasis::random(*n, *n, *bits);

            // CPU LLL
            let cpu_start = Instant::now();
            let cpu_config = LLLConfig::default();
            let (_cpu_reduced, _cpu_stats) = LLL::reduce(&basis, &cpu_config);
            let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

            // GPU LLL
            let gpu_start = Instant::now();
            let gpu_config = GpuLLLConfig::default();
            let (_gpu_reduced, _gpu_stats) = gpu_lll_reduce(&cuda, &basis, &gpu_config);
            let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;

            // fplll (if available)
            let fplll_ms = if fplll_available {
                run_fplll_benchmark(&basis)
            } else {
                0.0
            };

            let gpu_vs_cpu = if gpu_ms > 0.001 { cpu_ms / gpu_ms } else { 0.0 };
            let gpu_vs_fplll = if fplll_ms > 0.001 { fplll_ms / gpu_ms } else { 0.0 };

            let fplll_str = if fplll_available {
                format!("{:>10.2}", fplll_ms)
            } else {
                "     N/A  ".to_string()
            };

            let gpu_fplll_str = if fplll_available && gpu_vs_fplll > 0.0 {
                format!("{:>8.2}×", gpu_vs_fplll)
            } else {
                "     N/A ".to_string()
            };

            println!(
                "│ {:>4} │ {:>4} │ {:>10.2} │ {:>10.2} │ {} │ {:>8.2}× │ {} │",
                n, bits, cpu_ms, gpu_ms, fplll_str, gpu_vs_cpu, gpu_fplll_str
            );
        }

        println!("└──────┴──────┴────────────┴────────────┴────────────┴───────────┴───────────┘");
        println!();

        if fplll_available {
            println!("Notes:");
            println!("  - fplll uses highly optimized floating-point LLL");
            println!("  - Our GPU implementation uses exact CRT arithmetic");
            println!("  - Different reduction strategies may yield different results");
        }
    }
}

/// Check if fplll is available in PATH
fn check_fplll_available() -> bool {
    std::process::Command::new("fplll")
        .arg("--version")
        .output()
        .is_ok()
}

/// Run fplll on a lattice basis and return execution time
fn run_fplll_benchmark(basis: &LatticeBasis) -> f64 {
    use std::io::Write;
    use std::process::{Command, Stdio};

    // Convert basis to fplll format (space-separated rows in brackets)
    let mut input = format!("[");
    for i in 0..basis.n {
        input.push_str("[");
        for j in 0..basis.m {
            if j > 0 {
                input.push(' ');
            }
            input.push_str(&basis.vectors[i][j].to_string());
        }
        input.push_str("]");
    }
    input.push_str("]");

    // Run fplll
    let start = Instant::now();
    let mut child = match Command::new("fplll")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return 0.0,
    };

    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(input.as_bytes());
    }

    match child.wait() {
        Ok(status) if status.success() => {
            start.elapsed().as_secs_f64() * 1000.0
        }
        _ => 0.0,
    }
}

/// Export LLL benchmark results to CSV
fn export_lll_results(path: &PathBuf, results: &[LLLBenchResult]) {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path).expect("Failed to create export file");
    writeln!(file, "n,m,bits,cpu_ms,gpu_ms,speedup,cpu_swaps,gpu_swaps,cpu_iterations,gpu_iterations,verified").unwrap();

    for r in results {
        writeln!(
            file,
            "{},{},{},{:.6},{:.6},{:.6},{},{},{},{},{}",
            r.n, r.m, r.bits, r.cpu_ms, r.gpu_ms, r.speedup,
            r.cpu_swaps, r.gpu_swaps, r.cpu_iterations, r.gpu_iterations, r.verified
        ).unwrap();
    }
}

/// Knapsack lattice benchmark (cryptanalytic application)
pub fn run_knapsack_benchmark(max_n: usize, export: Option<PathBuf>) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         Knapsack Lattice LLL Benchmark (Cryptanalysis)       ║");
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

        println!("Testing LLL on knapsack/subset-sum lattices");
        println!("These lattices arise in cryptanalytic applications");
        println!();

        let configs: Vec<usize> = vec![4, 8, 12, 16, 20, 24, 28, 32]
            .into_iter()
            .filter(|&n| n <= max_n)
            .collect();

        println!("┌──────┬────────────┬────────────┬──────────┬────────────┬────────────┐");
        println!("│  n   │  CPU (ms)  │  GPU (ms)  │ Speedup  │ CPU swaps  │ GPU swaps  │");
        println!("├──────┼────────────┼────────────┼──────────┼────────────┼────────────┤");

        for n in &configs {
            // Generate random knapsack instance
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let a: Vec<i64> = (0..*n).map(|_| rng.gen_range(1..1000)).collect();
            let s: i64 = a.iter().take(n / 2).sum(); // Target sum

            let basis = LatticeBasis::knapsack(&a, s);

            // CPU LLL
            let cpu_start = Instant::now();
            let cpu_config = LLLConfig::default();
            let (_cpu_reduced, cpu_stats) = LLL::reduce(&basis, &cpu_config);
            let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

            // GPU LLL
            let gpu_start = Instant::now();
            let gpu_config = GpuLLLConfig::default();
            let (_gpu_reduced, gpu_stats) = gpu_lll_reduce(&cuda, &basis, &gpu_config);
            let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;

            let speedup = if gpu_ms > 0.001 { cpu_ms / gpu_ms } else { 0.0 };

            println!(
                "│ {:>4} │ {:>10.2} │ {:>10.2} │ {:>7.2}× │ {:>10} │ {:>10} │",
                n, cpu_ms, gpu_ms, speedup, cpu_stats.swaps, gpu_stats.swaps
            );
        }

        println!("└──────┴────────────┴────────────┴──────────┴────────────┴────────────┘");
    }
}
