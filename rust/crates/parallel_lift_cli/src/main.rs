//! Parallel Lift CLI
//!
//! CRT-accelerated exact arithmetic for ZK preprocessing.
//!
//! # Usage
//! ```bash
//! # ZK preprocessing scenario
//! parallel-lift zk-preprocess --scenario ledger --size 64 --k 16 --backend metal
//!
//! # Scaling sweeps (paper figures)
//! parallel-lift sweep --export results/
//!
//! # Compare CRT approach vs naive BigInt
//! parallel-lift compare-crt
//! ```

mod benchmark;
mod scenarios;
mod sweep;
mod sparse_bench;
mod adaptive_bench;
mod pipeline_bench;
mod r1cs_bench;
mod gpu_crt_bench;
mod lll_bench;
mod v2_bench;
mod hensel_bench;

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "parallel-lift")]
#[command(about = "CRT-accelerated exact arithmetic for ZK preprocessing")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run ZK preprocessing scenarios
    ZkPreprocess {
        /// Scenario to run
        #[arg(long, value_enum)]
        scenario: Scenario,

        /// Matrix/constraint size
        #[arg(long, default_value = "64")]
        size: usize,

        /// Number of RHS vectors (witness columns)
        #[arg(long, default_value = "16")]
        k: usize,

        /// Backend to use
        #[arg(long, value_enum, default_value = "cpu")]
        backend: BackendChoice,

        /// Export results to CSV
        #[arg(long)]
        export: Option<PathBuf>,

        /// Number of iterations for timing
        #[arg(long, default_value = "1")]
        iterations: usize,
    },

    /// Run scaling benchmarks
    Benchmark {
        /// Benchmark type
        #[arg(long, value_enum)]
        bench: BenchType,

        /// Maximum size
        #[arg(long, default_value = "256")]
        max_size: usize,

        /// Backend to use
        #[arg(long, value_enum, default_value = "cpu")]
        backend: BackendChoice,

        /// Export results to CSV
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Run comprehensive scaling sweeps (paper figures)
    Sweep {
        /// Export directory for CSV files
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Compare CRT-lifted approach vs naive BigInt GPU
    CompareCrt,

    /// Verify correctness with test vectors
    Verify {
        /// Path to test vectors CSV
        #[arg(long)]
        vectors: PathBuf,
    },

    /// Benchmark sparse vs dense solvers on R1CS-like matrices
    SparseBench {
        /// Maximum matrix size
        #[arg(long, default_value = "512")]
        max_size: usize,

        /// Average nonzeros per row (R1CS typically 3-5)
        #[arg(long, default_value = "5")]
        nnz_per_row: usize,

        /// Number of RHS vectors
        #[arg(long, default_value = "16")]
        k: usize,

        /// Export results to CSV
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Benchmark adaptive vs fixed CRT prime selection
    AdaptiveBench {
        /// Maximum matrix size
        #[arg(long, default_value = "128")]
        max_size: usize,

        /// Number of RHS vectors
        #[arg(long, default_value = "16")]
        k: usize,

        /// Export results to CSV
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Benchmark pipelined vs sequential execution
    PipelineBench {
        /// Maximum matrix size
        #[arg(long, default_value = "128")]
        max_size: usize,

        /// Number of RHS vectors
        #[arg(long, default_value = "8")]
        k: usize,

        /// Export results to CSV
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Benchmark GPU vs CPU Wiedemann sparse solver
    GpuWiedemannBench {
        /// Maximum matrix size
        #[arg(long, default_value = "512")]
        max_size: usize,

        /// Average nonzeros per row (R1CS typically 3-5)
        #[arg(long, default_value = "5")]
        nnz_per_row: usize,

        /// Export results to CSV
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Benchmark R1CS constraint system solving
    R1csBench {
        /// Path to R1CS file (circom binary format)
        #[arg(long)]
        r1cs: Option<PathBuf>,

        /// Maximum constraint count for synthetic R1CS
        #[arg(long, default_value = "2048")]
        max_size: usize,

        /// Export results to CSV
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Display GPU device information
    GpuInfo,

    /// Benchmark GPU CRT reconstruction vs CPU
    GpuCrtBench {
        /// Maximum number of primes
        #[arg(long, default_value = "512")]
        max_primes: usize,

        /// Maximum number of values to reconstruct
        #[arg(long, default_value = "4096")]
        max_values: usize,

        /// Export results to CSV
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Full pipeline benchmark with GPU CRT
    GpuFullBench {
        /// Maximum matrix size
        #[arg(long, default_value = "256")]
        max_size: usize,

        /// Number of RHS vectors
        #[arg(long, default_value = "16")]
        k: usize,

        /// Export results to CSV
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Benchmark GPU LLL lattice reduction
    LllBench {
        /// Maximum lattice dimension
        #[arg(long, default_value = "30")]
        max_dim: usize,

        /// Maximum entry bit size
        #[arg(long, default_value = "20")]
        max_bits: usize,

        /// Export results to CSV
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Benchmark LLL against fplll
    LllFplllBench {
        /// Maximum lattice dimension
        #[arg(long, default_value = "50")]
        max_dim: usize,

        /// Maximum entry bit size
        #[arg(long, default_value = "20")]
        max_bits: usize,

        /// Export results to CSV
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Benchmark LLL on knapsack lattices (cryptanalysis)
    LllKnapsackBench {
        /// Maximum knapsack size
        #[arg(long, default_value = "32")]
        max_n: usize,

        /// Export results to CSV
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Benchmark PCIe transfer overhead (H→D, Compute, D→H breakdown)
    TransferBench {
        /// Matrix sizes to benchmark (comma-separated)
        #[arg(long, default_value = "64,128,256,512")]
        sizes: String,

        /// Number of RHS vectors
        #[arg(long, default_value = "16")]
        k: usize,

        /// Number of trials per configuration
        #[arg(long, default_value = "5")]
        trials: usize,

        /// Export results to CSV
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Benchmark sparse solver with varying sparsity levels
    SparsityBench {
        /// Matrix size
        #[arg(long, default_value = "256")]
        size: usize,

        /// Sparsity levels to test (comma-separated, e.g., "0.50,0.80,0.95,0.99")
        #[arg(long, default_value = "0.50,0.80,0.95,0.99")]
        sparsities: String,

        /// Number of RHS vectors
        #[arg(long, default_value = "16")]
        k: usize,

        /// Number of trials per configuration
        #[arg(long, default_value = "3")]
        trials: usize,

        /// Export results to CSV
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// V2 Benchmark: Compare 31-bit vs 62-bit primes
    V2Bench {
        /// Matrix sizes to benchmark (comma-separated)
        #[arg(long, default_value = "32,64,128,256")]
        sizes: String,

        /// Number of RHS vectors
        #[arg(long, default_value = "16")]
        k: usize,

        /// Number of trials per configuration
        #[arg(long, default_value = "5")]
        trials: usize,

        /// Export results to CSV
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Hensel Lifting Benchmark: Compare Hensel vs CRT vs IML
    HenselBench {
        /// Matrix sizes to benchmark (comma-separated)
        #[arg(long, default_value = "32,64,128,256")]
        sizes: String,

        /// Number of RHS vectors
        #[arg(long, default_value = "16")]
        k: usize,

        /// Number of trials per configuration
        #[arg(long, default_value = "5")]
        trials: usize,

        /// Export results to CSV
        #[arg(long)]
        export: Option<PathBuf>,
    },
}

#[derive(Clone, Copy, ValueEnum)]
pub enum Scenario {
    /// Merkle ledger witness generation
    Ledger,
    /// Range proof preprocessing
    Range,
    /// Polynomial commitment setup
    Poly,
    /// Sparse constraint matrix (R1CS-like)
    Sparse,
    /// Banded constraint matrix
    Banded,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum BackendChoice {
    Cpu,
    #[cfg(target_os = "macos")]
    Metal,
    #[cfg(feature = "cuda")]
    Cuda,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum BenchType {
    /// Determinant scaling
    Det,
    /// Single-RHS solve
    Solve,
    /// Multi-RHS solve
    MultiRhs,
    /// Matrix inverse
    Inverse,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::ZkPreprocess {
            scenario,
            size,
            k,
            backend,
            export,
            iterations,
        } => {
            run_zk_preprocess(scenario, size, k, backend, export, iterations);
        }
        Commands::Benchmark {
            bench,
            max_size,
            backend,
            export,
        } => {
            run_benchmark(bench, max_size, backend, export);
        }
        Commands::Sweep { export } => {
            sweep::run_scaling_sweeps(export);
        }
        Commands::CompareCrt => {
            sweep::run_crt_vs_naive_comparison();
        }
        Commands::Verify { vectors } => {
            run_verify(vectors);
        }
        Commands::SparseBench {
            max_size,
            nnz_per_row,
            k,
            export,
        } => {
            sparse_bench::run_sparse_benchmark(max_size, nnz_per_row, k, export);
        }
        Commands::AdaptiveBench {
            max_size,
            k,
            export,
        } => {
            adaptive_bench::run_adaptive_benchmark(max_size, k, export);
        }
        Commands::PipelineBench {
            max_size,
            k,
            export,
        } => {
            pipeline_bench::run_pipeline_benchmark(max_size, k, export);
        }
        Commands::GpuWiedemannBench {
            max_size,
            nnz_per_row,
            export,
        } => {
            sparse_bench::run_gpu_wiedemann_benchmark(max_size, nnz_per_row, export);
        }
        Commands::R1csBench {
            r1cs,
            max_size,
            export,
        } => {
            r1cs_bench::run_r1cs_benchmark(r1cs, max_size, export);
        }
        Commands::GpuInfo => {
            run_gpu_info();
        }
        Commands::GpuCrtBench {
            max_primes,
            max_values,
            export,
        } => {
            gpu_crt_bench::run_gpu_crt_benchmark(max_primes, max_values, export);
        }
        #[cfg(feature = "cuda")]
        Commands::GpuFullBench {
            max_size,
            k,
            export,
        } => {
            gpu_crt_bench::run_full_pipeline_benchmark(max_size, k, export);
        }
        #[cfg(not(feature = "cuda"))]
        Commands::GpuFullBench { .. } => {
            println!("CUDA feature not enabled. Compile with --features cuda");
        }
        Commands::LllBench {
            max_dim,
            max_bits,
            export,
        } => {
            lll_bench::run_lll_benchmark(max_dim, max_bits, export);
        }
        Commands::LllFplllBench {
            max_dim,
            max_bits,
            export,
        } => {
            lll_bench::run_fplll_comparison(max_dim, max_bits, export);
        }
        Commands::LllKnapsackBench { max_n, export } => {
            lll_bench::run_knapsack_benchmark(max_n, export);
        }
        #[cfg(feature = "cuda")]
        Commands::TransferBench {
            sizes,
            k,
            trials,
            export,
        } => {
            run_transfer_benchmark(&sizes, k, trials, export);
        }
        #[cfg(not(feature = "cuda"))]
        Commands::TransferBench { .. } => {
            println!("CUDA feature not enabled. Compile with --features cuda");
        }
        Commands::SparsityBench {
            size,
            sparsities,
            k,
            trials,
            export,
        } => {
            sparse_bench::run_sparsity_benchmark(size, &sparsities, k, trials, export);
        }
        #[cfg(feature = "cuda")]
        Commands::V2Bench {
            sizes,
            k,
            trials,
            export,
        } => {
            v2_bench::run_v2_benchmark(&sizes, k, trials, export);
        }
        #[cfg(not(feature = "cuda"))]
        Commands::V2Bench { .. } => {
            println!("CUDA feature not enabled. Compile with --features cuda");
        }
        #[cfg(feature = "cuda")]
        Commands::HenselBench {
            sizes,
            k,
            trials,
            export,
        } => {
            hensel_bench::run_hensel_benchmark(&sizes, k, trials, export);
        }
        #[cfg(not(feature = "cuda"))]
        Commands::HenselBench { .. } => {
            println!("CUDA feature not enabled. Compile with --features cuda");
        }
    }
}

fn run_gpu_info() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                     GPU Device Information                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    #[cfg(target_os = "macos")]
    {
        use parallel_lift_metal::MultiGpuManager;

        match MultiGpuManager::new() {
            Some(mgr) => {
                println!("Metal GPU Backend: Available");
                println!();
                mgr.print_info();
                println!();

                // Test work partitioning
                let num_gpus = mgr.num_gpus();
                if num_gpus > 1 {
                    println!("Multi-GPU Work Distribution:");
                    let partitions = mgr.partition_work(16);
                    for (i, partition) in partitions.iter().enumerate() {
                        println!("  GPU {}: {} primes", i, partition.len());
                    }
                    println!();
                    println!("Note: With {} GPUs, work can be distributed for faster parallel processing.", num_gpus);
                } else {
                    println!("Single GPU mode - all primes processed on one device.");
                }
            }
            None => {
                println!("Metal GPU Backend: Not available");
                println!();
                println!("Possible reasons:");
                println!("  - No Metal-compatible GPU found");
                println!("  - Metal framework not installed");
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        println!("Metal GPU Backend: Not supported on this platform");
        println!();
        println!("Metal is only available on macOS.");
    }

    // CUDA backend info
    #[cfg(feature = "cuda")]
    {
        use parallel_lift_cuda::CudaBackend;
        use parallel_lift_core::Backend;

        println!();
        match CudaBackend::try_new() {
            Ok(backend) => {
                println!("CUDA GPU Backend: Available");
                println!("  Successfully initialized CUDA device 0");
                println!("  Backend name: {}", backend.name());
            }
            Err(e) => {
                println!("CUDA GPU Backend: Not available");
                println!();
                println!("Error details: {:?}", e);
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!();
        println!("CUDA GPU Backend: Not compiled (enable with --features cuda)");
    }
}

fn run_zk_preprocess(
    scenario: Scenario,
    size: usize,
    k: usize,
    backend: BackendChoice,
    export: Option<PathBuf>,
    iterations: usize,
) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║             Parallel Lift - ZK Preprocessing                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let scenario_name = match scenario {
        Scenario::Ledger => "Merkle Ledger Witness",
        Scenario::Range => "Range Proof Setup",
        Scenario::Poly => "Polynomial Commitment",
        Scenario::Sparse => "Sparse R1CS Constraints",
        Scenario::Banded => "Banded Constraint Matrix",
    };

    let backend_name = match backend {
        BackendChoice::Cpu => "CPU",
        #[cfg(target_os = "macos")]
        BackendChoice::Metal => "Metal GPU",
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => "CUDA GPU",
    };

    println!("Scenario:   {}", scenario_name);
    println!("Size:       {} × {}", size, size);
    println!("RHS cols:   {}", k);
    println!("Backend:    {}", backend_name);
    println!("Iterations: {}", iterations);
    println!();

    let results = match scenario {
        Scenario::Ledger => scenarios::run_ledger_scenario(size, k, backend, iterations),
        Scenario::Range => scenarios::run_range_scenario(size, k, backend, iterations),
        Scenario::Poly => scenarios::run_poly_scenario(size, k, backend, iterations),
        Scenario::Sparse => scenarios::run_sparse_scenario(size, k, backend, iterations),
        Scenario::Banded => scenarios::run_banded_scenario(size, k, backend, iterations),
    };

    // Print results
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ Results                                                       │");
    println!("├──────────────────────────────────────────────────────────────┤");
    println!(
        "│ Primes used:      {:>6}                                      │",
        results.num_primes
    );
    println!(
        "│ Total time:       {:>10.3} ms                                │",
        results.total_ms
    );
    println!(
        "│ ├─ Residue:       {:>10.3} ms ({:>5.1}%)                      │",
        results.residue_ms,
        100.0 * results.residue_ms / results.total_ms
    );
    println!(
        "│ ├─ Solve:         {:>10.3} ms ({:>5.1}%)                      │",
        results.solve_ms,
        100.0 * results.solve_ms / results.total_ms
    );
    println!(
        "│ └─ CRT:           {:>10.3} ms ({:>5.1}%)                      │",
        results.crt_ms,
        100.0 * results.crt_ms / results.total_ms
    );
    println!(
        "│ Result hash:      {:>16}                      │",
        &results.result_hash[..16.min(results.result_hash.len())]
    );
    println!(
        "│ Verified:         {:>6}                                      │",
        if results.verified { "✓" } else { "✗" }
    );
    println!("└──────────────────────────────────────────────────────────────┘");

    // Export if requested
    if let Some(path) = export {
        export_results(&path, &results);
        println!("\nResults exported to: {}", path.display());
    }
}

fn run_benchmark(bench: BenchType, max_size: usize, backend: BackendChoice, export: Option<PathBuf>) {
    println!("Running {:?} benchmark up to size {}...", bench, max_size);
    benchmark::run_scaling_benchmark(bench, max_size, backend, export);
}

fn run_verify(vectors: PathBuf) {
    println!("Verifying against test vectors: {}", vectors.display());
    // TODO: Implement verification against golden CSVs
    println!("Verification not yet implemented");
}

fn export_results(path: &PathBuf, results: &scenarios::ScenarioResult) {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path).expect("Failed to create export file");
    writeln!(file, "metric,value").unwrap();
    writeln!(file, "num_primes,{}", results.num_primes).unwrap();
    writeln!(file, "total_ms,{:.6}", results.total_ms).unwrap();
    writeln!(file, "residue_ms,{:.6}", results.residue_ms).unwrap();
    writeln!(file, "solve_ms,{:.6}", results.solve_ms).unwrap();
    writeln!(file, "crt_ms,{:.6}", results.crt_ms).unwrap();
    writeln!(file, "result_hash,{}", results.result_hash).unwrap();
    writeln!(file, "verified,{}", results.verified).unwrap();
}

/// Run PCIe transfer timing benchmark
/// Measures H→D, Compute, D→H breakdown for multi-RHS solve
#[cfg(feature = "cuda")]
fn run_transfer_benchmark(sizes_str: &str, k: usize, trials: usize, export: Option<PathBuf>) {
    use parallel_lift_cuda::CudaBackend;
    use parallel_lift_core::PrimeGenerator;
    use num_bigint::BigInt;
    use rand::Rng;
    use std::fs::File;
    use std::io::Write;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           PCIe Transfer Timing Benchmark                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let cuda = match CudaBackend::new() {
        Some(c) => c,
        None => {
            println!("CUDA backend not available");
            return;
        }
    };

    // Parse sizes
    let sizes: Vec<usize> = sizes_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    println!("Configuration:");
    println!("  Matrix sizes: {:?}", sizes);
    println!("  RHS vectors:  {}", k);
    println!("  Trials:       {}", trials);
    println!();

    // Results storage
    let mut all_results = Vec::new();

    println!("┌───────┬─────────┬────────────┬────────────┬────────────┬────────────┬────────────┬───────────┐");
    println!("│   n   │ Primes  │ Prepare(ms)│   H→D(ms) │ Compute(ms)│   D→H(ms) │ Total(ms)  │ Transfer% │");
    println!("├───────┼─────────┼────────────┼────────────┼────────────┼────────────┼────────────┼───────────┤");

    for &n in &sizes {
        let mut rng = rand::thread_rng();

        // Generate random matrix (diagonally dominant)
        let matrix: Vec<u32> = (0..n * n)
            .map(|i| {
                if i / n == i % n {
                    rng.gen_range(100..1000u32)
                } else {
                    rng.gen_range(0..50u32)
                }
            })
            .collect();

        // Generate RHS columns
        let b_cols: Vec<Vec<u32>> = (0..k)
            .map(|_| (0..n).map(|_| rng.gen_range(0..100u32)).collect())
            .collect();

        // Estimate primes needed (48-bit entries, n×n matrix)
        let num_primes = (n * 64 / 30).max(4);
        let primes = PrimeGenerator::generate_31bit_primes(num_primes);

        // Run trials
        let mut trial_results = Vec::new();
        for _ in 0..trials {
            let (_, timing) = cuda.gpu_batch_multi_rhs_solve_timed(
                &matrix, &b_cols, n, k, &primes,
            );
            trial_results.push(timing);
        }

        // Compute median timing
        trial_results.sort_by(|a, b| a.total_ms.partial_cmp(&b.total_ms).unwrap());
        let median = &trial_results[trials / 2];

        let transfer_pct = 100.0 * (median.htod_ms + median.dtoh_ms) / median.total_ms;

        println!(
            "│ {:>5} │ {:>7} │ {:>10.3} │ {:>10.3} │ {:>10.3} │ {:>10.3} │ {:>10.3} │ {:>8.1}% │",
            n,
            primes.len(),
            median.prepare_ms,
            median.htod_ms,
            median.compute_ms,
            median.dtoh_ms,
            median.total_ms,
            transfer_pct
        );

        all_results.push((n, primes.len(), median.clone()));
    }

    println!("└───────┴─────────┴────────────┴────────────┴────────────┴────────────┴────────────┴───────────┘");

    // Print data transfer sizes
    println!();
    println!("Transfer Sizes:");
    println!("┌───────┬─────────┬───────────────┬───────────────┬────────────────┐");
    println!("│   n   │ Primes  │    H→D (MB)   │    D→H (MB)   │ Bandwidth Req. │");
    println!("├───────┼─────────┼───────────────┼───────────────┼────────────────┤");

    for (n, num_primes, timing) in &all_results {
        let htod_mb = timing.htod_bytes as f64 / (1024.0 * 1024.0);
        let dtoh_mb = timing.dtoh_bytes as f64 / (1024.0 * 1024.0);
        // Estimate bandwidth requirement (GB/s needed to complete transfer in 1ms)
        let bw_gbps = (htod_mb + dtoh_mb) / timing.total_ms;

        println!(
            "│ {:>5} │ {:>7} │ {:>13.2} │ {:>13.2} │ {:>11.2} GB/s │",
            n, num_primes, htod_mb, dtoh_mb, bw_gbps
        );
    }

    println!("└───────┴─────────┴───────────────┴───────────────┴────────────────┘");

    // Export if requested
    if let Some(path) = export {
        let mut file = File::create(&path).expect("Failed to create export file");
        writeln!(file, "n,primes,prepare_ms,htod_ms,compute_ms,dtoh_ms,total_ms,transfer_pct,htod_mb,dtoh_mb").unwrap();

        for (n, num_primes, timing) in &all_results {
            let transfer_pct = 100.0 * (timing.htod_ms + timing.dtoh_ms) / timing.total_ms;
            let htod_mb = timing.htod_bytes as f64 / (1024.0 * 1024.0);
            let dtoh_mb = timing.dtoh_bytes as f64 / (1024.0 * 1024.0);

            writeln!(
                file,
                "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2},{:.2},{:.2}",
                n, num_primes, timing.prepare_ms, timing.htod_ms, timing.compute_ms,
                timing.dtoh_ms, timing.total_ms, transfer_pct, htod_mb, dtoh_mb
            ).unwrap();
        }

        println!();
        println!("Results exported to: {}", path.display());
    }
}
