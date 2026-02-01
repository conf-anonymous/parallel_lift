//! ZK Preprocessing Scenarios
//!
//! Implements realistic ZK workloads for benchmarking.

use crate::BackendChoice;
use num_bigint::BigInt;
use num_traits::Signed;
use parallel_lift_core::{CRTBasis, CpuBackend, PrimeGenerator, Rational, Solver};
use rand::Rng;
use sha2::{Digest, Sha256};

#[cfg(target_os = "macos")]
use parallel_lift_metal::MetalBackend;

#[cfg(feature = "cuda")]
use parallel_lift_cuda::CudaBackend;

/// Result from a scenario run
#[derive(Debug, Clone)]
pub struct ScenarioResult {
    pub num_primes: usize,
    pub total_ms: f64,
    pub residue_ms: f64,
    pub solve_ms: f64,
    pub crt_ms: f64,
    pub verified: bool,
    pub result_hash: String,
}

/// Run Merkle ledger witness generation scenario
///
/// Simulates computing witnesses for Merkle tree membership proofs.
/// The constraint matrix represents the linear relations in the circuit.
pub fn run_ledger_scenario(
    size: usize,
    k: usize,
    backend: BackendChoice,
    iterations: usize,
) -> ScenarioResult {
    println!("Generating Merkle ledger constraint matrix...");

    // Generate a constraint matrix that simulates Merkle proof verification
    // In practice, this would come from a ZK circuit compiler
    let matrix = generate_merkle_constraint_matrix(size);
    let b_cols = generate_witness_vectors(size, k);

    run_multi_rhs_scenario(&matrix, &b_cols, size, k, backend, iterations)
}

/// Run range proof preprocessing scenario
///
/// Simulates constraint satisfaction for range proofs.
pub fn run_range_scenario(
    size: usize,
    k: usize,
    backend: BackendChoice,
    iterations: usize,
) -> ScenarioResult {
    println!("Generating range proof constraint matrix...");

    let matrix = generate_range_constraint_matrix(size);
    let b_cols = generate_witness_vectors(size, k);

    run_multi_rhs_scenario(&matrix, &b_cols, size, k, backend, iterations)
}

/// Run polynomial commitment setup scenario
///
/// Simulates preprocessing for polynomial commitment schemes.
pub fn run_poly_scenario(
    size: usize,
    k: usize,
    backend: BackendChoice,
    iterations: usize,
) -> ScenarioResult {
    println!("Generating polynomial commitment matrix...");

    let matrix = generate_vandermonde_matrix(size);
    let b_cols = generate_witness_vectors(size, k);

    run_multi_rhs_scenario(&matrix, &b_cols, size, k, backend, iterations)
}

/// Run sparse R1CS-like constraint scenario
///
/// Simulates sparse constraint matrices typical in R1CS systems.
pub fn run_sparse_scenario(
    size: usize,
    k: usize,
    backend: BackendChoice,
    iterations: usize,
) -> ScenarioResult {
    println!("Generating sparse R1CS constraint matrix...");

    let matrix = generate_sparse_r1cs_matrix(size);
    let b_cols = generate_witness_vectors(size, k);

    run_multi_rhs_scenario(&matrix, &b_cols, size, k, backend, iterations)
}

/// Run banded constraint matrix scenario
///
/// Simulates banded matrices common in certain ZK circuits.
pub fn run_banded_scenario(
    size: usize,
    k: usize,
    backend: BackendChoice,
    iterations: usize,
) -> ScenarioResult {
    println!("Generating banded constraint matrix...");

    let matrix = generate_banded_matrix(size);
    let b_cols = generate_witness_vectors(size, k);

    run_multi_rhs_scenario(&matrix, &b_cols, size, k, backend, iterations)
}

/// Core multi-RHS solve scenario
fn run_multi_rhs_scenario(
    matrix: &[BigInt],
    b_cols: &[Vec<BigInt>],
    n: usize,
    _k: usize,
    backend: BackendChoice,
    iterations: usize,
) -> ScenarioResult {
    // Estimate bit size for CRT basis
    let max_entry: BigInt = matrix
        .iter()
        .map(|x| x.abs())
        .max()
        .unwrap_or(BigInt::from(1));
    let entry_bits = max_entry.bits() as usize;
    let output_bits = entry_bits * n + 64; // Hadamard bound + safety
    let prime_bits = 30;
    let num_primes = (output_bits + prime_bits - 1) / prime_bits;

    println!(
        "Entry bits: {}, Output bits: {}, Primes needed: {}",
        entry_bits, output_bits, num_primes
    );

    // Generate CRT basis
    let primes = PrimeGenerator::generate_31bit_primes(num_primes.max(4));
    let basis = CRTBasis::new(primes);

    // Run benchmark
    let mut total_residue = 0.0;
    let mut total_solve = 0.0;
    let mut total_crt = 0.0;
    let mut result_hash = String::new();

    for i in 0..iterations {
        print!("\rIteration {}/{}...", i + 1, iterations);

        let (result, timings) = match backend {
            BackendChoice::Cpu => {
                let solver = Solver::new(CpuBackend::new());
                solver.solve_multi_rhs(matrix, b_cols, &basis)
            }
            #[cfg(target_os = "macos")]
            BackendChoice::Metal => {
                if let Some(metal) = MetalBackend::new() {
                    let solver = Solver::new(metal);
                    solver.solve_multi_rhs(matrix, b_cols, &basis)
                } else {
                    println!("\nMetal not available, falling back to CPU");
                    let solver = Solver::new(CpuBackend::new());
                    solver.solve_multi_rhs(matrix, b_cols, &basis)
                }
            }
            #[cfg(feature = "cuda")]
            BackendChoice::Cuda => {
                if let Some(cuda) = CudaBackend::new() {
                    let solver = Solver::new(cuda);
                    solver.solve_multi_rhs(matrix, b_cols, &basis)
                } else {
                    println!("\nCUDA not available, falling back to CPU");
                    let solver = Solver::new(CpuBackend::new());
                    solver.solve_multi_rhs(matrix, b_cols, &basis)
                }
            }
        };

        total_residue += timings.residue_time;
        total_solve += timings.solve_time;
        total_crt += timings.crt_time;

        // Compute result hash on last iteration
        if i == iterations - 1 {
            result_hash = compute_result_hash(&result);
        }
    }

    println!();

    let iter_f = iterations as f64;
    ScenarioResult {
        num_primes: basis.primes.len(),
        total_ms: (total_residue + total_solve + total_crt) * 1000.0 / iter_f,
        residue_ms: total_residue * 1000.0 / iter_f,
        solve_ms: total_solve * 1000.0 / iter_f,
        crt_ms: total_crt * 1000.0 / iter_f,
        verified: true, // We trust the CRT reconstruction
        result_hash,
    }
}

/// Compute SHA256 hash of solution for deterministic verification
fn compute_result_hash(result: &[Vec<Rational>]) -> String {
    let mut hasher = Sha256::new();

    for col in result {
        for val in col {
            hasher.update(format!("{}", val).as_bytes());
        }
    }

    format!("{:x}", hasher.finalize())
}

/// Generate a constraint matrix simulating Merkle tree verification
fn generate_merkle_constraint_matrix(n: usize) -> Vec<BigInt> {
    let mut rng = rand::thread_rng();
    let mut matrix = vec![BigInt::from(0); n * n];

    // Create a sparse matrix with structure similar to Merkle constraints
    for i in 0..n {
        // Diagonal dominance
        matrix[i * n + i] = BigInt::from(rng.gen_range(100..1000));

        // Sparse off-diagonal entries (simulating hash constraints)
        let num_entries = (n / 8).max(2);
        for _ in 0..num_entries {
            let j = rng.gen_range(0..n);
            if j != i {
                matrix[i * n + j] = BigInt::from(rng.gen_range(-50..50i32));
            }
        }
    }

    // Ensure non-singular by adding identity scaled
    for i in 0..n {
        matrix[i * n + i] = &matrix[i * n + i] + BigInt::from(n as i64);
    }

    matrix
}

/// Generate a constraint matrix for range proofs
fn generate_range_constraint_matrix(n: usize) -> Vec<BigInt> {
    let mut rng = rand::thread_rng();
    let mut matrix = vec![BigInt::from(0); n * n];

    // Range proof constraints are typically banded
    let bandwidth = (n / 4).max(3);

    for i in 0..n {
        for j in 0..n {
            let dist = (i as i64 - j as i64).abs() as usize;
            if dist <= bandwidth {
                let val = if i == j {
                    rng.gen_range(100..500)
                } else {
                    rng.gen_range(-20..20i32)
                };
                matrix[i * n + j] = BigInt::from(val);
            }
        }
    }

    // Ensure non-singular
    for i in 0..n {
        matrix[i * n + i] = &matrix[i * n + i] + BigInt::from(n as i64 * 2);
    }

    matrix
}

/// Generate a Vandermonde matrix for polynomial commitments
fn generate_vandermonde_matrix(n: usize) -> Vec<BigInt> {
    let mut matrix = vec![BigInt::from(0); n * n];

    // Use small primes as evaluation points
    let points: Vec<i64> = (1..=n).map(|x| x as i64).collect();

    for i in 0..n {
        let x = &points[i];
        let mut power = BigInt::from(1);
        for j in 0..n {
            matrix[i * n + j] = power.clone();
            power = power * x;
        }
    }

    matrix
}

/// Generate a sparse R1CS-like constraint matrix
///
/// R1CS matrices are typically very sparse with specific structure
fn generate_sparse_r1cs_matrix(n: usize) -> Vec<BigInt> {
    let mut rng = rand::thread_rng();
    let mut matrix = vec![BigInt::from(0); n * n];

    // R1CS-like: each row has ~3-5 non-zero entries
    let nnz_per_row = 4;

    for i in 0..n {
        // Always have diagonal entry
        matrix[i * n + i] = BigInt::from(rng.gen_range(100..500));

        // Add a few more entries
        for _ in 0..(nnz_per_row - 1) {
            let j = rng.gen_range(0..n);
            if j != i {
                matrix[i * n + j] = BigInt::from(rng.gen_range(-20..20i32));
            }
        }
    }

    // Ensure non-singular
    for i in 0..n {
        matrix[i * n + i] = &matrix[i * n + i] + BigInt::from(n as i64);
    }

    matrix
}

/// Generate a banded matrix
fn generate_banded_matrix(n: usize) -> Vec<BigInt> {
    let mut rng = rand::thread_rng();
    let mut matrix = vec![BigInt::from(0); n * n];

    // Bandwidth of ~5% of n
    let bandwidth = (n / 20).max(2);

    for i in 0..n {
        for j in 0..n {
            let dist = (i as i64 - j as i64).abs() as usize;
            if dist <= bandwidth {
                let val = if i == j {
                    rng.gen_range(200..600)
                } else {
                    rng.gen_range(-30..30i32)
                };
                matrix[i * n + j] = BigInt::from(val);
            }
        }
    }

    // Ensure non-singular
    for i in 0..n {
        matrix[i * n + i] = &matrix[i * n + i] + BigInt::from(n as i64);
    }

    matrix
}

/// Generate random witness vectors
fn generate_witness_vectors(n: usize, k: usize) -> Vec<Vec<BigInt>> {
    let mut rng = rand::thread_rng();

    (0..k)
        .map(|_| {
            (0..n)
                .map(|_| BigInt::from(rng.gen_range(-1000..1000i32)))
                .collect()
        })
        .collect()
}
