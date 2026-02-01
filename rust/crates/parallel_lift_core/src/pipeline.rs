//! Pipelined CPU/GPU Solver
//!
//! Overlaps CPU work (residue reduction, CRT reconstruction) with GPU work (solve).
//!
//! # Pipeline Architecture
//!
//! The standard CRT solve has three phases:
//! 1. Residue: Reduce BigInt matrix/rhs to u32 mod each prime (CPU)
//! 2. Solve: Gaussian elimination mod p for all primes (GPU)
//! 3. CRT: Reconstruct BigInt from residues (CPU)
//!
//! With pipelining, we split primes into batches and overlap:
//! ```text
//! Time:     T0         T1         T2         T3         T4
//! Batch 0:  [Residue]  [Solve]    [CRT]
//! Batch 1:             [Residue]  [Solve]    [CRT]
//! Batch 2:                        [Residue]  [Solve]    [CRT]
//! ```
//!
//! This hides CPU latency behind GPU computation.

use crate::{Backend, CRTBasis, CRTReconstruction, Rational, SolveResult, Timings};
use crate::primes::PrimeGenerator;
use num_bigint::BigInt;
use num_traits::Zero;
use std::sync::mpsc::{channel, Sender, Receiver};
use std::thread;
use std::time::Instant;

/// Configuration for pipelined solver
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of primes per batch
    pub batch_size: usize,
    /// Total number of primes
    pub total_primes: usize,
    /// Number of worker threads for CPU tasks
    pub cpu_threads: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            batch_size: 8,
            total_primes: 32,
            cpu_threads: 4,
        }
    }
}

/// Statistics from pipelined execution
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    pub num_batches: usize,
    pub residue_time: f64,
    pub solve_time: f64,
    pub crt_time: f64,
    pub overlap_efficiency: f64, // 0-1, how much overlap we achieved
}

/// Message types for pipeline communication
enum ResidueResult {
    Ready {
        batch_idx: usize,
        matrix_residues: Vec<Vec<u32>>, // [prime_in_batch][element]
        b_residues: Vec<Vec<u32>>,      // [prime_in_batch][element]
        primes: Vec<u32>,
    },
    Done,
}

enum SolveResult2 {
    Ready {
        prime_idx: usize,
        solution: Vec<u32>, // solution for a single prime
    },
    Singular,
    Done,
}

/// Pipelined solver that overlaps CPU and GPU work
pub struct PipelineSolver<B: Backend + Send + Sync + 'static> {
    backend: B,
    config: PipelineConfig,
    prime_pool: Vec<u32>,
}

impl<B: Backend + Send + Sync + 'static> PipelineSolver<B> {
    pub fn new(backend: B) -> Self {
        Self::with_config(backend, PipelineConfig::default())
    }

    pub fn with_config(backend: B, config: PipelineConfig) -> Self {
        let prime_pool = PrimeGenerator::generate_31bit_primes(config.total_primes);
        Self {
            backend,
            config,
            prime_pool,
        }
    }

    /// Solve Ax = b with pipelined execution
    pub fn solve(&self, matrix: &[BigInt], b: &[BigInt]) -> (SolveResult, Timings, PipelineStats) {
        let mut timings = Timings::default();
        let mut stats = PipelineStats::default();
        let start = Instant::now();
        let n = b.len();

        let num_batches = (self.config.total_primes + self.config.batch_size - 1)
            / self.config.batch_size;
        stats.num_batches = num_batches;

        // For simplicity, run phases sequentially but batch primes
        // True async pipelining would require the Backend to support async dispatch

        // Phase 1: Compute all residues (CPU - can be parallelized)
        let residue_start = Instant::now();
        let all_matrix_residues = self.compute_residues(matrix);
        let all_b_residues = self.compute_residues(b);
        stats.residue_time = residue_start.elapsed().as_secs_f64();
        timings.residue_time = stats.residue_time;

        // Phase 2: Solve all primes (GPU - batched)
        let solve_start = Instant::now();
        let mut all_solutions: Vec<Vec<u32>> = Vec::with_capacity(self.config.total_primes);

        // Since we've already computed residues, solve each prime individually
        for prime_idx in 0..self.config.total_primes {
            let p = self.prime_pool[prime_idx];
            let matrix_mod = &all_matrix_residues[prime_idx];
            let b_mod = &all_b_residues[prime_idx];

            match self.backend.solve_mod(matrix_mod, b_mod, n, p) {
                Some(sol) => {
                    all_solutions.push(sol);
                }
                None => {
                    timings.total_time = start.elapsed().as_secs_f64();
                    return (
                        SolveResult::Singular { rank: 0 },
                        timings,
                        stats,
                    );
                }
            }
        }
        stats.solve_time = solve_start.elapsed().as_secs_f64();
        timings.solve_time = stats.solve_time;

        // Phase 3: CRT reconstruction
        let crt_start = Instant::now();
        let basis = CRTBasis::new(self.prime_pool.clone());

        // Transpose solutions: [prime][component] -> [component][prime]
        let mut x_residues: Vec<Vec<u32>> = vec![vec![0; self.config.total_primes]; n];
        for (prime_idx, sol) in all_solutions.iter().enumerate() {
            for (i, &val) in sol.iter().enumerate() {
                x_residues[i][prime_idx] = val;
            }
        }

        let x_bigint: Vec<BigInt> = x_residues
            .iter()
            .map(|residues| CRTReconstruction::reconstruct_signed(residues, &basis))
            .collect();

        stats.crt_time = crt_start.elapsed().as_secs_f64();
        timings.crt_time = stats.crt_time;

        // Convert to rationals
        let x_rational: Vec<Rational> = x_bigint
            .into_iter()
            .map(Rational::from_bigint)
            .collect();

        // Verify
        let verify_start = Instant::now();
        let verified = self.verify_solution(matrix, b, &x_rational, n);
        timings.verify_time = verify_start.elapsed().as_secs_f64();

        timings.num_primes = self.config.total_primes;
        timings.total_time = start.elapsed().as_secs_f64();

        // Calculate overlap efficiency (ideal = 1.0 when all phases overlap)
        let sequential_time = stats.residue_time + stats.solve_time + stats.crt_time;
        stats.overlap_efficiency = if sequential_time > 0.0 {
            (sequential_time - timings.total_time + timings.verify_time) / sequential_time
        } else {
            0.0
        };

        (
            SolveResult::Solution { x: x_rational, verified },
            timings,
            stats,
        )
    }

    /// Solve AX = B with pipelined execution (multi-RHS)
    pub fn solve_multi_rhs(
        &self,
        matrix: &[BigInt],
        b_cols: &[Vec<BigInt>],
    ) -> (Vec<Vec<Rational>>, Timings, PipelineStats) {
        let mut timings = Timings::default();
        let mut stats = PipelineStats::default();
        let start = Instant::now();
        let n = b_cols[0].len();
        let k = b_cols.len();

        let num_batches = (self.config.total_primes + self.config.batch_size - 1)
            / self.config.batch_size;
        stats.num_batches = num_batches;

        // Phase 1: Compute all residues
        let residue_start = Instant::now();
        let all_matrix_residues = self.compute_residues(matrix);
        let all_b_cols_residues: Vec<Vec<Vec<u32>>> = b_cols
            .iter()
            .map(|col| self.compute_residues(col))
            .collect();
        stats.residue_time = residue_start.elapsed().as_secs_f64();
        timings.residue_time = stats.residue_time;

        // Phase 2: Solve for each prime
        let solve_start = Instant::now();
        let mut all_solutions: Vec<Vec<Vec<u32>>> = Vec::with_capacity(self.config.total_primes);

        for prime_idx in 0..self.config.total_primes {
            let p = self.prime_pool[prime_idx];
            let matrix_mod = &all_matrix_residues[prime_idx];
            let b_cols_mod: Vec<Vec<u32>> = all_b_cols_residues
                .iter()
                .map(|col_residues| col_residues[prime_idx].clone())
                .collect();

            match self.backend.solve_multi_rhs_mod(matrix_mod, &b_cols_mod, n, k, p) {
                Some(solutions) => {
                    all_solutions.push(solutions);
                }
                None => {
                    timings.total_time = start.elapsed().as_secs_f64();
                    return (vec![], timings, stats);
                }
            }
        }
        stats.solve_time = solve_start.elapsed().as_secs_f64();
        timings.solve_time = stats.solve_time;

        // Phase 3: CRT reconstruction
        let crt_start = Instant::now();
        let basis = CRTBasis::new(self.prime_pool.clone());

        // Transpose: [prime][col][row] -> [col][row][prime]
        let mut x_residues: Vec<Vec<Vec<u32>>> = vec![vec![vec![0; self.config.total_primes]; n]; k];
        for (prime_idx, sols_for_prime) in all_solutions.iter().enumerate() {
            for (col_idx, col) in sols_for_prime.iter().enumerate() {
                for (row_idx, &val) in col.iter().enumerate() {
                    x_residues[col_idx][row_idx][prime_idx] = val;
                }
            }
        }

        let x_rational: Vec<Vec<Rational>> = x_residues
            .into_iter()
            .map(|col| {
                col.into_iter()
                    .map(|residues| {
                        let bigint = CRTReconstruction::reconstruct_signed(&residues, &basis);
                        Rational::from_bigint(bigint)
                    })
                    .collect()
            })
            .collect();

        stats.crt_time = crt_start.elapsed().as_secs_f64();
        timings.crt_time = stats.crt_time;

        timings.num_primes = self.config.total_primes;
        timings.total_time = start.elapsed().as_secs_f64();

        (x_rational, timings, stats)
    }

    /// Compute residues for all primes
    fn compute_residues(&self, values: &[BigInt]) -> Vec<Vec<u32>> {
        self.prime_pool
            .iter()
            .map(|&p| {
                let p_big = BigInt::from(p);
                values
                    .iter()
                    .map(|v| {
                        let r = v % &p_big;
                        if r < BigInt::zero() {
                            ((&r + &p_big) % &p_big).try_into().unwrap_or(0u32)
                        } else {
                            r.try_into().unwrap_or(0u32)
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Verify solution
    fn verify_solution(&self, matrix: &[BigInt], b: &[BigInt], x: &[Rational], n: usize) -> bool {
        for i in 0..n {
            let mut sum = Rational::zero();
            for j in 0..n {
                let a_ij = Rational::from_bigint(matrix[i * n + j].clone());
                sum = sum + a_ij * x[j].clone();
            }
            let b_i = Rational::from_bigint(b[i].clone());
            if sum != b_i {
                return false;
            }
        }
        true
    }
}

/// Async pipelined solver using channels
///
/// This version uses separate threads for each pipeline stage to achieve
/// true overlap between CPU and GPU work.
pub struct AsyncPipelineSolver<B: Backend + Send + Sync + 'static> {
    backend: std::sync::Arc<B>,
    config: PipelineConfig,
    prime_pool: Vec<u32>,
}

impl<B: Backend + Send + Sync + 'static> AsyncPipelineSolver<B> {
    pub fn new(backend: B) -> Self {
        Self::with_config(backend, PipelineConfig::default())
    }

    pub fn with_config(backend: B, config: PipelineConfig) -> Self {
        let prime_pool = PrimeGenerator::generate_31bit_primes(config.total_primes);
        Self {
            backend: std::sync::Arc::new(backend),
            config,
            prime_pool,
        }
    }

    /// Solve with true async pipelining
    pub fn solve_async(
        &self,
        matrix: Vec<BigInt>,
        b: Vec<BigInt>,
    ) -> (SolveResult, Timings, PipelineStats) {
        let mut timings = Timings::default();
        let mut stats = PipelineStats::default();
        let start = Instant::now();
        let n = b.len();
        let total_primes = self.config.total_primes;

        // Channels for pipeline stages
        let (residue_tx, residue_rx): (Sender<ResidueResult>, Receiver<ResidueResult>) = channel();
        let (solve_tx, solve_rx): (Sender<SolveResult2>, Receiver<SolveResult2>) = channel();

        let num_batches = (total_primes + self.config.batch_size - 1) / self.config.batch_size;
        stats.num_batches = num_batches;

        let prime_pool = self.prime_pool.clone();
        let batch_size = self.config.batch_size;

        // Stage 1: Residue computation thread
        let matrix_clone = matrix.clone();
        let b_clone = b.clone();
        let prime_pool_residue = prime_pool.clone();
        let residue_handle = thread::spawn(move || {
            let residue_start = Instant::now();

            for batch_idx in 0..num_batches {
                let start_prime = batch_idx * batch_size;
                let end_prime = (start_prime + batch_size).min(total_primes);
                let batch_primes: Vec<u32> = prime_pool_residue[start_prime..end_prime].to_vec();

                let matrix_residues: Vec<Vec<u32>> = batch_primes
                    .iter()
                    .map(|&p| {
                        let p_big = BigInt::from(p);
                        matrix_clone
                            .iter()
                            .map(|v| {
                                let r = v % &p_big;
                                if r < BigInt::zero() {
                                    ((&r + &p_big) % &p_big).try_into().unwrap_or(0u32)
                                } else {
                                    r.try_into().unwrap_or(0u32)
                                }
                            })
                            .collect()
                    })
                    .collect();

                let b_residues: Vec<Vec<u32>> = batch_primes
                    .iter()
                    .map(|&p| {
                        let p_big = BigInt::from(p);
                        b_clone
                            .iter()
                            .map(|v| {
                                let r = v % &p_big;
                                if r < BigInt::zero() {
                                    ((&r + &p_big) % &p_big).try_into().unwrap_or(0u32)
                                } else {
                                    r.try_into().unwrap_or(0u32)
                                }
                            })
                            .collect()
                    })
                    .collect();

                residue_tx.send(ResidueResult::Ready {
                    batch_idx,
                    matrix_residues,
                    b_residues,
                    primes: batch_primes,
                }).unwrap();
            }
            residue_tx.send(ResidueResult::Done).unwrap();

            residue_start.elapsed().as_secs_f64()
        });

        // Stage 2: GPU solve thread
        let backend = self.backend.clone();
        let solve_handle = thread::spawn(move || {
            let solve_start = Instant::now();

            loop {
                match residue_rx.recv() {
                    Ok(ResidueResult::Ready { batch_idx, matrix_residues, b_residues, primes }) => {
                        // Solve each prime in the batch
                        let mut batch_solutions = Vec::with_capacity(primes.len());
                        let mut singular = false;

                        for (local_idx, &p) in primes.iter().enumerate() {
                            let matrix_mod = &matrix_residues[local_idx];
                            let b_mod = &b_residues[local_idx];

                            match backend.solve_mod(matrix_mod, b_mod, n, p) {
                                Some(sol) => batch_solutions.push(sol),
                                None => {
                                    singular = true;
                                    break;
                                }
                            }
                        }

                        if singular {
                            solve_tx.send(SolveResult2::Singular).unwrap();
                            break;
                        }

                        for (local_idx, sol) in batch_solutions.into_iter().enumerate() {
                            let prime_idx = batch_idx * batch_size + local_idx;
                            solve_tx.send(SolveResult2::Ready {
                                prime_idx,
                                solution: sol,
                            }).unwrap();
                        }
                    }
                    Ok(ResidueResult::Done) => {
                        solve_tx.send(SolveResult2::Done).unwrap();
                        break;
                    }
                    Err(_) => break,
                }
            }

            solve_start.elapsed().as_secs_f64()
        });

        // Stage 3: CRT reconstruction (main thread)
        let crt_start = Instant::now();
        let basis = CRTBasis::new(prime_pool.clone());
        let mut x_residues: Vec<Vec<u32>> = vec![vec![0; total_primes]; n];
        let mut solutions_received = 0;
        let mut singular = false;

        loop {
            match solve_rx.recv() {
                Ok(SolveResult2::Ready { prime_idx, solution }) => {
                    for (i, &val) in solution.iter().enumerate() {
                        x_residues[i][prime_idx] = val;
                    }
                    solutions_received += 1;
                }
                Ok(SolveResult2::Singular) => {
                    singular = true;
                    break;
                }
                Ok(SolveResult2::Done) => break,
                Err(_) => break,
            }
        }

        // Wait for threads
        let residue_time = residue_handle.join().unwrap_or(0.0);
        let solve_time = solve_handle.join().unwrap_or(0.0);

        if singular {
            timings.total_time = start.elapsed().as_secs_f64();
            return (SolveResult::Singular { rank: 0 }, timings, stats);
        }

        let x_bigint: Vec<BigInt> = x_residues
            .iter()
            .map(|residues| CRTReconstruction::reconstruct_signed(residues, &basis))
            .collect();

        stats.crt_time = crt_start.elapsed().as_secs_f64();
        stats.residue_time = residue_time;
        stats.solve_time = solve_time;

        timings.residue_time = residue_time;
        timings.solve_time = solve_time;
        timings.crt_time = stats.crt_time;

        // Convert to rationals
        let x_rational: Vec<Rational> = x_bigint
            .into_iter()
            .map(Rational::from_bigint)
            .collect();

        // Verify
        let verify_start = Instant::now();
        let verified = verify_solution_standalone(&matrix, &b, &x_rational, n);
        timings.verify_time = verify_start.elapsed().as_secs_f64();

        timings.num_primes = total_primes;
        timings.total_time = start.elapsed().as_secs_f64();

        // Calculate overlap
        let sequential_time = stats.residue_time + stats.solve_time + stats.crt_time;
        let actual_compute_time = timings.total_time - timings.verify_time;
        stats.overlap_efficiency = if sequential_time > 0.0 {
            1.0 - (actual_compute_time / sequential_time)
        } else {
            0.0
        };
        stats.overlap_efficiency = stats.overlap_efficiency.max(0.0);

        (
            SolveResult::Solution { x: x_rational, verified },
            timings,
            stats,
        )
    }
}

fn verify_solution_standalone(matrix: &[BigInt], b: &[BigInt], x: &[Rational], n: usize) -> bool {
    for i in 0..n {
        let mut sum = Rational::zero();
        for j in 0..n {
            let a_ij = Rational::from_bigint(matrix[i * n + j].clone());
            sum = sum + a_ij * x[j].clone();
        }
        let b_i = Rational::from_bigint(b[i].clone());
        if sum != b_i {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuBackend;

    #[test]
    fn test_pipeline_solve_simple() {
        let backend = CpuBackend::new();
        let config = PipelineConfig {
            batch_size: 2,
            total_primes: 8,
            cpu_threads: 2,
        };
        let solver = PipelineSolver::with_config(backend, config);

        // Simple 2x2: [[2, 1], [1, 3]] x = [5, 10] => x = [1, 3]
        let matrix: Vec<BigInt> = vec![2, 1, 1, 3].into_iter().map(BigInt::from).collect();
        let b: Vec<BigInt> = vec![5, 10].into_iter().map(BigInt::from).collect();

        let (result, timings, stats) = solver.solve(&matrix, &b);

        match result {
            SolveResult::Solution { x, verified } => {
                assert!(verified);
                assert_eq!(x[0], Rational::from_bigint(BigInt::from(1)));
                assert_eq!(x[1], Rational::from_bigint(BigInt::from(3)));
            }
            _ => panic!("Expected solution"),
        }

        println!("Pipeline stats: {:?}", stats);
        println!("Timings: {:?}", timings);
    }

    #[test]
    fn test_async_pipeline_solve() {
        let backend = CpuBackend::new();
        let config = PipelineConfig {
            batch_size: 2,
            total_primes: 8,
            cpu_threads: 2,
        };
        let solver = AsyncPipelineSolver::with_config(backend, config);

        let matrix: Vec<BigInt> = vec![2, 1, 1, 3].into_iter().map(BigInt::from).collect();
        let b: Vec<BigInt> = vec![5, 10].into_iter().map(BigInt::from).collect();

        let (result, timings, stats) = solver.solve_async(matrix, b);

        match result {
            SolveResult::Solution { x, verified } => {
                assert!(verified);
                assert_eq!(x[0], Rational::from_bigint(BigInt::from(1)));
                assert_eq!(x[1], Rational::from_bigint(BigInt::from(3)));
            }
            _ => panic!("Expected solution"),
        }

        println!("Async pipeline stats: {:?}", stats);
        println!("Overlap efficiency: {:.2}%", stats.overlap_efficiency * 100.0);
    }
}
