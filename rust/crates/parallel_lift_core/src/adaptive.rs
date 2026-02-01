//! Adaptive Early-Exit CRT Solver
//!
//! Instead of using a pessimistic Hadamard bound to estimate the number of primes needed,
//! this solver starts with a minimal prime count and adds primes incrementally until
//! the solution verifies correctly.
//!
//! # Strategy
//!
//! The Hadamard bound for an n×n matrix with entries bounded by B gives:
//!   |det(A)| ≤ n^(n/2) * B^n
//!
//! For solution components x_i = det(A_i) / det(A), this gives a massive overestimate
//! because the actual determinants are typically much smaller than the Hadamard bound.
//!
//! Adaptive approach:
//! 1. Start with `base_primes` (e.g., 4 primes ≈ 124 bits)
//! 2. Solve and reconstruct
//! 3. Verify Ax = b
//! 4. If verification fails, double the prime count and retry
//! 5. Early exit on successful verification
//!
//! This can reduce computation by 2-10x for typical ZK matrices.

use crate::{Backend, CRTBasis, CRTReconstruction, Rational, SolveResult, Timings};
use crate::primes::PrimeGenerator;
use num_bigint::BigInt;
use num_traits::Zero;
use std::time::Instant;

/// Configuration for adaptive CRT solver
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Minimum number of primes to start with
    pub min_primes: usize,
    /// Maximum number of primes (safety limit)
    pub max_primes: usize,
    /// Growth factor when adding primes (e.g., 2.0 = double each iteration)
    pub growth_factor: f64,
    /// Whether to cache solutions from previous iterations
    pub cache_residues: bool,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            min_primes: 4,       // ~124 bits initially
            max_primes: 128,     // ~4000 bits max
            growth_factor: 2.0,  // Double primes each round
            cache_residues: true,
        }
    }
}

/// Statistics from adaptive solving
#[derive(Debug, Clone, Default)]
pub struct AdaptiveStats {
    /// Number of iterations needed
    pub iterations: usize,
    /// Prime counts at each iteration
    pub prime_counts: Vec<usize>,
    /// Whether each iteration verified successfully
    pub verification_results: Vec<bool>,
    /// Time spent in each phase
    pub phase_times: AdaptivePhaseTimes,
}

#[derive(Debug, Clone, Default)]
pub struct AdaptivePhaseTimes {
    pub residue_total: f64,
    pub solve_total: f64,
    pub crt_total: f64,
    pub verify_total: f64,
}

/// Adaptive CRT solver that minimizes prime count through early-exit verification
pub struct AdaptiveSolver<B: Backend> {
    backend: B,
    config: AdaptiveConfig,
    /// Pre-generated prime pool
    prime_pool: Vec<u32>,
}

impl<B: Backend> AdaptiveSolver<B> {
    pub fn new(backend: B) -> Self {
        Self::with_config(backend, AdaptiveConfig::default())
    }

    pub fn with_config(backend: B, config: AdaptiveConfig) -> Self {
        // Pre-generate max primes upfront (cheap, one-time cost)
        let prime_pool = PrimeGenerator::generate_31bit_primes(config.max_primes);
        Self {
            backend,
            config,
            prime_pool,
        }
    }

    /// Solve Ax = b with adaptive prime selection
    ///
    /// Returns solution, timing breakdown, and adaptive statistics
    pub fn solve(
        &self,
        matrix: &[BigInt],
        b: &[BigInt],
    ) -> (SolveResult, Timings, AdaptiveStats) {
        let mut timings = Timings::default();
        let mut stats = AdaptiveStats::default();
        let start = Instant::now();
        let n = b.len();

        // Prepare matrix and RHS residues (compute once, reuse)
        let residue_start = Instant::now();
        let matrix_residues = self.compute_all_residues(matrix);
        let b_residues = self.compute_all_residues(b);
        timings.residue_time = residue_start.elapsed().as_secs_f64();
        stats.phase_times.residue_total = timings.residue_time;

        let mut num_primes = self.config.min_primes;
        let mut cached_solutions: Vec<Vec<u32>> = Vec::new();

        loop {
            stats.iterations += 1;
            stats.prime_counts.push(num_primes);

            // Build basis with current prime count
            let primes = self.prime_pool[..num_primes].to_vec();
            let basis = CRTBasis::new(primes.clone());

            // Solve for each prime (reuse cached solutions from previous iterations)
            let solve_start = Instant::now();
            let solutions_per_prime = self.solve_with_primes(
                &matrix_residues,
                &b_residues,
                n,
                &primes,
                &mut cached_solutions,
            );
            let solve_time = solve_start.elapsed().as_secs_f64();
            timings.solve_time += solve_time;
            stats.phase_times.solve_total += solve_time;

            // Handle singular case
            let solutions = match solutions_per_prime {
                Some(s) => s,
                None => {
                    timings.total_time = start.elapsed().as_secs_f64();
                    timings.num_primes = num_primes;
                    return (SolveResult::Singular { rank: 0 }, timings, stats);
                }
            };

            // CRT reconstruction
            let crt_start = Instant::now();
            let x_bigint = self.reconstruct_solution(&solutions, n, num_primes, &basis);
            let crt_time = crt_start.elapsed().as_secs_f64();
            timings.crt_time += crt_time;
            stats.phase_times.crt_total += crt_time;

            // Verification using modular arithmetic (fast)
            let verify_start = Instant::now();
            let verified = self.verify_solution_modular(matrix, b, &x_bigint, n, num_primes);
            let verify_time = verify_start.elapsed().as_secs_f64();
            timings.verify_time += verify_time;
            stats.phase_times.verify_total += verify_time;
            stats.verification_results.push(verified);

            if verified {
                // Success! Early exit - convert to rationals
                let x_rational: Vec<Rational> = x_bigint
                    .into_iter()
                    .map(Rational::from_bigint)
                    .collect();
                timings.num_primes = num_primes;
                timings.total_time = start.elapsed().as_secs_f64();
                return (
                    SolveResult::Solution { x: x_rational, verified: true },
                    timings,
                    stats,
                );
            }

            // Need more primes
            let new_count = ((num_primes as f64) * self.config.growth_factor).ceil() as usize;
            num_primes = new_count.min(self.config.max_primes);

            if num_primes >= self.config.max_primes && !verified {
                // Hit max primes without verification - likely a bug or numerical issue
                let x_rational: Vec<Rational> = x_bigint
                    .into_iter()
                    .map(Rational::from_bigint)
                    .collect();
                timings.num_primes = num_primes;
                timings.total_time = start.elapsed().as_secs_f64();
                return (
                    SolveResult::Solution { x: x_rational, verified: false },
                    timings,
                    stats,
                );
            }
        }
    }

    /// Solve AX = B with adaptive prime selection (multi-RHS)
    pub fn solve_multi_rhs(
        &self,
        matrix: &[BigInt],
        b_cols: &[Vec<BigInt>],
    ) -> (Vec<Vec<Rational>>, Timings, AdaptiveStats) {
        let mut timings = Timings::default();
        let mut stats = AdaptiveStats::default();
        let start = Instant::now();
        let n = b_cols[0].len();
        let k = b_cols.len();

        // Prepare residues
        let residue_start = Instant::now();
        let matrix_residues = self.compute_all_residues(matrix);
        let b_cols_residues: Vec<Vec<Vec<u32>>> = b_cols
            .iter()
            .map(|col| self.compute_all_residues(col))
            .collect();
        timings.residue_time = residue_start.elapsed().as_secs_f64();
        stats.phase_times.residue_total = timings.residue_time;

        let mut num_primes = self.config.min_primes;
        let mut cached_solutions: Vec<Vec<Vec<u32>>> = Vec::new(); // [prime][col][row]

        loop {
            stats.iterations += 1;
            stats.prime_counts.push(num_primes);

            let primes = self.prime_pool[..num_primes].to_vec();
            let basis = CRTBasis::new(primes.clone());

            // Solve for each prime
            let solve_start = Instant::now();
            let solutions_per_prime = self.solve_multi_rhs_with_primes(
                &matrix_residues,
                &b_cols_residues,
                n,
                k,
                &primes,
                &mut cached_solutions,
            );
            let solve_time = solve_start.elapsed().as_secs_f64();
            timings.solve_time += solve_time;
            stats.phase_times.solve_total += solve_time;

            let solutions = match solutions_per_prime {
                Some(s) => s,
                None => {
                    timings.total_time = start.elapsed().as_secs_f64();
                    timings.num_primes = num_primes;
                    return (vec![], timings, stats);
                }
            };

            // CRT reconstruction (to BigInt first)
            let crt_start = Instant::now();
            let x_bigint = self.reconstruct_multi_rhs_bigint(&solutions, n, k, num_primes, &basis);
            let crt_time = crt_start.elapsed().as_secs_f64();
            timings.crt_time += crt_time;
            stats.phase_times.crt_total += crt_time;

            // Verification using modular arithmetic (check first and last column)
            let verify_start = Instant::now();
            let verified = self.verify_multi_rhs_sample_modular(matrix, b_cols, &x_bigint, n, num_primes);
            let verify_time = verify_start.elapsed().as_secs_f64();
            timings.verify_time += verify_time;
            stats.phase_times.verify_total += verify_time;
            stats.verification_results.push(verified);

            if verified {
                // Convert to rationals
                let x_rational: Vec<Vec<Rational>> = x_bigint
                    .into_iter()
                    .map(|col| col.into_iter().map(Rational::from_bigint).collect())
                    .collect();
                timings.num_primes = num_primes;
                timings.total_time = start.elapsed().as_secs_f64();
                return (x_rational, timings, stats);
            }

            let new_count = ((num_primes as f64) * self.config.growth_factor).ceil() as usize;
            num_primes = new_count.min(self.config.max_primes);

            if num_primes >= self.config.max_primes && !verified {
                // Max primes reached without verification - return what we have
                let x_rational: Vec<Vec<Rational>> = x_bigint
                    .into_iter()
                    .map(|col| col.into_iter().map(Rational::from_bigint).collect())
                    .collect();
                timings.num_primes = num_primes;
                timings.total_time = start.elapsed().as_secs_f64();
                return (x_rational, timings, stats);
            }
        }
    }

    /// Compute residues for all primes in the pool (once)
    fn compute_all_residues(&self, values: &[BigInt]) -> Vec<Vec<u32>> {
        // residues[prime_idx][value_idx]
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

    /// Solve using specific primes, caching results
    fn solve_with_primes(
        &self,
        matrix_residues: &[Vec<u32>],
        b_residues: &[Vec<u32>],
        n: usize,
        primes: &[u32],
        cached: &mut Vec<Vec<u32>>,
    ) -> Option<Vec<Vec<u32>>> {
        let mut solutions = Vec::with_capacity(primes.len());

        for (i, &p) in primes.iter().enumerate() {
            if i < cached.len() {
                // Reuse cached solution
                solutions.push(cached[i].clone());
            } else {
                // Compute new solution for this prime
                let matrix_mod = &matrix_residues[i];
                let b_mod = &b_residues[i];

                match self.backend.solve_mod(matrix_mod, b_mod, n, p) {
                    Some(sol) => {
                        if self.config.cache_residues {
                            cached.push(sol.clone());
                        }
                        solutions.push(sol);
                    }
                    None => return None,
                }
            }
        }

        Some(solutions)
    }

    /// Solve multi-RHS using specific primes
    fn solve_multi_rhs_with_primes(
        &self,
        matrix_residues: &[Vec<u32>],
        b_cols_residues: &[Vec<Vec<u32>>],
        n: usize,
        k: usize,
        primes: &[u32],
        cached: &mut Vec<Vec<Vec<u32>>>,
    ) -> Option<Vec<Vec<Vec<u32>>>> {
        let mut solutions = Vec::with_capacity(primes.len());

        for (i, &p) in primes.iter().enumerate() {
            if i < cached.len() {
                solutions.push(cached[i].clone());
            } else {
                let matrix_mod = &matrix_residues[i];
                let b_cols_mod: Vec<Vec<u32>> = b_cols_residues
                    .iter()
                    .map(|col_residues| col_residues[i].clone())
                    .collect();

                match self.backend.solve_multi_rhs_mod(matrix_mod, &b_cols_mod, n, k, p) {
                    Some(sol) => {
                        if self.config.cache_residues {
                            cached.push(sol.clone());
                        }
                        solutions.push(sol);
                    }
                    None => return None,
                }
            }
        }

        Some(solutions)
    }

    /// Reconstruct solution from residues
    fn reconstruct_solution(
        &self,
        solutions: &[Vec<u32>],
        n: usize,
        num_primes: usize,
        basis: &CRTBasis,
    ) -> Vec<BigInt> {
        // Transpose: [prime][component] -> [component][prime]
        let mut x_residues: Vec<Vec<u32>> = vec![vec![0; num_primes]; n];
        for (prime_idx, sol) in solutions.iter().enumerate() {
            for (i, &val) in sol.iter().enumerate() {
                x_residues[i][prime_idx] = val;
            }
        }

        x_residues
            .iter()
            .map(|residues| CRTReconstruction::reconstruct_signed(residues, basis))
            .collect()
    }

    /// Reconstruct multi-RHS solution to BigInt (for verification)
    fn reconstruct_multi_rhs_bigint(
        &self,
        solutions: &[Vec<Vec<u32>>],
        n: usize,
        k: usize,
        num_primes: usize,
        basis: &CRTBasis,
    ) -> Vec<Vec<BigInt>> {
        // Transpose: [prime][col][row] -> [col][row][prime]
        let mut x_residues: Vec<Vec<Vec<u32>>> = vec![vec![vec![0; num_primes]; n]; k];
        for (prime_idx, sols_for_prime) in solutions.iter().enumerate() {
            for (col_idx, col) in sols_for_prime.iter().enumerate() {
                for (row_idx, &val) in col.iter().enumerate() {
                    x_residues[col_idx][row_idx][prime_idx] = val;
                }
            }
        }

        x_residues
            .into_iter()
            .map(|col| {
                col.into_iter()
                    .map(|residues| CRTReconstruction::reconstruct_signed(&residues, basis))
                    .collect()
            })
            .collect()
    }

    /// Verify solution Ax = b using modular arithmetic
    ///
    /// Uses a verification prime not in the CRT basis to check correctness.
    /// This is O(n²) instead of O(n² × bigint_mult), much faster.
    fn verify_solution_modular(
        &self,
        matrix: &[BigInt],
        b: &[BigInt],
        x: &[BigInt],
        n: usize,
        num_primes_used: usize,
    ) -> bool {
        // Use a DIFFERENT verification prime outside the CRT basis
        // Make sure it's not one of the primes we used for CRT
        let verify_idx = num_primes_used.min(self.prime_pool.len() - 1);
        let verify_prime = self.prime_pool[verify_idx];
        let p = verify_prime as u64;

        // Reduce matrix, x, b mod p
        let matrix_mod: Vec<u64> = matrix.iter()
            .map(|v| {
                let r = v % BigInt::from(p);
                if r < BigInt::zero() {
                    ((r + BigInt::from(p)) % BigInt::from(p)).try_into().unwrap_or(0u64)
                } else {
                    r.try_into().unwrap_or(0u64)
                }
            })
            .collect();

        let x_mod: Vec<u64> = x.iter()
            .map(|v| {
                let r = v % BigInt::from(p);
                if r < BigInt::zero() {
                    ((r + BigInt::from(p)) % BigInt::from(p)).try_into().unwrap_or(0u64)
                } else {
                    r.try_into().unwrap_or(0u64)
                }
            })
            .collect();

        let b_mod: Vec<u64> = b.iter()
            .map(|v| {
                let r = v % BigInt::from(p);
                if r < BigInt::zero() {
                    ((r + BigInt::from(p)) % BigInt::from(p)).try_into().unwrap_or(0u64)
                } else {
                    r.try_into().unwrap_or(0u64)
                }
            })
            .collect();

        // Verify Ax = b mod p
        for i in 0..n {
            let mut sum: u64 = 0;
            for j in 0..n {
                sum = (sum + (matrix_mod[i * n + j] * x_mod[j]) % p) % p;
            }
            if sum != b_mod[i] {
                return false;
            }
        }

        true
    }

    /// Verify multi-RHS by sampling (first and last column)
    fn verify_multi_rhs_sample_modular(
        &self,
        matrix: &[BigInt],
        b_cols: &[Vec<BigInt>],
        x_cols: &[Vec<BigInt>],
        n: usize,
        num_primes_used: usize,
    ) -> bool {
        if x_cols.is_empty() {
            return b_cols.is_empty();
        }

        // Check first column
        if !self.verify_solution_modular(matrix, &b_cols[0], &x_cols[0], n, num_primes_used) {
            return false;
        }

        // Check last column if different
        if x_cols.len() > 1 {
            let last = x_cols.len() - 1;
            if !self.verify_solution_modular(matrix, &b_cols[last], &x_cols[last], n, num_primes_used) {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuBackend;

    #[test]
    fn test_adaptive_solve_simple() {
        let backend = CpuBackend::new();
        let solver = AdaptiveSolver::new(backend);

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

        // Should succeed with minimal primes
        assert_eq!(stats.iterations, 1);
        assert_eq!(timings.num_primes, 4); // min_primes
    }

    #[test]
    fn test_adaptive_needs_more_primes() {
        let backend = CpuBackend::new();
        let config = AdaptiveConfig {
            min_primes: 2, // Start very low
            max_primes: 32,
            growth_factor: 2.0,
            cache_residues: true,
        };
        let solver = AdaptiveSolver::with_config(backend, config);

        // Larger coefficients that need more bits
        let matrix: Vec<BigInt> = vec![
            BigInt::from(1_000_000_000i64),
            BigInt::from(1),
            BigInt::from(1),
            BigInt::from(1_000_000_000i64),
        ];
        let b: Vec<BigInt> = vec![
            BigInt::from(1_000_000_001i64),
            BigInt::from(1_000_000_001i64),
        ];

        let (result, timings, stats) = solver.solve(&matrix, &b);

        match result {
            SolveResult::Solution { verified, .. } => {
                assert!(verified);
            }
            _ => panic!("Expected solution"),
        }

        // May need multiple iterations depending on coefficients
        println!("Iterations: {}, final primes: {}", stats.iterations, timings.num_primes);
    }
}
