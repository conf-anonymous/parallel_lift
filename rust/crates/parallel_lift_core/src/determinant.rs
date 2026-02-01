//! CRT-based exact determinant computation
//!
//! Computes determinant of integer matrices using CRT.

use crate::{Backend, CRTBasis, CRTReconstruction, Timings};
use num_bigint::BigInt;
use std::time::Instant;

/// CRT-based determinant calculator
pub struct Determinant<B: Backend> {
    backend: B,
}

impl<B: Backend> Determinant<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Compute determinant of an integer matrix exactly
    ///
    /// # Arguments
    /// * `matrix` - n×n matrix in row-major order (flattened)
    /// * `n` - Matrix dimension
    /// * `basis` - CRT prime basis
    ///
    /// # Returns
    /// * Exact determinant as BigInt
    /// * Timing breakdown
    pub fn compute(&self, matrix: &[BigInt], n: usize, basis: &CRTBasis) -> (BigInt, Timings) {
        let mut timings = Timings::default();
        let start = Instant::now();

        // 1. Reduce matrix to residues
        let residue_start = Instant::now();
        let matrix_residues = self.reduce_matrix(matrix, basis);
        timings.residue_time = residue_start.elapsed().as_secs_f64();

        // 2. Compute determinant mod each prime
        let det_start = Instant::now();
        let det_residues: Vec<u32> = basis
            .primes
            .iter()
            .enumerate()
            .map(|(prime_idx, &p)| {
                let matrix_mod: Vec<u32> = matrix_residues
                    .iter()
                    .map(|residues| residues[prime_idx])
                    .collect();
                self.backend.determinant_mod(&matrix_mod, n, p)
            })
            .collect();
        timings.det_time = det_start.elapsed().as_secs_f64();

        // 3. CRT reconstruction
        let crt_start = Instant::now();
        let determinant = CRTReconstruction::reconstruct_signed(&det_residues, basis);
        timings.crt_time = crt_start.elapsed().as_secs_f64();

        timings.num_primes = basis.primes.len();
        timings.total_time = start.elapsed().as_secs_f64();

        (determinant, timings)
    }

    /// Batch compute determinants using backend's batch method
    ///
    /// Uses the backend's batch_determinant_mod for better GPU utilization.
    pub fn compute_batched(&self, matrix: &[BigInt], n: usize, basis: &CRTBasis) -> (BigInt, Timings) {
        let mut timings = Timings::default();
        let start = Instant::now();

        // 1. Reduce matrix - we'll reduce once and dispatch all primes
        let residue_start = Instant::now();
        // For batch, we need a flat u32 matrix (just use first prime's reduction as base)
        let matrix_u32: Vec<u32> = matrix
            .iter()
            .map(|v| {
                let abs_v = if v < &BigInt::from(0) { -v } else { v.clone() };
                abs_v.try_into().unwrap_or(u32::MAX)
            })
            .collect();
        timings.residue_time = residue_start.elapsed().as_secs_f64();

        // 2. Batch compute determinants
        let det_start = Instant::now();
        let det_residues = self.backend.batch_determinant_mod(&matrix_u32, n, &basis.primes);
        timings.det_time = det_start.elapsed().as_secs_f64();

        // 3. CRT reconstruction
        let crt_start = Instant::now();
        let determinant = CRTReconstruction::reconstruct_signed(&det_residues, basis);
        timings.crt_time = crt_start.elapsed().as_secs_f64();

        timings.num_primes = basis.primes.len();
        timings.total_time = start.elapsed().as_secs_f64();

        (determinant, timings)
    }

    /// Reduce matrix to residues mod each prime
    fn reduce_matrix(&self, matrix: &[BigInt], basis: &CRTBasis) -> Vec<Vec<u32>> {
        matrix
            .iter()
            .map(|val| {
                basis.primes
                    .iter()
                    .map(|&p| {
                        let p_big = BigInt::from(p);
                        let r = val % &p_big;
                        if r < BigInt::from(0) {
                            (r + &p_big).try_into().unwrap_or(0)
                        } else {
                            r.try_into().unwrap_or(0)
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuBackend, primes::PrimeGenerator};

    #[test]
    fn test_determinant_2x2() {
        let backend = CpuBackend::new();
        let det_calc = Determinant::new(backend);

        // [[1, 2], [3, 4]], det = -2
        let matrix: Vec<BigInt> = vec![1, 2, 3, 4].into_iter().map(BigInt::from).collect();

        let primes = PrimeGenerator::generate_31bit_primes(4);
        let basis = CRTBasis::new(primes);

        let (det, _timings) = det_calc.compute(&matrix, 2, &basis);
        assert_eq!(det, BigInt::from(-2));
    }

    #[test]
    fn test_determinant_3x3() {
        let backend = CpuBackend::new();
        let det_calc = Determinant::new(backend);

        // [[1, 2, 3], [4, 5, 6], [7, 8, 9]], det = 0 (singular)
        let matrix: Vec<BigInt> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9]
            .into_iter()
            .map(BigInt::from)
            .collect();

        let primes = PrimeGenerator::generate_31bit_primes(4);
        let basis = CRTBasis::new(primes);

        let (det, _timings) = det_calc.compute(&matrix, 3, &basis);
        assert_eq!(det, BigInt::from(0));
    }

    #[test]
    fn test_determinant_nonsingular_3x3() {
        let backend = CpuBackend::new();
        let det_calc = Determinant::new(backend);

        // [[1, 2, 3], [4, 5, 6], [7, 8, 10]], det = -3
        let matrix: Vec<BigInt> = vec![1, 2, 3, 4, 5, 6, 7, 8, 10]
            .into_iter()
            .map(BigInt::from)
            .collect();

        let primes = PrimeGenerator::generate_31bit_primes(4);
        let basis = CRTBasis::new(primes);

        let (det, _timings) = det_calc.compute(&matrix, 3, &basis);
        assert_eq!(det, BigInt::from(-3));
    }
}
