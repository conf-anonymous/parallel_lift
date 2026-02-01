//! LLL Lattice Reduction Algorithm
//!
//! GPU-accelerated LLL using CRT-based exact arithmetic.
//!
//! # The LLL Algorithm
//!
//! Given a basis B = [b_1, ..., b_n], LLL produces a δ-reduced basis satisfying:
//! 1. **Size reduction**: |μ_ij| ≤ 1/2 for all j < i
//! 2. **Lovász condition**: δ ||b*_i||² ≤ ||b*_{i+1} + μ_{i+1,i} b*_i||²
//!
//! # Complexity
//!
//! Standard LLL: O(n^4 × log B × log log B) where B is max entry bit size
//! Our CRT version: Same asymptotic, but with parallel modular operations

use num_bigint::BigInt;
use num_traits::{Zero, One};
use num_integer::Integer;
use std::time::Instant;
use crate::primes::PrimeGenerator;
use crate::backend::Backend;
use super::basis::LatticeBasis;
use super::gram_schmidt::GramSchmidt;

/// LLL configuration parameters
#[derive(Debug, Clone)]
pub struct LLLConfig {
    /// Lovász parameter δ (default 0.75 = 3/4)
    /// Must be in (1/4, 1). Higher values give better reduction but slower.
    pub delta_num: i64,
    pub delta_den: i64,
    /// Maximum iterations (safety limit)
    pub max_iterations: usize,
    /// Whether to use deep insertions (BKZ-style optimization)
    pub deep_insertions: bool,
    /// Verbosity level (0 = silent, 1 = summary, 2 = detailed)
    pub verbose: u32,
}

impl Default for LLLConfig {
    fn default() -> Self {
        Self {
            delta_num: 3,
            delta_den: 4,
            max_iterations: 1_000_000,
            deep_insertions: false,
            verbose: 0,
        }
    }
}

impl LLLConfig {
    /// Create config with δ = 0.99 (strong reduction)
    pub fn strong() -> Self {
        Self {
            delta_num: 99,
            delta_den: 100,
            ..Default::default()
        }
    }

    /// Create config with δ = 0.5 (fast but weaker reduction)
    pub fn fast() -> Self {
        Self {
            delta_num: 1,
            delta_den: 2,
            ..Default::default()
        }
    }
}

/// Statistics from LLL execution
#[derive(Debug, Clone, Default)]
pub struct LLLStats {
    /// Number of size reductions performed
    pub size_reductions: usize,
    /// Number of swaps performed
    pub swaps: usize,
    /// Total iterations
    pub iterations: usize,
    /// Time for Gram-Schmidt computation (seconds)
    pub gs_time: f64,
    /// Time for size reduction (seconds)
    pub reduce_time: f64,
    /// Time for swap operations (seconds)
    pub swap_time: f64,
    /// Total time (seconds)
    pub total_time: f64,
    /// Final basis quality (log of first vector norm / Gaussian heuristic)
    pub hermite_factor: f64,
}

/// LLL lattice reduction algorithm
pub struct LLL;

impl LLL {
    /// Reduce a lattice basis using the LLL algorithm
    ///
    /// # Arguments
    /// * `basis` - The input lattice basis
    /// * `config` - LLL configuration parameters
    ///
    /// # Returns
    /// The reduced basis and execution statistics
    pub fn reduce(basis: &LatticeBasis, config: &LLLConfig) -> (LatticeBasis, LLLStats) {
        let start = Instant::now();
        let mut stats = LLLStats::default();

        // Clone basis for modification
        let mut b = basis.clone();
        let n = b.n;

        if n <= 1 {
            stats.total_time = start.elapsed().as_secs_f64();
            return (b, stats);
        }

        // Compute initial Gram-Schmidt
        let gs_start = Instant::now();
        let mut gs = GramSchmidt::compute(&b);
        stats.gs_time = gs_start.elapsed().as_secs_f64();

        // Main LLL loop
        let mut k = 1usize;

        while k < n && stats.iterations < config.max_iterations {
            stats.iterations += 1;

            // Size reduce b_k with respect to b_{k-1}
            let reduce_start = Instant::now();
            Self::size_reduce(&mut b, &mut gs, k, k - 1, &mut stats);
            stats.reduce_time += reduce_start.elapsed().as_secs_f64();

            // Check Lovász condition
            if gs.check_lovasz(k, config.delta_num, config.delta_den) {
                // Size reduce b_k with respect to all b_j for j < k-1
                for j in (0..k - 1).rev() {
                    let reduce_start = Instant::now();
                    Self::size_reduce(&mut b, &mut gs, k, j, &mut stats);
                    stats.reduce_time += reduce_start.elapsed().as_secs_f64();
                }
                k += 1;
            } else {
                // Swap b_k and b_{k-1}
                let swap_start = Instant::now();
                b.swap(k, k - 1);
                // Recompute Gram-Schmidt (simpler and more robust than incremental update)
                gs = GramSchmidt::compute(&b);
                stats.swaps += 1;
                stats.swap_time += swap_start.elapsed().as_secs_f64();

                // Go back
                k = if k > 1 { k - 1 } else { 1 };
            }

            // Progress reporting
            if config.verbose >= 2 && stats.iterations % 1000 == 0 {
                eprintln!(
                    "LLL iteration {}: k={}, swaps={}, reductions={}",
                    stats.iterations, k, stats.swaps, stats.size_reductions
                );
            }
        }

        stats.total_time = start.elapsed().as_secs_f64();

        // Compute Hermite factor
        let first_norm_sq = b.norm_squared(0);
        let det_approx = Self::estimate_determinant(&b);
        if det_approx > BigInt::zero() {
            let det_root_n = Self::nth_root_approx(&det_approx, n);
            if det_root_n > BigInt::zero() {
                let first_norm = Self::sqrt_approx(&first_norm_sq);
                let ratio = Self::bigint_to_f64(&first_norm) / Self::bigint_to_f64(&det_root_n);
                stats.hermite_factor = ratio;
            }
        }

        if config.verbose >= 1 {
            eprintln!(
                "LLL completed: {} iterations, {} swaps, {} reductions, {:.3}s",
                stats.iterations, stats.swaps, stats.size_reductions, stats.total_time
            );
        }

        (b, stats)
    }

    /// Perform size reduction: b_k = b_k - round(μ_kj) * b_j
    fn size_reduce(
        basis: &mut LatticeBasis,
        gs: &mut GramSchmidt,
        k: usize,
        j: usize,
        stats: &mut LLLStats,
    ) {
        if !gs.needs_size_reduction(k, j) {
            return;
        }

        let mu = gs.get_mu(k, j);

        // q = round(μ_kj) = floor(μ_kj + 1/2)
        // For exact rational: q = floor((2*num + den) / (2*den))
        let two_num: BigInt = &mu.numerator * 2;
        let two_den: BigInt = &mu.denominator * 2;
        let q: BigInt = (&two_num + &mu.denominator).div_floor(&two_den);

        if q.is_zero() {
            return;
        }

        // b_k = b_k - q * b_j
        basis.reduce_vector(k, j, &q);

        // Update Gram-Schmidt
        gs.update_size_reduction(k, j, &q);

        stats.size_reductions += 1;
    }

    /// Estimate lattice determinant from basis
    fn estimate_determinant(basis: &LatticeBasis) -> BigInt {
        // For square bases, det = product of norms (approximately)
        // This is a rough estimate for the Hermite factor computation
        let mut det = BigInt::one();
        for i in 0..basis.n {
            let norm_sq = basis.norm_squared(i);
            det = det * Self::sqrt_approx(&norm_sq);
        }
        det
    }

    /// Integer square root approximation
    fn sqrt_approx(n: &BigInt) -> BigInt {
        if n <= &BigInt::zero() {
            return BigInt::zero();
        }
        // Newton's method
        let mut x: BigInt = n.clone();
        let mut y: BigInt = (&x + 1) / 2;
        while y < x {
            x = y.clone();
            y = (&x + n / &x) / 2;
        }
        x
    }

    /// Approximate n-th root of a BigInt
    fn nth_root_approx(n: &BigInt, root: usize) -> BigInt {
        if n <= &BigInt::zero() || root == 0 {
            return BigInt::zero();
        }
        if root == 1 {
            return n.clone();
        }
        if root == 2 {
            return Self::sqrt_approx(n);
        }

        // Binary search for n^(1/root)
        let bits = n.bits() as usize;
        let mut lo = BigInt::one();
        let mut hi = BigInt::one() << ((bits / root) + 1);

        while lo < &hi - 1 {
            let mid = (&lo + &hi) / 2;
            let mid_pow = Self::pow_bigint(&mid, root);
            if mid_pow <= *n {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        lo
    }

    /// BigInt power
    fn pow_bigint(base: &BigInt, exp: usize) -> BigInt {
        let mut result = BigInt::one();
        let mut b = base.clone();
        let mut e = exp;
        while e > 0 {
            if e & 1 == 1 {
                result = result * &b;
            }
            b = &b * &b;
            e >>= 1;
        }
        result
    }

    /// Convert BigInt to f64 (lossy, for reporting only)
    fn bigint_to_f64(n: &BigInt) -> f64 {
        let (sign, digits) = n.to_u64_digits();
        if digits.is_empty() {
            return 0.0;
        }
        let mut result = 0.0f64;
        let mut multiplier = 1.0f64;
        for &d in &digits {
            result += d as f64 * multiplier;
            multiplier *= (1u64 << 63) as f64 * 2.0;
        }
        if sign == num_bigint::Sign::Minus {
            -result
        } else {
            result
        }
    }

    /// Check if a basis is LLL-reduced
    pub fn is_reduced(basis: &LatticeBasis, config: &LLLConfig) -> bool {
        let gs = GramSchmidt::compute(basis);
        let n = basis.n;

        // Check size reduction
        for i in 1..n {
            for j in 0..i {
                if gs.needs_size_reduction(i, j) {
                    return false;
                }
            }
        }

        // Check Lovász condition
        for k in 1..n {
            if !gs.check_lovasz(k, config.delta_num, config.delta_den) {
                return false;
            }
        }

        true
    }
}

/// GPU-accelerated LLL using batch modular operations
pub struct LLLBatched;

impl LLLBatched {
    /// Reduce a lattice basis using GPU-accelerated LLL
    ///
    /// This version uses CRT representation for Gram-Schmidt computations
    /// and batches operations across primes for GPU parallelism.
    ///
    /// # Arguments
    /// * `basis` - The input lattice basis
    /// * `config` - LLL configuration parameters
    /// * `backend` - GPU backend for parallel computation
    ///
    /// # Returns
    /// The reduced basis and execution statistics
    pub fn reduce<B: Backend>(
        basis: &LatticeBasis,
        config: &LLLConfig,
        _backend: &B,
    ) -> (LatticeBasis, LLLStats) {
        // For now, use the CPU version as a baseline
        // GPU acceleration will be added for the inner product computations
        // and Gram-Schmidt updates

        // Estimate number of primes needed
        let gs_bits = basis.estimate_gs_bits();
        let num_primes = PrimeGenerator::estimate_primes_needed(gs_bits);

        if config.verbose >= 1 {
            eprintln!(
                "LLLBatched: n={}, m={}, estimated bits={}, primes={}",
                basis.n, basis.m, gs_bits, num_primes
            );
        }

        // TODO: Implement GPU-accelerated version
        // For now, fall back to CPU version
        LLL::reduce(basis, config)
    }

    /// Batch compute inner products using GPU
    ///
    /// For a basis with n vectors of dimension m, compute the Gram matrix
    /// G[i,j] = <b_i, b_j> using GPU parallelism.
    pub fn batch_inner_products<B: Backend>(
        basis: &LatticeBasis,
        backend: &B,
        primes: &[u32],
    ) -> Vec<Vec<u32>> {
        let n = basis.n;
        let m = basis.m;

        // Result: inner_products[prime_idx][i*n + j]
        let mut results = vec![vec![0u32; n * n]; primes.len()];

        for (p_idx, &p) in primes.iter().enumerate() {
            let residues = basis.to_residues(p);
            let p64 = p as u64;

            // Compute all inner products mod p
            for i in 0..n {
                for j in 0..=i {
                    let mut sum = 0u64;
                    for k in 0..m {
                        let a = residues[i * m + k] as u64;
                        let b = residues[j * m + k] as u64;
                        sum = (sum + (a * b) % p64) % p64;
                    }
                    results[p_idx][i * n + j] = sum as u32;
                    results[p_idx][j * n + i] = sum as u32;
                }
            }
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lll_simple() {
        // Simple 2D lattice that needs reduction
        let basis = LatticeBasis::from_rows(&[
            vec![1i64, 1],
            vec![0, 1],
        ]);

        let config = LLLConfig::default();
        let (reduced, stats) = LLL::reduce(&basis, &config);

        // Should be LLL-reduced
        assert!(LLL::is_reduced(&reduced, &config));
        assert!(stats.total_time < 1.0); // Should be fast
    }

    #[test]
    fn test_lll_knapsack() {
        // Small knapsack lattice
        let a = vec![3i64, 5, 7];
        let s = 12i64; // 3 + 5 + 7 = 15, or 5 + 7 = 12
        let basis = LatticeBasis::knapsack(&a, s);

        let config = LLLConfig::default();
        let (reduced, stats) = LLL::reduce(&basis, &config);

        // Should be LLL-reduced
        assert!(LLL::is_reduced(&reduced, &config));

        // Check that basis is smaller
        let original_norm = basis.norm_squared(0);
        let reduced_norm = reduced.norm_squared(0);

        // The reduced basis should have shorter first vector
        // (or at least not much longer)
        eprintln!(
            "Original first vector norm²: {}, Reduced: {}",
            original_norm, reduced_norm
        );
        eprintln!("Stats: {:?}", stats);
    }

    #[test]
    fn test_lll_random() {
        // Random small lattice
        let basis = LatticeBasis::random(5, 5, 8);

        let config = LLLConfig::default();
        let (reduced, stats) = LLL::reduce(&basis, &config);

        // Should be LLL-reduced
        assert!(LLL::is_reduced(&reduced, &config));
        eprintln!("Random lattice stats: {:?}", stats);
    }

    #[test]
    fn test_lll_identity() {
        // Identity lattice (already reduced)
        let basis = LatticeBasis::from_rows(&[
            vec![1i64, 0, 0],
            vec![0, 1, 0],
            vec![0, 0, 1],
        ]);

        let config = LLLConfig::default();
        let (reduced, stats) = LLL::reduce(&basis, &config);

        // Should remain the same (no swaps needed)
        assert_eq!(stats.swaps, 0);
        assert!(LLL::is_reduced(&reduced, &config));
    }

    #[test]
    fn test_lll_strong() {
        // Test with stronger reduction parameter
        let basis = LatticeBasis::from_rows(&[
            vec![1i64, 1, 1],
            vec![-1, 0, 2],
            vec![3, 5, 6],
        ]);

        let config_weak = LLLConfig::fast();
        let config_strong = LLLConfig::strong();

        let (reduced_weak, _) = LLL::reduce(&basis, &config_weak);
        let (reduced_strong, _) = LLL::reduce(&basis, &config_strong);

        // Both should be reduced under their respective parameters
        assert!(LLL::is_reduced(&reduced_weak, &config_weak));
        assert!(LLL::is_reduced(&reduced_strong, &config_strong));

        // Strong reduction may produce better results
        let weak_norm = reduced_weak.norm_squared(0);
        let strong_norm = reduced_strong.norm_squared(0);

        eprintln!("Weak δ=0.5 norm²: {}", weak_norm);
        eprintln!("Strong δ=0.99 norm²: {}", strong_norm);
    }

    #[test]
    fn test_lll_dimension_10() {
        // Larger test case
        let basis = LatticeBasis::random(10, 10, 16);

        let config = LLLConfig {
            verbose: 1,
            ..Default::default()
        };
        let (reduced, stats) = LLL::reduce(&basis, &config);

        assert!(LLL::is_reduced(&reduced, &config));
        eprintln!("10×10 stats: {:?}", stats);
    }
}
