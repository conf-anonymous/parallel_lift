//! Gram-Schmidt orthogonalization via CRT
//!
//! Computes the Gram-Schmidt orthogonalization of a lattice basis using
//! CRT arithmetic for GPU acceleration.
//!
//! # The Gram-Schmidt Process
//!
//! Given basis B = [b_1, ..., b_n], compute orthogonal vectors b*_i and coefficients μ_ij:
//!
//! ```text
//! b*_1 = b_1
//! b*_i = b_i - Σ_{j<i} μ_ij b*_j
//! μ_ij = <b_i, b*_j> / <b*_j, b*_j>
//! ```
//!
//! # CRT Representation
//!
//! The key insight is that all Gram-Schmidt computations involve exact rationals.
//! We represent:
//! - Numerators of μ_ij: <b_i, b*_j> (integers)
//! - Denominators: <b*_j, b*_j> (integers)
//!
//! All arithmetic is done mod each prime independently.

use num_bigint::BigInt;
use num_traits::{Zero, Signed};
use crate::rational::Rational;
use crate::primes::CRTBasis;
use crate::crt::CRTReconstruction;
use super::basis::LatticeBasis;

/// Gram-Schmidt orthogonalization data (exact rational representation)
#[derive(Debug, Clone)]
pub struct GramSchmidt {
    /// Gram-Schmidt coefficients μ_ij (i > j)
    /// Stored as lower triangular matrix: mu[i][j] for j < i
    pub mu: Vec<Vec<Rational>>,
    /// Squared norms ||b*_i||^2 (exact rationals)
    pub b_star_norms_sq: Vec<Rational>,
    /// Dimension
    pub n: usize,
}

impl GramSchmidt {
    /// Compute Gram-Schmidt orthogonalization from a lattice basis
    ///
    /// This is the basic CPU implementation using exact rational arithmetic.
    pub fn compute(basis: &LatticeBasis) -> Self {
        let n = basis.n;

        // Initialize μ matrix (lower triangular)
        let mut mu: Vec<Vec<Rational>> = (0..n)
            .map(|i| vec![Rational::zero(); i])
            .collect();

        // B* squared norms
        let mut b_star_norms_sq = vec![Rational::zero(); n];

        // Track the orthogonalized vectors (stored as exact rationals)
        // b*_i = b_i - sum_{j<i} μ_ij * b*_j
        // But we only need <b*_i, b*_i> and <b_k, b*_i> for k > i

        // Inner products <b_i, b_j> (precompute for efficiency)
        let inner_products: Vec<Vec<BigInt>> = (0..n)
            .map(|i| {
                (0..=i)
                    .map(|j| basis.inner_product(i, j))
                    .collect()
            })
            .collect();

        // First vector
        b_star_norms_sq[0] = Rational::from_bigint(inner_products[0][0].clone());

        // Process remaining vectors
        for i in 1..n {
            // Compute μ_ij and b*_i
            // <b_i, b*_j> = <b_i, b_j> - sum_{k<j} μ_jk * <b_i, b*_k>
            //
            // We use the recurrence:
            // <b_i, b*_j> = <b_i, b_j> - sum_{k<j} μ_jk * <b_i, b*_k>
            // μ_ij = <b_i, b*_j> / ||b*_j||^2

            let mut inner_with_b_star: Vec<Rational> = vec![Rational::zero(); i];

            for j in 0..i {
                // <b_i, b*_j>
                let mut inner_i_bstarj = Rational::from_bigint(inner_products[i][j].clone());

                for k in 0..j {
                    // inner_i_bstarj -= μ_jk * <b_i, b*_k>
                    let mu_jk = &mu[j][k];
                    let prod = mu_jk * &inner_with_b_star[k];
                    inner_i_bstarj = inner_i_bstarj - prod;
                }

                inner_with_b_star[j] = inner_i_bstarj.clone();

                // μ_ij = <b_i, b*_j> / ||b*_j||^2
                mu[i][j] = inner_i_bstarj / b_star_norms_sq[j].clone();
            }

            // ||b*_i||^2 = <b_i, b_i> - sum_{j<i} μ_ij * <b_i, b*_j>
            let mut b_star_i_sq = Rational::from_bigint(inner_products[i][i].clone());
            for j in 0..i {
                let prod = &mu[i][j] * &inner_with_b_star[j];
                b_star_i_sq = b_star_i_sq - prod;
            }
            b_star_norms_sq[i] = b_star_i_sq;
        }

        Self { mu, b_star_norms_sq, n }
    }

    /// Get μ_ij (returns 0 for i <= j)
    pub fn get_mu(&self, i: usize, j: usize) -> &Rational {
        if j < i {
            &self.mu[i][j]
        } else {
            panic!("μ_ij only defined for j < i")
        }
    }

    /// Get ||b*_i||^2
    pub fn get_norm_sq(&self, i: usize) -> &Rational {
        &self.b_star_norms_sq[i]
    }

    /// Check if μ_ij needs size reduction (|μ_ij| > 1/2)
    pub fn needs_size_reduction(&self, i: usize, j: usize) -> bool {
        let mu = self.get_mu(i, j);
        // |μ| > 1/2  ⟺  |2*num| > |den|
        let two_num: BigInt = &mu.numerator * 2;
        two_num.abs() > mu.denominator.abs()
    }

    /// Check Lovász condition at position k
    ///
    /// The condition is: δ ||b*_{k-1}||^2 ≤ ||b*_k||^2 + μ_{k,k-1}^2 ||b*_{k-1}||^2
    /// Equivalently: δ ||b*_{k-1}||^2 ≤ ||b*_k + μ_{k,k-1} b*_{k-1}||^2
    ///
    /// Using δ = 3/4 (Lovász's original choice)
    pub fn check_lovasz(&self, k: usize, delta_num: i64, delta_den: i64) -> bool {
        if k == 0 {
            return true;
        }

        // LHS: δ * ||b*_{k-1}||^2
        let lhs_num = BigInt::from(delta_num) * &self.b_star_norms_sq[k - 1].numerator;
        let lhs_den = BigInt::from(delta_den) * &self.b_star_norms_sq[k - 1].denominator;

        // RHS: ||b*_k||^2 + μ_{k,k-1}^2 * ||b*_{k-1}||^2
        // = (||b*_k||^2 * ||b*_{k-1}||^2.den + μ_{k,k-1}^2 * ||b*_{k-1}||^2 * ||b*_k||^2.den)
        //   / (||b*_k||^2.den * ||b*_{k-1}||^2.den)

        let mu_k = &self.mu[k][k - 1];
        let b_star_k = &self.b_star_norms_sq[k];
        let b_star_km1 = &self.b_star_norms_sq[k - 1];

        // μ^2 as exact rational
        let mu_sq = Rational::new(
            &mu_k.numerator * &mu_k.numerator,
            &mu_k.denominator * &mu_k.denominator,
        );

        // RHS = b_star_k + μ^2 * b_star_km1
        let term2 = Rational::new(
            &mu_sq.numerator * &b_star_km1.numerator,
            &mu_sq.denominator * &b_star_km1.denominator,
        );
        let rhs = Rational::new(
            &b_star_k.numerator * &term2.denominator + &term2.numerator * &b_star_k.denominator,
            &b_star_k.denominator * &term2.denominator,
        );

        let rhs_num = rhs.numerator;
        let rhs_den = rhs.denominator;

        // Compare: lhs_num/lhs_den ≤ rhs_num/rhs_den
        // ⟺ lhs_num * rhs_den ≤ rhs_num * lhs_den
        lhs_num * rhs_den <= rhs_num * lhs_den
    }

    /// Update Gram-Schmidt after size reduction b_k = b_k - q * b_j
    pub fn update_size_reduction(&mut self, k: usize, j: usize, q: &BigInt) {
        // μ_kj -= q
        let q_rat = Rational::from_bigint(q.clone());
        self.mu[k][j] = self.mu[k][j].clone() - q_rat.clone();

        // μ_ki -= q * μ_ji for i < j
        for i in 0..j {
            let mu_ji = self.mu[j][i].clone();
            let prod = Rational::new(q * &mu_ji.numerator, mu_ji.denominator);
            self.mu[k][i] = self.mu[k][i].clone() - prod;
        }
    }

    /// Update Gram-Schmidt after swapping basis vectors k and k-1
    ///
    /// This is the key update that maintains GS data after a Lovász swap.
    pub fn update_swap(&mut self, basis: &LatticeBasis, k: usize) {
        // After swap, need to update:
        // - μ values involving k and k-1
        // - ||b*_{k-1}||^2 and ||b*_k||^2

        // Save old values
        let mu_k_km1 = self.mu[k][k - 1].clone();
        let b_star_km1_sq = self.b_star_norms_sq[k - 1].clone();
        let b_star_k_sq = self.b_star_norms_sq[k].clone();

        // New ||b*_{k-1}||^2 = ||b*_k||^2 + μ_{k,k-1}^2 * ||b*_{k-1}||^2
        let mu_sq = Rational::new(
            &mu_k_km1.numerator * &mu_k_km1.numerator,
            &mu_k_km1.denominator * &mu_k_km1.denominator,
        );
        let term = &mu_sq * &b_star_km1_sq;
        let new_b_star_km1_sq = b_star_k_sq.clone() + term;

        // New μ_{k-1,j} for j < k-1 are the old μ_{k,j}
        // New μ_{k,j} = (μ_{k-1,j} * ||b*_{k-1}||^2 + μ_{k,k-1} * μ_{k,j} * ||b*_{k-1}||^2) / new_||b*_{k-1}||^2
        // But actually for j < k-1: new μ_k,j = old μ_{k-1,j}

        // Swap μ rows for indices < k-1
        if k >= 2 {
            for j in 0..k - 1 {
                let tmp = self.mu[k - 1][j].clone();
                self.mu[k - 1][j] = self.mu[k][j].clone();
                self.mu[k][j] = tmp;
            }
        }

        // New μ_{k,k-1} = μ_{k,k-1} * old_||b*_{k-1}||^2 / new_||b*_{k-1}||^2
        let new_mu_k_km1 = Rational::new(
            &mu_k_km1.numerator * &b_star_km1_sq.numerator * &new_b_star_km1_sq.denominator,
            &mu_k_km1.denominator * &b_star_km1_sq.denominator * &new_b_star_km1_sq.numerator,
        );
        self.mu[k][k - 1] = new_mu_k_km1.clone();

        // New ||b*_k||^2 = old_||b*_{k-1}||^2 * old_||b*_k||^2 / new_||b*_{k-1}||^2
        let new_b_star_k_sq = Rational::new(
            &b_star_km1_sq.numerator * &b_star_k_sq.numerator * &new_b_star_km1_sq.denominator,
            &b_star_km1_sq.denominator * &b_star_k_sq.denominator * &new_b_star_km1_sq.numerator,
        );

        self.b_star_norms_sq[k - 1] = new_b_star_km1_sq;
        self.b_star_norms_sq[k] = new_b_star_k_sq;

        // Update μ_{i,k-1} and μ_{i,k} for i > k
        for i in k + 1..self.n {
            let mu_i_km1 = self.mu[i][k - 1].clone();
            let mu_i_k = self.mu[i][k].clone();

            // new μ_{i,k-1} = μ_{i,k} + μ_{k,k-1} * μ_{i,k-1}
            let prod = &mu_k_km1 * &mu_i_km1;
            self.mu[i][k - 1] = mu_i_k.clone() + prod;

            // new μ_{i,k} = μ_{i,k-1} - new_μ_{k,k-1} * new_μ_{i,k-1}
            let new_prod = &new_mu_k_km1 * &self.mu[i][k - 1];
            self.mu[i][k] = mu_i_km1 - new_prod;
        }
    }
}

/// CRT-based Gram-Schmidt computation for GPU acceleration
///
/// Stores Gram-Schmidt data in CRT representation for parallel computation.
#[derive(Debug, Clone)]
pub struct GramSchmidtCRT {
    /// μ numerators in CRT form: mu_num[prime_idx][i*(i-1)/2 + j] for j < i
    pub mu_num: Vec<Vec<u32>>,
    /// μ denominators in CRT form: mu_den[prime_idx][i*(i-1)/2 + j] for j < i
    pub mu_den: Vec<Vec<u32>>,
    /// ||b*||^2 numerators in CRT form
    pub b_star_norms_sq_num: Vec<Vec<u32>>,
    /// ||b*||^2 denominators in CRT form
    pub b_star_norms_sq_den: Vec<Vec<u32>>,
    /// The CRT basis
    pub basis: CRTBasis,
    /// Dimension
    pub n: usize,
}

impl GramSchmidtCRT {
    /// Create CRT representation from exact Gram-Schmidt data
    pub fn from_exact(gs: &GramSchmidt, crt_basis: CRTBasis) -> Self {
        let n = gs.n;
        let num_primes = crt_basis.len();

        // Allocate storage
        let mu_size = n * (n - 1) / 2;
        let mut mu_num = vec![vec![0u32; mu_size]; num_primes];
        let mut mu_den = vec![vec![0u32; mu_size]; num_primes];
        let mut b_star_norms_sq_num = vec![vec![0u32; n]; num_primes];
        let mut b_star_norms_sq_den = vec![vec![0u32; n]; num_primes];

        // Convert μ values (both numerator and denominator)
        for i in 1..n {
            for j in 0..i {
                let idx = i * (i - 1) / 2 + j;
                let mu = gs.get_mu(i, j);

                for (p_idx, &p) in crt_basis.primes.iter().enumerate() {
                    let p_big = BigInt::from(p);
                    let num_mod = ((&mu.numerator % &p_big) + &p_big) % &p_big;
                    let den_mod = ((&mu.denominator % &p_big) + &p_big) % &p_big;
                    mu_num[p_idx][idx] = num_mod.try_into().unwrap_or(0);
                    mu_den[p_idx][idx] = den_mod.try_into().unwrap_or(0);
                }
            }
        }

        // Convert ||b*||^2 values
        for i in 0..n {
            let norm = gs.get_norm_sq(i);

            for (p_idx, &p) in crt_basis.primes.iter().enumerate() {
                let p_big = BigInt::from(p);
                let num_mod = ((&norm.numerator % &p_big) + &p_big) % &p_big;
                let den_mod = ((&norm.denominator % &p_big) + &p_big) % &p_big;
                b_star_norms_sq_num[p_idx][i] = num_mod.try_into().unwrap_or(0);
                b_star_norms_sq_den[p_idx][i] = den_mod.try_into().unwrap_or(0);
            }
        }

        Self {
            mu_num,
            mu_den,
            b_star_norms_sq_num,
            b_star_norms_sq_den,
            basis: crt_basis,
            n,
        }
    }

    /// Convert back to exact Gram-Schmidt data
    pub fn to_exact(&self) -> GramSchmidt {
        let n = self.n;
        let num_primes = self.basis.len();

        // Reconstruct μ values
        let mut mu: Vec<Vec<Rational>> = (0..n)
            .map(|i| vec![Rational::zero(); i])
            .collect();

        for i in 1..n {
            for j in 0..i {
                let idx = i * (i - 1) / 2 + j;

                // Reconstruct numerator
                let num_residues: Vec<u32> = (0..num_primes)
                    .map(|p| self.mu_num[p][idx])
                    .collect();
                let num = CRTReconstruction::reconstruct_signed(&num_residues, &self.basis);

                // Reconstruct denominator
                let den_residues: Vec<u32> = (0..num_primes)
                    .map(|p| self.mu_den[p][idx])
                    .collect();
                let den = CRTReconstruction::reconstruct_signed(&den_residues, &self.basis);

                mu[i][j] = Rational::new(num, den);
            }
        }

        // Reconstruct ||b*||^2 values
        let mut b_star_norms_sq = vec![Rational::zero(); n];
        for i in 0..n {
            let num_residues: Vec<u32> = (0..num_primes)
                .map(|p| self.b_star_norms_sq_num[p][i])
                .collect();
            let den_residues: Vec<u32> = (0..num_primes)
                .map(|p| self.b_star_norms_sq_den[p][i])
                .collect();

            let num = CRTReconstruction::reconstruct_signed(&num_residues, &self.basis);
            let den = CRTReconstruction::reconstruct_signed(&den_residues, &self.basis);

            b_star_norms_sq[i] = Rational::new(num, den);
        }

        GramSchmidt { mu, b_star_norms_sq, n }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gram_schmidt_basic() {
        // Simple 2D basis
        let basis = LatticeBasis::from_rows(&[
            vec![3i64, 1],
            vec![2, 2],
        ]);

        let gs = GramSchmidt::compute(&basis);

        // ||b*_0||^2 = <b_0, b_0> = 9 + 1 = 10
        assert_eq!(gs.b_star_norms_sq[0], Rational::from_int(10i64));

        // μ_10 = <b_1, b_0> / ||b*_0||^2 = (6 + 2) / 10 = 4/5
        assert_eq!(
            gs.mu[1][0],
            Rational::new(BigInt::from(4), BigInt::from(5))
        );

        // ||b*_1||^2 = ||b_1||^2 - μ_10^2 * ||b*_0||^2
        //            = 8 - (16/25) * 10 = 8 - 32/5 = 8/5
        assert_eq!(
            gs.b_star_norms_sq[1],
            Rational::new(BigInt::from(8), BigInt::from(5))
        );
    }

    #[test]
    fn test_gram_schmidt_3d() {
        let basis = LatticeBasis::from_rows(&[
            vec![1i64, 1, 1],
            vec![-1, 0, 2],
            vec![3, 5, 6],
        ]);

        let gs = GramSchmidt::compute(&basis);

        // Verify orthogonality properties
        assert_eq!(gs.n, 3);

        // All norms should be positive
        for i in 0..3 {
            let norm = &gs.b_star_norms_sq[i];
            assert!(
                norm.numerator > BigInt::zero() && norm.denominator > BigInt::zero(),
                "Norm at {} should be positive: {:?}",
                i,
                norm
            );
        }
    }

    #[test]
    fn test_lovasz_condition() {
        let basis = LatticeBasis::from_rows(&[
            vec![1i64, 0],
            vec![0, 1],
        ]);

        let gs = GramSchmidt::compute(&basis);

        // For identity basis, Lovász condition should always hold
        assert!(gs.check_lovasz(1, 3, 4)); // δ = 3/4
    }

    #[test]
    fn test_crt_roundtrip() {
        let basis = LatticeBasis::from_rows(&[
            vec![3i64, 1],
            vec![2, 2],
        ]);

        let gs = GramSchmidt::compute(&basis);
        let crt_basis = CRTBasis::with_primes(5);
        let gs_crt = GramSchmidtCRT::from_exact(&gs, crt_basis);
        let gs_recovered = gs_crt.to_exact();

        // Check values match
        assert_eq!(gs.n, gs_recovered.n);
        assert_eq!(gs.b_star_norms_sq[0], gs_recovered.b_star_norms_sq[0]);
        assert_eq!(gs.b_star_norms_sq[1], gs_recovered.b_star_norms_sq[1]);
        assert_eq!(gs.mu[1][0], gs_recovered.mu[1][0]);
    }
}
