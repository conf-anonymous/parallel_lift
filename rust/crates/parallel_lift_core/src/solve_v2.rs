//! V2 CRT-based solver with 62-bit primes
//!
//! Key improvements over v1:
//! 1. Uses 62-bit primes instead of 31-bit (halves prime count)
//! 2. Dedicated 64-bit CUDA kernels with 128-bit intermediate arithmetic

use crate::{Rational, Timings};
use num_bigint::BigInt;
use num_traits::{One, Zero};
use std::time::Instant;

/// 62-bit prime generator for V2 solver
///
/// Using larger primes halves the number of primes needed for CRT reconstruction.
pub struct PrimeGenerator62;

impl PrimeGenerator62 {
    /// Generate `count` 62-bit primes suitable for modular arithmetic
    ///
    /// These primes are near 2^62, leaving room for 2-bit overflow in products.
    pub fn generate_62bit_primes(count: usize) -> Vec<u64> {
        let mut primes = Vec::with_capacity(count);
        // Start just below 2^62 to allow 2-bit headroom in 128-bit products
        let mut candidate = (1u64 << 62) - 57; // 2^62 - 57 is prime

        while primes.len() < count {
            if Self::is_prime_u64(candidate) {
                primes.push(candidate);
            }
            candidate = candidate.saturating_sub(2); // Skip even numbers
            if candidate < (1u64 << 61) {
                // Safety: don't go below 2^61
                break;
            }
        }

        primes
    }

    /// Miller-Rabin primality test for 64-bit integers
    pub fn is_prime_u64(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 || n == 3 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }

        // Write n-1 as 2^r * d
        let mut d = n - 1;
        let mut r = 0u32;
        while d % 2 == 0 {
            d /= 2;
            r += 1;
        }

        // Deterministic witnesses for 64-bit numbers
        let witnesses: [u64; 7] = [2, 325, 9375, 28178, 450775, 9780504, 1795265022];

        for &a in &witnesses {
            if a % n == 0 {
                continue;
            }
            if !Self::miller_rabin_witness(a, d, r, n) {
                return false;
            }
        }
        true
    }

    fn miller_rabin_witness(a: u64, d: u64, r: u32, n: u64) -> bool {
        let mut x = Self::mod_pow_u128(a, d, n);

        if x == 1 || x == n - 1 {
            return true;
        }

        for _ in 0..r - 1 {
            x = Self::mod_mul_u128(x, x, n);
            if x == n - 1 {
                return true;
            }
        }
        false
    }

    /// Modular exponentiation using 128-bit intermediate
    pub fn mod_pow_u128(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
        let mut result = 1u64;
        base %= modulus;

        while exp > 0 {
            if exp % 2 == 1 {
                result = Self::mod_mul_u128(result, base, modulus);
            }
            exp /= 2;
            base = Self::mod_mul_u128(base, base, modulus);
        }
        result
    }

    /// Modular multiplication using 128-bit intermediate
    pub fn mod_mul_u128(a: u64, b: u64, modulus: u64) -> u64 {
        ((a as u128 * b as u128) % modulus as u128) as u64
    }

    /// Estimate number of 62-bit primes needed for a given bit width
    pub fn estimate_primes_needed(bit_width: usize) -> usize {
        // Each 62-bit prime contributes ~62 bits to the product
        (bit_width + 61) / 62 + 1
    }

    /// Estimate primes needed for an n×n matrix solve
    pub fn primes_for_matrix_size(n: usize, entry_bits: usize) -> usize {
        // Hadamard bound: |det(A)| <= n^(n/2) * max_entry^n
        // Solution bound is roughly similar
        // log2(bound) ≈ n * log2(n) + n * entry_bits
        let log2_bound = (n as f64 * (n as f64).log2() + n as f64 * entry_bits as f64) as usize;
        Self::estimate_primes_needed(log2_bound)
    }
}

/// CRT basis with 62-bit primes
#[derive(Debug, Clone)]
pub struct CRTBasis62 {
    /// The 62-bit prime moduli
    pub primes: Vec<u64>,
    /// Product of all primes (M)
    pub product: BigInt,
    /// M/2 for signed reconstruction
    pub half_product: BigInt,
    /// Partial products for Garner's algorithm
    pub partial_products: Vec<BigInt>,
    /// Precomputed inverses: partial_products[i]^(-1) mod primes[i]
    pub garner_inverses: Vec<u64>,
}

impl CRTBasis62 {
    /// Create a new CRT basis with the given 62-bit primes
    pub fn new(primes: Vec<u64>) -> Self {
        let k = primes.len();

        // Compute partial products: partial_products[i] = prod(primes[0..i])
        let mut partial_products = vec![BigInt::one(); k];
        for i in 1..k {
            partial_products[i] = &partial_products[i - 1] * BigInt::from(primes[i - 1]);
        }

        // Total product
        let product = if k > 0 {
            &partial_products[k - 1] * BigInt::from(primes[k - 1])
        } else {
            BigInt::one()
        };
        let half_product = &product / 2;

        // Precompute Garner inverses
        let mut garner_inverses = vec![0u64; k];
        for i in 1..k {
            let pp_mod = Self::bigint_mod_u64(&partial_products[i], primes[i]);
            garner_inverses[i] = PrimeGenerator62::mod_pow_u128(pp_mod, primes[i] - 2, primes[i]);
        }

        Self {
            primes,
            product,
            half_product,
            partial_products,
            garner_inverses,
        }
    }

    /// Reduce BigInt modulo u64 prime
    fn bigint_mod_u64(n: &BigInt, p: u64) -> u64 {
        let (_, digits) = n.to_u64_digits();
        if digits.is_empty() {
            return 0;
        }

        // Compute n mod p using Horner's method
        // n = d[k-1] * 2^(64*(k-1)) + ... + d[1] * 2^64 + d[0]
        let mut result = 0u64;
        let base_mod = {
            // 2^64 mod p
            let two32 = (1u64 << 32) % p;
            PrimeGenerator62::mod_mul_u128(two32, two32, p)
        };

        // Process from most significant digit
        for &digit in digits.iter().rev() {
            result = PrimeGenerator62::mod_mul_u128(result, base_mod, p);
            result = (result + (digit % p)) % p;
        }

        result
    }

    /// Create a basis with `count` 62-bit primes
    pub fn with_primes(count: usize) -> Self {
        let primes = PrimeGenerator62::generate_62bit_primes(count);
        Self::new(primes)
    }

    /// Create a basis for solving n×n matrices with given entry bit width
    pub fn for_matrix(n: usize, entry_bits: usize) -> Self {
        let count = PrimeGenerator62::primes_for_matrix_size(n, entry_bits);
        Self::with_primes(count)
    }

    /// Number of primes in this basis
    pub fn len(&self) -> usize {
        self.primes.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.primes.is_empty()
    }

    /// Reduce a BigInt to residues for all primes
    pub fn reduce(&self, value: &BigInt) -> Vec<u64> {
        self.primes
            .iter()
            .map(|&p| Self::bigint_mod_u64(value, p))
            .collect()
    }
}

/// CRT reconstruction for 62-bit residues
pub struct CRTReconstruction62;

impl CRTReconstruction62 {
    /// Reconstruct BigInt from 62-bit residues using Garner's algorithm
    pub fn reconstruct(residues: &[u64], basis: &CRTBasis62) -> BigInt {
        let k = residues.len();
        if k == 0 {
            return BigInt::zero();
        }

        // Garner's algorithm
        let mut result = BigInt::from(residues[0]);

        for i in 1..k {
            let p_i = basis.primes[i];
            let res_mod = CRTBasis62::bigint_mod_u64(&result, p_i);

            let diff = if residues[i] >= res_mod {
                residues[i] - res_mod
            } else {
                p_i - (res_mod - residues[i])
            };

            let coeff = PrimeGenerator62::mod_mul_u128(diff, basis.garner_inverses[i], p_i);
            result += &basis.partial_products[i] * BigInt::from(coeff);
        }

        result
    }

    /// Reconstruct as signed integer (centered around 0)
    pub fn reconstruct_signed(residues: &[u64], basis: &CRTBasis62) -> BigInt {
        let unsigned = Self::reconstruct(residues, basis);
        if unsigned > basis.half_product {
            unsigned - &basis.product
        } else {
            unsigned
        }
    }
}

/// V2 Timings with breakdown
#[derive(Debug, Clone, Default)]
pub struct V2Timings {
    pub total_ms: f64,
    pub prepare_ms: f64,
    pub htod_ms: f64,
    pub compute_ms: f64,
    pub dtoh_ms: f64,
    pub crt_ms: f64,
    pub num_primes: usize,
    pub htod_bytes: usize,
    pub dtoh_bytes: usize,
}

impl V2Timings {
    /// Convert to standard Timings struct
    pub fn to_timings(&self) -> Timings {
        Timings {
            total_time: self.total_ms / 1000.0,
            residue_time: self.prepare_ms / 1000.0,
            solve_time: self.compute_ms / 1000.0,
            det_time: 0.0,
            crt_time: self.crt_ms / 1000.0,
            verify_time: 0.0,
            num_primes: self.num_primes,
        }
    }
}

/// V2 Solver result
pub struct V2SolveResult {
    /// Solution matrix (k columns, each with n elements)
    pub solutions: Vec<Vec<Rational>>,
    /// Detailed timing breakdown
    pub timings: V2Timings,
}

/// Convert BigInt matrix to u64 flat array (taking absolute value, for reduction)
pub fn bigint_matrix_to_u64(matrix: &[BigInt]) -> Vec<u64> {
    matrix
        .iter()
        .map(|v| {
            let (_, digits) = v.to_u64_digits();
            digits.first().copied().unwrap_or(0)
        })
        .collect()
}

/// Convert BigInt vector to u64 vector
pub fn bigint_vec_to_u64(vec: &[BigInt]) -> Vec<u64> {
    vec.iter()
        .map(|v| {
            let (_, digits) = v.to_u64_digits();
            digits.first().copied().unwrap_or(0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_62bit_prime_generation() {
        let primes = PrimeGenerator62::generate_62bit_primes(10);
        assert_eq!(primes.len(), 10);
        for &p in &primes {
            assert!(PrimeGenerator62::is_prime_u64(p));
            assert!(p >= (1u64 << 61)); // Should be at least 61 bits
        }
    }

    #[test]
    fn test_crt_basis_62() {
        let basis = CRTBasis62::with_primes(5);
        assert_eq!(basis.len(), 5);
        assert!(!basis.product.is_zero());
    }

    #[test]
    fn test_crt_reconstruction_62_small() {
        let primes = PrimeGenerator62::generate_62bit_primes(3);
        let basis = CRTBasis62::new(primes.clone());

        // Test value: 42
        let value = BigInt::from(42);
        let residues: Vec<u64> = primes.iter().map(|&p| 42u64 % p).collect();

        let reconstructed = CRTReconstruction62::reconstruct(&residues, &basis);
        assert_eq!(reconstructed, value);
    }

    #[test]
    fn test_crt_reconstruction_62_large() {
        let primes = PrimeGenerator62::generate_62bit_primes(5);
        let basis = CRTBasis62::new(primes.clone());

        // Test with a larger value
        let value = BigInt::from(1234567890123456789u64);
        let residues = basis.reduce(&value);

        let reconstructed = CRTReconstruction62::reconstruct(&residues, &basis);
        assert_eq!(reconstructed, value);
    }

    #[test]
    fn test_crt_reconstruction_62_signed() {
        let primes = PrimeGenerator62::generate_62bit_primes(3);
        let basis = CRTBasis62::new(primes.clone());

        // Test with negative value
        let value = BigInt::from(-42);
        // For negative value, we compute residues as (p + (value mod p)) mod p
        let residues: Vec<u64> = primes
            .iter()
            .map(|&p| {
                let pbi = BigInt::from(p);
                let r = ((&value % &pbi) + &pbi) % &pbi;
                r.to_u64_digits().1.first().copied().unwrap_or(0)
            })
            .collect();

        let reconstructed = CRTReconstruction62::reconstruct_signed(&residues, &basis);
        assert_eq!(reconstructed, value);
    }

    #[test]
    fn test_primes_for_matrix() {
        // For n=256 with 10-bit entries, we should need fewer primes than 31-bit version
        let count_62 = PrimeGenerator62::primes_for_matrix_size(256, 10);
        let count_31 = (256 * 256 / 10 + 256 * 10 + 30) / 31 + 1; // Rough 31-bit estimate

        // 62-bit should need roughly half as many primes
        assert!(count_62 < count_31);
        println!("n=256: 62-bit needs {} primes, 31-bit estimate ~{}", count_62, count_31);
    }
}
