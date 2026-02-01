//! Chinese Remainder Theorem reconstruction
//!
//! Implements Garner's algorithm for efficient CRT reconstruction.

use num_bigint::BigInt;
use num_traits::Zero;
use crate::primes::CRTBasis;

/// CRT reconstruction utilities
pub struct CRTReconstruction;

impl CRTReconstruction {
    /// Reconstruct a BigInt from its residues using Garner's algorithm
    ///
    /// # Arguments
    /// * `residues` - Residues modulo each prime in the basis
    /// * `basis` - The CRT basis with precomputed values
    ///
    /// # Returns
    /// The unique integer x in [0, M) where M = product of all primes
    pub fn reconstruct(residues: &[u32], basis: &CRTBasis) -> BigInt {
        debug_assert_eq!(residues.len(), basis.len());

        if residues.is_empty() {
            return BigInt::zero();
        }

        if residues.len() == 1 {
            return BigInt::from(residues[0]);
        }

        // Garner's algorithm (incremental CRT)
        let mut result = BigInt::from(residues[0]);

        for i in 1..residues.len() {
            let mi = BigInt::from(basis.primes[i]);
            let ri = BigInt::from(residues[i]);

            // t = (r_i - result) * inv mod m_i
            let result_mod_mi = ((&result % &mi) + &mi) % &mi;
            let diff = ((&ri - &result_mod_mi) % &mi + &mi) % &mi;
            let t = (&diff * &basis.garner_inverses[i]) % &mi;

            // result += partial_products[i] * t
            result = &result + &basis.partial_products[i] * t;
        }

        result
    }

    /// Reconstruct a signed BigInt (in symmetric range [-M/2, M/2))
    pub fn reconstruct_signed(residues: &[u32], basis: &CRTBasis) -> BigInt {
        let unsigned = Self::reconstruct(residues, basis);

        if unsigned > basis.half_product {
            unsigned - &basis.product
        } else {
            unsigned
        }
    }

    /// Batch reconstruct multiple values using the same basis
    ///
    /// More efficient than calling reconstruct multiple times due to
    /// shared precomputation.
    pub fn batch_reconstruct(all_residues: &[Vec<u32>], basis: &CRTBasis) -> Vec<BigInt> {
        all_residues
            .iter()
            .map(|r| Self::reconstruct_signed(r, basis))
            .collect()
    }

    /// Optimized batch reconstruction with direct memory access
    ///
    /// For cases where residues are stored in a flat array:
    /// residues[prime_idx * num_values + value_idx]
    pub fn batch_reconstruct_flat(
        residues: &[u32],
        basis: &CRTBasis,
        num_values: usize,
    ) -> Vec<BigInt> {
        // Use the optimized version with u64 arithmetic
        Self::batch_reconstruct_flat_optimized(residues, basis, num_values)
    }

    /// Highly optimized batch reconstruction using u64 arithmetic in inner loop
    ///
    /// Key optimizations:
    /// 1. Uses precomputed u32 inverses instead of BigInt
    /// 2. Uses u64 modular arithmetic for the Garner step
    /// 3. Only converts to BigInt for the final accumulation
    pub fn batch_reconstruct_flat_optimized(
        residues: &[u32],
        basis: &CRTBasis,
        num_values: usize,
    ) -> Vec<BigInt> {
        let num_primes = basis.len();
        debug_assert_eq!(residues.len(), num_primes * num_values);

        let mut results = vec![BigInt::zero(); num_values];

        // Initialize with first prime's residues
        for v in 0..num_values {
            results[v] = BigInt::from(residues[v]);
        }

        // Garner iterations with optimized inner loop
        for i in 1..num_primes {
            let mi = basis.primes[i] as u64;
            let inv = basis.garner_inverses_u32[i] as u64;
            let prod = &basis.partial_products[i];
            let base_offset = i * num_values;

            for v in 0..num_values {
                let ri = residues[base_offset + v] as u64;

                // Compute result mod mi using BigInt (necessary for large results)
                // But use u64 for the modular arithmetic part
                let result_mod_mi = Self::bigint_mod_u64(&results[v], mi);

                // t = (ri - result_mod_mi) * inv mod mi
                // Use u64 arithmetic - much faster than BigInt
                let diff = if ri >= result_mod_mi {
                    ri - result_mod_mi
                } else {
                    mi - result_mod_mi + ri
                };
                let t = (diff * inv) % mi;

                // Accumulate: result += partial_products[i] * t
                if t != 0 {
                    results[v] = &results[v] + prod * t;
                }
            }
        }

        // Convert to signed representation
        for v in 0..num_values {
            if results[v] > basis.half_product {
                results[v] = &results[v] - &basis.product;
            }
        }

        results
    }

    /// Compute n mod m for BigInt n and u64 m
    #[inline]
    fn bigint_mod_u64(n: &BigInt, m: u64) -> u64 {
        let m_big = BigInt::from(m);
        let r = n % &m_big;
        // Handle negative case
        let r = if r < BigInt::zero() { r + m_big } else { r };
        // Convert to u64
        let (_, digits) = r.to_u64_digits();
        if digits.is_empty() {
            0
        } else {
            digits[0]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crt_reconstruction() {
        let basis = CRTBasis::with_primes(3);

        // Test value: 12345
        let x = 12345i64;
        let residues: Vec<u32> = basis
            .primes
            .iter()
            .map(|&p| (x.rem_euclid(p as i64)) as u32)
            .collect();

        let reconstructed = CRTReconstruction::reconstruct(&residues, &basis);
        assert_eq!(reconstructed, BigInt::from(x));
    }

    #[test]
    fn test_signed_reconstruction() {
        let basis = CRTBasis::with_primes(3);

        // Test negative value: -12345
        let x = -12345i64;
        let residues: Vec<u32> = basis
            .primes
            .iter()
            .map(|&p| (x.rem_euclid(p as i64)) as u32)
            .collect();

        let reconstructed = CRTReconstruction::reconstruct_signed(&residues, &basis);
        assert_eq!(reconstructed, BigInt::from(x));
    }
}
