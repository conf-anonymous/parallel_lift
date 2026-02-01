//! Prime generation and CRT basis management
//!
//! Provides utilities for generating coprime moduli suitable for CRT computation.

use num_bigint::BigInt;
use num_traits::{One, Zero};

/// Prime generator for CRT moduli
pub struct PrimeGenerator;

impl PrimeGenerator {
    /// Generate `count` 31-bit primes suitable for modular arithmetic
    ///
    /// These primes are chosen to be:
    /// - Large enough to avoid overflow in 64-bit intermediate products
    /// - Small enough for efficient modular operations
    pub fn generate_31bit_primes(count: usize) -> Vec<u32> {
        let mut primes = Vec::with_capacity(count);
        let mut candidate = (1u32 << 31) - 1; // Start near 2^31

        while primes.len() < count {
            if Self::is_prime_u32(candidate) {
                primes.push(candidate);
            }
            candidate -= 2; // Skip even numbers
        }

        primes
    }

    /// Simple primality test for 32-bit integers
    fn is_prime_u32(n: u32) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }

        let sqrt_n = (n as f64).sqrt() as u32 + 1;
        for i in (3..=sqrt_n).step_by(2) {
            if n % i == 0 {
                return false;
            }
        }
        true
    }

    /// Estimate number of primes needed to represent a value with given bit width
    pub fn estimate_primes_needed(bit_width: usize) -> usize {
        // Each 31-bit prime contributes ~31 bits to the product
        // Add margin for safety
        (bit_width + 30) / 31 + 1
    }
}

/// CRT basis: a set of coprime moduli with precomputed values
#[derive(Debug, Clone)]
pub struct CRTBasis {
    /// The prime moduli
    pub primes: Vec<u32>,
    /// Product of all primes (M)
    pub product: BigInt,
    /// M/2 for signed reconstruction
    pub half_product: BigInt,
    /// Partial products for Garner's algorithm (BigInt for final reconstruction)
    pub partial_products: Vec<BigInt>,
    /// Precomputed inverses for Garner's algorithm (BigInt)
    pub garner_inverses: Vec<BigInt>,
    /// Precomputed inverses as u32 for fast modular arithmetic
    /// garner_inverses_u32[i] = partial_products[i]^(-1) mod primes[i]
    pub garner_inverses_u32: Vec<u32>,
    /// partial_products[i] mod primes[i] as u64 (for fast modular reduction)
    pub partial_products_mod: Vec<u64>,
}

impl CRTBasis {
    /// Create a new CRT basis with the given primes
    pub fn new(primes: Vec<u32>) -> Self {
        let k = primes.len();

        // Compute partial products: partial_products[i] = prod(primes[0..i])
        let mut partial_products = vec![BigInt::one(); k];
        for i in 1..k {
            partial_products[i] = &partial_products[i - 1] * BigInt::from(primes[i - 1]);
        }

        // Total product
        let product = &partial_products[k - 1] * BigInt::from(primes[k - 1]);
        let half_product = &product / 2;

        // Precompute Garner inverses: (partial_products[i])^(-1) mod primes[i]
        let mut garner_inverses = vec![BigInt::zero(); k];
        let mut garner_inverses_u32 = vec![0u32; k];
        let mut partial_products_mod = vec![0u64; k];

        for i in 1..k {
            let mi = BigInt::from(primes[i]);
            let pp_mod_mi = &partial_products[i] % &mi;
            garner_inverses[i] = Self::mod_inverse(&pp_mod_mi, &mi)
                .expect("Primes should be coprime");

            // Also store as u32 for fast modular arithmetic
            let pp_mod_u64 = Self::bigint_to_u64(&pp_mod_mi);
            partial_products_mod[i] = pp_mod_u64;
            garner_inverses_u32[i] = Self::mod_inverse_u32(pp_mod_u64 as u32, primes[i]);
        }

        Self {
            primes,
            product,
            half_product,
            partial_products,
            garner_inverses,
            garner_inverses_u32,
            partial_products_mod,
        }
    }

    /// Convert BigInt to u64 (assumes it fits)
    fn bigint_to_u64(n: &BigInt) -> u64 {
        let (_, digits) = n.to_u64_digits();
        if digits.is_empty() {
            0
        } else {
            digits[0]
        }
    }

    /// Modular inverse for u32 using Fermat's Little Theorem
    /// a^(-1) = a^(p-2) mod p for prime p
    fn mod_inverse_u32(a: u32, p: u32) -> u32 {
        if a == 0 {
            return 0;
        }
        let mut result = 1u64;
        let mut base = a as u64;
        let mut exp = p - 2;
        let p64 = p as u64;

        while exp > 0 {
            if exp & 1 == 1 {
                result = (result * base) % p64;
            }
            base = (base * base) % p64;
            exp >>= 1;
        }
        result as u32
    }

    /// Create a basis with `count` 31-bit primes
    pub fn with_primes(count: usize) -> Self {
        let primes = PrimeGenerator::generate_31bit_primes(count);
        Self::new(primes)
    }

    /// Modular inverse using extended Euclidean algorithm
    fn mod_inverse(a: &BigInt, m: &BigInt) -> Option<BigInt> {
        let (g, x, _) = Self::extended_gcd(a, m);
        if g != BigInt::one() {
            return None;
        }
        Some(((x % m) + m) % m)
    }

    /// Extended Euclidean algorithm
    fn extended_gcd(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
        if b.is_zero() {
            (a.clone(), BigInt::one(), BigInt::zero())
        } else {
            let (g, x, y) = Self::extended_gcd(b, &(a % b));
            (g, y.clone(), x - (a / b) * y)
        }
    }

    /// Number of primes in this basis
    pub fn len(&self) -> usize {
        self.primes.len()
    }

    /// Check if the basis is empty
    pub fn is_empty(&self) -> bool {
        self.primes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prime_generation() {
        let primes = PrimeGenerator::generate_31bit_primes(10);
        assert_eq!(primes.len(), 10);
        for p in &primes {
            assert!(PrimeGenerator::is_prime_u32(*p));
            assert!(*p >= (1 << 30)); // Should be at least 30 bits
        }
    }

    #[test]
    fn test_crt_basis_creation() {
        let basis = CRTBasis::with_primes(3);
        assert_eq!(basis.len(), 3);
        assert!(!basis.product.is_zero());
    }
}
