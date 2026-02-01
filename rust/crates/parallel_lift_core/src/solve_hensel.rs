//! Hensel (Dixon) p-adic lifting for exact linear system solving
//!
//! This implements Dixon's algorithm which uses p-adic lifting to solve
//! Ax = b over the integers. The key advantage over CRT is that we:
//! 1. Factor A mod p ONCE (compute A⁻¹ mod p)
//! 2. Iteratively lift the solution: O(log(bound)) iterations
//! 3. Each iteration is O(n²) matrix-vector multiply, not O(n³) factorization
//!
//! Complexity comparison for n×n system with k RHS:
//! - CRT: O(num_primes × n³) for factorization
//! - Hensel: O(n³) for inverse + O(iterations × n²) for lifting
//!
//! For n=256 with 546 primes, this is a massive improvement.

use num_bigint::BigInt;
use num_traits::{One, Zero, Signed};

/// Configuration for Hensel lifting
#[derive(Debug, Clone)]
pub struct HenselConfig {
    /// The prime to use for lifting (should be > 2n for stability)
    pub prime: u32,
    /// Maximum number of lifting iterations
    pub max_iterations: usize,
    /// Target bit precision (determines when to stop lifting)
    pub target_bits: usize,
}

impl Default for HenselConfig {
    fn default() -> Self {
        Self {
            // Use a large prime that's safe for 32-bit arithmetic
            prime: 2147483629, // Largest 31-bit prime
            max_iterations: 100,
            target_bits: 4096,
        }
    }
}

impl HenselConfig {
    /// Create config for a specific matrix size and entry bound
    pub fn for_matrix(n: usize, entry_bits: usize) -> Self {
        // Hadamard bound: |det(A)| <= n^(n/2) * max_entry^n
        // Solution bound is similar
        let solution_bits = (n as f64 * (n as f64).log2() / 2.0
            + n as f64 * entry_bits as f64
            + 64.0) as usize; // Add safety margin

        // Each iteration doubles precision, so iterations = log2(target_bits / 31)
        let iterations = (solution_bits as f64 / 31.0).log2().ceil() as usize + 2;

        Self {
            prime: 2147483629,
            max_iterations: iterations.max(10),
            target_bits: solution_bits,
        }
    }

    /// Estimate number of iterations needed
    pub fn estimate_iterations(&self) -> usize {
        // Each iteration doubles precision
        // Start with ~31 bits (one prime), need target_bits
        ((self.target_bits as f64) / 31.0).log2().ceil() as usize + 1
    }
}

/// Result of Hensel lifting
#[derive(Debug, Clone)]
pub struct HenselResult {
    /// Solutions as BigInt (one vector per RHS column)
    pub solutions: Vec<Vec<BigInt>>,
    /// Number of lifting iterations performed
    pub iterations: usize,
    /// Final precision achieved (bits)
    pub precision_bits: usize,
}

/// Timing breakdown for Hensel lifting
#[derive(Debug, Clone, Default)]
pub struct HenselTimings {
    /// Time to compute A⁻¹ mod p
    pub inverse_ms: f64,
    /// Time for all lifting iterations
    pub lifting_ms: f64,
    /// Time for rational reconstruction
    pub reconstruction_ms: f64,
    /// Total time
    pub total_ms: f64,
    /// Number of iterations
    pub iterations: usize,
}

/// State for p-adic lifting on CPU
///
/// Stores the solution in p-adic form: x = Σ x_i × p^i
/// where each x_i is a vector of n residues mod p
#[derive(Debug, Clone)]
pub struct HenselState {
    /// The prime used for lifting
    pub prime: u32,
    /// A⁻¹ mod p (n×n matrix, row-major)
    pub a_inv: Vec<u32>,
    /// Matrix dimension
    pub n: usize,
    /// Number of RHS vectors
    pub k: usize,
    /// P-adic digits of solution: digits[iter][rhs_idx][row]
    /// Each digit is the coefficient of p^iter in the solution
    pub digits: Vec<Vec<Vec<u32>>>,
    /// Current precision (number of digits computed)
    pub precision: usize,
}

impl HenselState {
    /// Create a new Hensel state (CPU version for reference)
    pub fn new(a_inv: Vec<u32>, n: usize, k: usize, prime: u32) -> Self {
        Self {
            prime,
            a_inv,
            n,
            k,
            digits: Vec::new(),
            precision: 0,
        }
    }

    /// Compute matrix-vector product: y = A * x mod p (where A is original matrix)
    fn matvec_mod(&self, a: &[u32], x: &[u32]) -> Vec<u32> {
        let p = self.prime as u64;
        let mut y = vec![0u32; self.n];

        for i in 0..self.n {
            let mut sum = 0u64;
            for j in 0..self.n {
                let aij = a[i * self.n + j] as u64;
                let xj = x[j] as u64;
                sum = (sum + aij * xj) % p;
            }
            y[i] = sum as u32;
        }
        y
    }

    /// Compute matrix-vector product with A⁻¹: y = A⁻¹ * x mod p
    fn inv_matvec_mod(&self, x: &[u32]) -> Vec<u32> {
        let p = self.prime as u64;
        let mut y = vec![0u32; self.n];

        for i in 0..self.n {
            let mut sum = 0u64;
            for j in 0..self.n {
                let aij = self.a_inv[i * self.n + j] as u64;
                let xj = x[j] as u64;
                sum = (sum + aij * xj) % p;
            }
            y[i] = sum as u32;
        }
        y
    }

    /// Perform one lifting iteration (CPU reference implementation)
    ///
    /// Given current approximation x_approx, compute next digit:
    /// 1. residual = (b - A * x_approx) / p^precision mod p
    /// 2. digit = A⁻¹ * residual mod p
    /// 3. x_new = x_approx + p^precision * digit
    pub fn lift_iteration(
        &mut self,
        a: &[u32],
        b_cols: &[Vec<u32>],
        x_approx: &mut [Vec<BigInt>],
    ) {
        let p = BigInt::from(self.prime);
        let p_power = p.pow(self.precision as u32);

        let mut new_digits = Vec::with_capacity(self.k);

        for col_idx in 0..self.k {
            // Compute A * x_approx for this column
            let mut ax = vec![BigInt::zero(); self.n];
            for i in 0..self.n {
                for j in 0..self.n {
                    let aij = BigInt::from(a[i * self.n + j]);
                    ax[i] += &aij * &x_approx[col_idx][j];
                }
            }

            // Compute residual = (b - Ax) / p^precision
            let mut residual = vec![0u32; self.n];
            for i in 0..self.n {
                let bi = BigInt::from(b_cols[col_idx][i]);
                let diff = &bi - &ax[i];

                // Exact division by p^precision
                let quotient = &diff / &p_power;

                // Take mod p (handle negative values)
                let q_mod = ((&quotient % &p) + &p) % &p;
                residual[i] = q_mod.to_u32_digits().1.first().copied().unwrap_or(0);
            }

            // Compute digit = A⁻¹ * residual mod p
            let digit = self.inv_matvec_mod(&residual);

            // Update x_approx: x = x + p^precision * digit
            for i in 0..self.n {
                let delta = &p_power * BigInt::from(digit[i]);
                x_approx[col_idx][i] += delta;
            }

            new_digits.push(digit);
        }

        self.digits.push(new_digits);
        self.precision += 1;
    }

    /// Convert p-adic representation to signed integers
    ///
    /// Given x mod p^k, if x > p^k/2, return x - p^k (negative)
    pub fn to_signed(&self, x: &BigInt) -> BigInt {
        let p = BigInt::from(self.prime);
        let p_power = p.pow(self.precision as u32);
        let half = &p_power / 2;

        if x > &half {
            x - &p_power
        } else {
            x.clone()
        }
    }
}

/// CPU reference implementation of Hensel lifting
///
/// This is used for testing and as a fallback when GPU is not available.
pub fn hensel_solve_cpu(
    a: &[i64],
    b_cols: &[Vec<i64>],
    n: usize,
    k: usize,
    config: &HenselConfig,
) -> Option<HenselResult> {
    let p = config.prime;
    let p64 = p as u64;

    // Convert A to u32 mod p
    let a_mod: Vec<u32> = a.iter().map(|&x| {
        let x_mod = ((x % p as i64) + p as i64) as u64 % p64;
        x_mod as u32
    }).collect();

    // Convert b to u32 mod p
    let b_mod: Vec<Vec<u32>> = b_cols.iter().map(|col| {
        col.iter().map(|&x| {
            let x_mod = ((x % p as i64) + p as i64) as u64 % p64;
            x_mod as u32
        }).collect()
    }).collect();

    // Compute A⁻¹ mod p using Gauss-Jordan elimination
    let a_inv = matrix_inverse_mod_cpu(&a_mod, n, p)?;

    // Initialize solution approximations to zero
    let mut x_approx: Vec<Vec<BigInt>> = (0..k)
        .map(|_| vec![BigInt::zero(); n])
        .collect();

    // Create lifting state
    let mut state = HenselState::new(a_inv.clone(), n, k, p);

    // Initial solution: x_0 = A⁻¹ * b mod p
    let mut initial_digits = Vec::with_capacity(k);
    for col_idx in 0..k {
        let x0 = state.inv_matvec_mod(&b_mod[col_idx]);
        for i in 0..n {
            x_approx[col_idx][i] = BigInt::from(x0[i]);
        }
        initial_digits.push(x0);
    }
    state.digits.push(initial_digits);
    state.precision = 1;

    // Lifting iterations
    let iterations_needed = config.estimate_iterations();
    for _ in 0..iterations_needed.min(config.max_iterations) {
        state.lift_iteration(&a_mod, &b_mod, &mut x_approx);
    }

    // Convert to signed representation
    let solutions: Vec<Vec<BigInt>> = x_approx.iter().map(|col| {
        col.iter().map(|x| state.to_signed(x)).collect()
    }).collect();

    let precision_bits = (state.precision as f64 * (p as f64).log2()) as usize;

    Some(HenselResult {
        solutions,
        iterations: state.precision,
        precision_bits,
    })
}

/// Compute matrix inverse mod p using Gauss-Jordan elimination (CPU)
fn matrix_inverse_mod_cpu(a: &[u32], n: usize, p: u32) -> Option<Vec<u32>> {
    let p64 = p as u64;

    // Create augmented matrix [A | I]
    let mut aug = vec![0u32; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = a[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1; // Identity on the right
    }

    // Gauss-Jordan elimination
    for col in 0..n {
        // Find pivot
        let mut pivot_row = None;
        for row in col..n {
            if aug[row * 2 * n + col] != 0 {
                pivot_row = Some(row);
                break;
            }
        }

        let pivot_row = pivot_row?; // Matrix is singular

        // Swap rows if needed
        if pivot_row != col {
            for j in 0..2 * n {
                aug.swap(col * 2 * n + j, pivot_row * 2 * n + j);
            }
        }

        // Scale pivot row
        let pivot = aug[col * 2 * n + col] as u64;
        let pivot_inv = mod_pow(pivot, p64 - 2, p64);
        for j in 0..2 * n {
            aug[col * 2 * n + j] = ((aug[col * 2 * n + j] as u64 * pivot_inv) % p64) as u32;
        }

        // Eliminate column
        for row in 0..n {
            if row != col && aug[row * 2 * n + col] != 0 {
                let factor = aug[row * 2 * n + col] as u64;
                for j in 0..2 * n {
                    let sub = (factor * aug[col * 2 * n + j] as u64) % p64;
                    aug[row * 2 * n + j] = ((aug[row * 2 * n + j] as u64 + p64 - sub) % p64) as u32;
                }
            }
        }
    }

    // Extract inverse from right half
    let mut inv = vec![0u32; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }

    Some(inv)
}

/// Modular exponentiation
fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % modulus;
        }
        exp /= 2;
        base = (base * base) % modulus;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_inverse_cpu() {
        // 2x2 matrix: [[2, 1], [1, 3]]
        // det = 5, inverse mod 101: [[61, 67], [67, 41]]
        let a = vec![2u32, 1, 1, 3];
        let p = 101u32;

        let inv = matrix_inverse_mod_cpu(&a, 2, p).unwrap();

        // Verify A * A⁻¹ = I mod p
        let mut product = vec![0u32; 4];
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0u64;
                for k in 0..2 {
                    sum += a[i * 2 + k] as u64 * inv[k * 2 + j] as u64;
                }
                product[i * 2 + j] = (sum % p as u64) as u32;
            }
        }

        assert_eq!(product[0], 1); // [0,0] = 1
        assert_eq!(product[1], 0); // [0,1] = 0
        assert_eq!(product[2], 0); // [1,0] = 0
        assert_eq!(product[3], 1); // [1,1] = 1
    }

    #[test]
    fn test_hensel_simple() {
        // Simple 2x2 system: 2x + y = 5, x + 3y = 10
        // Solution: x = 1, y = 3
        let a = vec![2i64, 1, 1, 3];
        let b = vec![vec![5i64, 10]];

        let config = HenselConfig {
            prime: 101,
            max_iterations: 10,
            target_bits: 64,
        };

        let result = hensel_solve_cpu(&a, &b, 2, 1, &config).unwrap();

        assert_eq!(result.solutions[0][0], BigInt::from(1));
        assert_eq!(result.solutions[0][1], BigInt::from(3));
    }

    #[test]
    fn test_hensel_negative() {
        // System with negative solution: x + y = 0, x - y = 2
        // Solution: x = 1, y = -1
        let a = vec![1i64, 1, 1, -1];
        let b = vec![vec![0i64, 2]];

        let config = HenselConfig {
            prime: 101,
            max_iterations: 10,
            target_bits: 64,
        };

        let result = hensel_solve_cpu(&a, &b, 2, 1, &config).unwrap();

        assert_eq!(result.solutions[0][0], BigInt::from(1));
        assert_eq!(result.solutions[0][1], BigInt::from(-1));
    }

    #[test]
    fn test_hensel_multi_rhs() {
        // Same matrix, two RHS
        // 2x + y = 5, x + 3y = 10 => x=1, y=3
        // 2x + y = 3, x + 3y = 2 => x=7/5, y=1/5 (not integer, but let's try integers)
        // Actually: 2x + y = 4, x + 3y = 6 => x=6/5... need integer solutions

        // Use: 2x + y = 7, x + 3y = 8 => 5x = 13 (not integer)
        // Use: 2x + y = 8, x + 3y = 7 => 5x = 17 (not integer)
        // Use integer RHS: 2x + y = 5, x + 3y = 10 AND 2x + y = 9, x + 3y = 12
        // Second: 5x = 15 => x = 3, y = 3
        let a = vec![2i64, 1, 1, 3];
        let b = vec![
            vec![5i64, 10],  // x=1, y=3
            vec![9i64, 12],  // x=3, y=3
        ];

        let config = HenselConfig {
            prime: 101,
            max_iterations: 10,
            target_bits: 64,
        };

        let result = hensel_solve_cpu(&a, &b, 2, 2, &config).unwrap();

        assert_eq!(result.solutions[0][0], BigInt::from(1));
        assert_eq!(result.solutions[0][1], BigInt::from(3));
        assert_eq!(result.solutions[1][0], BigInt::from(3));
        assert_eq!(result.solutions[1][1], BigInt::from(3));
    }
}
