//! Backend trait for CPU/GPU dispatch
//!
//! Defines the interface for modular arithmetic backends.

/// Backend trait for modular arithmetic operations
///
/// Implementations can target CPU, Metal GPU, or CUDA GPU.
pub trait Backend: Send + Sync {
    /// Name of this backend (for logging)
    fn name(&self) -> &'static str;

    /// Compute determinant of matrix A mod prime p
    ///
    /// # Arguments
    /// * `matrix` - Flattened n×n matrix mod p
    /// * `n` - Matrix dimension
    /// * `p` - Prime modulus
    ///
    /// # Returns
    /// det(A) mod p
    fn determinant_mod(&self, matrix: &[u32], n: usize, p: u32) -> u32;

    /// Solve Ax = b mod prime p
    ///
    /// # Arguments
    /// * `matrix` - Flattened n×n matrix mod p
    /// * `b` - RHS vector mod p
    /// * `n` - Dimension
    /// * `p` - Prime modulus
    ///
    /// # Returns
    /// Solution x mod p, or None if singular
    fn solve_mod(&self, matrix: &[u32], b: &[u32], n: usize, p: u32) -> Option<Vec<u32>>;

    /// Solve AX = B (multi-RHS) mod prime p
    ///
    /// # Arguments
    /// * `matrix` - Flattened n×n matrix mod p
    /// * `b_cols` - k RHS vectors, each of length n
    /// * `n` - Matrix dimension
    /// * `k` - Number of RHS vectors
    /// * `p` - Prime modulus
    ///
    /// # Returns
    /// Solution matrix X (k columns, each of length n), or None if singular
    fn solve_multi_rhs_mod(
        &self,
        matrix: &[u32],
        b_cols: &[Vec<u32>],
        n: usize,
        k: usize,
        p: u32,
    ) -> Option<Vec<Vec<u32>>>;

    /// Batch compute determinants mod multiple primes
    ///
    /// Default implementation calls determinant_mod sequentially.
    /// GPU backends override this to dispatch all primes in parallel.
    fn batch_determinant_mod(&self, matrix: &[u32], n: usize, primes: &[u32]) -> Vec<u32> {
        primes
            .iter()
            .map(|&p| {
                // Reduce matrix mod p
                let matrix_mod: Vec<u32> = matrix.iter().map(|&x| x % p).collect();
                self.determinant_mod(&matrix_mod, n, p)
            })
            .collect()
    }

    /// Batch solve Ax = b mod multiple primes
    ///
    /// Default implementation calls solve_mod sequentially.
    /// GPU backends override this to dispatch all primes in parallel.
    ///
    /// # Returns
    /// Vector of solutions, one per prime, or None if singular for any prime
    fn batch_solve_mod(
        &self,
        matrix: &[u32],
        b: &[u32],
        n: usize,
        primes: &[u32],
    ) -> Option<Vec<Vec<u32>>> {
        primes
            .iter()
            .map(|&p| {
                let matrix_mod: Vec<u32> = matrix.iter().map(|&x| x % p).collect();
                let b_mod: Vec<u32> = b.iter().map(|&x| x % p).collect();
                self.solve_mod(&matrix_mod, &b_mod, n, p)
            })
            .collect()
    }

    /// Batch solve AX = B (multi-RHS) mod multiple primes
    ///
    /// This is the key optimization for ZK preprocessing: factor A once and solve for all k RHS.
    /// Default implementation calls solve_multi_rhs_mod sequentially.
    /// GPU backends override this to dispatch all primes in parallel.
    ///
    /// # Returns
    /// For each prime: a vector of k solution vectors
    fn batch_multi_rhs_solve_mod(
        &self,
        matrix: &[u32],
        b_cols: &[Vec<u32>],
        n: usize,
        k: usize,
        primes: &[u32],
    ) -> Option<Vec<Vec<Vec<u32>>>> {
        primes
            .iter()
            .map(|&p| {
                let matrix_mod: Vec<u32> = matrix.iter().map(|&x| x % p).collect();
                let b_cols_mod: Vec<Vec<u32>> = b_cols
                    .iter()
                    .map(|col| col.iter().map(|&x| x % p).collect())
                    .collect();
                self.solve_multi_rhs_mod(&matrix_mod, &b_cols_mod, n, k, p)
            })
            .collect()
    }
}

/// CPU backend implementation using standard algorithms
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }

    /// Modular inverse using extended Euclidean algorithm
    fn mod_inverse(a: u64, p: u64) -> Option<u64> {
        if a == 0 {
            return None;
        }

        let mut t: i64 = 0;
        let mut new_t: i64 = 1;
        let mut r = p as i64;
        let mut new_r = (a % p) as i64;

        while new_r != 0 {
            let q = r / new_r;
            let tmp_t = t - q * new_t;
            t = new_t;
            new_t = tmp_t;
            let tmp_r = r - q * new_r;
            r = new_r;
            new_r = tmp_r;
        }

        if r > 1 {
            return None; // Not invertible
        }

        if t < 0 {
            t += p as i64;
        }

        Some(t as u64)
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "CPU"
    }

    fn determinant_mod(&self, matrix: &[u32], n: usize, p: u32) -> u32 {
        // Gaussian elimination with modular arithmetic
        let p64 = p as u64;
        let mut m: Vec<u64> = matrix.iter().map(|&x| (x % p) as u64).collect();
        let mut det: u64 = 1;
        let mut sign = 1i64;

        for col in 0..n {
            // Find pivot
            let mut pivot = None;
            for row in col..n {
                if m[row * n + col] != 0 {
                    pivot = Some(row);
                    break;
                }
            }

            let pivot = match pivot {
                Some(p) => p,
                None => return 0, // Singular
            };

            // Swap rows if needed
            if pivot != col {
                for j in 0..n {
                    m.swap(col * n + j, pivot * n + j);
                }
                sign = -sign;
            }

            let pivot_val = m[col * n + col];
            det = (det * pivot_val) % p64;

            // Eliminate below
            let pivot_inv = Self::mod_inverse(pivot_val, p64).unwrap();
            for row in (col + 1)..n {
                let factor = (m[row * n + col] * pivot_inv) % p64;
                for j in col..n {
                    let sub = (factor * m[col * n + j]) % p64;
                    m[row * n + j] = (m[row * n + j] + p64 - sub) % p64;
                }
            }
        }

        if sign < 0 {
            ((p64 - det) % p64) as u32
        } else {
            det as u32
        }
    }

    fn solve_mod(&self, matrix: &[u32], b: &[u32], n: usize, p: u32) -> Option<Vec<u32>> {
        // Gaussian elimination with back-substitution
        let p64 = p as u64;
        let mut m: Vec<u64> = matrix.iter().map(|&x| (x % p) as u64).collect();
        let mut rhs: Vec<u64> = b.iter().map(|&x| (x % p) as u64).collect();

        // Forward elimination
        for col in 0..n {
            // Find pivot
            let mut pivot = None;
            for row in col..n {
                if m[row * n + col] != 0 {
                    pivot = Some(row);
                    break;
                }
            }

            let pivot = pivot?;

            // Swap rows
            if pivot != col {
                for j in 0..n {
                    m.swap(col * n + j, pivot * n + j);
                }
                rhs.swap(col, pivot);
            }

            // Scale pivot row
            let pivot_val = m[col * n + col];
            let pivot_inv = Self::mod_inverse(pivot_val, p64)?;

            for j in 0..n {
                m[col * n + j] = (m[col * n + j] * pivot_inv) % p64;
            }
            rhs[col] = (rhs[col] * pivot_inv) % p64;

            // Eliminate
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = m[row * n + col];
                if factor != 0 {
                    for j in 0..n {
                        let sub = (factor * m[col * n + j]) % p64;
                        m[row * n + j] = (m[row * n + j] + p64 - sub) % p64;
                    }
                    let sub = (factor * rhs[col]) % p64;
                    rhs[row] = (rhs[row] + p64 - sub) % p64;
                }
            }
        }

        Some(rhs.into_iter().map(|x| x as u32).collect())
    }

    fn solve_multi_rhs_mod(
        &self,
        matrix: &[u32],
        b_cols: &[Vec<u32>],
        n: usize,
        _k: usize,
        p: u32,
    ) -> Option<Vec<Vec<u32>>> {
        // Simple implementation: solve each RHS sequentially
        // A GPU backend would parallelize this
        b_cols
            .iter()
            .map(|b| self.solve_mod(matrix, b, n, p))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_determinant() {
        let backend = CpuBackend::new();

        // 2x2 matrix: [[1, 2], [3, 4]], det = -2
        let matrix = vec![1, 2, 3, 4];
        let p = 101u32;

        let det = backend.determinant_mod(&matrix, 2, p);
        // det = -2 mod 101 = 99
        assert_eq!(det, 99);
    }

    #[test]
    fn test_cpu_solve() {
        let backend = CpuBackend::new();

        // Ax = b where A = [[2, 1], [1, 3]], b = [5, 10]
        // Solution: x = [1, 3]
        // 2*1 + 1*3 = 5 ✓
        // 1*1 + 3*3 = 10 ✓
        let matrix = vec![2, 1, 1, 3];
        let b = vec![5, 10];
        let p = 101u32;

        let x = backend.solve_mod(&matrix, &b, 2, p).unwrap();
        assert_eq!(x, vec![1, 3]);
    }
}
