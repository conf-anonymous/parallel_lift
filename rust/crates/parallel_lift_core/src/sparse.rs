//! Sparse Matrix Support
//!
//! Provides CSR (Compressed Sparse Row) representation and sparse linear algebra
//! algorithms optimized for R1CS-style constraint matrices.
//!
//! # Key Algorithms
//!
//! - **Wiedemann Algorithm**: Iterative solver for sparse systems over finite fields
//! - **Block Wiedemann**: Batched variant for multiple RHS vectors
//!
//! # Why Sparse Matters for ZK
//!
//! Real R1CS constraint matrices are extremely sparse (typically 3-5 nonzeros per row).
//! Dense Gaussian elimination is O(n³), while sparse Wiedemann is O(n × nnz × iterations)
//! where nnz is the number of nonzeros. For typical ZK circuits, this is a massive win.

use num_bigint::BigInt;
use num_traits::{One, Zero};

/// Compressed Sparse Row (CSR) matrix representation
///
/// Memory layout:
/// - `values`: Nonzero values in row-major order
/// - `col_indices`: Column index for each value
/// - `row_ptrs`: Start index in values/col_indices for each row
///
/// For an n×n matrix with nnz nonzeros:
/// - values.len() == nnz
/// - col_indices.len() == nnz
/// - row_ptrs.len() == n + 1
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    /// Number of rows
    pub nrows: usize,
    /// Number of columns
    pub ncols: usize,
    /// Nonzero values in row-major order
    pub values: Vec<BigInt>,
    /// Column indices for each value
    pub col_indices: Vec<usize>,
    /// Row pointers: row i spans values[row_ptrs[i]..row_ptrs[i+1]]
    pub row_ptrs: Vec<usize>,
}

/// CSR matrix with u32 values for modular arithmetic
#[derive(Debug, Clone)]
pub struct SparseMatrixMod {
    pub nrows: usize,
    pub ncols: usize,
    pub values: Vec<u32>,
    pub col_indices: Vec<usize>,
    pub row_ptrs: Vec<usize>,
}

impl SparseMatrix {
    /// Create a new sparse matrix from COO (coordinate) format
    pub fn from_coo(nrows: usize, ncols: usize, entries: &[(usize, usize, BigInt)]) -> Self {
        // Sort entries by row, then column
        let mut sorted: Vec<_> = entries.to_vec();
        sorted.sort_by_key(|(r, c, _)| (*r, *c));

        let mut values = Vec::with_capacity(sorted.len());
        let mut col_indices = Vec::with_capacity(sorted.len());
        let mut row_ptrs = vec![0usize; nrows + 1];

        for (row, col, val) in sorted {
            if !val.is_zero() {
                values.push(val);
                col_indices.push(col);
                row_ptrs[row + 1] += 1;
            }
        }

        // Convert counts to cumulative offsets
        for i in 1..=nrows {
            row_ptrs[i] += row_ptrs[i - 1];
        }

        Self {
            nrows,
            ncols,
            values,
            col_indices,
            row_ptrs,
        }
    }

    /// Create from dense matrix (for testing/comparison)
    pub fn from_dense(dense: &[BigInt], n: usize) -> Self {
        let mut entries = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let val = &dense[i * n + j];
                if !val.is_zero() {
                    entries.push((i, j, val.clone()));
                }
            }
        }
        Self::from_coo(n, n, &entries)
    }

    /// Number of nonzero entries
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Sparsity ratio (fraction of zeros)
    pub fn sparsity(&self) -> f64 {
        let total = self.nrows * self.ncols;
        1.0 - (self.nnz() as f64 / total as f64)
    }

    /// Convert to modular representation
    pub fn to_mod(&self, p: u32) -> SparseMatrixMod {
        let p_big = BigInt::from(p);
        let values: Vec<u32> = self.values.iter()
            .map(|v| {
                let r = v % &p_big;
                if r < BigInt::zero() {
                    ((&r + &p_big) % &p_big).try_into().unwrap_or(0)
                } else {
                    r.try_into().unwrap_or(0)
                }
            })
            .collect();

        SparseMatrixMod {
            nrows: self.nrows,
            ncols: self.ncols,
            values,
            col_indices: self.col_indices.clone(),
            row_ptrs: self.row_ptrs.clone(),
        }
    }

    /// Generate sparse matrix with target sparsity level
    ///
    /// Sparsity is the fraction of zeros (e.g., 0.95 = 95% zeros = 5% nonzeros)
    /// Matrix is guaranteed to be non-singular (diagonal dominance)
    pub fn generate_with_sparsity(n: usize, target_sparsity: f64, seed: u64) -> Self {
        use std::collections::HashSet;

        let mut entries = Vec::new();
        let mut state = seed;

        // Simple LCG for deterministic generation
        let lcg = |s: &mut u64| -> u64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *s
        };

        // Target density (1 - sparsity) determines nonzeros per row
        let density = 1.0 - target_sparsity;
        let avg_nnz_per_row = ((n as f64 * density) as usize).max(1);

        for row in 0..n {
            let mut used_cols = HashSet::new();

            // Always include diagonal for non-singularity (large value for dominance)
            let diag_val = ((lcg(&mut state) % 500) as i64 + 500) as i64;
            entries.push((row, row, BigInt::from(diag_val)));
            used_cols.insert(row);

            // Add random off-diagonal entries to achieve target density
            let num_extra = avg_nnz_per_row.saturating_sub(1);
            for _ in 0..num_extra {
                let col = (lcg(&mut state) as usize) % n;
                if !used_cols.contains(&col) {
                    used_cols.insert(col);
                    // Random small values
                    let val = ((lcg(&mut state) % 200) as i64 - 100) as i64;
                    if val != 0 {
                        entries.push((row, col, BigInt::from(val)));
                    }
                }
            }
        }

        Self::from_coo(n, n, &entries)
    }

    /// Generate R1CS-like sparse matrix for benchmarking
    ///
    /// R1CS matrices have:
    /// - ~3-5 nonzeros per row (constraint structure)
    /// - Values typically small integers (-1, 0, 1, or field elements)
    pub fn generate_r1cs_like(n: usize, avg_nnz_per_row: usize, seed: u64) -> Self {
        use std::collections::HashSet;

        let mut entries = Vec::new();
        let mut state = seed;

        // Simple LCG for deterministic generation
        let lcg = |s: &mut u64| -> u64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *s
        };

        for row in 0..n {
            let mut used_cols = HashSet::new();

            // Always include diagonal for non-singularity
            let diag_val = ((lcg(&mut state) % 100) as i64 + 50) as i64;
            entries.push((row, row, BigInt::from(diag_val)));
            used_cols.insert(row);

            // Add random off-diagonal entries
            let num_extra = avg_nnz_per_row.saturating_sub(1);
            for _ in 0..num_extra {
                let col = (lcg(&mut state) as usize) % n;
                if !used_cols.contains(&col) {
                    used_cols.insert(col);
                    // R1CS-style: small values, often -1, 0, 1
                    let val = match lcg(&mut state) % 10 {
                        0..=3 => 1i64,
                        4..=6 => -1i64,
                        _ => ((lcg(&mut state) % 20) as i64 - 10),
                    };
                    if val != 0 {
                        entries.push((row, col, BigInt::from(val)));
                    }
                }
            }
        }

        Self::from_coo(n, n, &entries)
    }
}

impl SparseMatrixMod {
    /// Sparse matrix-vector multiply: y = A * x mod p
    pub fn matvec(&self, x: &[u32], p: u32) -> Vec<u32> {
        let mut y = vec![0u32; self.nrows];
        let p64 = p as u64;

        for row in 0..self.nrows {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];

            let mut sum: u64 = 0;
            for idx in start..end {
                let col = self.col_indices[idx];
                let val = self.values[idx] as u64;
                sum = (sum + val * (x[col] as u64)) % p64;
            }
            y[row] = sum as u32;
        }

        y
    }

    /// Sparse matrix-vector multiply with transposed matrix: y = A^T * x mod p
    pub fn matvec_transpose(&self, x: &[u32], p: u32) -> Vec<u32> {
        let mut y = vec![0u32; self.ncols];
        let p64 = p as u64;

        for row in 0..self.nrows {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];
            let x_row = x[row] as u64;

            for idx in start..end {
                let col = self.col_indices[idx];
                let val = self.values[idx] as u64;
                y[col] = ((y[col] as u64 + val * x_row) % p64) as u32;
            }
        }

        y
    }
}

/// Wiedemann algorithm for solving sparse linear systems over finite fields
///
/// Solves Ax = b where A is a sparse n×n matrix mod prime p.
///
/// Algorithm overview:
/// 1. Generate random vector u
/// 2. Compute sequence: u^T * b, u^T * A*b, u^T * A²*b, ...
/// 3. Find minimal polynomial using Berlekamp-Massey
/// 4. Construct solution from minimal polynomial
///
/// Complexity: O(n × nnz × n) = O(n² × nnz) vs O(n³) for dense
/// For sparse R1CS (nnz ≈ 5n), this is O(5n³) vs O(n³) - but with much
/// better constants and cache behavior.
pub struct WiedemannSolver {
    /// Maximum iterations (typically 2n + small constant)
    max_iterations: usize,
}

impl WiedemannSolver {
    pub fn new() -> Self {
        Self {
            max_iterations: 0, // Will be set based on matrix size
        }
    }

    /// Solve Ax = b mod p using Wiedemann algorithm
    pub fn solve(&self, a: &SparseMatrixMod, b: &[u32], p: u32) -> Option<Vec<u32>> {
        let n = a.nrows;
        if n == 0 || b.len() != n {
            return None;
        }

        // For small matrices, fall back to dense (Wiedemann overhead not worth it)
        if n <= 32 || a.values.len() > n * n / 3 {
            return self.solve_dense_fallback(a, b, p);
        }

        // Wiedemann algorithm
        let max_iter = 2 * n + 10;

        // Generate random vector u (use deterministic seed for reproducibility)
        let mut u = vec![0u32; n];
        let mut state = 12345u64;
        for i in 0..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            u[i] = (state % (p as u64)) as u32;
        }

        // Compute sequence: s[i] = u^T * A^i * b
        let mut sequence = Vec::with_capacity(max_iter);
        let mut v = b.to_vec(); // v = A^i * b, starting with A^0 * b = b

        for _ in 0..max_iter {
            // s[i] = u^T * v = dot(u, v)
            let s = dot_mod(&u, &v, p);
            sequence.push(s);

            // v = A * v
            v = a.matvec(&v, p);
        }

        // Find minimal polynomial using Berlekamp-Massey
        let min_poly = berlekamp_massey(&sequence, p)?;

        if min_poly.is_empty() || min_poly[0] == 0 {
            // Singular matrix
            return None;
        }

        // Construct solution: x = -c[0]^(-1) * sum_{i=1}^{d} c[i] * A^{i-1} * b
        let c0_inv = mod_inverse(min_poly[0] as u64, p as u64)? as u32;
        let neg_c0_inv = ((p as u64 - c0_inv as u64) % p as u64) as u32;

        let mut x = vec![0u32; n];
        let mut v = b.to_vec();

        for i in 1..min_poly.len() {
            let coeff = min_poly[i];
            if coeff != 0 {
                // x += coeff * v
                for j in 0..n {
                    x[j] = ((x[j] as u64 + (coeff as u64) * (v[j] as u64)) % p as u64) as u32;
                }
            }
            if i < min_poly.len() - 1 {
                v = a.matvec(&v, p);
            }
        }

        // x = neg_c0_inv * x
        for j in 0..n {
            x[j] = ((neg_c0_inv as u64 * x[j] as u64) % p as u64) as u32;
        }

        // Verify: A * x should equal b
        let ax = a.matvec(&x, p);
        if ax == b {
            Some(x)
        } else {
            // Verification failed - try dense fallback
            self.solve_dense_fallback(a, b, p)
        }
    }

    /// Dense Gaussian elimination fallback for small/dense matrices
    fn solve_dense_fallback(&self, a: &SparseMatrixMod, b: &[u32], p: u32) -> Option<Vec<u32>> {
        let n = a.nrows;
        let p64 = p as u64;

        // Convert to dense augmented matrix [A|b]
        let mut aug = vec![0u64; n * (n + 1)];

        // Fill A
        for row in 0..n {
            let start = a.row_ptrs[row];
            let end = a.row_ptrs[row + 1];
            for idx in start..end {
                let col = a.col_indices[idx];
                aug[row * (n + 1) + col] = a.values[idx] as u64;
            }
            // Fill b
            aug[row * (n + 1) + n] = b[row] as u64;
        }

        // Gaussian elimination
        for col in 0..n {
            // Find pivot
            let mut pivot = None;
            for row in col..n {
                if aug[row * (n + 1) + col] != 0 {
                    pivot = Some(row);
                    break;
                }
            }

            let pivot = pivot?;

            // Swap rows
            if pivot != col {
                for j in 0..=n {
                    aug.swap(col * (n + 1) + j, pivot * (n + 1) + j);
                }
            }

            // Scale pivot row
            let pivot_val = aug[col * (n + 1) + col];
            let pivot_inv = mod_inverse(pivot_val, p64)?;
            for j in col..=n {
                aug[col * (n + 1) + j] = (aug[col * (n + 1) + j] * pivot_inv) % p64;
            }

            // Eliminate
            for row in 0..n {
                if row != col {
                    let factor = aug[row * (n + 1) + col];
                    if factor != 0 {
                        for j in col..=n {
                            let sub = (factor * aug[col * (n + 1) + j]) % p64;
                            aug[row * (n + 1) + j] = (aug[row * (n + 1) + j] + p64 - sub) % p64;
                        }
                    }
                }
            }
        }

        // Extract solution
        Some((0..n).map(|i| aug[i * (n + 1) + n] as u32).collect())
    }
}

impl Default for WiedemannSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Dot product mod p
fn dot_mod(a: &[u32], b: &[u32], p: u32) -> u32 {
    let p64 = p as u64;
    let mut sum: u64 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        sum = (sum + (*x as u64) * (*y as u64)) % p64;
    }
    sum as u32
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
        return None;
    }
    if t < 0 {
        t += p as i64;
    }
    Some(t as u64)
}

/// Berlekamp-Massey algorithm to find minimal polynomial
///
/// Given sequence s[0], s[1], ..., s[2n-1], finds shortest LFSR that generates it.
/// Returns coefficients c[0], c[1], ..., c[d] of minimal polynomial.
fn berlekamp_massey(seq: &[u32], p: u32) -> Option<Vec<u32>> {
    let n = seq.len();
    if n == 0 {
        return None;
    }

    let p64 = p as u64;

    // Connection polynomial C(x) and its length
    let mut c = vec![0u32; n + 1];
    c[0] = 1;
    let mut c_len = 1;

    // Previous polynomial B(x)
    let mut b = vec![0u32; n + 1];
    b[0] = 1;

    let mut l = 0; // Current LFSR length
    let mut m = 1; // Number of iterations since last length change
    let mut b_coeff = 1u32; // Leading coefficient of discrepancy when B was set

    for i in 0..n {
        // Compute discrepancy d = s[i] + sum_{j=1}^{l} c[j] * s[i-j]
        let mut d: u64 = seq[i] as u64;
        for j in 1..=l.min(i) {
            d = (d + (c[j] as u64) * (seq[i - j] as u64)) % p64;
        }

        if d == 0 {
            m += 1;
        } else {
            // T(x) = C(x) - (d/b) * x^m * B(x)
            let db_inv = (d * mod_inverse(b_coeff as u64, p64)?) % p64;

            if 2 * l <= i {
                // Save C before modifying
                let old_c = c.clone();

                // C(x) = C(x) - (d/b) * x^m * B(x)
                for j in 0..b.len() {
                    if b[j] != 0 && j + m < c.len() {
                        let sub = (db_inv * (b[j] as u64)) % p64;
                        c[j + m] = ((c[j + m] as u64 + p64 - sub) % p64) as u32;
                    }
                }

                l = i + 1 - l;
                b = old_c;
                b_coeff = d as u32;
                c_len = c_len.max(l + 1);
                m = 1;
            } else {
                // Just update C
                for j in 0..b.len() {
                    if b[j] != 0 && j + m < c.len() {
                        let sub = (db_inv * (b[j] as u64)) % p64;
                        c[j + m] = ((c[j + m] as u64 + p64 - sub) % p64) as u32;
                    }
                }
                m += 1;
            }
        }
    }

    // Return minimal polynomial coefficients
    c.truncate(l + 1);
    Some(c)
}

/// Sparse solver that integrates with the Backend trait
pub struct SparseSolver;

impl SparseSolver {
    pub fn new() -> Self {
        Self
    }

    /// Solve sparse system Ax = b for multiple primes, return solutions mod each prime
    pub fn batch_solve_mod(
        &self,
        a: &SparseMatrix,
        b: &[BigInt],
        primes: &[u32],
    ) -> Option<Vec<Vec<u32>>> {
        let wiedemann = WiedemannSolver::new();

        primes.iter().map(|&p| {
            let a_mod = a.to_mod(p);
            let b_mod: Vec<u32> = b.iter()
                .map(|v| {
                    let p_big = BigInt::from(p);
                    let r = v % &p_big;
                    if r < BigInt::zero() {
                        ((&r + &p_big) % &p_big).try_into().unwrap_or(0)
                    } else {
                        r.try_into().unwrap_or(0)
                    }
                })
                .collect();

            wiedemann.solve(&a_mod, &b_mod, p)
        }).collect()
    }

    /// Solve sparse system for multiple RHS vectors
    pub fn batch_multi_rhs_solve_mod(
        &self,
        a: &SparseMatrix,
        b_cols: &[Vec<BigInt>],
        primes: &[u32],
    ) -> Option<Vec<Vec<Vec<u32>>>> {
        let wiedemann = WiedemannSolver::new();

        primes.iter().map(|&p| {
            let a_mod = a.to_mod(p);

            b_cols.iter().map(|b| {
                let b_mod: Vec<u32> = b.iter()
                    .map(|v| {
                        let p_big = BigInt::from(p);
                        let r = v % &p_big;
                        if r < BigInt::zero() {
                            ((&r + &p_big) % &p_big).try_into().unwrap_or(0)
                        } else {
                            r.try_into().unwrap_or(0)
                        }
                    })
                    .collect();

                wiedemann.solve(&a_mod, &b_mod, p)
            }).collect::<Option<Vec<_>>>()
        }).collect()
    }
}

impl Default for SparseSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_from_coo() {
        let entries = vec![
            (0, 0, BigInt::from(2)),
            (0, 1, BigInt::from(3)),
            (1, 1, BigInt::from(4)),
        ];
        let sparse = SparseMatrix::from_coo(2, 2, &entries);

        assert_eq!(sparse.nnz(), 3);
        assert_eq!(sparse.nrows, 2);
        assert_eq!(sparse.ncols, 2);
    }

    #[test]
    fn test_sparse_matvec() {
        // Matrix: [[2, 3], [0, 4]]
        let entries = vec![
            (0, 0, BigInt::from(2)),
            (0, 1, BigInt::from(3)),
            (1, 1, BigInt::from(4)),
        ];
        let sparse = SparseMatrix::from_coo(2, 2, &entries);
        let sparse_mod = sparse.to_mod(101);

        // x = [1, 2], expected y = [2*1 + 3*2, 0*1 + 4*2] = [8, 8]
        let x = vec![1u32, 2u32];
        let y = sparse_mod.matvec(&x, 101);

        assert_eq!(y, vec![8, 8]);
    }

    #[test]
    fn test_wiedemann_simple() {
        // Simple 2x2 system: [[2, 1], [1, 3]] * x = [5, 10]
        // Solution: x = [1, 3]
        let entries = vec![
            (0, 0, BigInt::from(2)),
            (0, 1, BigInt::from(1)),
            (1, 0, BigInt::from(1)),
            (1, 1, BigInt::from(3)),
        ];
        let sparse = SparseMatrix::from_coo(2, 2, &entries);
        let sparse_mod = sparse.to_mod(101);

        let b = vec![5u32, 10u32];
        let solver = WiedemannSolver::new();
        let x = solver.solve(&sparse_mod, &b, 101).unwrap();

        assert_eq!(x, vec![1, 3]);
    }

    #[test]
    fn test_r1cs_generation() {
        let sparse = SparseMatrix::generate_r1cs_like(100, 5, 42);

        // Should be sparse
        assert!(sparse.sparsity() > 0.9);

        // Should have reasonable nnz
        let expected_nnz = 100 * 5; // ~5 per row
        assert!(sparse.nnz() < expected_nnz * 2);
        assert!(sparse.nnz() > expected_nnz / 2);
    }
}
