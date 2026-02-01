//! Lattice basis representation
//!
//! Provides a lattice basis data structure with CRT support for GPU acceleration.

use num_bigint::BigInt;
use num_traits::{Zero, One, Signed};
use std::fmt;

/// A lattice basis represented as a matrix of row vectors
///
/// Each row b_i is a basis vector in Z^m.
/// The lattice L(B) = {Σ x_i b_i : x_i ∈ Z}
#[derive(Debug, Clone)]
pub struct LatticeBasis {
    /// Basis vectors as rows (n vectors of dimension m)
    pub vectors: Vec<Vec<BigInt>>,
    /// Number of basis vectors (rank)
    pub n: usize,
    /// Dimension of the ambient space
    pub m: usize,
}

impl LatticeBasis {
    /// Create a new lattice basis from row vectors
    ///
    /// # Arguments
    /// * `rows` - Vector of basis vectors (as row vectors)
    ///
    /// # Panics
    /// Panics if rows have inconsistent dimensions or if the basis is empty
    pub fn new(vectors: Vec<Vec<BigInt>>) -> Self {
        assert!(!vectors.is_empty(), "Basis cannot be empty");
        let m = vectors[0].len();
        assert!(m > 0, "Vectors cannot be empty");
        assert!(
            vectors.iter().all(|v| v.len() == m),
            "All vectors must have the same dimension"
        );

        let n = vectors.len();
        Self { vectors, n, m }
    }

    /// Create a lattice basis from integer slices
    pub fn from_rows<T: Into<BigInt> + Clone>(rows: &[Vec<T>]) -> Self {
        let vectors: Vec<Vec<BigInt>> = rows
            .iter()
            .map(|row| row.iter().map(|x| x.clone().into()).collect())
            .collect();
        Self::new(vectors)
    }

    /// Create a lattice basis from a flat array (row-major order)
    pub fn from_flat<T: Into<BigInt> + Clone>(data: &[T], n: usize, m: usize) -> Self {
        assert_eq!(data.len(), n * m, "Data size must match n × m");
        let vectors: Vec<Vec<BigInt>> = (0..n)
            .map(|i| {
                (0..m)
                    .map(|j| data[i * m + j].clone().into())
                    .collect()
            })
            .collect();
        Self { vectors, n, m }
    }

    /// Create a random lattice basis for testing
    ///
    /// # Arguments
    /// * `n` - Number of basis vectors (rank)
    /// * `m` - Dimension of ambient space
    /// * `bits` - Maximum bit size of entries
    pub fn random(n: usize, m: usize, bits: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let bound = BigInt::one() << bits;

        let vectors: Vec<Vec<BigInt>> = (0..n)
            .map(|_| {
                (0..m)
                    .map(|_| {
                        let val: i64 = rng.gen_range(-(1i64 << (bits - 1))..(1i64 << (bits - 1)));
                        BigInt::from(val)
                    })
                    .collect()
            })
            .collect();

        Self { vectors, n, m }
    }

    /// Create a knapsack/subset-sum lattice for testing
    ///
    /// Given a = [a_1, ..., a_n] and target s, creates the lattice:
    /// ```text
    /// [ 2  0  0 ... 0  a_1 ]
    /// [ 0  2  0 ... 0  a_2 ]
    /// [ 0  0  2 ... 0  a_3 ]
    /// [ ...                ]
    /// [ 1  1  1 ... 1   s  ]
    /// ```
    pub fn knapsack(a: &[i64], s: i64) -> Self {
        let n = a.len() + 1;
        let m = a.len() + 1;

        let mut vectors = vec![vec![BigInt::zero(); m]; n];

        // First n-1 rows: diagonal 2's with a_i in last column
        for i in 0..a.len() {
            vectors[i][i] = BigInt::from(2);
            vectors[i][m - 1] = BigInt::from(a[i]);
        }

        // Last row: all 1's with s in last column
        for j in 0..a.len() {
            vectors[n - 1][j] = BigInt::one();
        }
        vectors[n - 1][m - 1] = BigInt::from(s);

        Self { vectors, n, m }
    }

    /// Get vector at index i
    pub fn get(&self, i: usize) -> &[BigInt] {
        &self.vectors[i]
    }

    /// Get mutable reference to vector at index i
    pub fn get_mut(&mut self, i: usize) -> &mut Vec<BigInt> {
        &mut self.vectors[i]
    }

    /// Swap two basis vectors
    pub fn swap(&mut self, i: usize, j: usize) {
        self.vectors.swap(i, j);
    }

    /// Compute inner product <b_i, b_j>
    pub fn inner_product(&self, i: usize, j: usize) -> BigInt {
        self.vectors[i]
            .iter()
            .zip(self.vectors[j].iter())
            .map(|(a, b)| a * b)
            .fold(BigInt::zero(), |acc, x| acc + x)
    }

    /// Compute squared norm ||b_i||^2
    pub fn norm_squared(&self, i: usize) -> BigInt {
        self.inner_product(i, i)
    }

    /// Update b_i = b_i - q * b_j (size reduction step)
    pub fn reduce_vector(&mut self, i: usize, j: usize, q: &BigInt) {
        for k in 0..self.m {
            self.vectors[i][k] = &self.vectors[i][k] - q * &self.vectors[j][k];
        }
    }

    /// Get maximum absolute entry (for bound estimation)
    pub fn max_entry(&self) -> BigInt {
        self.vectors
            .iter()
            .flat_map(|v| v.iter())
            .map(|x| x.abs())
            .max()
            .unwrap_or_else(BigInt::zero)
    }

    /// Estimate bit size needed for Gram-Schmidt denominators
    ///
    /// The Gram-Schmidt coefficients μ_ij have denominators bounded by
    /// det(B_{1..j}^T B_{1..j}), which is at most (n * max_entry^2)^(n/2)
    pub fn estimate_gs_bits(&self) -> usize {
        let max_entry = self.max_entry();
        let max_bits = max_entry.bits() as usize;

        // Upper bound: each GS step can double the bit size
        // Conservative estimate: n * (2 * max_bits + log2(m))
        let entry_bits = 2 * max_bits + (self.m as f64).log2().ceil() as usize;
        self.n * entry_bits + self.n * 32 // Extra margin
    }

    /// Convert to residues mod prime
    pub fn to_residues(&self, p: u32) -> Vec<u32> {
        let p_big = BigInt::from(p);
        let mut residues = vec![0u32; self.n * self.m];

        for i in 0..self.n {
            for j in 0..self.m {
                let r = &self.vectors[i][j] % &p_big;
                let r = if r < BigInt::zero() { r + &p_big } else { r };
                residues[i * self.m + j] = r.try_into().unwrap_or(0);
            }
        }

        residues
    }

    /// Flatten to row-major BigInt array
    pub fn to_flat(&self) -> Vec<BigInt> {
        self.vectors.iter().flat_map(|v| v.clone()).collect()
    }
}

impl fmt::Display for LatticeBasis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "LatticeBasis ({}×{}):", self.n, self.m)?;
        for (i, v) in self.vectors.iter().enumerate() {
            write!(f, "  b_{}: [", i)?;
            for (j, x) in v.iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", x)?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basis_creation() {
        let basis = LatticeBasis::from_rows(&[
            vec![1i64, 0, 3],
            vec![0, 1, 5],
            vec![0, 0, 7],
        ]);

        assert_eq!(basis.n, 3);
        assert_eq!(basis.m, 3);
    }

    #[test]
    fn test_inner_product() {
        let basis = LatticeBasis::from_rows(&[
            vec![1i64, 2, 3],
            vec![4, 5, 6],
        ]);

        // <b_0, b_0> = 1 + 4 + 9 = 14
        assert_eq!(basis.norm_squared(0), BigInt::from(14));

        // <b_0, b_1> = 4 + 10 + 18 = 32
        assert_eq!(basis.inner_product(0, 1), BigInt::from(32));
    }

    #[test]
    fn test_knapsack_lattice() {
        let a = vec![1i64, 2, 3];
        let s = 5i64;
        let basis = LatticeBasis::knapsack(&a, s);

        assert_eq!(basis.n, 4);
        assert_eq!(basis.m, 4);

        // Check structure
        assert_eq!(basis.vectors[0][0], BigInt::from(2));
        assert_eq!(basis.vectors[0][3], BigInt::from(1));
        assert_eq!(basis.vectors[3][3], BigInt::from(5));
    }

    #[test]
    fn test_to_residues() {
        let basis = LatticeBasis::from_rows(&[
            vec![10i64, -3, 7],
            vec![-2, 5, 11],
        ]);

        let p = 7u32;
        let residues = basis.to_residues(p);

        // 10 mod 7 = 3
        assert_eq!(residues[0], 3);
        // -3 mod 7 = 4
        assert_eq!(residues[1], 4);
    }
}
