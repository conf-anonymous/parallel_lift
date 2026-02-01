//! Dense matrix operations
//!
//! Row-major dense matrix representation.

use num_bigint::BigInt;
use num_traits::Zero;
use crate::rational::Rational;

/// Dense matrix in row-major order
#[derive(Debug, Clone)]
pub struct Matrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: Clone> Matrix<T> {
    /// Create a matrix from a flat vector (row-major order)
    pub fn from_flat(data: Vec<T>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { data, rows, cols }
    }

    /// Get matrix dimensions
    pub fn dims(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Get number of rows
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get number of columns
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Access element at (i, j)
    pub fn get(&self, i: usize, j: usize) -> &T {
        &self.data[i * self.cols + j]
    }

    /// Mutable access to element at (i, j)
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut T {
        &mut self.data[i * self.cols + j]
    }

    /// Get underlying data as slice
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get mutable underlying data
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consume and return underlying data
    pub fn into_data(self) -> Vec<T> {
        self.data
    }

    /// Get a row as a slice
    pub fn row(&self, i: usize) -> &[T] {
        let start = i * self.cols;
        &self.data[start..start + self.cols]
    }
}

impl Matrix<BigInt> {
    /// Create a zero matrix
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![BigInt::zero(); rows * cols],
            rows,
            cols,
        }
    }

    /// Create an identity matrix
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            *m.get_mut(i, i) = BigInt::from(1);
        }
        m
    }

    /// Reduce all entries modulo p
    pub fn reduce_mod(&self, p: u32) -> Vec<u32> {
        let p_big = BigInt::from(p);
        self.data
            .iter()
            .map(|x| {
                let reduced = ((x % &p_big) + &p_big) % &p_big;
                // Safe because reduced is in [0, p)
                reduced.try_into().unwrap_or(0)
            })
            .collect()
    }
}

impl Matrix<Rational> {
    /// Create a zero rational matrix
    pub fn zeros_rational(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![Rational::zero(); rows * cols],
            rows,
            cols,
        }
    }
}

impl<T: Clone + Default> Matrix<T> {
    /// Create a matrix filled with default values
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![T::default(); rows * cols],
            rows,
            cols,
        }
    }
}

/// Generate a random BigInt matrix for testing
pub fn random_matrix(rows: usize, cols: usize, bound: i64) -> Matrix<BigInt> {
    use std::time::{SystemTime, UNIX_EPOCH};

    // Simple PRNG for testing (not cryptographically secure)
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    let mut state = seed;

    let next_random = |s: &mut u64| -> i64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = (*s >> 33) as i64;
        (val % bound) - bound / 2
    };

    let mut state_copy = state;
    let data: Vec<BigInt> = (0..rows * cols)
        .map(|_| BigInt::from(next_random(&mut state_copy)))
        .collect();

    Matrix::from_flat(data, rows, cols)
}

/// Generate a random BigInt vector for testing
pub fn random_vector(n: usize, bound: i64) -> Vec<BigInt> {
    use std::time::{SystemTime, UNIX_EPOCH};

    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    let mut state = seed.wrapping_add(12345); // Different seed from matrix

    let next_random = |s: &mut u64| -> i64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = (*s >> 33) as i64;
        (val % bound) - bound / 2
    };

    (0..n).map(|_| BigInt::from(next_random(&mut state))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_access() {
        let data: Vec<BigInt> = (0..6).map(BigInt::from).collect();
        let m = Matrix::from_flat(data, 2, 3);

        assert_eq!(m.get(0, 0), &BigInt::from(0));
        assert_eq!(m.get(0, 2), &BigInt::from(2));
        assert_eq!(m.get(1, 0), &BigInt::from(3));
        assert_eq!(m.get(1, 2), &BigInt::from(5));
    }

    #[test]
    fn test_identity() {
        let id = Matrix::identity(3);
        assert_eq!(id.get(0, 0), &BigInt::from(1));
        assert_eq!(id.get(1, 1), &BigInt::from(1));
        assert_eq!(id.get(0, 1), &BigInt::from(0));
    }
}
