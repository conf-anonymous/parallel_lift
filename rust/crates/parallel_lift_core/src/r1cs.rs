//! R1CS File Format Support
//!
//! Provides loading and parsing of R1CS (Rank-1 Constraint System) files.
//! Supports the iden3/circom binary R1CS format.
//!
//! # R1CS Format Overview
//!
//! An R1CS system represents constraints as A × B - C = 0, where:
//! - A, B, C are sparse matrices
//! - Each row represents one constraint
//! - Columns correspond to wires (variables)
//!
//! # File Format
//!
//! Based on iden3/r1csfile specification:
//! - Magic: "r1cs" (0x72 0x31 0x63 0x73)
//! - Version: 4 bytes little-endian
//! - Sections: Header, Constraints, Wire-to-Label map

use std::io::{Read, Cursor};
use std::path::Path;
use std::fs::File;
use num_bigint::BigInt;
use num_traits::Zero;
use crate::sparse::SparseMatrix;

/// Error types for R1CS parsing
#[derive(Debug)]
pub enum R1csError {
    IoError(std::io::Error),
    InvalidMagic,
    UnsupportedVersion(u32),
    InvalidSectionType(u32),
    ParseError(String),
}

impl From<std::io::Error> for R1csError {
    fn from(e: std::io::Error) -> Self {
        R1csError::IoError(e)
    }
}

/// R1CS constraint system parsed from file
#[derive(Debug, Clone)]
pub struct R1csFile {
    /// Prime field modulus
    pub prime: BigInt,
    /// Number of wires (including constant wire 0)
    pub n_wires: usize,
    /// Number of public outputs
    pub n_pub_out: usize,
    /// Number of public inputs
    pub n_pub_in: usize,
    /// Number of private inputs
    pub n_prv_in: usize,
    /// Total number of constraints
    pub n_constraints: usize,
    /// A matrix in COO format: (row, col, value)
    pub a: Vec<(usize, usize, BigInt)>,
    /// B matrix in COO format
    pub b: Vec<(usize, usize, BigInt)>,
    /// C matrix in COO format
    pub c: Vec<(usize, usize, BigInt)>,
}

impl R1csFile {
    /// Load an R1CS file from disk
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, R1csError> {
        let mut file = File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Self::parse(&data)
    }

    /// Parse R1CS data from bytes
    pub fn parse(data: &[u8]) -> Result<Self, R1csError> {
        let mut cursor = Cursor::new(data);

        // Magic number: "r1cs" = 0x72 0x31 0x63 0x73
        let magic = read_u32_le(&mut cursor)?;
        if magic != 0x73633172 { // little-endian "r1cs"
            return Err(R1csError::InvalidMagic);
        }

        // Version
        let version = read_u32_le(&mut cursor)?;
        if version != 1 {
            return Err(R1csError::UnsupportedVersion(version));
        }

        // Number of sections
        let n_sections = read_u32_le(&mut cursor)?;

        // First pass: find section locations
        let mut section_offsets: Vec<(u32, u64, u64)> = Vec::new(); // (type, offset, size)
        for _ in 0..n_sections {
            let section_type = read_u32_le(&mut cursor)?;
            let section_size = read_u64_le(&mut cursor)?;
            let section_start = cursor.position();
            section_offsets.push((section_type, section_start, section_size));
            cursor.set_position(section_start + section_size);
        }

        // Parse header first (section type 1)
        let mut header: Option<R1csHeader> = None;
        for &(section_type, offset, _size) in &section_offsets {
            if section_type == 0x00000001 {
                cursor.set_position(offset);
                header = Some(Self::parse_header(&mut cursor)?);
                break;
            }
        }

        let header = header.ok_or_else(|| R1csError::ParseError("Missing header section".into()))?;

        // Parse constraints (section type 2) now that we have header
        let mut constraints: Vec<R1csConstraint> = Vec::new();
        for &(section_type, offset, _size) in &section_offsets {
            if section_type == 0x00000002 {
                cursor.set_position(offset);
                constraints = Self::parse_constraints(&mut cursor, &header)?;
                break;
            }
        }

        // Convert constraints to COO format
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();

        for (row, constraint) in constraints.iter().enumerate() {
            for (col, val) in &constraint.a {
                if !val.is_zero() {
                    a.push((row, *col, val.clone()));
                }
            }
            for (col, val) in &constraint.b {
                if !val.is_zero() {
                    b.push((row, *col, val.clone()));
                }
            }
            for (col, val) in &constraint.c {
                if !val.is_zero() {
                    c.push((row, *col, val.clone()));
                }
            }
        }

        Ok(R1csFile {
            prime: header.prime,
            n_wires: header.n_wires,
            n_pub_out: header.n_pub_out,
            n_pub_in: header.n_pub_in,
            n_prv_in: header.n_prv_in,
            n_constraints: header.n_constraints,
            a,
            b,
            c,
        })
    }

    fn parse_header(cursor: &mut Cursor<&[u8]>) -> Result<R1csHeader, R1csError> {
        // Field element size (must be multiple of 8)
        let field_size = read_u32_le(cursor)? as usize;

        // Prime modulus (field_size bytes, little-endian)
        let mut prime_bytes = vec![0u8; field_size];
        cursor.read_exact(&mut prime_bytes)?;
        let prime = BigInt::from_bytes_le(num_bigint::Sign::Plus, &prime_bytes);

        // Wire counts
        let n_wires = read_u32_le(cursor)? as usize;
        let n_pub_out = read_u32_le(cursor)? as usize;
        let n_pub_in = read_u32_le(cursor)? as usize;
        let n_prv_in = read_u32_le(cursor)? as usize;

        // Label count (ignored for now)
        let _n_labels = read_u64_le(cursor)?;

        // Constraint count
        let n_constraints = read_u32_le(cursor)? as usize;

        Ok(R1csHeader {
            field_size,
            prime,
            n_wires,
            n_pub_out,
            n_pub_in,
            n_prv_in,
            n_constraints,
        })
    }

    fn parse_constraints(
        cursor: &mut Cursor<&[u8]>,
        header: &R1csHeader,
    ) -> Result<Vec<R1csConstraint>, R1csError> {
        let mut constraints = Vec::with_capacity(header.n_constraints);

        for _ in 0..header.n_constraints {
            let a = Self::parse_linear_combination(cursor, header.field_size)?;
            let b = Self::parse_linear_combination(cursor, header.field_size)?;
            let c = Self::parse_linear_combination(cursor, header.field_size)?;

            constraints.push(R1csConstraint { a, b, c });
        }

        Ok(constraints)
    }

    fn parse_linear_combination(
        cursor: &mut Cursor<&[u8]>,
        field_size: usize,
    ) -> Result<Vec<(usize, BigInt)>, R1csError> {
        let n_factors = read_u32_le(cursor)? as usize;
        let mut factors = Vec::with_capacity(n_factors);

        for _ in 0..n_factors {
            let wire_id = read_u32_le(cursor)? as usize;

            let mut coeff_bytes = vec![0u8; field_size];
            cursor.read_exact(&mut coeff_bytes)?;
            let coeff = BigInt::from_bytes_le(num_bigint::Sign::Plus, &coeff_bytes);

            factors.push((wire_id, coeff));
        }

        Ok(factors)
    }

    /// Convert A matrix to SparseMatrix format
    pub fn a_matrix(&self) -> SparseMatrix {
        SparseMatrix::from_coo(self.n_constraints, self.n_wires, &self.a)
    }

    /// Convert B matrix to SparseMatrix format
    pub fn b_matrix(&self) -> SparseMatrix {
        SparseMatrix::from_coo(self.n_constraints, self.n_wires, &self.b)
    }

    /// Convert C matrix to SparseMatrix format
    pub fn c_matrix(&self) -> SparseMatrix {
        SparseMatrix::from_coo(self.n_constraints, self.n_wires, &self.c)
    }

    /// Print statistics about the R1CS system
    pub fn stats(&self) -> R1csStats {
        let a_nnz = self.a.len();
        let b_nnz = self.b.len();
        let c_nnz = self.c.len();
        let total_nnz = a_nnz + b_nnz + c_nnz;
        let total_possible = self.n_constraints * self.n_wires * 3;
        let sparsity = if total_possible > 0 {
            1.0 - (total_nnz as f64 / total_possible as f64)
        } else {
            1.0
        };

        R1csStats {
            n_constraints: self.n_constraints,
            n_wires: self.n_wires,
            n_pub_in: self.n_pub_in,
            n_pub_out: self.n_pub_out,
            n_prv_in: self.n_prv_in,
            a_nnz,
            b_nnz,
            c_nnz,
            total_nnz,
            sparsity,
            avg_nnz_per_constraint: if self.n_constraints > 0 {
                total_nnz as f64 / self.n_constraints as f64
            } else {
                0.0
            },
        }
    }
}

/// Header section data
struct R1csHeader {
    field_size: usize,
    prime: BigInt,
    n_wires: usize,
    n_pub_out: usize,
    n_pub_in: usize,
    n_prv_in: usize,
    n_constraints: usize,
}

/// A single R1CS constraint: A × B - C = 0
struct R1csConstraint {
    /// Linear combination A
    a: Vec<(usize, BigInt)>,
    /// Linear combination B
    b: Vec<(usize, BigInt)>,
    /// Linear combination C
    c: Vec<(usize, BigInt)>,
}

/// Statistics about an R1CS system
#[derive(Debug, Clone)]
pub struct R1csStats {
    pub n_constraints: usize,
    pub n_wires: usize,
    pub n_pub_in: usize,
    pub n_pub_out: usize,
    pub n_prv_in: usize,
    pub a_nnz: usize,
    pub b_nnz: usize,
    pub c_nnz: usize,
    pub total_nnz: usize,
    pub sparsity: f64,
    pub avg_nnz_per_constraint: f64,
}

impl std::fmt::Display for R1csStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "R1CS Statistics:")?;
        writeln!(f, "  Constraints:       {:>10}", self.n_constraints)?;
        writeln!(f, "  Wires:             {:>10}", self.n_wires)?;
        writeln!(f, "  Public inputs:     {:>10}", self.n_pub_in)?;
        writeln!(f, "  Public outputs:    {:>10}", self.n_pub_out)?;
        writeln!(f, "  Private inputs:    {:>10}", self.n_prv_in)?;
        writeln!(f, "  A matrix NNZ:      {:>10}", self.a_nnz)?;
        writeln!(f, "  B matrix NNZ:      {:>10}", self.b_nnz)?;
        writeln!(f, "  C matrix NNZ:      {:>10}", self.c_nnz)?;
        writeln!(f, "  Total NNZ:         {:>10}", self.total_nnz)?;
        writeln!(f, "  Sparsity:          {:>10.2}%", self.sparsity * 100.0)?;
        writeln!(f, "  Avg NNZ/constraint:{:>10.1}", self.avg_nnz_per_constraint)?;
        Ok(())
    }
}

/// Read a little-endian u32
fn read_u32_le(cursor: &mut Cursor<&[u8]>) -> Result<u32, R1csError> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

/// Read a little-endian u64
fn read_u64_le(cursor: &mut Cursor<&[u8]>) -> Result<u64, R1csError> {
    let mut buf = [0u8; 8];
    cursor.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// Generate a synthetic R1CS system for testing/benchmarking
///
/// Creates an R1CS that mimics typical ZK circuit characteristics:
/// - ~3 nonzeros per constraint in A
/// - ~1-2 nonzeros per constraint in B
/// - ~1 nonzero per constraint in C
pub fn generate_synthetic_r1cs(n_constraints: usize, n_wires: usize, seed: u64) -> R1csFile {
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut c = Vec::new();

    let mut state = seed;
    let lcg = |s: &mut u64| -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        *s
    };

    // BN254 prime (commonly used in ZK)
    let prime = BigInt::parse_bytes(
        b"21888242871839275222246405745257275088548364400416034343698204186575808495617",
        10,
    ).unwrap();

    for row in 0..n_constraints {
        // A: typically 2-4 nonzeros
        let a_nnz = 2 + (lcg(&mut state) % 3) as usize;
        for _ in 0..a_nnz {
            let col = (lcg(&mut state) as usize) % n_wires;
            let val = BigInt::from((lcg(&mut state) % 100) as i64 + 1);
            a.push((row, col, val));
        }

        // B: typically 1-2 nonzeros
        let b_nnz = 1 + (lcg(&mut state) % 2) as usize;
        for _ in 0..b_nnz {
            let col = (lcg(&mut state) as usize) % n_wires;
            let val = BigInt::from((lcg(&mut state) % 100) as i64 + 1);
            b.push((row, col, val));
        }

        // C: typically 1 nonzero
        let col = (lcg(&mut state) as usize) % n_wires;
        let val = BigInt::from((lcg(&mut state) % 100) as i64 + 1);
        c.push((row, col, val));
    }

    R1csFile {
        prime,
        n_wires,
        n_pub_out: 1,
        n_pub_in: 1,
        n_prv_in: n_wires - 3, // wire 0 is constant, pub_in + pub_out + prv_in = n_wires - 1
        n_constraints,
        a,
        b,
        c,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_r1cs() {
        let r1cs = generate_synthetic_r1cs(100, 50, 42);

        assert_eq!(r1cs.n_constraints, 100);
        assert_eq!(r1cs.n_wires, 50);
        assert!(r1cs.a.len() > 0);
        assert!(r1cs.b.len() > 0);
        assert!(r1cs.c.len() > 0);

        let stats = r1cs.stats();
        println!("{}", stats);

        // Should be sparse
        assert!(stats.sparsity > 0.9);
        // Should have ~3-5 nonzeros per constraint on average
        assert!(stats.avg_nnz_per_constraint > 3.0);
        assert!(stats.avg_nnz_per_constraint < 10.0);
    }

    #[test]
    fn test_to_sparse_matrix() {
        let r1cs = generate_synthetic_r1cs(50, 30, 42);

        let a_sparse = r1cs.a_matrix();
        assert_eq!(a_sparse.nrows, 50);
        assert_eq!(a_sparse.ncols, 30);
        assert!(a_sparse.nnz() > 0);
    }
}
