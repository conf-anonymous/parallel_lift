//! Lattice basis reduction algorithms
//!
//! GPU-accelerated exact arithmetic for lattice operations using CRT.
//!
//! # Overview
//!
//! This module provides LLL and BKZ lattice basis reduction algorithms
//! accelerated via CRT-based GPU computation. The key insight is that
//! Gram-Schmidt coefficients are rationals with bounded denominators,
//! allowing exact CRT representation.
//!
//! # Key Components
//!
//! - [`LatticeBasis`] - Lattice basis representation with CRT support
//! - [`GramSchmidt`] - Gram-Schmidt orthogonalization via CRT
//! - [`LLL`] - LLL lattice reduction algorithm
//!
//! # Example
//!
//! ```ignore
//! use parallel_lift_core::lattice::{LatticeBasis, LLL, LLLConfig};
//! use parallel_lift_core::CpuBackend;
//!
//! let basis = LatticeBasis::from_rows(&[
//!     vec![1, 0, 3],
//!     vec![0, 1, 5],
//!     vec![0, 0, 7],
//! ]);
//!
//! let backend = CpuBackend::new();
//! let config = LLLConfig::default();
//! let reduced = LLL::reduce(&basis, &config, &backend);
//! ```

pub mod basis;
pub mod gram_schmidt;
pub mod lll;

pub use basis::LatticeBasis;
pub use gram_schmidt::{GramSchmidt, GramSchmidtCRT};
pub use lll::{LLL, LLLConfig, LLLStats};
