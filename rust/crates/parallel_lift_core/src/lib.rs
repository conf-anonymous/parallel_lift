//! Parallel Lift Core Library
//!
//! CRT-based exact arithmetic primitives for GPU-accelerated computation.
//!
//! # Overview
//!
//! This library provides the core building blocks for exact integer linear algebra
//! using the Chinese Remainder Theorem (CRT) to convert big-integer arithmetic
//! into parallel modular operations.
//!
//! # Key Components
//!
//! - [`primes`] - Prime generation and CRT basis management
//! - [`crt`] - CRT reconstruction algorithms (Garner's algorithm)
//! - [`matrix`] - Dense matrix operations
//! - [`rational`] - Exact rational number type
//! - [`backend`] - Backend trait for CPU/GPU dispatch
//! - [`lattice`] - Lattice basis reduction (LLL, BKZ)

pub mod primes;
pub mod crt;
pub mod matrix;
pub mod rational;
pub mod backend;
pub mod solve;
pub mod determinant;
pub mod sparse;
pub mod adaptive;
pub mod pipeline;
pub mod r1cs;
pub mod lattice;
pub mod fhe_bridge;
pub mod solve_v2;
pub mod solve_hensel;

pub use primes::{PrimeGenerator, CRTBasis};
pub use solve_v2::{PrimeGenerator62, CRTBasis62, CRTReconstruction62, V2Timings, V2SolveResult};
pub use crt::CRTReconstruction;
pub use matrix::Matrix;
pub use rational::Rational;
pub use backend::{Backend, CpuBackend};
pub use solve::Solver;
pub use determinant::Determinant;
pub use sparse::{SparseMatrix, SparseMatrixMod, SparseSolver, WiedemannSolver};
pub use adaptive::{AdaptiveSolver, AdaptiveConfig, AdaptiveStats};
pub use pipeline::{PipelineSolver, PipelineConfig, PipelineStats, AsyncPipelineSolver};
pub use r1cs::{R1csFile, R1csStats, R1csError, generate_synthetic_r1cs};
pub use lattice::{LatticeBasis, GramSchmidt, LLL, LLLConfig, LLLStats};
pub use fhe_bridge::{FheCrtContext, FheLinearSolver, RnsUtils, FheGpuPrecomputed};
pub use solve_hensel::{HenselConfig, HenselResult, HenselTimings, HenselState, hensel_solve_cpu};

/// Timing breakdown for operations
#[derive(Debug, Clone, Default)]
pub struct Timings {
    pub total_time: f64,
    pub residue_time: f64,
    pub solve_time: f64,
    pub det_time: f64,
    pub crt_time: f64,
    pub verify_time: f64,
    pub num_primes: usize,
}

/// Result types for linear algebra operations
#[derive(Debug, Clone)]
pub enum SolveResult {
    /// Unique solution found
    Solution { x: Vec<Rational>, verified: bool },
    /// System is singular (no unique solution)
    Singular { rank: usize },
    /// System is inconsistent (overdetermined, no solution)
    Inconsistent { rank_a: usize, rank_aug: usize },
}

#[derive(Debug, Clone)]
pub enum InverseResult {
    /// Inverse computed successfully
    Success { matrix: Vec<Rational>, determinant: num_bigint::BigInt },
    /// Matrix is singular
    Singular { rank: usize },
}
