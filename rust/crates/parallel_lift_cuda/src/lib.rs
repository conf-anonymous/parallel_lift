//! CUDA GPU Backend for Parallel Lift
//!
//! Provides GPU-accelerated modular arithmetic using NVIDIA CUDA.
//!
//! # Features
//! - Parallel determinant computation across primes
//! - Batched Gaussian elimination
//! - Multi-RHS linear system solving
//! - GPU-accelerated sparse matrix-vector multiply
//!
//! # Requirements
//! - NVIDIA GPU with CUDA support
//! - CUDA toolkit installed (nvcc)
//! - Linux operating system (tested on RunPod with RTX 4090)

#[cfg(not(feature = "stub"))]
mod cuda_backend;
#[cfg(not(feature = "stub"))]
mod error;
#[cfg(not(feature = "stub"))]
pub mod fhe_gpu;

#[cfg(not(feature = "stub"))]
pub use cuda_backend::{
    CudaBackend, GpuCrtPrecomputed, GpuCrtPrecomputed64, GpuGramSchmidtResult,
    GpuLLL, GpuLLLConfig, GpuLLLStats, TransferTiming, basis_to_residues,
    gpu_lll_reduce, residues_to_basis, residues_to_basis_cpu,
    residues_to_basis_64, residues_to_basis_64_timed, residues_to_basis_64_cpu,
};
#[cfg(not(feature = "stub"))]
pub use error::{CudaError, Result};
#[cfg(not(feature = "stub"))]
pub use fhe_gpu::{FheGpuContext, FheBenchmark};

// Stub for non-CUDA systems
#[cfg(feature = "stub")]
pub struct CudaBackend;

#[cfg(feature = "stub")]
impl CudaBackend {
    pub fn new() -> Option<Self> {
        None
    }
}
