//! Metal GPU Backend for Parallel Lift
//!
//! Provides GPU-accelerated modular arithmetic using Apple Metal.
//!
//! # Features
//! - Parallel determinant computation across primes
//! - Batched Gaussian elimination
//! - Multi-RHS linear system solving
//! - GPU-accelerated sparse matrix-vector multiply
//! - GPU Wiedemann solver for sparse linear systems
//! - Multi-GPU dispatch for systems with multiple GPUs
//!
//! # Requirements
//! - macOS with Apple Silicon (M1/M2/M3/M4)
//! - Metal framework

#[cfg(target_os = "macos")]
mod metal_backend;
#[cfg(target_os = "macos")]
mod shaders;
#[cfg(target_os = "macos")]
mod gpu_wiedemann;
#[cfg(target_os = "macos")]
mod multi_gpu;

#[cfg(target_os = "macos")]
pub use metal_backend::MetalBackend;
#[cfg(target_os = "macos")]
pub use gpu_wiedemann::{GpuWiedemannSolver, GpuWiedemannStats};
#[cfg(target_os = "macos")]
pub use multi_gpu::{MultiGpuManager, GpuDeviceInfo, MultiGpuStats};

// Stub for non-macOS platforms
#[cfg(not(target_os = "macos"))]
pub struct MetalBackend;

#[cfg(not(target_os = "macos"))]
impl MetalBackend {
    pub fn new() -> Option<Self> {
        None
    }
}
