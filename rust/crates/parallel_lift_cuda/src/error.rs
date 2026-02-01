//! Error types for CUDA backend

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaError {
    #[error("CUDA driver error: {0}")]
    Driver(#[from] cudarc::driver::DriverError),

    #[error("CUDA not available on this system")]
    NotAvailable,

    #[error("Failed to load PTX kernel: {0}")]
    PtxLoad(String),

    #[error("Failed to get kernel function: {0}")]
    KernelNotFound(String),

    #[error("Kernel launch failed: {0}")]
    LaunchFailed(String),

    #[error("Memory allocation failed: {0}")]
    MemoryAlloc(String),

    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch { expected: usize, actual: usize },
}

pub type Result<T> = std::result::Result<T, CudaError>;
