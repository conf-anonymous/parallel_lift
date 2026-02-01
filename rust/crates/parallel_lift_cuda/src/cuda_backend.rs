//! CUDA backend implementation
//!
//! GPU-accelerated backend for CRT-based exact arithmetic using NVIDIA CUDA.
//! Key design: One thread per prime - batch all primes and dispatch to GPU in parallel.

use cudarc::driver::{CudaDevice, CudaSlice, CudaFunction, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use parallel_lift_core::Backend;
use std::sync::Arc;
use std::time::Instant;

use crate::error::{CudaError, Result};

/// Timing breakdown for GPU operations
///
/// Measures Host→Device transfer, kernel compute, and Device→Host transfer
/// times separately to analyze PCIe transfer overhead.
#[derive(Debug, Clone)]
pub struct TransferTiming {
    /// Time to prepare data on CPU (reduction mod p, etc.)
    pub prepare_ms: f64,
    /// Time to transfer data from Host to Device
    pub htod_ms: f64,
    /// Time for GPU kernel computation
    pub compute_ms: f64,
    /// Time to transfer results from Device to Host
    pub dtoh_ms: f64,
    /// Total end-to-end time
    pub total_ms: f64,
    /// Number of bytes transferred Host→Device
    pub htod_bytes: usize,
    /// Number of bytes transferred Device→Host
    pub dtoh_bytes: usize,
}

/// PTX kernel source compiled at build time
const KERNELS_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

/// CUDA GPU backend for modular arithmetic
///
/// This backend accelerates CRT-based exact arithmetic by:
/// 1. Batching matrices for all primes and dispatching to GPU
/// 2. Each GPU thread handles one prime independently
/// 3. For multi-RHS, factors A once and solves all k RHS vectors
pub struct CudaBackend {
    device: Arc<CudaDevice>,
}

impl CudaBackend {
    /// Create a new CUDA backend
    ///
    /// Returns None if CUDA is not available
    pub fn new() -> Option<Self> {
        Self::try_new().ok()
    }

    /// Try to create a new CUDA backend, returning detailed error on failure
    pub fn try_new() -> Result<Self> {
        let device = CudaDevice::new(0)?;

        // Load compiled PTX
        device.load_ptx(
            Ptx::from_src(KERNELS_PTX),
            "parallel_lift",
            &[
                "modular_determinant",
                "modular_determinant_small",
                "modular_determinant_tiled",
                "modular_solve",
                "modular_solve_multi_rhs",
                "modular_solve_multi_rhs_tiled",
                "sparse_matvec_csr",
                "sparse_matvec_csr_single",
                "crt_garner_step",
                "bigint_mod_prime",
                "bigint_compare_half",
                "crt_reconstruct_full",
                "crt_to_signed",
                // Gram-Schmidt / LLL kernels
                "batch_gram_matrix",
                "batch_inner_product_gs",
                "batch_size_reduce",
                "batch_swap_vectors",
                "batch_squared_norms",
                "gram_schmidt_step",
                "check_lovasz_condition",
                // V2: 64-bit prime kernels
                "modular_solve_multi_rhs_64",
                "modular_solve_multi_rhs_tiled_64",
                // V2: 64-bit CRT reconstruction
                "crt_reconstruct_full_64",
                "crt_to_signed_64",
                // Hensel (Dixon) p-adic lifting
                "hensel_matrix_inverse",
                "hensel_batch_matrix_inverse",
                "hensel_matvec",
                "hensel_initial_solve",
                "hensel_lift_iteration",
                "hensel_lift_simple",
                "hensel_solve_full",
                // GPU-native Hensel lifting (defect-based)
                "hensel_gpu_init_defect",
                "hensel_gpu_compute_digit",
                "hensel_gpu_update_defect",
                "hensel_gpu_compute_digit_tiled",
                "hensel_gpu_update_defect_tiled",
                "hensel_gpu_lift_step",
                // All-on-GPU lifting with offset (no per-iteration transfers)
                "hensel_gpu_compute_digit_offset",
                "hensel_gpu_update_defect_offset",
                // Fully fused kernel (single launch for all iterations)
                "hensel_gpu_lift_all_fused",
            ],
        )?;

        Ok(Self {
            device,
        })
    }

    /// Get a kernel function by name
    fn get_kernel(&self, name: &str) -> Result<CudaFunction> {
        self.device
            .get_func("parallel_lift", name)
            .ok_or_else(|| CudaError::KernelNotFound(name.to_string()))
    }

    /// GPU batched determinant: compute det(A) mod p for all primes in parallel
    fn gpu_batch_determinant(&self, matrix: &[u32], n: usize, primes: &[u32]) -> Vec<u32> {
        let num_primes = primes.len();
        let nn = n * n;

        // Prepare batched matrices: reduce matrix mod each prime
        let mut matrices_data = vec![0u32; num_primes * nn];
        for (pi, &p) in primes.iter().enumerate() {
            for i in 0..nn {
                matrices_data[pi * nn + i] = matrix[i] % p;
            }
        }

        // Allocate device memory
        let d_matrices = self.device.htod_copy(matrices_data).unwrap();
        let d_primes = self.device.htod_copy(primes.to_vec()).unwrap();
        let d_results: CudaSlice<u32> = self.device.alloc_zeros(num_primes).unwrap();
        let d_singular: CudaSlice<u32> = self.device.alloc_zeros(num_primes).unwrap();
        let d_workspace: CudaSlice<u32> = self.device.alloc_zeros(num_primes * nn).unwrap();

        // Choose kernel based on matrix size
        let use_tiled = n >= 32;

        if use_tiled {
            // Tiled: 1 threadblock (16x16) per prime
            let kernel = self.get_kernel("modular_determinant_tiled").unwrap();
            let config = LaunchConfig {
                grid_dim: (num_primes as u32, 1, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                kernel
                    .launch(
                        config,
                        (
                            &d_matrices,
                            &d_primes,
                            &d_results,
                            &d_singular,
                            &d_workspace,
                            n as u32,
                            num_primes as u32,
                        ),
                    )
                    .unwrap();
            }
        } else if n <= 16 {
            // Small: thread-local storage
            let kernel = self.get_kernel("modular_determinant_small").unwrap();
            let threads_per_block = 256.min(num_primes);
            let blocks = (num_primes + threads_per_block - 1) / threads_per_block;

            let config = LaunchConfig {
                grid_dim: (blocks as u32, 1, 1),
                block_dim: (threads_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                kernel
                    .launch(
                        config,
                        (
                            &d_matrices,
                            &d_primes,
                            &d_results,
                            &d_singular,
                            n as u32,
                            num_primes as u32,
                        ),
                    )
                    .unwrap();
            }
        } else {
            // Serial: 1 thread per prime
            let kernel = self.get_kernel("modular_determinant").unwrap();
            let threads_per_block = 256.min(num_primes);
            let blocks = (num_primes + threads_per_block - 1) / threads_per_block;

            let config = LaunchConfig {
                grid_dim: (blocks as u32, 1, 1),
                block_dim: (threads_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                kernel
                    .launch(
                        config,
                        (
                            &d_matrices,
                            &d_primes,
                            &d_results,
                            &d_singular,
                            &d_workspace,
                            n as u32,
                            num_primes as u32,
                        ),
                    )
                    .unwrap();
            }
        }

        // Copy results back
        self.device.dtoh_sync_copy(&d_results).unwrap()
    }

    /// GPU batched solve: solve Ax = b mod p for all primes in parallel
    fn gpu_batch_solve(
        &self,
        matrix: &[u32],
        b: &[u32],
        n: usize,
        primes: &[u32],
    ) -> Option<Vec<Vec<u32>>> {
        let num_primes = primes.len();
        let aug_width = n + 1;
        let aug_stride = n * aug_width;

        // Prepare batched augmented matrices [A|b] for each prime
        let mut augmented_data = vec![0u32; num_primes * aug_stride];
        for (pi, &p) in primes.iter().enumerate() {
            let offset = pi * aug_stride;
            for row in 0..n {
                // Copy A row
                for col in 0..n {
                    augmented_data[offset + row * aug_width + col] = matrix[row * n + col] % p;
                }
                // Copy b element
                augmented_data[offset + row * aug_width + n] = b[row] % p;
            }
        }

        // Allocate device memory
        let d_augmented = self.device.htod_copy(augmented_data).unwrap();
        let d_primes = self.device.htod_copy(primes.to_vec()).unwrap();
        let d_solutions: CudaSlice<u32> = self.device.alloc_zeros(num_primes * n).unwrap();
        let d_singular: CudaSlice<u32> = self.device.alloc_zeros(num_primes).unwrap();
        let d_workspace: CudaSlice<u32> = self.device.alloc_zeros(num_primes * aug_stride).unwrap();

        let kernel = self.get_kernel("modular_solve").unwrap();
        let threads_per_block = 256.min(num_primes);
        let blocks = (num_primes + threads_per_block - 1) / threads_per_block;

        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel
                .launch(
                    config,
                    (
                        &d_augmented,
                        &d_primes,
                        &d_solutions,
                        &d_singular,
                        &d_workspace,
                        n as u32,
                        num_primes as u32,
                    ),
                )
                .unwrap();
        }

        // Read results
        let solutions: Vec<u32> = self.device.dtoh_sync_copy(&d_solutions).unwrap();
        let singular_flags: Vec<u32> = self.device.dtoh_sync_copy(&d_singular).unwrap();

        // Check for singular matrices
        if singular_flags.iter().any(|&f| f != 0) {
            return None;
        }

        // Split into per-prime solutions
        let result: Vec<Vec<u32>> = (0..num_primes)
            .map(|pi| solutions[pi * n..(pi + 1) * n].to_vec())
            .collect();

        Some(result)
    }

    /// GPU batched multi-RHS solve: solve AX = B mod p for all primes
    /// B has k columns, so we factor A once and solve for all k RHS vectors
    fn gpu_batch_multi_rhs_solve(
        &self,
        matrix: &[u32],
        b_cols: &[Vec<u32>],
        n: usize,
        k: usize,
        primes: &[u32],
    ) -> Option<Vec<Vec<Vec<u32>>>> {
        let num_primes = primes.len();
        let aug_width = n + k;
        let aug_stride = n * aug_width;

        // Prepare batched augmented matrices [A|B] for each prime
        let mut augmented_data = vec![0u32; num_primes * aug_stride];
        for (pi, &p) in primes.iter().enumerate() {
            let offset = pi * aug_stride;
            for row in 0..n {
                // Copy A row
                for col in 0..n {
                    augmented_data[offset + row * aug_width + col] = matrix[row * n + col] % p;
                }
                // Copy B columns
                for col_idx in 0..k {
                    augmented_data[offset + row * aug_width + n + col_idx] =
                        b_cols[col_idx][row] % p;
                }
            }
        }

        // Allocate device memory
        let d_augmented = self.device.htod_copy(augmented_data).unwrap();
        let d_primes = self.device.htod_copy(primes.to_vec()).unwrap();
        let d_solutions: CudaSlice<u32> = self.device.alloc_zeros(num_primes * n * k).unwrap();
        let d_singular: CudaSlice<u32> = self.device.alloc_zeros(num_primes).unwrap();
        let d_workspace: CudaSlice<u32> = self.device.alloc_zeros(num_primes * aug_stride).unwrap();

        // Choose kernel based on matrix size
        let use_tiled = n >= 32;

        if use_tiled {
            // Tiled: 1 threadblock (16x16) per prime
            let kernel = self.get_kernel("modular_solve_multi_rhs_tiled").unwrap();
            let config = LaunchConfig {
                grid_dim: (num_primes as u32, 1, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                kernel
                    .launch(
                        config,
                        (
                            &d_augmented,
                            &d_primes,
                            &d_solutions,
                            &d_singular,
                            &d_workspace,
                            n as u32,
                            k as u32,
                            num_primes as u32,
                        ),
                    )
                    .unwrap();
            }
        } else {
            let kernel = self.get_kernel("modular_solve_multi_rhs").unwrap();
            let threads_per_block = 256.min(num_primes);
            let blocks = (num_primes + threads_per_block - 1) / threads_per_block;

            let config = LaunchConfig {
                grid_dim: (blocks as u32, 1, 1),
                block_dim: (threads_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                kernel
                    .launch(
                        config,
                        (
                            &d_augmented,
                            &d_primes,
                            &d_solutions,
                            &d_singular,
                            &d_workspace,
                            n as u32,
                            k as u32,
                            num_primes as u32,
                        ),
                    )
                    .unwrap();
            }
        }

        // Read results
        let solutions: Vec<u32> = self.device.dtoh_sync_copy(&d_solutions).unwrap();
        let singular_flags: Vec<u32> = self.device.dtoh_sync_copy(&d_singular).unwrap();

        // Check for singular matrices
        if singular_flags.iter().any(|&f| f != 0) {
            return None;
        }

        // Parse results: solutions are stored column-major per prime
        // solutions[pi * n * k + col * n + row] = X[col][row] mod primes[pi]
        let mut result = Vec::with_capacity(num_primes);
        for pi in 0..num_primes {
            let prime_offset = pi * n * k;
            let mut cols = Vec::with_capacity(k);
            for col_idx in 0..k {
                let col_offset = prime_offset + col_idx * n;
                cols.push(solutions[col_offset..col_offset + n].to_vec());
            }
            result.push(cols);
        }

        Some(result)
    }

    /// V2: GPU batch multi-RHS solve with 64-bit primes
    ///
    /// Uses 62-bit primes for roughly 2x fewer primes than 31-bit version.
    /// Returns solutions as u64 residues per prime.
    pub fn gpu_batch_multi_rhs_solve_64(
        &self,
        matrix: &[u64],
        b_cols: &[Vec<u64>],
        n: usize,
        k: usize,
        primes: &[u64],
    ) -> Option<Vec<Vec<Vec<u64>>>> {
        let num_primes = primes.len();
        let aug_width = n + k;
        let aug_stride = n * aug_width;

        // Prepare batched augmented matrices [A|B] for each prime
        let mut augmented_data = vec![0u64; num_primes * aug_stride];
        for (pi, &p) in primes.iter().enumerate() {
            let offset = pi * aug_stride;
            for row in 0..n {
                // Copy A row (already reduced mod p)
                for col in 0..n {
                    augmented_data[offset + row * aug_width + col] = matrix[row * n + col] % p;
                }
                // Copy B columns
                for col_idx in 0..k {
                    augmented_data[offset + row * aug_width + n + col_idx] =
                        b_cols[col_idx][row] % p;
                }
            }
        }

        // Allocate device memory
        let d_augmented = self.device.htod_copy(augmented_data).unwrap();
        let d_primes = self.device.htod_copy(primes.to_vec()).unwrap();
        let d_solutions: CudaSlice<u64> = self.device.alloc_zeros(num_primes * n * k).unwrap();
        let d_singular: CudaSlice<u32> = self.device.alloc_zeros(num_primes).unwrap();
        let d_workspace: CudaSlice<u64> = self.device.alloc_zeros(num_primes * aug_stride).unwrap();

        // Choose kernel based on matrix size
        let use_tiled = n >= 32;

        if use_tiled {
            // Tiled: 1 threadblock (16x16) per prime
            let kernel = self.get_kernel("modular_solve_multi_rhs_tiled_64").unwrap();
            let config = LaunchConfig {
                grid_dim: (num_primes as u32, 1, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                kernel
                    .launch(
                        config,
                        (
                            &d_augmented,
                            &d_primes,
                            &d_solutions,
                            &d_singular,
                            &d_workspace,
                            n as u32,
                            k as u32,
                            num_primes as u32,
                        ),
                    )
                    .unwrap();
            }
        } else {
            let kernel = self.get_kernel("modular_solve_multi_rhs_64").unwrap();
            let threads_per_block = 256.min(num_primes);
            let blocks = (num_primes + threads_per_block - 1) / threads_per_block;

            let config = LaunchConfig {
                grid_dim: (blocks as u32, 1, 1),
                block_dim: (threads_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                kernel
                    .launch(
                        config,
                        (
                            &d_augmented,
                            &d_primes,
                            &d_solutions,
                            &d_singular,
                            &d_workspace,
                            n as u32,
                            k as u32,
                            num_primes as u32,
                        ),
                    )
                    .unwrap();
            }
        }

        // Read results
        let solutions: Vec<u64> = self.device.dtoh_sync_copy(&d_solutions).unwrap();
        let singular_flags: Vec<u32> = self.device.dtoh_sync_copy(&d_singular).unwrap();

        // Check for singular matrices
        if singular_flags.iter().any(|&f| f != 0) {
            return None;
        }

        // Parse results: solutions are stored column-major per prime
        let mut result = Vec::with_capacity(num_primes);
        for pi in 0..num_primes {
            let prime_offset = pi * n * k;
            let mut cols = Vec::with_capacity(k);
            for col_idx in 0..k {
                let col_offset = prime_offset + col_idx * n;
                cols.push(solutions[col_offset..col_offset + n].to_vec());
            }
            result.push(cols);
        }

        Some(result)
    }

    /// V2: GPU batch multi-RHS solve with 64-bit primes and timing
    pub fn gpu_batch_multi_rhs_solve_64_timed(
        &self,
        matrix: &[u64],
        b_cols: &[Vec<u64>],
        n: usize,
        k: usize,
        primes: &[u64],
    ) -> Option<(Vec<Vec<Vec<u64>>>, TransferTiming)> {
        use std::time::Instant;
        let total_start = Instant::now();

        let num_primes = primes.len();
        let aug_width = n + k;
        let aug_stride = n * aug_width;

        // Prepare phase
        let prepare_start = Instant::now();
        let mut augmented_data = vec![0u64; num_primes * aug_stride];
        for (pi, &p) in primes.iter().enumerate() {
            let offset = pi * aug_stride;
            for row in 0..n {
                for col in 0..n {
                    augmented_data[offset + row * aug_width + col] = matrix[row * n + col] % p;
                }
                for col_idx in 0..k {
                    augmented_data[offset + row * aug_width + n + col_idx] =
                        b_cols[col_idx][row] % p;
                }
            }
        }
        let prepare_ms = prepare_start.elapsed().as_secs_f64() * 1000.0;

        // H→D transfer
        let htod_start = Instant::now();
        let d_augmented = self.device.htod_copy(augmented_data).unwrap();
        let d_primes = self.device.htod_copy(primes.to_vec()).unwrap();
        let d_solutions: CudaSlice<u64> = self.device.alloc_zeros(num_primes * n * k).unwrap();
        let d_singular: CudaSlice<u32> = self.device.alloc_zeros(num_primes).unwrap();
        let d_workspace: CudaSlice<u64> = self.device.alloc_zeros(num_primes * aug_stride).unwrap();
        let htod_ms = htod_start.elapsed().as_secs_f64() * 1000.0;
        let htod_bytes = (num_primes * aug_stride + num_primes) * 8;

        // Compute phase
        let compute_start = Instant::now();
        let use_tiled = n >= 32;

        if use_tiled {
            let kernel = self.get_kernel("modular_solve_multi_rhs_tiled_64").unwrap();
            let config = LaunchConfig {
                grid_dim: (num_primes as u32, 1, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                kernel.launch(config, (
                    &d_augmented, &d_primes, &d_solutions, &d_singular, &d_workspace,
                    n as u32, k as u32, num_primes as u32,
                )).unwrap();
            }
        } else {
            let kernel = self.get_kernel("modular_solve_multi_rhs_64").unwrap();
            let threads_per_block = 256.min(num_primes);
            let blocks = (num_primes + threads_per_block - 1) / threads_per_block;
            let config = LaunchConfig {
                grid_dim: (blocks as u32, 1, 1),
                block_dim: (threads_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                kernel.launch(config, (
                    &d_augmented, &d_primes, &d_solutions, &d_singular, &d_workspace,
                    n as u32, k as u32, num_primes as u32,
                )).unwrap();
            }
        }
        self.device.synchronize().unwrap();
        let compute_ms = compute_start.elapsed().as_secs_f64() * 1000.0;

        // D→H transfer
        let dtoh_start = Instant::now();
        let solutions: Vec<u64> = self.device.dtoh_sync_copy(&d_solutions).unwrap();
        let singular_flags: Vec<u32> = self.device.dtoh_sync_copy(&d_singular).unwrap();
        let dtoh_ms = dtoh_start.elapsed().as_secs_f64() * 1000.0;
        let dtoh_bytes = num_primes * n * k * 8 + num_primes * 4;

        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        let timing = TransferTiming {
            prepare_ms,
            htod_ms,
            compute_ms,
            dtoh_ms,
            total_ms,
            htod_bytes,
            dtoh_bytes,
        };

        if singular_flags.iter().any(|&f| f != 0) {
            return None;
        }

        let mut result = Vec::with_capacity(num_primes);
        for pi in 0..num_primes {
            let prime_offset = pi * n * k;
            let mut cols = Vec::with_capacity(k);
            for col_idx in 0..k {
                let col_offset = prime_offset + col_idx * n;
                cols.push(solutions[col_offset..col_offset + n].to_vec());
            }
            result.push(cols);
        }

        Some((result, timing))
    }

    /// CPU fallback for single-prime determinant
    fn cpu_determinant(&self, matrix: &[u32], n: usize, p: u32) -> u32 {
        let p64 = p as u64;
        let mut m: Vec<u64> = matrix.iter().map(|&x| (x % p) as u64).collect();
        let mut det: u64 = 1;
        let mut sign = 1i64;

        for col in 0..n {
            // Find pivot
            let mut pivot_row = None;
            for row in col..n {
                if m[row * n + col] != 0 {
                    pivot_row = Some(row);
                    break;
                }
            }

            let pivot_row = match pivot_row {
                Some(r) => r,
                None => return 0,
            };

            if pivot_row != col {
                for j in 0..n {
                    m.swap(col * n + j, pivot_row * n + j);
                }
                sign = -sign;
            }

            let pivot_val = m[col * n + col];
            det = (det * pivot_val) % p64;

            let pivot_inv = match Self::mod_inverse(pivot_val, p64) {
                Some(inv) => inv,
                None => return 0,
            };

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

    fn cpu_solve(&self, matrix: &[u32], b: &[u32], n: usize, p: u32) -> Option<Vec<u32>> {
        let p64 = p as u64;
        let mut m: Vec<u64> = matrix.iter().map(|&x| (x % p) as u64).collect();
        let mut rhs: Vec<u64> = b.iter().map(|&x| (x % p) as u64).collect();

        // Forward elimination
        for col in 0..n {
            let mut pivot_row = None;
            for row in col..n {
                if m[row * n + col] != 0 {
                    pivot_row = Some(row);
                    break;
                }
            }

            let pivot_row = pivot_row?;

            if pivot_row != col {
                for j in 0..n {
                    m.swap(col * n + j, pivot_row * n + j);
                }
                rhs.swap(col, pivot_row);
            }

            let pivot_val = m[col * n + col];
            let pivot_inv = Self::mod_inverse(pivot_val, p64)?;

            for j in 0..n {
                m[col * n + j] = (m[col * n + j] * pivot_inv) % p64;
            }
            rhs[col] = (rhs[col] * pivot_inv) % p64;

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

    /// GPU sparse matrix-vector multiply: y = A * x mod p for single prime
    pub fn sparse_matvec_single(
        &self,
        row_ptr: &[u32],
        col_idx: &[u32],
        values: &[u32],
        x: &[u32],
        p: u32,
    ) -> Vec<u32> {
        let n = row_ptr.len() - 1;
        let _nnz = col_idx.len();

        // For very small matrices, CPU might be faster
        if n < 64 || _nnz < 256 {
            return self.cpu_sparse_matvec(row_ptr, col_idx, values, x, p);
        }

        let d_row_ptr = self.device.htod_copy(row_ptr.to_vec()).unwrap();
        let d_col_idx = self.device.htod_copy(col_idx.to_vec()).unwrap();
        let d_values = self.device.htod_copy(values.to_vec()).unwrap();
        let d_x = self.device.htod_copy(x.to_vec()).unwrap();
        let d_y: CudaSlice<u32> = self.device.alloc_zeros(n).unwrap();

        let kernel = self.get_kernel("sparse_matvec_csr_single").unwrap();
        let threads_per_block = 256.min(n);
        let blocks = (n + threads_per_block - 1) / threads_per_block;

        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel
                .launch(
                    config,
                    (&d_row_ptr, &d_col_idx, &d_values, &d_x, &d_y, p, n as u32),
                )
                .unwrap();
        }

        self.device.dtoh_sync_copy(&d_y).unwrap()
    }

    /// CPU fallback for sparse matvec
    fn cpu_sparse_matvec(
        &self,
        row_ptr: &[u32],
        col_idx: &[u32],
        values: &[u32],
        x: &[u32],
        p: u32,
    ) -> Vec<u32> {
        let n = row_ptr.len() - 1;
        let p64 = p as u64;
        let mut y = vec![0u32; n];

        for row in 0..n {
            let start = row_ptr[row] as usize;
            let end = row_ptr[row + 1] as usize;
            let mut sum = 0u64;

            for j in start..end {
                let col = col_idx[j] as usize;
                sum += (values[j] as u64) * (x[col] as u64);
            }

            y[row] = (sum % p64) as u32;
        }

        y
    }

    /// GPU-accelerated CRT reconstruction
    ///
    /// Reconstructs BigInts from residues using Garner's algorithm on the GPU.
    /// Returns (limbs, signs) where each value is represented as:
    /// - limbs: Vec<u32> little-endian limbs
    /// - signs: bool indicating if the value is negative (for signed reconstruction)
    pub fn gpu_crt_reconstruct(
        &self,
        residues: &[u32],      // [num_values * num_primes] row-major
        num_values: usize,
        precomputed: &GpuCrtPrecomputed,
    ) -> (Vec<Vec<u32>>, Vec<bool>) {
        let num_primes = precomputed.primes.len();
        let max_acc_limbs = precomputed.max_limbs;

        // Upload input data to GPU
        let d_residues = self.device.htod_copy(residues.to_vec()).unwrap();
        let d_primes = self.device.htod_copy(precomputed.primes.clone()).unwrap();
        let d_inverses = self.device.htod_copy(precomputed.garner_inverses.clone()).unwrap();
        let d_pp_limbs = self.device.htod_copy(precomputed.pp_limbs.clone()).unwrap();
        let d_pp_offsets = self.device.htod_copy(precomputed.pp_offsets.clone()).unwrap();
        let d_pp_sizes = self.device.htod_copy(precomputed.pp_sizes.clone()).unwrap();
        let d_pow2_mod = self.device.htod_copy(precomputed.pow2_mod.clone()).unwrap();

        // Allocate output buffers
        let d_output_limbs: CudaSlice<u32> = self.device
            .alloc_zeros(num_values * max_acc_limbs)
            .unwrap();
        let d_output_sizes: CudaSlice<u32> = self.device.alloc_zeros(num_values).unwrap();

        // Launch CRT reconstruction kernel
        let kernel = self.get_kernel("crt_reconstruct_full").unwrap();
        let threads_per_block = 256;
        let blocks = (num_values + threads_per_block - 1) / threads_per_block;

        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel
                .launch(
                    config,
                    (
                        &d_residues,
                        &d_primes,
                        &d_inverses,
                        &d_pp_limbs,
                        &d_pp_offsets,
                        &d_pp_sizes,
                        &d_pow2_mod,
                        &d_output_limbs,
                        &d_output_sizes,
                        num_values as u32,
                        num_primes as u32,
                        max_acc_limbs as u32,
                    ),
                )
                .unwrap();
        }

        // Now apply sign detection
        let d_half_product = self.device.htod_copy(precomputed.half_product_limbs.clone()).unwrap();
        let d_full_product = self.device.htod_copy(precomputed.product_limbs.clone()).unwrap();
        let d_signs: CudaSlice<u32> = self.device.alloc_zeros(num_values).unwrap();

        let sign_kernel = self.get_kernel("crt_to_signed").unwrap();

        unsafe {
            sign_kernel
                .launch(
                    config,
                    (
                        &d_output_limbs,
                        &d_half_product,
                        &d_full_product,
                        &d_signs,
                        num_values as u32,
                        max_acc_limbs as u32,
                        precomputed.half_product_limbs.len() as u32,
                        precomputed.product_limbs.len() as u32,
                    ),
                )
                .unwrap();
        }

        // Copy results back
        let output_limbs: Vec<u32> = self.device.dtoh_sync_copy(&d_output_limbs).unwrap();
        let output_sizes: Vec<u32> = self.device.dtoh_sync_copy(&d_output_sizes).unwrap();
        let signs: Vec<u32> = self.device.dtoh_sync_copy(&d_signs).unwrap();

        // Parse into per-value limbs
        let mut results = Vec::with_capacity(num_values);
        for v in 0..num_values {
            let size = output_sizes[v] as usize;
            let start = v * max_acc_limbs;
            // Trim trailing zeros
            let mut limbs = output_limbs[start..start + size].to_vec();
            while limbs.len() > 1 && limbs.last() == Some(&0) {
                limbs.pop();
            }
            results.push(limbs);
        }

        let sign_flags: Vec<bool> = signs.iter().map(|&s| s != 0).collect();

        (results, sign_flags)
    }

    /// GPU-accelerated CRT reconstruction for 64-bit primes (V2)
    ///
    /// Reconstructs BigInts from 64-bit residues using Garner's algorithm on the GPU.
    /// Returns (limbs, signs) where each value is represented as:
    /// - limbs: Vec<u32> little-endian limbs
    /// - signs: bool indicating if the value is negative (for signed reconstruction)
    pub fn gpu_crt_reconstruct_64(
        &self,
        residues: &[u64],      // [num_values * num_primes] row-major
        num_values: usize,
        precomputed: &GpuCrtPrecomputed64,
    ) -> (Vec<Vec<u32>>, Vec<bool>) {
        let num_primes = precomputed.primes.len();
        let max_acc_limbs = precomputed.max_limbs;

        // Upload input data to GPU
        let d_residues = self.device.htod_copy(residues.to_vec()).unwrap();
        let d_primes = self.device.htod_copy(precomputed.primes.clone()).unwrap();
        let d_inverses = self.device.htod_copy(precomputed.garner_inverses.clone()).unwrap();
        let d_pp_limbs = self.device.htod_copy(precomputed.pp_limbs.clone()).unwrap();
        let d_pp_offsets = self.device.htod_copy(precomputed.pp_offsets.clone()).unwrap();
        let d_pp_sizes = self.device.htod_copy(precomputed.pp_sizes.clone()).unwrap();
        let d_pow2_mod = self.device.htod_copy(precomputed.pow2_mod.clone()).unwrap();

        // Allocate output buffers
        let d_output_limbs: CudaSlice<u32> = self.device
            .alloc_zeros(num_values * max_acc_limbs)
            .unwrap();
        let d_output_sizes: CudaSlice<u32> = self.device.alloc_zeros(num_values).unwrap();

        // Launch CRT reconstruction kernel
        let kernel = self.get_kernel("crt_reconstruct_full_64").unwrap();
        let threads_per_block = 256;
        let blocks = (num_values + threads_per_block - 1) / threads_per_block;

        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel
                .launch(
                    config,
                    (
                        &d_residues,
                        &d_primes,
                        &d_inverses,
                        &d_pp_limbs,
                        &d_pp_offsets,
                        &d_pp_sizes,
                        &d_pow2_mod,
                        &d_output_limbs,
                        &d_output_sizes,
                        num_values as u32,
                        num_primes as u32,
                        max_acc_limbs as u32,
                    ),
                )
                .unwrap();
        }

        // Now apply sign detection (same kernel works for both 32-bit and 64-bit CRT output)
        let d_half_product = self.device.htod_copy(precomputed.half_product_limbs.clone()).unwrap();
        let d_full_product = self.device.htod_copy(precomputed.product_limbs.clone()).unwrap();
        let d_signs: CudaSlice<u32> = self.device.alloc_zeros(num_values).unwrap();

        let sign_kernel = self.get_kernel("crt_to_signed_64").unwrap();

        unsafe {
            sign_kernel
                .launch(
                    config,
                    (
                        &d_output_limbs,
                        &d_half_product,
                        &d_full_product,
                        &d_signs,
                        num_values as u32,
                        max_acc_limbs as u32,
                        precomputed.half_product_limbs.len() as u32,
                        precomputed.product_limbs.len() as u32,
                    ),
                )
                .unwrap();
        }

        // Copy results back
        let output_limbs: Vec<u32> = self.device.dtoh_sync_copy(&d_output_limbs).unwrap();
        let output_sizes: Vec<u32> = self.device.dtoh_sync_copy(&d_output_sizes).unwrap();
        let signs: Vec<u32> = self.device.dtoh_sync_copy(&d_signs).unwrap();

        // Parse into per-value limbs
        let mut results = Vec::with_capacity(num_values);
        for v in 0..num_values {
            let size = output_sizes[v] as usize;
            let start = v * max_acc_limbs;
            // Trim trailing zeros
            let mut limbs = output_limbs[start..start + size].to_vec();
            while limbs.len() > 1 && limbs.last() == Some(&0) {
                limbs.pop();
            }
            results.push(limbs);
        }

        let sign_flags: Vec<bool> = signs.iter().map(|&s| s != 0).collect();

        (results, sign_flags)
    }

    /// GPU-accelerated CRT reconstruction for 64-bit primes with timing
    pub fn gpu_crt_reconstruct_64_timed(
        &self,
        residues: &[u64],
        num_values: usize,
        precomputed: &GpuCrtPrecomputed64,
    ) -> ((Vec<Vec<u32>>, Vec<bool>), f64) {
        let start = Instant::now();
        let result = self.gpu_crt_reconstruct_64(residues, num_values, precomputed);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        (result, elapsed_ms)
    }

    /// Access the underlying CUDA device for advanced operations
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    // =========================================================================
    // Hensel (Dixon) p-adic Lifting Operations
    // =========================================================================

    /// Compute matrix inverse mod p using GPU
    ///
    /// Returns A⁻¹ mod p, or None if the matrix is singular mod p.
    pub fn gpu_matrix_inverse_mod(
        &self,
        matrix: &[u32],
        n: usize,
        p: u32,
    ) -> Option<Vec<u32>> {
        // For matrices requiring too much shared memory (> 48KB), use CPU
        // Shared memory: n * 2n * 4 bytes
        let shared_mem_needed = n * 2 * n * std::mem::size_of::<u32>();
        if n <= 4 || shared_mem_needed > 48 * 1024 {
            return Self::cpu_matrix_inverse_mod(matrix, n, p);
        }

        // Upload matrix to GPU
        let d_matrix = self.device.htod_copy(matrix.to_vec()).unwrap();
        let d_inverse: CudaSlice<u32> = self.device.alloc_zeros(n * n).unwrap();
        let d_singular: CudaSlice<u32> = self.device.alloc_zeros(1).unwrap();

        let kernel = self.get_kernel("hensel_matrix_inverse").unwrap();

        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (n as u32, 1, 1),
            shared_mem_bytes: shared_mem_needed as u32,
        };

        unsafe {
            kernel
                .launch(
                    config,
                    (
                        &d_matrix,
                        &d_inverse,
                        &d_singular,
                        p,
                        n as u32,
                    ),
                )
                .unwrap();
        }

        // Check singularity
        let singular: Vec<u32> = self.device.dtoh_sync_copy(&d_singular).unwrap();
        if singular[0] != 0 {
            return None;
        }

        // Copy result back
        let inverse: Vec<u32> = self.device.dtoh_sync_copy(&d_inverse).unwrap();
        Some(inverse)
    }

    /// CPU fallback for matrix inverse mod p
    fn cpu_matrix_inverse_mod(matrix: &[u32], n: usize, p: u32) -> Option<Vec<u32>> {
        let p64 = p as u64;

        // Create augmented matrix [A | I]
        let mut aug = vec![0u32; n * 2 * n];
        for i in 0..n {
            for j in 0..n {
                aug[i * 2 * n + j] = matrix[i * n + j] % p;
            }
            aug[i * 2 * n + n + i] = 1;
        }

        // Gauss-Jordan elimination
        for col in 0..n {
            // Find pivot
            let mut pivot_row = None;
            for row in col..n {
                if aug[row * 2 * n + col] != 0 {
                    pivot_row = Some(row);
                    break;
                }
            }

            let pivot_row = pivot_row?;

            // Swap rows
            if pivot_row != col {
                for j in 0..2 * n {
                    aug.swap(col * 2 * n + j, pivot_row * 2 * n + j);
                }
            }

            // Scale pivot row
            let pivot = aug[col * 2 * n + col] as u64;
            let pivot_inv = Self::mod_pow_u64(pivot, p64 - 2, p64);
            for j in 0..2 * n {
                aug[col * 2 * n + j] = ((aug[col * 2 * n + j] as u64 * pivot_inv) % p64) as u32;
            }

            // Eliminate column
            for row in 0..n {
                if row != col && aug[row * 2 * n + col] != 0 {
                    let factor = aug[row * 2 * n + col] as u64;
                    for j in 0..2 * n {
                        let sub = (factor * aug[col * 2 * n + j] as u64) % p64;
                        aug[row * 2 * n + j] = ((aug[row * 2 * n + j] as u64 + p64 - sub) % p64) as u32;
                    }
                }
            }
        }

        // Extract inverse
        let mut inv = vec![0u32; n * n];
        for i in 0..n {
            for j in 0..n {
                inv[i * n + j] = aug[i * 2 * n + n + j];
            }
        }

        Some(inv)
    }

    /// Modular exponentiation helper
    fn mod_pow_u64(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
        let mut result = 1u64;
        base %= modulus;
        while exp > 0 {
            if exp % 2 == 1 {
                result = (result * base) % modulus;
            }
            exp /= 2;
            base = (base * base) % modulus;
        }
        result
    }

    /// GPU-accelerated Hensel (Dixon) lifting for exact linear system solving
    ///
    /// This solves Ax = b over the integers using p-adic lifting:
    /// 1. Compute A⁻¹ mod p once
    /// 2. Iteratively lift the solution keeping everything in p-adic form
    /// 3. Convert p-adic representation to BigInt at the end
    ///
    /// Key optimization: All lifting iterations stay in modular arithmetic (u32).
    /// BigInt is only used for final reconstruction.
    ///
    /// Returns (solutions, timings) where solutions[col][row] is the solution component.
    pub fn gpu_hensel_solve(
        &self,
        matrix: &[i64],      // n×n matrix (row-major)
        b_cols: &[Vec<i64>], // k RHS vectors
        n: usize,
        k: usize,
        config: &parallel_lift_core::solve_hensel::HenselConfig,
    ) -> Option<(Vec<Vec<num_bigint::BigInt>>, parallel_lift_core::solve_hensel::HenselTimings)> {
        use num_bigint::BigInt;

        let start_total = Instant::now();
        let p = config.prime;
        let p64 = p as u64;

        // Convert matrix to u32 mod p (keep original for residual computation)
        let matrix_mod: Vec<u32> = matrix.iter().map(|&x| {
            let x_mod = ((x % p as i64) + p as i64) as u64 % p64;
            x_mod as u32
        }).collect();

        // Convert b to u32 mod p
        let b_mod: Vec<Vec<u32>> = b_cols.iter().map(|col| {
            col.iter().map(|&x| {
                let x_mod = ((x % p as i64) + p as i64) as u64 % p64;
                x_mod as u32
            }).collect()
        }).collect();

        // Step 1: Compute A⁻¹ mod p
        let start_inverse = Instant::now();
        let a_inv = self.gpu_matrix_inverse_mod(&matrix_mod, n, p)?;
        let inverse_ms = start_inverse.elapsed().as_secs_f64() * 1000.0;

        // Step 2: Lifting iterations - keep everything in p-adic form
        let start_lifting = Instant::now();
        let iterations_needed = config.estimate_iterations().min(config.max_iterations);

        // Store x as p-adic digits: x_digits[iteration][col][row]
        // x = sum_{i=0}^{iter-1} x_digits[i] * p^i
        let mut x_digits: Vec<Vec<Vec<u32>>> = Vec::with_capacity(iterations_needed);

        // Initial solve: x_0 = A⁻¹ * b mod p
        let mut initial_digits = Vec::with_capacity(k);
        for col_idx in 0..k {
            let mut digit = vec![0u32; n];
            for i in 0..n {
                let mut sum = 0u64;
                for j in 0..n {
                    sum = (sum + a_inv[i * n + j] as u64 * b_mod[col_idx][j] as u64) % p64;
                }
                digit[i] = sum as u32;
            }
            initial_digits.push(digit);
        }
        x_digits.push(initial_digits);

        // Lifting iterations
        // For each iteration i, we compute:
        // residual = (b - A*x) / p^i mod p
        // new_digit = A⁻¹ * residual mod p
        //
        // To compute (b - A*x) / p^i mod p efficiently:
        // We track the "remainder" from previous iterations

        // For proper lifting, we need to track the full residual
        // r_0 = b - A*x_0 (divisible by p, since x_0 = A⁻¹b mod p)
        // r_1 = (b - A*x_0 - A*x_1*p) / p = (r_0 - A*x_1*p) / p (divisible by p)
        // etc.

        // Simpler approach: compute A*x using p-adic digits and track high/low parts
        for iter in 1..iterations_needed {
            let mut new_digits = Vec::with_capacity(k);

            for col_idx in 0..k {
                // Compute A * x mod p^(iter+1)
                // Where x = sum_{d=0}^{iter-1} x_digits[d][col] * p^d

                // We need the coefficient of p^iter in (b - A*x)
                // This requires computing A*x with enough precision

                // For efficiency, compute incrementally:
                // (b - A*x_new) / p^iter = ((b - A*x_old) / p^{iter-1} - A*x_{iter-1}) / p mod p

                // Use 64-bit arithmetic to track carries
                let mut ax_at_iter = vec![0i64; n];

                // Contribution from each previous digit
                // A * x_digits[d] * p^d, we need coefficient of p^iter
                // Only d <= iter contributes, and d=iter contributes directly

                // Compute A * x_digits[iter-1] (the previous digit)
                let prev_digit = &x_digits[iter - 1][col_idx];
                for i in 0..n {
                    for j in 0..n {
                        let aij = matrix[i * n + j];
                        ax_at_iter[i] += aij * prev_digit[j] as i64;
                    }
                }

                // The residual at level iter is:
                // (b - A*(x_0 + x_1*p + ... + x_{iter-1}*p^{iter-1})) / p^iter
                // = ((b - A*x_0)/p - A*x_1 - ... - A*x_{iter-1}*p^{iter-2}) / p^{iter-1}
                // Recursively...

                // For simplicity, use the "correction" approach:
                // We know x satisfies A*x ≡ b (mod p^iter) from previous iterations
                // So (b - A*x) is divisible by p^iter
                // The residual mod p is ((b - A*x) / p^iter) mod p

                // This is tricky to compute without BigInt...
                // Let's use a different formulation.

                // At iteration iter, we have x = x_0 + x_1*p + ... + x_{iter-1}*p^{iter-1}
                // We want: y = A⁻¹ * ((b - A*x) / p^iter) mod p
                // Then: x_iter = y, and new x = x + y*p^iter

                // The key insight: (b - A*x) / p^iter mod p can be computed as:
                // Let r = b - A*x (as actual integers)
                // Then r / p^iter mod p = (r mod p^{iter+1}) / p^iter

                // To avoid BigInt, we can track r iteratively:
                // r_0 = b (mod p^{max_iter})
                // After each iteration: r_{i+1} = (r_i - p^i * A * digit_i)

                // For now, use a simpler (slower) approach that's still modular:
                // Compute residual using multi-precision modular arithmetic

                // Actually, the simplest correct approach without BigInt is to use
                // the recurrence: digit_i = A⁻¹ * ((b - A*(sum of lower digits)) / p^i) mod p

                // This can be done by tracking a "running residual" as 64-bit values
                // But it gets complicated with carries...

                // For this implementation, let's compute the residual directly
                // using 64-bit arithmetic with careful carry handling

                // Compute b - A*(accumulated x) at the level of p^iter
                let mut residual = vec![0u32; n];

                // b component (stays constant)
                let b_val = &b_cols[col_idx];

                // We need (b - A*x) / p^iter mod p
                // Where x = sum_{d=0}^{iter-1} digit_d * p^d

                // Key: A*x mod p^{iter+1} can be computed modularly
                // A*x = sum_{d=0}^{iter-1} A*digit_d * p^d
                // Coefficient of p^iter in A*x comes from carries of lower terms

                // Simplified: compute coefficient of p^iter in (A*digit_0 + p*A*digit_1 + ...)
                // and subtract from b's coefficient of p^iter (which is 0 since b < p^iter typically)

                // For correctness with arbitrary b, use 128-bit accumulator
                for i in 0..n {
                    // Compute (b[i] - sum over j,d of: A[i,j] * digit_d[j] * p^d) / p^iter mod p
                    let mut accum: i128 = b_val[i] as i128;

                    let mut p_power: i128 = 1;
                    for d in 0..iter {
                        let digit = &x_digits[d][col_idx];
                        for j in 0..n {
                            accum -= matrix[i * n + j] as i128 * digit[j] as i128 * p_power;
                        }
                        p_power *= p as i128;
                    }

                    // Now accum = b[i] - (A*x)[i], and it should be divisible by p^iter
                    // Extract coefficient of p^iter
                    let p_iter = (p as i128).pow(iter as u32);
                    let coeff = accum / p_iter;

                    // Take mod p (handle negative values)
                    let coeff_mod = ((coeff % p as i128) + p as i128) % p as i128;
                    residual[i] = coeff_mod as u32;
                }

                // Compute new digit = A⁻¹ * residual mod p
                let mut new_digit = vec![0u32; n];
                for i in 0..n {
                    let mut sum = 0u64;
                    for j in 0..n {
                        sum = (sum + a_inv[i * n + j] as u64 * residual[j] as u64) % p64;
                    }
                    new_digit[i] = sum as u32;
                }

                new_digits.push(new_digit);
            }

            x_digits.push(new_digits);
        }

        let lifting_ms = start_lifting.elapsed().as_secs_f64() * 1000.0;

        // Step 3: Convert p-adic representation to BigInt
        let start_recon = Instant::now();
        let p_big = BigInt::from(p);
        let final_p_power = p_big.pow(iterations_needed as u32);
        let half_p_power = &final_p_power / 2;

        let mut solutions: Vec<Vec<BigInt>> = Vec::with_capacity(k);
        for col_idx in 0..k {
            let mut col = Vec::with_capacity(n);
            for row in 0..n {
                // x = sum_{d=0}^{iter-1} x_digits[d][col][row] * p^d
                let mut x = BigInt::from(0);
                let mut p_power = BigInt::from(1);
                for d in 0..iterations_needed {
                    x += BigInt::from(x_digits[d][col_idx][row]) * &p_power;
                    p_power *= &p_big;
                }

                // Convert to signed
                if x > half_p_power {
                    x -= &final_p_power;
                }
                col.push(x);
            }
            solutions.push(col);
        }

        let reconstruction_ms = start_recon.elapsed().as_secs_f64() * 1000.0;
        let total_ms = start_total.elapsed().as_secs_f64() * 1000.0;

        let timings = parallel_lift_core::solve_hensel::HenselTimings {
            inverse_ms,
            lifting_ms,
            reconstruction_ms,
            total_ms,
            iterations: iterations_needed,
        };

        Some((solutions, timings))
    }

    /// GPU-accelerated Hensel solve with timed result
    pub fn gpu_hensel_solve_timed(
        &self,
        matrix: &[i64],
        b_cols: &[Vec<i64>],
        n: usize,
        k: usize,
        config: &parallel_lift_core::solve_hensel::HenselConfig,
    ) -> Option<(Vec<Vec<num_bigint::BigInt>>, parallel_lift_core::solve_hensel::HenselTimings)> {
        // Use GPU-native version for better performance
        self.gpu_hensel_solve_native(matrix, b_cols, n, k, config)
    }

    /// Fully GPU-native Hensel lifting using defect-based algorithm
    ///
    /// This keeps ALL lifting iterations on GPU using 64-bit signed arithmetic.
    /// The "defect" d_i = (b - A*x)/p^i stays bounded, avoiding BigInt.
    ///
    /// Recurrence:
    ///   digit_i = A⁻¹ * (d_i mod p) mod p
    ///   d_{i+1} = (d_i - A * digit_i) / p
    pub fn gpu_hensel_solve_native(
        &self,
        matrix: &[i64],
        b_cols: &[Vec<i64>],
        n: usize,
        k: usize,
        config: &parallel_lift_core::solve_hensel::HenselConfig,
    ) -> Option<(Vec<Vec<num_bigint::BigInt>>, parallel_lift_core::solve_hensel::HenselTimings)> {
        use num_bigint::BigInt;

        let start_total = Instant::now();
        let p = config.prime;
        let p64 = p as u64;

        // Convert matrix to u32 mod p (for inverse computation)
        let matrix_mod: Vec<u32> = matrix.iter().map(|&x| {
            let x_mod = ((x % p as i64) + p as i64) as u64 % p64;
            x_mod as u32
        }).collect();

        // Step 1: Compute A⁻¹ mod p
        let start_inverse = Instant::now();
        let a_inv = self.gpu_matrix_inverse_mod(&matrix_mod, n, p)?;
        let inverse_ms = start_inverse.elapsed().as_secs_f64() * 1000.0;

        // Step 2: GPU lifting iterations - FULLY FUSED single kernel launch
        let start_lifting = Instant::now();
        let iterations_needed = config.estimate_iterations().min(config.max_iterations);

        // Initialize b (flattened column-major: [col][row])
        let b_flat: Vec<i64> = b_cols.iter().flat_map(|col| col.iter().copied()).collect();

        // Upload everything to GPU ONCE
        let d_a_inv = self.device.htod_copy(a_inv.clone()).unwrap();
        let d_a: CudaSlice<i64> = self.device.htod_copy(matrix.to_vec()).unwrap();
        let d_b: CudaSlice<i64> = self.device.htod_copy(b_flat).unwrap();

        // Allocate space for ALL digits on GPU [iterations][k][n]
        let total_digit_space = iterations_needed * k * n;
        let d_all_digits: CudaSlice<u32> = self.device.alloc_zeros(total_digit_space).unwrap();

        // FULLY FUSED: Single kernel launch for ALL iterations
        // Shared memory: 2 * n * sizeof(u32) for digit + defect_modp
        let shared_mem = (2 * n * std::mem::size_of::<u32>()) as u32;

        let fused_config = LaunchConfig {
            grid_dim: (k as u32, 1, 1),
            block_dim: (n as u32, 1, 1),
            shared_mem_bytes: shared_mem,
        };

        // Single kernel launch runs ALL iterations
        unsafe {
            self.get_kernel("hensel_gpu_lift_all_fused").unwrap().launch(
                fused_config,
                (
                    &d_a_inv,
                    &d_a,
                    &d_b,
                    &d_all_digits,
                    p,
                    n as u32,
                    k as u32,
                    iterations_needed as u32,
                ),
            ).unwrap();
        }

        // Synchronize and download ALL digits at once
        self.device.synchronize().unwrap();
        let all_digits_flat: Vec<u32> = self.device.dtoh_sync_copy(&d_all_digits).unwrap();

        // Reshape to [iterations][k][n]
        let mut x_digits: Vec<Vec<Vec<u32>>> = Vec::with_capacity(iterations_needed);
        for iter in 0..iterations_needed {
            let iter_offset = iter * k * n;
            let mut digit_vec = Vec::with_capacity(k);
            for col_idx in 0..k {
                let start = iter_offset + col_idx * n;
                digit_vec.push(all_digits_flat[start..start + n].to_vec());
            }
            x_digits.push(digit_vec);
        }

        let lifting_ms = start_lifting.elapsed().as_secs_f64() * 1000.0;

        // Step 3: Convert p-adic representation to BigInt
        let start_recon = Instant::now();
        let p_big = BigInt::from(p);
        let final_p_power = p_big.pow(iterations_needed as u32);
        let half_p_power = &final_p_power / 2;

        let mut solutions: Vec<Vec<BigInt>> = Vec::with_capacity(k);
        for col_idx in 0..k {
            let mut col = Vec::with_capacity(n);
            for row in 0..n {
                // x = sum_{d=0}^{iter-1} x_digits[d][col][row] * p^d
                let mut x = BigInt::from(0);
                let mut p_power = BigInt::from(1);
                for d in 0..iterations_needed {
                    x += BigInt::from(x_digits[d][col_idx][row]) * &p_power;
                    p_power *= &p_big;
                }

                // Convert to signed
                if x > half_p_power {
                    x -= &final_p_power;
                }
                col.push(x);
            }
            solutions.push(col);
        }

        let reconstruction_ms = start_recon.elapsed().as_secs_f64() * 1000.0;
        let total_ms = start_total.elapsed().as_secs_f64() * 1000.0;

        let timings = parallel_lift_core::solve_hensel::HenselTimings {
            inverse_ms,
            lifting_ms,
            reconstruction_ms,
            total_ms,
            iterations: iterations_needed,
        };

        Some((solutions, timings))
    }

    // =========================================================================
    // Gram-Schmidt / LLL GPU Operations
    // =========================================================================

    /// Compute Gram matrix for all primes in parallel on GPU
    ///
    /// Input: basis vectors [num_primes][n][m] stored as residues mod each prime
    /// Output: gram[prime_idx][i][j] = <b_i, b_j> mod primes[prime_idx]
    pub fn gpu_batch_gram_matrix(
        &self,
        basis: &[u32],      // [num_primes * n * m] flattened
        n: usize,           // number of basis vectors
        m: usize,           // dimension of each vector
        primes: &[u32],
    ) -> Vec<u32> {
        let num_primes = primes.len();
        let total_gram_elements = num_primes * n * n;

        // Upload data to GPU
        let d_basis = self.device.htod_copy(basis.to_vec()).unwrap();
        let d_primes = self.device.htod_copy(primes.to_vec()).unwrap();
        let d_gram: CudaSlice<u32> = self.device.alloc_zeros(total_gram_elements).unwrap();

        let kernel = self.get_kernel("batch_gram_matrix").unwrap();
        let threads_per_block = 256;
        let blocks = (total_gram_elements + threads_per_block - 1) / threads_per_block;

        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel
                .launch(
                    config,
                    (
                        &d_basis,
                        &d_gram,
                        &d_primes,
                        n as u32,
                        m as u32,
                        num_primes as u32,
                    ),
                )
                .unwrap();
        }

        self.device.dtoh_sync_copy(&d_gram).unwrap()
    }

    /// Compute squared norms for all basis vectors across all primes
    ///
    /// Output: norms_sq[prime_idx * n + i] = ||b_i||^2 mod primes[prime_idx]
    pub fn gpu_batch_squared_norms(
        &self,
        basis: &[u32],      // [num_primes * n * m]
        n: usize,
        m: usize,
        primes: &[u32],
    ) -> Vec<u32> {
        let num_primes = primes.len();
        let total_norms = num_primes * n;

        let d_basis = self.device.htod_copy(basis.to_vec()).unwrap();
        let d_primes = self.device.htod_copy(primes.to_vec()).unwrap();
        let d_norms: CudaSlice<u32> = self.device.alloc_zeros(total_norms).unwrap();

        let kernel = self.get_kernel("batch_squared_norms").unwrap();
        let threads_per_block = 256;
        let blocks = (total_norms + threads_per_block - 1) / threads_per_block;

        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel
                .launch(
                    config,
                    (
                        &d_basis,
                        &d_norms,
                        &d_primes,
                        n as u32,
                        m as u32,
                        num_primes as u32,
                    ),
                )
                .unwrap();
        }

        self.device.dtoh_sync_copy(&d_norms).unwrap()
    }

    /// GPU Gram-Schmidt orthogonalization for all primes
    ///
    /// Performs full Gram-Schmidt orthogonalization on the basis vectors.
    /// Returns:
    /// - b_star: orthogonalized vectors [num_primes * n * m]
    /// - mu_num: μ coefficient numerators [num_primes * n*(n-1)/2]
    /// - mu_den: μ coefficient denominators [num_primes * n*(n-1)/2]
    /// - norms_sq: ||b*_i||^2 for each vector [num_primes * n]
    pub fn gpu_gram_schmidt(
        &self,
        basis: &[u32],      // [num_primes * n * m]
        n: usize,
        m: usize,
        primes: &[u32],
    ) -> GpuGramSchmidtResult {
        let num_primes = primes.len();
        let mu_size = n * (n - 1) / 2;

        // Allocate device memory
        let d_basis = self.device.htod_copy(basis.to_vec()).unwrap();
        let d_primes = self.device.htod_copy(primes.to_vec()).unwrap();
        let d_b_star: CudaSlice<u32> = self.device.alloc_zeros(num_primes * n * m).unwrap();
        let d_mu_num: CudaSlice<u32> = self.device.alloc_zeros(num_primes * mu_size).unwrap();
        let d_mu_den: CudaSlice<u32> = self.device.alloc_zeros(num_primes * mu_size).unwrap();
        let d_norms_sq: CudaSlice<u32> = self.device.alloc_zeros(num_primes * n).unwrap();

        let threads_per_block = 256.min(num_primes);
        let blocks = (num_primes + threads_per_block - 1) / threads_per_block;

        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Run Gram-Schmidt step by step (each step depends on previous)
        for i in 0..n {
            let kernel = self.get_kernel("gram_schmidt_step").unwrap();
            unsafe {
                kernel
                    .launch(
                        config,
                        (
                            &d_basis,
                            &d_b_star,
                            &d_mu_num,
                            &d_mu_den,
                            &d_norms_sq,
                            &d_primes,
                            i as u32,
                            n as u32,
                            m as u32,
                            num_primes as u32,
                        ),
                    )
                    .unwrap();
            }
        }

        GpuGramSchmidtResult {
            b_star: self.device.dtoh_sync_copy(&d_b_star).unwrap(),
            mu_num: self.device.dtoh_sync_copy(&d_mu_num).unwrap(),
            mu_den: self.device.dtoh_sync_copy(&d_mu_den).unwrap(),
            norms_sq: self.device.dtoh_sync_copy(&d_norms_sq).unwrap(),
            n,
            m,
            num_primes,
        }
    }

    /// Swap two adjacent basis vectors on GPU for all primes
    pub fn gpu_batch_swap_vectors(
        &self,
        basis: &mut [u32],  // [num_primes * n * m] - modified in place
        k: usize,           // swap b_k with b_{k-1}
        n: usize,
        m: usize,
        primes: &[u32],
    ) {
        let num_primes = primes.len();
        let total_elements = num_primes * m;

        let mut d_basis = self.device.htod_copy(basis.to_vec()).unwrap();
        let d_primes = self.device.htod_copy(primes.to_vec()).unwrap();

        let kernel = self.get_kernel("batch_swap_vectors").unwrap();
        let threads_per_block = 256;
        let blocks = (total_elements + threads_per_block - 1) / threads_per_block;

        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel
                .launch(
                    config,
                    (
                        &mut d_basis,
                        k as u32,
                        &d_primes,
                        n as u32,
                        m as u32,
                        num_primes as u32,
                    ),
                )
                .unwrap();
        }

        // Copy back to host
        let result = self.device.dtoh_sync_copy(&d_basis).unwrap();
        basis.copy_from_slice(&result);
    }

    /// Check Lovász condition for index k across all primes
    /// Returns true only if condition is satisfied for all primes (consensus)
    pub fn gpu_check_lovasz(
        &self,
        norms_sq: &[u32],   // [num_primes * n]
        mu_num: &[u32],     // [num_primes * mu_size]
        mu_den: &[u32],     // [num_primes * mu_size]
        k: usize,
        n: usize,
        delta_num: u32,
        delta_den: u32,
        primes: &[u32],
    ) -> bool {
        let num_primes = primes.len();

        let d_norms = self.device.htod_copy(norms_sq.to_vec()).unwrap();
        let d_mu_num = self.device.htod_copy(mu_num.to_vec()).unwrap();
        let d_mu_den = self.device.htod_copy(mu_den.to_vec()).unwrap();
        let d_primes = self.device.htod_copy(primes.to_vec()).unwrap();
        let d_satisfied: CudaSlice<u32> = self.device.alloc_zeros(num_primes).unwrap();

        let kernel = self.get_kernel("check_lovasz_condition").unwrap();
        let threads_per_block = 256.min(num_primes);
        let blocks = (num_primes + threads_per_block - 1) / threads_per_block;

        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel
                .launch(
                    config,
                    (
                        &d_norms,
                        &d_mu_num,
                        &d_mu_den,
                        &d_satisfied,
                        &d_primes,
                        k as u32,
                        delta_num,
                        delta_den,
                        n as u32,
                        num_primes as u32,
                    ),
                )
                .unwrap();
        }

        let satisfied: Vec<u32> = self.device.dtoh_sync_copy(&d_satisfied).unwrap();

        // Consensus: all primes must agree
        satisfied.iter().all(|&s| s == 1)
    }

    /// Apply size reduction on GPU: b_k = b_k - round(mu) * b_j
    ///
    /// This kernel applies the size reduction step for all primes in parallel.
    /// The mu_rounded value is computed from the exact mu coefficient.
    pub fn gpu_batch_size_reduce(
        &self,
        basis: &mut [u32],  // [num_primes * n * m] - modified in place
        mu_rounded: &[u32], // [num_primes] - rounded mu coefficient for each prime
        k: usize,           // vector being reduced
        j: usize,           // vector to subtract
        n: usize,
        m: usize,
        primes: &[u32],
    ) {
        let num_primes = primes.len();
        let total_elements = num_primes * m;

        let mut d_basis = self.device.htod_copy(basis.to_vec()).unwrap();
        let d_mu_rounded = self.device.htod_copy(mu_rounded.to_vec()).unwrap();
        let d_primes = self.device.htod_copy(primes.to_vec()).unwrap();

        let kernel = self.get_kernel("batch_size_reduce").unwrap();
        let threads_per_block = 256;
        let blocks = (total_elements + threads_per_block - 1) / threads_per_block;

        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel
                .launch(
                    config,
                    (
                        &mut d_basis,
                        &d_mu_rounded,
                        k as u32,
                        j as u32,
                        &d_primes,
                        n as u32,
                        m as u32,
                        num_primes as u32,
                    ),
                )
                .unwrap();
        }

        // Copy back to host
        let result = self.device.dtoh_sync_copy(&d_basis).unwrap();
        basis.copy_from_slice(&result);
    }

    /// GPU batched multi-RHS solve with detailed timing breakdown
    ///
    /// Same as `gpu_batch_multi_rhs_solve` but returns timing information
    /// for analyzing PCIe transfer overhead.
    pub fn gpu_batch_multi_rhs_solve_timed(
        &self,
        matrix: &[u32],
        b_cols: &[Vec<u32>],
        n: usize,
        k: usize,
        primes: &[u32],
    ) -> (Option<Vec<Vec<Vec<u32>>>>, TransferTiming) {
        let total_start = Instant::now();
        let num_primes = primes.len();
        let aug_width = n + k;
        let aug_stride = n * aug_width;

        // Phase 1: Prepare data on CPU (reduction mod p)
        let prepare_start = Instant::now();
        let mut augmented_data = vec![0u32; num_primes * aug_stride];
        for (pi, &p) in primes.iter().enumerate() {
            let offset = pi * aug_stride;
            for row in 0..n {
                for col in 0..n {
                    augmented_data[offset + row * aug_width + col] = matrix[row * n + col] % p;
                }
                for col_idx in 0..k {
                    augmented_data[offset + row * aug_width + n + col_idx] =
                        b_cols[col_idx][row] % p;
                }
            }
        }
        let prepare_ms = prepare_start.elapsed().as_secs_f64() * 1000.0;

        // Phase 2: Host→Device transfer
        let htod_start = Instant::now();
        let d_augmented = self.device.htod_copy(augmented_data.clone()).unwrap();
        let d_primes = self.device.htod_copy(primes.to_vec()).unwrap();
        let d_solutions: CudaSlice<u32> = self.device.alloc_zeros(num_primes * n * k).unwrap();
        let d_singular: CudaSlice<u32> = self.device.alloc_zeros(num_primes).unwrap();
        let d_workspace: CudaSlice<u32> = self.device.alloc_zeros(num_primes * aug_stride).unwrap();
        let htod_ms = htod_start.elapsed().as_secs_f64() * 1000.0;

        // Calculate transfer sizes
        let htod_bytes = augmented_data.len() * 4 + primes.len() * 4;

        // Phase 3: GPU Compute
        let compute_start = Instant::now();
        let use_tiled = n >= 32;

        if use_tiled {
            let kernel = self.get_kernel("modular_solve_multi_rhs_tiled").unwrap();
            let config = LaunchConfig {
                grid_dim: (num_primes as u32, 1, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                kernel
                    .launch(
                        config,
                        (
                            &d_augmented,
                            &d_primes,
                            &d_solutions,
                            &d_singular,
                            &d_workspace,
                            n as u32,
                            k as u32,
                            num_primes as u32,
                        ),
                    )
                    .unwrap();
            }
        } else {
            let kernel = self.get_kernel("modular_solve_multi_rhs").unwrap();
            let threads_per_block = 256.min(num_primes);
            let blocks = (num_primes + threads_per_block - 1) / threads_per_block;

            let config = LaunchConfig {
                grid_dim: (blocks as u32, 1, 1),
                block_dim: (threads_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                kernel
                    .launch(
                        config,
                        (
                            &d_augmented,
                            &d_primes,
                            &d_solutions,
                            &d_singular,
                            &d_workspace,
                            n as u32,
                            k as u32,
                            num_primes as u32,
                        ),
                    )
                    .unwrap();
            }
        }

        // Synchronize to ensure kernel is complete before timing D→H
        // dtoh_sync_copy includes synchronization, but we need to measure compute time
        // We'll do a small dtoh to force synchronization
        let _sync: Vec<u32> = self.device.dtoh_sync_copy(&d_singular).unwrap();
        let compute_ms = compute_start.elapsed().as_secs_f64() * 1000.0;

        // Phase 4: Device→Host transfer
        let dtoh_start = Instant::now();
        let solutions: Vec<u32> = self.device.dtoh_sync_copy(&d_solutions).unwrap();
        let singular_flags = _sync; // Already fetched for sync
        let dtoh_ms = dtoh_start.elapsed().as_secs_f64() * 1000.0;

        let dtoh_bytes = num_primes * n * k * 4 + num_primes * 4;
        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        let timing = TransferTiming {
            prepare_ms,
            htod_ms,
            compute_ms,
            dtoh_ms,
            total_ms,
            htod_bytes,
            dtoh_bytes,
        };

        // Check for singular matrices
        if singular_flags.iter().any(|&f| f != 0) {
            return (None, timing);
        }

        // Parse results
        let mut result = Vec::with_capacity(num_primes);
        for pi in 0..num_primes {
            let prime_offset = pi * n * k;
            let mut cols = Vec::with_capacity(k);
            for col_idx in 0..k {
                let col_offset = prime_offset + col_idx * n;
                cols.push(solutions[col_offset..col_offset + n].to_vec());
            }
            result.push(cols);
        }

        (Some(result), timing)
    }
}

/// Result of GPU Gram-Schmidt orthogonalization
#[derive(Clone)]
pub struct GpuGramSchmidtResult {
    /// Orthogonalized vectors b*_i [num_primes * n * m]
    pub b_star: Vec<u32>,
    /// μ coefficient numerators [num_primes * n*(n-1)/2]
    pub mu_num: Vec<u32>,
    /// μ coefficient denominators [num_primes * n*(n-1)/2]
    pub mu_den: Vec<u32>,
    /// Squared norms ||b*_i||^2 [num_primes * n]
    pub norms_sq: Vec<u32>,
    pub n: usize,
    pub m: usize,
    pub num_primes: usize,
}

impl GpuGramSchmidtResult {
    /// Get μ_{i,j} numerator for a specific prime
    pub fn get_mu_num(&self, prime_idx: usize, i: usize, j: usize) -> u32 {
        assert!(j < i, "μ_{{{},{}}} only defined for j < i", i, j);
        let mu_idx = i * (i - 1) / 2 + j;
        self.mu_num[prime_idx * (self.n * (self.n - 1) / 2) + mu_idx]
    }

    /// Get μ_{i,j} denominator for a specific prime
    pub fn get_mu_den(&self, prime_idx: usize, i: usize, j: usize) -> u32 {
        assert!(j < i, "μ_{{{},{}}} only defined for j < i", i, j);
        let mu_idx = i * (i - 1) / 2 + j;
        self.mu_den[prime_idx * (self.n * (self.n - 1) / 2) + mu_idx]
    }

    /// Get ||b*_i||^2 for a specific prime
    pub fn get_norm_sq(&self, prime_idx: usize, i: usize) -> u32 {
        self.norms_sq[prime_idx * self.n + i]
    }
}

// ============================================================================
// GPU-Accelerated LLL Algorithm
// ============================================================================

/// Configuration for GPU LLL
#[derive(Clone, Debug)]
pub struct GpuLLLConfig {
    /// LLL parameter δ numerator (default: 3)
    pub delta_num: u32,
    /// LLL parameter δ denominator (default: 4)
    pub delta_den: u32,
    /// Maximum iterations before giving up
    pub max_iterations: usize,
    /// Enable verbose output
    pub verbose: bool,
}

impl Default for GpuLLLConfig {
    fn default() -> Self {
        Self {
            delta_num: 3,
            delta_den: 4,
            max_iterations: 100_000,
            verbose: false,
        }
    }
}

/// Statistics from GPU LLL reduction
#[derive(Clone, Debug, Default)]
pub struct GpuLLLStats {
    pub size_reductions: usize,
    pub swaps: usize,
    pub iterations: usize,
    pub gram_schmidt_time: f64,
    pub lovasz_check_time: f64,
    pub swap_time: f64,
    pub total_time: f64,
}

/// GPU-accelerated LLL lattice basis reduction
///
/// Uses CRT representation to perform exact arithmetic on GPU:
/// 1. Convert BigInt basis vectors to residues mod multiple primes
/// 2. Run LLL algorithm with GPU-accelerated Gram-Schmidt
/// 3. Reconstruct BigInt results using CRT
///
/// Key insight: Most LLL operations are inner products and vector updates,
/// which parallelize well across primes on GPU.
pub struct GpuLLL<'a> {
    backend: &'a CudaBackend,
    config: GpuLLLConfig,
}

impl<'a> GpuLLL<'a> {
    pub fn new(backend: &'a CudaBackend, config: GpuLLLConfig) -> Self {
        Self { backend, config }
    }

    /// Run LLL reduction on a lattice basis (already in residue form)
    ///
    /// Input:
    /// - basis_residues: [num_primes * n * m] basis vectors as residues
    /// - n: number of basis vectors
    /// - m: dimension of each vector
    /// - primes: the CRT primes
    ///
    /// Returns reduced basis and statistics
    pub fn reduce_from_residues(
        &self,
        mut basis_residues: Vec<u32>,
        n: usize,
        m: usize,
        primes: &[u32],
    ) -> (Vec<u32>, GpuLLLStats) {
        use std::time::Instant;

        let start = Instant::now();
        let mut stats = GpuLLLStats::default();
        let num_primes = primes.len();

        if n <= 1 {
            stats.total_time = start.elapsed().as_secs_f64();
            return (basis_residues, stats);
        }

        // Initial Gram-Schmidt computation
        let gs_start = Instant::now();
        let mut gs = self.backend.gpu_gram_schmidt(&basis_residues, n, m, primes);
        stats.gram_schmidt_time += gs_start.elapsed().as_secs_f64();

        let mut k = 1;

        while k < n && stats.iterations < self.config.max_iterations {
            stats.iterations += 1;

            // Size reduction: make |μ_{k,j}| ≤ 1/2 for j < k
            // Process from j = k-1 down to 0
            for j in (0..k).rev() {
                // Check if |μ_{k,j}| > 1/2 using consensus across primes
                // In exact arithmetic: |num/den| > 1/2 ⟺ |2*num| > |den|
                let needs_reduce = self.check_needs_size_reduction(&gs, k, j, primes);

                if needs_reduce {
                    // Compute rounded mu for each prime: round(mu_num / mu_den)
                    let mu_rounded = self.compute_rounded_mu(&gs, k, j, primes);

                    // Apply size reduction: b_k = b_k - round(mu) * b_j
                    self.backend.gpu_batch_size_reduce(
                        &mut basis_residues,
                        &mu_rounded,
                        k,
                        j,
                        n,
                        m,
                        primes,
                    );
                    stats.size_reductions += 1;

                    // Recompute Gram-Schmidt after modification
                    let gs_start = Instant::now();
                    gs = self.backend.gpu_gram_schmidt(&basis_residues, n, m, primes);
                    stats.gram_schmidt_time += gs_start.elapsed().as_secs_f64();
                }
            }

            // Check Lovász condition
            let lovasz_start = Instant::now();
            let lovasz_ok = self.backend.gpu_check_lovasz(
                &gs.norms_sq,
                &gs.mu_num,
                &gs.mu_den,
                k,
                n,
                self.config.delta_num,
                self.config.delta_den,
                primes,
            );
            stats.lovasz_check_time += lovasz_start.elapsed().as_secs_f64();

            if lovasz_ok {
                k += 1;
            } else {
                // Swap b_k and b_{k-1}
                let swap_start = Instant::now();
                self.backend.gpu_batch_swap_vectors(&mut basis_residues, k, n, m, primes);
                stats.swaps += 1;
                stats.swap_time += swap_start.elapsed().as_secs_f64();

                // Recompute Gram-Schmidt
                let gs_start = Instant::now();
                gs = self.backend.gpu_gram_schmidt(&basis_residues, n, m, primes);
                stats.gram_schmidt_time += gs_start.elapsed().as_secs_f64();

                k = if k > 1 { k - 1 } else { 1 };
            }

            if self.config.verbose && stats.iterations % 100 == 0 {
                println!(
                    "GPU LLL iteration {}: k={}, swaps={}, size_reductions={}",
                    stats.iterations, k, stats.swaps, stats.size_reductions
                );
            }
        }

        stats.total_time = start.elapsed().as_secs_f64();

        (basis_residues, stats)
    }

    /// Check if size reduction is needed: |μ_{k,j}| > 1/2
    /// Uses consensus across primes for robustness
    fn check_needs_size_reduction(
        &self,
        gs: &GpuGramSchmidtResult,
        k: usize,
        j: usize,
        primes: &[u32],
    ) -> bool {
        let num_primes = gs.num_primes;
        let mut votes_for_reduce = 0;

        for pi in 0..num_primes {
            let mu_n = gs.get_mu_num(pi, k, j) as u64;
            let mu_d = gs.get_mu_den(pi, k, j) as u64;
            let p = primes[pi] as u64;

            if mu_d == 0 {
                continue; // Skip degenerate case
            }

            // Check if |μ| > 1/2: equivalent to |2*num| > |den| in exact arithmetic
            // In modular arithmetic, we check if 2*mu_n > mu_d OR 2*(p - mu_n) > mu_d
            // (handling both positive and negative μ values)
            let two_mu_n = (2 * mu_n) % p;
            let two_neg_mu_n = (2 * (p - mu_n)) % p;

            // If mu_n represents a "small" value (close to 0 or close to p),
            // we check if twice that is larger than the denominator
            let needs_reduce = two_mu_n > mu_d && two_mu_n < p / 2
                || two_neg_mu_n > mu_d && two_neg_mu_n < p / 2;

            if needs_reduce {
                votes_for_reduce += 1;
            }
        }

        // Majority vote: if more than half the primes agree, reduce
        votes_for_reduce > num_primes / 2
    }

    /// Compute round(μ_{k,j}) for each prime
    /// Returns the rounded value mod each prime
    fn compute_rounded_mu(
        &self,
        gs: &GpuGramSchmidtResult,
        k: usize,
        j: usize,
        primes: &[u32],
    ) -> Vec<u32> {
        let num_primes = gs.num_primes;
        let mut result = vec![0u32; num_primes];

        for pi in 0..num_primes {
            let mu_n = gs.get_mu_num(pi, k, j) as u64;
            let mu_d = gs.get_mu_den(pi, k, j) as u64;
            let p = primes[pi] as u64;

            if mu_d == 0 {
                result[pi] = 0;
                continue;
            }

            // Compute mu_inv = inverse of mu_d mod p
            let mu_d_inv = mod_pow_u64(mu_d, p - 2, p);

            // mu_value = mu_n * mu_d_inv mod p
            let mu_value = (mu_n * mu_d_inv) % p;

            // Round: we need to determine if mu represents a value > 0.5 or < -0.5
            // In modular arithmetic, values near 0 are "small positive"
            // Values near p are "small negative"
            // The rounding is: round(x) = floor(x + 0.5)

            // For mu in [0, p), interpret:
            // - [0, p/2) as positive values
            // - [p/2, p) as negative values (i.e., value - p)

            // The rounded value is:
            // - If mu < p/4: round to 0 (small positive, |mu| < 0.5)
            // - If mu in [p/4, 3p/4]: round to nearest integer
            // - If mu > 3p/4: round to 0 (small negative, |mu| < 0.5)

            // For size reduction, we want round(mu) where |mu| > 0.5
            // This means mu_value in (p/4, 3p/4) roughly

            // Simple approach: treat values in [0, p/2) as positive, [p/2, p) as negative
            // round(x) = 1 if x in [p/4+1, p/2), round(x) = p-1 (i.e., -1) if x in [p/2, 3p/4)

            if mu_value < p / 2 {
                // Positive value: round(x) = floor(x + 0.5) = floor((2x+1)/2)
                // For |x| > 0.5, this gives 1 or more
                // Approximation: if x > 0.5, round to 1
                if mu_value > p / 4 {
                    result[pi] = 1; // round to 1
                } else {
                    result[pi] = 0;
                }
            } else {
                // Negative value (mu_value represents mu_value - p)
                // For |x| > 0.5 (i.e., x < -0.5), round to -1 = p - 1
                if mu_value < 3 * p / 4 {
                    result[pi] = (p - 1) as u32; // round to -1
                } else {
                    result[pi] = 0;
                }
            }
        }

        result
    }
}

/// Modular exponentiation for u64
fn mod_pow_u64(base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result = 1u64;
    let mut base = base % modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % modulus;
        }
        exp >>= 1;
        base = (base * base) % modulus;
    }
    result
}

/// Convert a lattice basis from BigInt to CRT residue form
///
/// Returns residues [num_primes * n * m] flattened
pub fn basis_to_residues(
    basis_vectors: &[num_bigint::BigInt], // [n * m] flat
    n: usize,
    m: usize,
    crt_basis: &parallel_lift_core::primes::CRTBasis,
) -> Vec<u32> {
    let num_primes = crt_basis.len();

    let mut residues = vec![0u32; num_primes * n * m];

    for (pi, &p) in crt_basis.primes.iter().enumerate() {
        let p_big = num_bigint::BigInt::from(p);
        for i in 0..n {
            for j in 0..m {
                let val = &basis_vectors[i * m + j];
                // Convert to positive residue
                let r = val % &p_big;
                let r_pos = if r < num_bigint::BigInt::from(0) {
                    r + &p_big
                } else {
                    r
                };
                residues[pi * n * m + i * m + j] = r_pos
                    .to_u32_digits()
                    .1
                    .first()
                    .copied()
                    .unwrap_or(0);
            }
        }
    }

    residues
}

/// Convert residues back to BigInt lattice basis using CRT reconstruction
///
/// Input: residues [num_primes * n * m] flattened
/// Returns: basis vectors [n * m] as BigInts
pub fn residues_to_basis(
    backend: &CudaBackend,
    residues: &[u32],
    n: usize,
    m: usize,
    precomputed: &GpuCrtPrecomputed,
) -> Vec<num_bigint::BigInt> {
    use num_bigint::{BigInt, Sign};

    let num_primes = precomputed.primes.len();
    let num_values = n * m;

    // Reorganize residues: from [num_primes][n][m] to [num_values][num_primes]
    // for CRT reconstruction
    let mut residues_by_value = vec![0u32; num_values * num_primes];
    for v in 0..num_values {
        for pi in 0..num_primes {
            residues_by_value[v * num_primes + pi] = residues[pi * n * m + v];
        }
    }

    // Use GPU CRT reconstruction
    let (limbs_vec, signs) = backend.gpu_crt_reconstruct(&residues_by_value, num_values, precomputed);

    // Convert limbs back to BigInt
    let mut result = Vec::with_capacity(num_values);
    for (v, limbs) in limbs_vec.into_iter().enumerate() {
        let is_negative = signs[v];
        let big = if limbs.iter().all(|&x| x == 0) {
            BigInt::from(0)
        } else {
            let sign = if is_negative { Sign::Minus } else { Sign::Plus };
            BigInt::from_slice(sign, &limbs)
        };
        result.push(big);
    }

    result
}

/// Convert residues back to BigInt using CPU-based CRT (fallback)
///
/// This is used when GPU CRT is not available or for validation
pub fn residues_to_basis_cpu(
    residues: &[u32],
    n: usize,
    m: usize,
    crt_basis: &parallel_lift_core::primes::CRTBasis,
) -> Vec<num_bigint::BigInt> {
    use num_bigint::BigInt;
    use parallel_lift_core::crt::CRTReconstruction;

    let num_primes = crt_basis.len();
    let num_values = n * m;

    let mut result = Vec::with_capacity(num_values);

    for v in 0..num_values {
        // Gather residues for this value from all primes
        let value_residues: Vec<u32> = (0..num_primes)
            .map(|pi| residues[pi * n * m + v])
            .collect();

        // Reconstruct using CRT
        let value = CRTReconstruction::reconstruct_signed(&value_residues, crt_basis);
        result.push(value);
    }

    result
}

/// Convert 64-bit residues to BigInt using GPU CRT (V2)
///
/// This is the GPU-accelerated CRT reconstruction for V2 solver using 62-bit primes.
pub fn residues_to_basis_64(
    backend: &CudaBackend,
    residues: &[u64],
    n: usize,
    m: usize,
    precomputed: &GpuCrtPrecomputed64,
) -> Vec<num_bigint::BigInt> {
    use num_bigint::{BigInt, Sign};

    let num_primes = precomputed.primes.len();
    let num_values = n * m;

    // Reorganize residues: from [num_primes][n][m] to [num_values][num_primes]
    // for CRT reconstruction
    let mut residues_by_value = vec![0u64; num_values * num_primes];
    for v in 0..num_values {
        for pi in 0..num_primes {
            residues_by_value[v * num_primes + pi] = residues[pi * n * m + v];
        }
    }

    // Use GPU CRT reconstruction
    let (limbs_vec, signs) = backend.gpu_crt_reconstruct_64(&residues_by_value, num_values, precomputed);

    // Convert limbs back to BigInt
    let mut result = Vec::with_capacity(num_values);
    for (v, limbs) in limbs_vec.into_iter().enumerate() {
        let is_negative = signs[v];
        let big = if limbs.iter().all(|&x| x == 0) {
            BigInt::from(0)
        } else {
            let sign = if is_negative { Sign::Minus } else { Sign::Plus };
            BigInt::from_slice(sign, &limbs)
        };
        result.push(big);
    }

    result
}

/// Convert 64-bit residues to BigInt using GPU CRT (V2) with timing
pub fn residues_to_basis_64_timed(
    backend: &CudaBackend,
    residues: &[u64],
    n: usize,
    m: usize,
    precomputed: &GpuCrtPrecomputed64,
) -> (Vec<num_bigint::BigInt>, f64) {
    use num_bigint::{BigInt, Sign};

    let num_primes = precomputed.primes.len();
    let num_values = n * m;

    // Reorganize residues
    let start = std::time::Instant::now();
    let mut residues_by_value = vec![0u64; num_values * num_primes];
    for v in 0..num_values {
        for pi in 0..num_primes {
            residues_by_value[v * num_primes + pi] = residues[pi * n * m + v];
        }
    }

    // Use GPU CRT reconstruction with timing
    let ((limbs_vec, signs), gpu_crt_ms) = backend.gpu_crt_reconstruct_64_timed(&residues_by_value, num_values, precomputed);

    // Convert limbs back to BigInt
    let mut result = Vec::with_capacity(num_values);
    for (v, limbs) in limbs_vec.into_iter().enumerate() {
        let is_negative = signs[v];
        let big = if limbs.iter().all(|&x| x == 0) {
            BigInt::from(0)
        } else {
            let sign = if is_negative { Sign::Minus } else { Sign::Plus };
            BigInt::from_slice(sign, &limbs)
        };
        result.push(big);
    }
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;

    (result, total_ms)
}

/// CPU-based CRT reconstruction for 64-bit residues (V2 fallback)
pub fn residues_to_basis_64_cpu(
    residues: &[u64],
    n: usize,
    m: usize,
    crt_basis: &parallel_lift_core::solve_v2::CRTBasis62,
) -> Vec<num_bigint::BigInt> {
    use parallel_lift_core::solve_v2::CRTReconstruction62;

    let num_primes = crt_basis.len();
    let num_values = n * m;

    let mut result = Vec::with_capacity(num_values);

    for v in 0..num_values {
        // Gather residues for this value from all primes
        let value_residues: Vec<u64> = (0..num_primes)
            .map(|pi| residues[pi * n * m + v])
            .collect();

        // Reconstruct using CRT
        let value = CRTReconstruction62::reconstruct_signed(&value_residues, crt_basis);
        result.push(value);
    }

    result
}

/// High-level GPU LLL reduction that takes BigInt basis and returns BigInt result
/// NOTE: This uses the original all-GPU approach which has convergence issues.
/// Use `gpu_lll_reduce_hybrid` for correct results.
pub fn gpu_lll_reduce(
    backend: &CudaBackend,
    basis: &parallel_lift_core::lattice::basis::LatticeBasis,
    config: &GpuLLLConfig,
) -> (parallel_lift_core::lattice::basis::LatticeBasis, GpuLLLStats) {
    // Delegate to hybrid implementation for correct convergence
    gpu_lll_reduce_hybrid(backend, basis, config)
}

/// Hybrid CPU/GPU LLL reduction
///
/// Hybrid CPU/GPU LLL implementation
///
/// This implementation uses:
/// - CPU for Gram-Schmidt orthogonalization (requires exact rational arithmetic)
/// - CPU for decision logic (size reduction check, Lovász condition)
/// - GPU support is reserved for future optimizations (e.g., batch inner products)
///
/// Note: Pure GPU GS doesn't work because modular division gives different results
/// than exact division. The μ values must be computed with exact rationals.
pub fn gpu_lll_reduce_hybrid(
    _backend: &CudaBackend,
    basis: &parallel_lift_core::lattice::basis::LatticeBasis,
    config: &GpuLLLConfig,
) -> (parallel_lift_core::lattice::basis::LatticeBasis, GpuLLLStats) {
    use parallel_lift_core::lattice::gram_schmidt::GramSchmidt;
    use num_bigint::BigInt;
    use num_traits::{Zero, Signed};
    use std::time::Instant;

    let start = Instant::now();
    let mut stats = GpuLLLStats::default();

    let n = basis.n;
    let m = basis.m;

    if n <= 1 {
        stats.total_time = start.elapsed().as_secs_f64();
        return (basis.clone(), stats);
    }

    // Clone basis for modification
    let mut b = basis.clone();

    // LLL delta parameter
    let delta_num = config.delta_num as i64;
    let delta_den = config.delta_den as i64;

    // Compute initial Gram-Schmidt (CPU, exact rationals)
    let gs_start = Instant::now();
    let mut gs = GramSchmidt::compute(&b);
    stats.gram_schmidt_time += gs_start.elapsed().as_secs_f64();

    let mut k = 1;
    let max_iterations = config.max_iterations;

    while k < n && stats.iterations < max_iterations {
        stats.iterations += 1;

        // Size reduce b_k with respect to b_{k-1}
        if gs.needs_size_reduction(k, k - 1) {
            let mu = gs.get_mu(k, k - 1);
            let q = round_rational(&mu.numerator, &mu.denominator);

            if !q.is_zero() {
                // b_k = b_k - q * b_{k-1}
                for col in 0..m {
                    b.vectors[k][col] = &b.vectors[k][col] - &q * &b.vectors[k - 1][col];
                }
                stats.size_reductions += 1;

                // Recompute GS after size reduction
                let gs_start = Instant::now();
                gs = GramSchmidt::compute(&b);
                stats.gram_schmidt_time += gs_start.elapsed().as_secs_f64();
            }
        }

        // Check Lovász condition
        let lovasz_start = Instant::now();
        let lovasz_ok = gs.check_lovasz(k, delta_num, delta_den);
        stats.lovasz_check_time += lovasz_start.elapsed().as_secs_f64();

        if lovasz_ok {
            // Size reduce with respect to earlier vectors
            for j in (0..k - 1).rev() {
                if gs.needs_size_reduction(k, j) {
                    let mu = gs.get_mu(k, j);
                    let q = round_rational(&mu.numerator, &mu.denominator);

                    if !q.is_zero() {
                        for col in 0..m {
                            b.vectors[k][col] = &b.vectors[k][col] - &q * &b.vectors[j][col];
                        }
                        stats.size_reductions += 1;
                    }
                }
            }

            // Recompute GS and move to next vector
            let gs_start = Instant::now();
            gs = GramSchmidt::compute(&b);
            stats.gram_schmidt_time += gs_start.elapsed().as_secs_f64();

            k += 1;
        } else {
            // Swap b_k and b_{k-1}
            let swap_start = Instant::now();
            b.swap(k, k - 1);
            stats.swaps += 1;
            stats.swap_time += swap_start.elapsed().as_secs_f64();

            // Recompute GS after swap
            let gs_start = Instant::now();
            gs = GramSchmidt::compute(&b);
            stats.gram_schmidt_time += gs_start.elapsed().as_secs_f64();

            k = if k > 1 { k - 1 } else { 1 };
        }

        if config.verbose && stats.iterations % 100 == 0 {
            println!(
                "Hybrid LLL iteration {}: k={}, swaps={}, size_reductions={}",
                stats.iterations, k, stats.swaps, stats.size_reductions
            );
        }
    }

    stats.total_time = start.elapsed().as_secs_f64();

    (b, stats)
}

/// Reconstruct Gram-Schmidt μ values and norms from GPU result using CRT
///
/// Returns: (mu_values, norms_sq) where
/// - mu_values[i][j] = (numerator, denominator) for μ_{i,j}
/// - norms_sq[i] = ||b*_i||² as BigInt
///
/// Note: This function is currently unused because modular GS doesn't give correct
/// exact values. Kept for reference and potential future GPU GS optimization.
#[allow(dead_code)]
fn reconstruct_gs_values(
    gs: &GpuGramSchmidtResult,
    crt_basis: &parallel_lift_core::primes::CRTBasis,
    n: usize,
) -> (Vec<Vec<(num_bigint::BigInt, num_bigint::BigInt)>>, Vec<num_bigint::BigInt>) {
    use num_bigint::BigInt;
    use parallel_lift_core::crt::CRTReconstruction;

    let num_primes = crt_basis.len();
    let mu_size = n * (n - 1) / 2;

    // Reconstruct μ numerators and denominators
    let mut mu_values: Vec<Vec<(BigInt, BigInt)>> = vec![vec![(BigInt::from(0), BigInt::from(1)); n]; n];

    for i in 1..n {
        for j in 0..i {
            let mu_idx = i * (i - 1) / 2 + j;

            // Gather residues for numerator
            let num_residues: Vec<u32> = (0..num_primes)
                .map(|pi| gs.mu_num[pi * mu_size + mu_idx])
                .collect();

            // Gather residues for denominator
            let den_residues: Vec<u32> = (0..num_primes)
                .map(|pi| gs.mu_den[pi * mu_size + mu_idx])
                .collect();

            // Reconstruct using CRT
            let num = CRTReconstruction::reconstruct_signed(&num_residues, crt_basis);
            let den = CRTReconstruction::reconstruct_signed(&den_residues, crt_basis);

            mu_values[i][j] = (num, den);
        }
    }

    // Reconstruct norms
    let mut norms_sq = Vec::with_capacity(n);
    for i in 0..n {
        let norm_residues: Vec<u32> = (0..num_primes)
            .map(|pi| gs.norms_sq[pi * n + i])
            .collect();
        let norm = CRTReconstruction::reconstruct_signed(&norm_residues, crt_basis);
        norms_sq.push(norm);
    }

    (mu_values, norms_sq)
}

/// Round a rational number (numerator/denominator) to nearest integer
fn round_rational(num: &num_bigint::BigInt, den: &num_bigint::BigInt) -> num_bigint::BigInt {
    use num_bigint::BigInt;
    use num_traits::{Zero, Signed};

    if den.is_zero() {
        return BigInt::zero();
    }

    // round(num/den) = floor(num/den + 1/2) = floor((2*num + den) / (2*den))
    let two = BigInt::from(2);
    let two_num = num * &two;
    let two_den = den * &two;

    // Handle signs properly
    if den.is_positive() {
        // (2*num + den) / (2*den)
        let adjusted = &two_num + den;
        if adjusted.is_negative() {
            // For negative values: floor division goes toward -infinity
            (&adjusted - &two_den + BigInt::from(1)) / &two_den
        } else {
            &adjusted / &two_den
        }
    } else {
        // den < 0: flip signs
        let neg_den = -den;
        let neg_two_den = &neg_den * &two;
        let adjusted = -&two_num + &neg_den;
        if adjusted.is_negative() {
            (&adjusted - &neg_two_den + BigInt::from(1)) / &neg_two_den
        } else {
            &adjusted / &neg_two_den
        }
    }
}

/// Check Lovász condition using exact BigInt arithmetic
///
/// Condition: δ ||b*_{k-1}||² ≤ ||b*_k||² + μ_{k,k-1}² ||b*_{k-1}||²
///
/// Note: Currently unused - using GramSchmidt::check_lovasz instead.
#[allow(dead_code)]
fn check_lovasz_cpu(
    mu_values: &[Vec<(num_bigint::BigInt, num_bigint::BigInt)>],
    norms_sq: &[num_bigint::BigInt],
    k: usize,
    delta_num: &num_bigint::BigInt,
    delta_den: &num_bigint::BigInt,
) -> bool {
    use num_bigint::BigInt;

    if k == 0 {
        return true;
    }

    let norm_k = &norms_sq[k];
    let norm_km1 = &norms_sq[k - 1];
    let (mu_num, mu_den) = &mu_values[k][k - 1];

    // LHS: δ ||b*_{k-1}||² = (delta_num / delta_den) * norm_km1
    // RHS: ||b*_k||² + μ_{k,k-1}² ||b*_{k-1}||²
    //    = norm_k + (mu_num² / mu_den²) * norm_km1

    // Cross-multiply to compare without division:
    // LHS * delta_den * mu_den² ≤ RHS * delta_den * mu_den²
    // delta_num * mu_den² * norm_km1 ≤ delta_den * (mu_den² * norm_k + mu_num² * norm_km1)

    let mu_den_sq = mu_den * mu_den;
    let mu_num_sq = mu_num * mu_num;

    let lhs = delta_num * &mu_den_sq * norm_km1;
    let rhs = delta_den * (&mu_den_sq * norm_k + &mu_num_sq * norm_km1);

    lhs <= rhs
}

/// Update residue basis after size reduction: b_k = b_k - q * b_j
///
/// Note: Currently unused - residue updates happen when using GPU LLL with CRT.
#[allow(dead_code)]
fn update_residues_size_reduce(
    residues: &mut [u32],
    q: &num_bigint::BigInt,
    k: usize,
    j: usize,
    n: usize,
    m: usize,
    crt_basis: &parallel_lift_core::primes::CRTBasis,
) {
    use num_bigint::BigInt;
    use num_traits::Zero;

    if q.is_zero() {
        return;
    }

    let num_primes = crt_basis.len();

    for (pi, &p) in crt_basis.primes.iter().enumerate() {
        let p_big = BigInt::from(p);

        // Compute q mod p
        let q_mod = {
            let r = q % &p_big;
            if r < BigInt::zero() {
                (r + &p_big).to_u32_digits().1.first().copied().unwrap_or(0)
            } else {
                r.to_u32_digits().1.first().copied().unwrap_or(0)
            }
        };

        // b_k[col] = b_k[col] - q * b_j[col] mod p
        for col in 0..m {
            let bk_idx = pi * n * m + k * m + col;
            let bj_idx = pi * n * m + j * m + col;

            let bk_val = residues[bk_idx] as u64;
            let bj_val = residues[bj_idx] as u64;
            let p64 = p as u64;

            // (bk - q * bj) mod p
            let product = (q_mod as u64 * bj_val) % p64;
            let result = if bk_val >= product {
                bk_val - product
            } else {
                p64 - product + bk_val
            };

            residues[bk_idx] = result as u32;
        }
    }
}

/// Precomputed data for GPU CRT reconstruction
#[derive(Clone)]
pub struct GpuCrtPrecomputed {
    pub primes: Vec<u32>,
    pub garner_inverses: Vec<u32>,
    pub pp_limbs: Vec<u32>,        // Packed partial product limbs
    pub pp_offsets: Vec<u32>,      // Offset into pp_limbs for each prime
    pub pp_sizes: Vec<u32>,        // Number of limbs for each partial product
    pub pow2_mod: Vec<u64>,        // [num_primes][max_limbs] 2^(32*j) mod p[i]
    pub max_limbs: usize,
    pub product_limbs: Vec<u32>,   // Full product M as limbs
    pub half_product_limbs: Vec<u32>,  // M/2 as limbs
}

impl GpuCrtPrecomputed {
    /// Create precomputed data from a CRT basis
    pub fn from_basis(basis: &parallel_lift_core::primes::CRTBasis) -> Self {
        let num_primes = basis.len();

        // Estimate max limbs needed: ~(num_primes * 31 bits) / 32 bits
        let max_limbs = (num_primes * 31 + 31) / 32 + 2;

        // Pack partial products into limb array
        let mut pp_limbs = Vec::new();
        let mut pp_offsets = Vec::with_capacity(num_primes);
        let mut pp_sizes = Vec::with_capacity(num_primes);

        for pp in &basis.partial_products {
            pp_offsets.push(pp_limbs.len() as u32);
            let limbs = Self::bigint_to_limbs(pp);
            pp_sizes.push(limbs.len() as u32);
            pp_limbs.extend(limbs);
        }

        // Precompute 2^(32*j) mod p[i] for all primes and limb positions
        let mut pow2_mod = vec![0u64; num_primes * max_limbs];
        for (i, &p) in basis.primes.iter().enumerate() {
            let p64 = p as u64;
            let mut power: u64 = 1;
            let multiplier = ((1u64 << 32) % p64) as u64;

            for j in 0..max_limbs {
                pow2_mod[i * max_limbs + j] = power;
                power = (power * multiplier) % p64;
            }
        }

        // Convert product and half_product to limbs
        let product_limbs = Self::bigint_to_limbs(&basis.product);
        let half_product_limbs = Self::bigint_to_limbs(&basis.half_product);

        Self {
            primes: basis.primes.clone(),
            garner_inverses: basis.garner_inverses_u32.clone(),
            pp_limbs,
            pp_offsets,
            pp_sizes,
            pow2_mod,
            max_limbs,
            product_limbs,
            half_product_limbs,
        }
    }

    /// Convert BigInt to little-endian u32 limbs
    fn bigint_to_limbs(n: &num_bigint::BigInt) -> Vec<u32> {
        use num_bigint::Sign;

        let (sign, digits) = n.to_u32_digits();
        if digits.is_empty() || sign == Sign::NoSign {
            vec![0]
        } else {
            digits
        }
    }
}

/// Precomputed data for GPU CRT reconstruction with 64-bit primes (V2)
#[derive(Clone)]
pub struct GpuCrtPrecomputed64 {
    pub primes: Vec<u64>,
    pub garner_inverses: Vec<u64>,
    pub pp_limbs: Vec<u32>,        // Packed partial product limbs (32-bit)
    pub pp_offsets: Vec<u32>,      // Offset into pp_limbs for each prime
    pub pp_sizes: Vec<u32>,        // Number of limbs for each partial product
    pub pow2_mod: Vec<u64>,        // [num_primes][max_limbs] 2^(32*j) mod p[i]
    pub max_limbs: usize,
    pub product_limbs: Vec<u32>,   // Full product M as limbs
    pub half_product_limbs: Vec<u32>,  // M/2 as limbs
}

impl GpuCrtPrecomputed64 {
    /// Create precomputed data from a V2 CRT basis (62-bit primes)
    pub fn from_basis(basis: &parallel_lift_core::solve_v2::CRTBasis62) -> Self {
        let num_primes = basis.len();

        // Estimate max limbs needed: ~(num_primes * 62 bits) / 32 bits
        let max_limbs = (num_primes * 62 + 31) / 32 + 4;

        // Pack partial products into limb array
        let mut pp_limbs = Vec::new();
        let mut pp_offsets = Vec::with_capacity(num_primes);
        let mut pp_sizes = Vec::with_capacity(num_primes);

        for pp in &basis.partial_products {
            pp_offsets.push(pp_limbs.len() as u32);
            let limbs = Self::bigint_to_limbs(pp);
            pp_sizes.push(limbs.len() as u32);
            pp_limbs.extend(limbs);
        }

        // Precompute 2^(32*j) mod p[i] for all primes and limb positions
        let mut pow2_mod = vec![0u64; num_primes * max_limbs];
        for (i, &p) in basis.primes.iter().enumerate() {
            let mut power: u64 = 1;
            let multiplier = Self::mod_2_32(p);

            for j in 0..max_limbs {
                pow2_mod[i * max_limbs + j] = power;
                power = Self::mod_mul_128(power, multiplier, p);
            }
        }

        // Convert product and half_product to limbs
        let product_limbs = Self::bigint_to_limbs(&basis.product);
        let half_product_limbs = Self::bigint_to_limbs(&basis.half_product);

        Self {
            primes: basis.primes.clone(),
            garner_inverses: basis.garner_inverses.clone(),
            pp_limbs,
            pp_offsets,
            pp_sizes,
            pow2_mod,
            max_limbs,
            product_limbs,
            half_product_limbs,
        }
    }

    /// Compute 2^32 mod p for 64-bit prime
    fn mod_2_32(p: u64) -> u64 {
        (1u64 << 32) % p
    }

    /// Modular multiplication with 128-bit intermediate
    fn mod_mul_128(a: u64, b: u64, p: u64) -> u64 {
        ((a as u128 * b as u128) % p as u128) as u64
    }

    /// Convert BigInt to little-endian u32 limbs
    fn bigint_to_limbs(n: &num_bigint::BigInt) -> Vec<u32> {
        use num_bigint::Sign;

        let (sign, digits) = n.to_u32_digits();
        if digits.is_empty() || sign == Sign::NoSign {
            vec![0]
        } else {
            digits
        }
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> &'static str {
        "CUDA"
    }

    fn determinant_mod(&self, matrix: &[u32], n: usize, p: u32) -> u32 {
        // For single prime, use CPU (GPU dispatch overhead not worth it)
        self.cpu_determinant(matrix, n, p)
    }

    fn solve_mod(&self, matrix: &[u32], b: &[u32], n: usize, p: u32) -> Option<Vec<u32>> {
        // For single prime, use CPU
        self.cpu_solve(matrix, b, n, p)
    }

    fn solve_multi_rhs_mod(
        &self,
        matrix: &[u32],
        b_cols: &[Vec<u32>],
        n: usize,
        _k: usize,
        p: u32,
    ) -> Option<Vec<Vec<u32>>> {
        // For single prime, solve each RHS using CPU
        b_cols.iter().map(|b| self.cpu_solve(matrix, b, n, p)).collect()
    }

    fn batch_determinant_mod(&self, matrix: &[u32], n: usize, primes: &[u32]) -> Vec<u32> {
        // Use GPU for batched operations - this is where we get the speedup!
        self.gpu_batch_determinant(matrix, n, primes)
    }

    fn batch_solve_mod(
        &self,
        matrix: &[u32],
        b: &[u32],
        n: usize,
        primes: &[u32],
    ) -> Option<Vec<Vec<u32>>> {
        // Use GPU for batched operations
        self.gpu_batch_solve(matrix, b, n, primes)
    }

    fn batch_multi_rhs_solve_mod(
        &self,
        matrix: &[u32],
        b_cols: &[Vec<u32>],
        n: usize,
        k: usize,
        primes: &[u32],
    ) -> Option<Vec<Vec<Vec<u32>>>> {
        // Use GPU for batched multi-RHS solve - the key ZK preprocessing optimization
        self.gpu_batch_multi_rhs_solve(matrix, b_cols, n, k, primes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_backend_creation() {
        if let Some(backend) = CudaBackend::new() {
            assert_eq!(backend.name(), "CUDA");
        }
        // Skip test if CUDA not available
    }

    #[test]
    fn test_cuda_determinant() {
        if let Some(backend) = CudaBackend::new() {
            let matrix = vec![1, 2, 3, 4];
            let det = backend.determinant_mod(&matrix, 2, 101);
            assert_eq!(det, 99); // -2 mod 101
        }
    }

    #[test]
    fn test_cuda_solve() {
        if let Some(backend) = CudaBackend::new() {
            // Ax = b where A = [[2, 1], [1, 3]], b = [5, 10]
            // Solution: x = [1, 3]
            let matrix = vec![2, 1, 1, 3];
            let b = vec![5, 10];
            let x = backend.solve_mod(&matrix, &b, 2, 101).unwrap();
            assert_eq!(x, vec![1, 3]);
        }
    }

    #[test]
    fn test_cuda_batch_determinant() {
        if let Some(backend) = CudaBackend::new() {
            // Matrix [[1, 2], [3, 4]] has det = -2
            let matrix = vec![1, 2, 3, 4];
            let primes = vec![101, 103, 107];

            let dets = backend.batch_determinant_mod(&matrix, 2, &primes);

            // -2 mod each prime
            assert_eq!(dets[0], 99); // -2 mod 101
            assert_eq!(dets[1], 101); // -2 mod 103
            assert_eq!(dets[2], 105); // -2 mod 107
        }
    }

    #[test]
    fn test_cuda_batch_solve() {
        if let Some(backend) = CudaBackend::new() {
            let matrix = vec![2, 1, 1, 3];
            let b = vec![5, 10];
            let primes = vec![101, 103, 107];

            let solutions = backend.batch_solve_mod(&matrix, &b, 2, &primes).unwrap();

            // Solution should be [1, 3] mod each prime
            for sol in solutions {
                assert_eq!(sol, vec![1, 3]);
            }
        }
    }

    #[test]
    fn test_cuda_batch_multi_rhs() {
        if let Some(backend) = CudaBackend::new() {
            // A = [[2, 1], [1, 3]]
            // B = [[5, 7], [10, 11]] (two RHS vectors as columns)
            // Solutions: [1, 3] and [2, 3]
            let matrix = vec![2, 1, 1, 3];
            let b_cols = vec![vec![5, 10], vec![7, 11]];
            let primes = vec![101, 103];

            let solutions = backend
                .batch_multi_rhs_solve_mod(&matrix, &b_cols, 2, 2, &primes)
                .unwrap();

            // Each prime should have solutions for both RHS
            for sol_per_prime in &solutions {
                assert_eq!(sol_per_prime[0], vec![1, 3]); // First RHS solution
                assert_eq!(sol_per_prime[1], vec![2, 3]); // Second RHS solution
            }
        }
    }

    #[test]
    fn test_gpu_gram_matrix() {
        if let Some(backend) = CudaBackend::new() {
            // Simple 2x2 basis: [[1, 0], [0, 1]]
            // Gram matrix should be identity: [[1, 0], [0, 1]]
            let n = 2;
            let m = 2;
            let primes = vec![101, 103];
            let num_primes = primes.len();

            // Basis: identity vectors [1,0] and [0,1] for each prime
            let mut basis = vec![0u32; num_primes * n * m];
            for pi in 0..num_primes {
                basis[pi * n * m + 0 * m + 0] = 1; // b_0 = [1, 0]
                basis[pi * n * m + 1 * m + 1] = 1; // b_1 = [0, 1]
            }

            let gram = backend.gpu_batch_gram_matrix(&basis, n, m, &primes);

            // Check Gram matrix for first prime
            assert_eq!(gram[0 * n * n + 0 * n + 0], 1); // <b0,b0> = 1
            assert_eq!(gram[0 * n * n + 0 * n + 1], 0); // <b0,b1> = 0
            assert_eq!(gram[0 * n * n + 1 * n + 0], 0); // <b1,b0> = 0
            assert_eq!(gram[0 * n * n + 1 * n + 1], 1); // <b1,b1> = 1
        }
    }

    #[test]
    fn test_gpu_squared_norms() {
        if let Some(backend) = CudaBackend::new() {
            // Basis: [[3, 4], [1, 0]]
            // Norms: ||[3,4]||^2 = 25, ||[1,0]||^2 = 1
            let n = 2;
            let m = 2;
            let primes = vec![101];
            let num_primes = 1;

            let mut basis = vec![0u32; num_primes * n * m];
            basis[0 * n * m + 0 * m + 0] = 3; // b_0 = [3, 4]
            basis[0 * n * m + 0 * m + 1] = 4;
            basis[0 * n * m + 1 * m + 0] = 1; // b_1 = [1, 0]
            basis[0 * n * m + 1 * m + 1] = 0;

            let norms = backend.gpu_batch_squared_norms(&basis, n, m, &primes);

            assert_eq!(norms[0], 25); // ||b_0||^2 = 9 + 16 = 25
            assert_eq!(norms[1], 1);  // ||b_1||^2 = 1
        }
    }

    #[test]
    fn test_gpu_gram_schmidt() {
        if let Some(backend) = CudaBackend::new() {
            // Basis: [[1, 1], [1, 0]]
            // After GS: b*_0 = [1, 1], b*_1 = [1, 0] - (1/2)[1, 1] = [0.5, -0.5]
            // But in modular arithmetic with exact fractions
            let n = 2;
            let m = 2;
            let primes = vec![101];
            let num_primes = 1;

            let mut basis = vec![0u32; num_primes * n * m];
            basis[0 * m + 0] = 1; // b_0 = [1, 1]
            basis[0 * m + 1] = 1;
            basis[1 * m + 0] = 1; // b_1 = [1, 0]
            basis[1 * m + 1] = 0;

            let gs = backend.gpu_gram_schmidt(&basis, n, m, &primes);

            // Check that b*_0 = b_0 (first vector unchanged)
            assert_eq!(gs.get_norm_sq(0, 0), 2); // ||b*_0||^2 = 1 + 1 = 2

            // μ_{1,0} = <b_1, b*_0> / ||b*_0||^2 = 1/2
            let mu_n = gs.get_mu_num(0, 1, 0);
            let mu_d = gs.get_mu_den(0, 1, 0);
            assert_eq!(mu_n, 1); // numerator = 1
            assert_eq!(mu_d, 2); // denominator = 2
        }
    }

    // =========================================================================
    // GPU LLL End-to-End Tests
    // =========================================================================

    #[test]
    fn test_gpu_lll_simple_2d() {
        use parallel_lift_core::lattice::basis::LatticeBasis;

        if let Some(backend) = CudaBackend::new() {
            // Simple 2D lattice that needs reduction
            let basis = LatticeBasis::from_rows(&[
                vec![1i64, 1],
                vec![0, 1],
            ]);

            let config = GpuLLLConfig::default();
            let (reduced, stats) = gpu_lll_reduce(&backend, &basis, &config);

            // Check that reduction completed
            assert!(stats.iterations > 0 || basis.n <= 1);
            assert_eq!(reduced.n, 2);
            assert_eq!(reduced.m, 2);

            // The reduced basis should have reasonable norms
            let orig_norm = basis.norm_squared(0);
            let reduced_norm = reduced.norm_squared(0);
            println!("GPU LLL 2D: original norm² = {}, reduced norm² = {}", orig_norm, reduced_norm);
            println!("Stats: {:?}", stats);
        }
    }

    #[test]
    fn test_gpu_lll_identity_basis() {
        use parallel_lift_core::lattice::basis::LatticeBasis;

        if let Some(backend) = CudaBackend::new() {
            // Identity basis (already reduced)
            let basis = LatticeBasis::from_rows(&[
                vec![1i64, 0, 0],
                vec![0, 1, 0],
                vec![0, 0, 1],
            ]);

            let config = GpuLLLConfig::default();
            let (reduced, stats) = gpu_lll_reduce(&backend, &basis, &config);

            // Identity should require minimal work
            println!("GPU LLL Identity: swaps = {}, iterations = {}", stats.swaps, stats.iterations);

            // Check norms are preserved
            for i in 0..3 {
                assert_eq!(reduced.norm_squared(i), basis.norm_squared(i));
            }
        }
    }

    #[test]
    fn test_gpu_lll_knapsack() {
        use parallel_lift_core::lattice::basis::LatticeBasis;

        if let Some(backend) = CudaBackend::new() {
            // Small knapsack lattice
            let a = vec![3i64, 5, 7];
            let s = 12i64;
            let basis = LatticeBasis::knapsack(&a, s);

            let config = GpuLLLConfig::default();
            let (reduced, stats) = gpu_lll_reduce(&backend, &basis, &config);

            println!(
                "GPU LLL Knapsack: n={}, swaps={}, reductions={}, time={:.3}s",
                basis.n, stats.swaps, stats.size_reductions, stats.total_time
            );

            // Check basis dimensions preserved
            assert_eq!(reduced.n, basis.n);
            assert_eq!(reduced.m, basis.m);
        }
    }

    #[test]
    fn test_gpu_lll_random_5x5() {
        use parallel_lift_core::lattice::basis::LatticeBasis;

        if let Some(backend) = CudaBackend::new() {
            // Random 5x5 lattice
            let basis = LatticeBasis::random(5, 5, 8);

            let config = GpuLLLConfig::default();
            let (reduced, stats) = gpu_lll_reduce(&backend, &basis, &config);

            println!(
                "GPU LLL 5x5 random: swaps={}, reductions={}, iterations={}, time={:.3}s",
                stats.swaps, stats.size_reductions, stats.iterations, stats.total_time
            );

            // Check basis dimensions preserved
            assert_eq!(reduced.n, 5);
            assert_eq!(reduced.m, 5);

            // The first vector should not have grown much larger
            let orig_norm = basis.norm_squared(0);
            let reduced_norm = reduced.norm_squared(0);
            println!("Original first norm²: {}", orig_norm);
            println!("Reduced first norm²: {}", reduced_norm);
        }
    }

    #[test]
    fn test_gpu_lll_random_10x10() {
        use parallel_lift_core::lattice::basis::LatticeBasis;

        if let Some(backend) = CudaBackend::new() {
            // Random 10x10 lattice - larger test
            let basis = LatticeBasis::random(10, 10, 12);

            let config = GpuLLLConfig {
                verbose: false,
                ..Default::default()
            };
            let (reduced, stats) = gpu_lll_reduce(&backend, &basis, &config);

            println!(
                "GPU LLL 10x10 random: swaps={}, reductions={}, iterations={}, time={:.3}s",
                stats.swaps, stats.size_reductions, stats.iterations, stats.total_time
            );
            println!(
                "  GS time: {:.3}s, Lovasz time: {:.3}s, Swap time: {:.3}s",
                stats.gram_schmidt_time, stats.lovasz_check_time, stats.swap_time
            );

            // Check basis dimensions preserved
            assert_eq!(reduced.n, 10);
            assert_eq!(reduced.m, 10);
        }
    }

    #[test]
    fn test_basis_to_residues_roundtrip() {
        use parallel_lift_core::lattice::basis::LatticeBasis;
        use parallel_lift_core::primes::CRTBasis;

        if let Some(backend) = CudaBackend::new() {
            // Create a simple basis
            let basis = LatticeBasis::from_rows(&[
                vec![10i64, -5, 3],
                vec![-2, 8, 1],
            ]);

            let n = basis.n;
            let m = basis.m;

            // Create CRT basis with enough primes
            let crt_basis = CRTBasis::with_primes(16);
            let precomputed = GpuCrtPrecomputed::from_basis(&crt_basis);

            // Convert to residues
            let flat_basis = basis.to_flat();
            let residues = basis_to_residues(&flat_basis, n, m, &crt_basis);

            // Convert back
            let recovered = residues_to_basis(&backend, &residues, n, m, &precomputed);

            // Check that values match
            for i in 0..n {
                for j in 0..m {
                    assert_eq!(
                        flat_basis[i * m + j], recovered[i * m + j],
                        "Mismatch at ({}, {}): expected {}, got {}",
                        i, j, flat_basis[i * m + j], recovered[i * m + j]
                    );
                }
            }
        }
    }

    #[test]
    fn test_gpu_size_reduce_kernel() {
        if let Some(backend) = CudaBackend::new() {
            // Test the size reduction kernel directly
            let n = 2;
            let m = 2;
            let primes = vec![101u32, 103];
            let num_primes = primes.len();

            // Basis: [[10, 0], [3, 1]] for each prime
            let mut basis = vec![0u32; num_primes * n * m];
            for pi in 0..num_primes {
                basis[pi * n * m + 0 * m + 0] = 10;
                basis[pi * n * m + 0 * m + 1] = 0;
                basis[pi * n * m + 1 * m + 0] = 3;
                basis[pi * n * m + 1 * m + 1] = 1;
            }

            // Apply size reduction: b_1 = b_1 - 0 * b_0 (should be no-op)
            let mu_rounded = vec![0u32; num_primes];
            backend.gpu_batch_size_reduce(&mut basis, &mu_rounded, 1, 0, n, m, &primes);

            // Verify no change when mu_rounded is 0
            assert_eq!(basis[0 * n * m + 1 * m + 0], 3);
            assert_eq!(basis[0 * n * m + 1 * m + 1], 1);
        }
    }
}
