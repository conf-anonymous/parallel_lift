//! Metal backend implementation
//!
//! GPU-accelerated backend for CRT-based exact arithmetic.
//! Key design: One thread per prime - batch all primes and dispatch to GPU in parallel.

use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize,
};
use parallel_lift_core::Backend;

use crate::shaders;

/// Metal GPU backend for modular arithmetic
///
/// This backend accelerates CRT-based exact arithmetic by:
/// 1. Batching matrices for all primes and dispatching to GPU
/// 2. Each GPU thread handles one prime independently
/// 3. For multi-RHS, factors A once and solves all k RHS vectors
pub struct MetalBackend {
    device: Device,
    command_queue: CommandQueue,
    #[allow(dead_code)]
    library: Library,
    // Pipeline states for various kernels
    determinant_pipeline: ComputePipelineState,
    determinant_small_pipeline: ComputePipelineState,
    determinant_tiled_pipeline: ComputePipelineState,  // Threadgroup-parallel version
    solve_pipeline: ComputePipelineState,
    multi_rhs_pipeline: ComputePipelineState,
    multi_rhs_tiled_pipeline: ComputePipelineState,  // Threadgroup-parallel version
    sparse_matvec_pipeline: ComputePipelineState,
    sparse_matvec_single_pipeline: ComputePipelineState,
    // CRT kernels
    crt_garner_step_pipeline: ComputePipelineState,
    bigint_mod_prime_pipeline: ComputePipelineState,
    bigint_compare_half_pipeline: ComputePipelineState,
}

impl MetalBackend {
    /// Create a new Metal backend
    ///
    /// Returns None if Metal is not available
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let command_queue = device.new_command_queue();

        // Compile shaders
        let options = metal::CompileOptions::new();
        let library = device
            .new_library_with_source(shaders::get_shader_source(), &options)
            .ok()?;

        // Create pipeline states for each kernel
        let determinant_fn = library.get_function("modular_determinant", None).ok()?;
        let determinant_pipeline = device
            .new_compute_pipeline_state_with_function(&determinant_fn)
            .ok()?;

        let determinant_small_fn = library
            .get_function("modular_determinant_small", None)
            .ok()?;
        let determinant_small_pipeline = device
            .new_compute_pipeline_state_with_function(&determinant_small_fn)
            .ok()?;

        let solve_fn = library.get_function("modular_solve", None).ok()?;
        let solve_pipeline = device
            .new_compute_pipeline_state_with_function(&solve_fn)
            .ok()?;

        let multi_rhs_fn = library.get_function("modular_solve_multi_rhs", None).ok()?;
        let multi_rhs_pipeline = device
            .new_compute_pipeline_state_with_function(&multi_rhs_fn)
            .ok()?;

        // Tiled (threadgroup-parallel) kernels for larger matrices
        let determinant_tiled_fn = library
            .get_function("modular_determinant_tiled", None)
            .ok()?;
        let determinant_tiled_pipeline = device
            .new_compute_pipeline_state_with_function(&determinant_tiled_fn)
            .ok()?;

        let multi_rhs_tiled_fn = library
            .get_function("modular_solve_multi_rhs_tiled", None)
            .ok()?;
        let multi_rhs_tiled_pipeline = device
            .new_compute_pipeline_state_with_function(&multi_rhs_tiled_fn)
            .ok()?;

        // Sparse kernels
        let sparse_matvec_fn = library.get_function("sparse_matvec_csr", None).ok()?;
        let sparse_matvec_pipeline = device
            .new_compute_pipeline_state_with_function(&sparse_matvec_fn)
            .ok()?;

        let sparse_matvec_single_fn = library.get_function("sparse_matvec_csr_single", None).ok()?;
        let sparse_matvec_single_pipeline = device
            .new_compute_pipeline_state_with_function(&sparse_matvec_single_fn)
            .ok()?;

        // CRT kernels
        let crt_garner_step_fn = library.get_function("crt_garner_step", None).ok()?;
        let crt_garner_step_pipeline = device
            .new_compute_pipeline_state_with_function(&crt_garner_step_fn)
            .ok()?;

        let bigint_mod_prime_fn = library.get_function("bigint_mod_prime", None).ok()?;
        let bigint_mod_prime_pipeline = device
            .new_compute_pipeline_state_with_function(&bigint_mod_prime_fn)
            .ok()?;

        let bigint_compare_half_fn = library.get_function("bigint_compare_half", None).ok()?;
        let bigint_compare_half_pipeline = device
            .new_compute_pipeline_state_with_function(&bigint_compare_half_fn)
            .ok()?;

        Some(Self {
            device,
            command_queue,
            library,
            determinant_pipeline,
            determinant_small_pipeline,
            determinant_tiled_pipeline,
            solve_pipeline,
            multi_rhs_pipeline,
            multi_rhs_tiled_pipeline,
            sparse_matvec_pipeline,
            sparse_matvec_single_pipeline,
            crt_garner_step_pipeline,
            bigint_mod_prime_pipeline,
            bigint_compare_half_pipeline,
        })
    }

    /// Create a buffer with data (host → GPU transfer)
    /// Uses StorageModeManaged for optimal CPU-to-GPU transfers with explicit synchronization
    fn create_buffer<T: Copy>(&self, data: &[T]) -> Buffer {
        let size = (data.len() * std::mem::size_of::<T>()) as u64;
        let buffer = self
            .device
            .new_buffer(size, MTLResourceOptions::StorageModeManaged);
        unsafe {
            let ptr = buffer.contents() as *mut T;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        // Notify Metal that CPU has modified the buffer contents
        buffer.did_modify_range(metal::NSRange::new(0, size));
        buffer
    }

    /// Create an empty buffer for GPU output (GPU → host transfer)
    /// Uses StorageModeShared for easy CPU readback
    fn create_empty_buffer(&self, byte_size: usize) -> Buffer {
        self.device
            .new_buffer(byte_size as u64, MTLResourceOptions::StorageModeShared)
    }

    /// Create a GPU-private workspace buffer (GPU-only, no CPU access needed)
    /// Uses StorageModePrivate for maximum GPU performance
    fn create_workspace_buffer(&self, byte_size: usize) -> Buffer {
        self.device
            .new_buffer(byte_size as u64, MTLResourceOptions::StorageModePrivate)
    }

    /// Read data from buffer
    fn read_buffer<T: Copy + Default>(&self, buffer: &Buffer, count: usize) -> Vec<T> {
        let mut result = vec![T::default(); count];
        unsafe {
            let ptr = buffer.contents() as *const T;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), count);
        }
        result
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

        // Create buffers
        // Input buffers: StorageModeManaged (CPU → GPU)
        let matrices_buffer = self.create_buffer(&matrices_data);
        let primes_buffer = self.create_buffer(primes);
        // Output buffers: StorageModeShared (GPU → CPU)
        let results_buffer = self.create_empty_buffer(num_primes * 4);
        let singular_buffer = self.create_empty_buffer(num_primes * 4);
        // Workspace: StorageModePrivate (GPU-only, maximum performance)
        let workspace_buffer = self.create_workspace_buffer(num_primes * nn * 4);

        // Create command buffer
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Choose pipeline and dispatch strategy based on matrix size
        // For n >= 32, use tiled kernel with threadgroup parallelism within each prime
        // For n <= 16, use small kernel with thread-local storage
        // For 16 < n < 32, use serial kernel (tiled overhead not worth it)
        let use_tiled = n >= 32;

        if use_tiled {
            // Tiled kernel: one threadgroup (16x16 = 256 threads) per prime
            encoder.set_compute_pipeline_state(&self.determinant_tiled_pipeline);
            encoder.set_buffer(0, Some(&matrices_buffer), 0);
            encoder.set_buffer(1, Some(&primes_buffer), 0);
            encoder.set_buffer(2, Some(&results_buffer), 0);
            encoder.set_buffer(3, Some(&singular_buffer), 0);

            let n_val = n as u32;
            let num_primes_val = num_primes as u32;
            encoder.set_bytes(4, 4, &n_val as *const u32 as *const _);
            encoder.set_bytes(5, 4, &num_primes_val as *const u32 as *const _);
            encoder.set_buffer(6, Some(&workspace_buffer), 0);

            // Dispatch: num_primes threadgroups, each with 16x16 threads
            let threadgroups = MTLSize::new(num_primes as u64, 1, 1);
            let threadgroup_size = MTLSize::new(16, 16, 1);  // 256 threads per threadgroup

            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        } else {
            // Serial kernel: one thread per prime
            let pipeline = if n <= 16 {
                &self.determinant_small_pipeline
            } else {
                &self.determinant_pipeline
            };

            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&matrices_buffer), 0);
            encoder.set_buffer(1, Some(&primes_buffer), 0);
            encoder.set_buffer(2, Some(&results_buffer), 0);
            encoder.set_buffer(3, Some(&singular_buffer), 0);

            let n_val = n as u32;
            let num_primes_val = num_primes as u32;
            encoder.set_bytes(4, 4, &n_val as *const u32 as *const _);
            encoder.set_bytes(5, 4, &num_primes_val as *const u32 as *const _);

            if n > 16 {
                encoder.set_buffer(6, Some(&workspace_buffer), 0);
            }

            let thread_count = MTLSize::new(num_primes as u64, 1, 1);
            let threadgroup_size = MTLSize::new(
                (num_primes as u64).min(pipeline.max_total_threads_per_threadgroup()),
                1,
                1,
            );

            encoder.dispatch_threads(thread_count, threadgroup_size);
        }

        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        self.read_buffer(&results_buffer, num_primes)
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
        let aug_stride = n * (n + 1);

        // Prepare batched augmented matrices [A|b] for each prime
        let mut augmented_data = vec![0u32; num_primes * aug_stride];
        for (pi, &p) in primes.iter().enumerate() {
            let offset = pi * aug_stride;
            for row in 0..n {
                // Copy A row
                for col in 0..n {
                    augmented_data[offset + row * (n + 1) + col] = matrix[row * n + col] % p;
                }
                // Copy b element
                augmented_data[offset + row * (n + 1) + n] = b[row] % p;
            }
        }

        // Create buffers
        // Input buffers: StorageModeManaged (CPU → GPU)
        let augmented_buffer = self.create_buffer(&augmented_data);
        let primes_buffer = self.create_buffer(primes);
        // Output buffers: StorageModeShared (GPU → CPU)
        let solutions_buffer = self.create_empty_buffer(num_primes * n * 4);
        let singular_buffer = self.create_empty_buffer(num_primes * 4);
        // Workspace: StorageModePrivate (GPU-only, maximum performance)
        let workspace_buffer = self.create_workspace_buffer(num_primes * aug_stride * 4);

        // Create command buffer
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.solve_pipeline);
        encoder.set_buffer(0, Some(&augmented_buffer), 0);
        encoder.set_buffer(1, Some(&primes_buffer), 0);
        encoder.set_buffer(2, Some(&solutions_buffer), 0);
        encoder.set_buffer(3, Some(&singular_buffer), 0);

        let n_val = n as u32;
        let num_primes_val = num_primes as u32;
        encoder.set_bytes(4, 4, &n_val as *const u32 as *const _);
        encoder.set_bytes(5, 4, &num_primes_val as *const u32 as *const _);
        encoder.set_buffer(6, Some(&workspace_buffer), 0);

        // Dispatch
        let thread_count = MTLSize::new(num_primes as u64, 1, 1);
        let threadgroup_size = MTLSize::new(
            (num_primes as u64).min(self.solve_pipeline.max_total_threads_per_threadgroup()),
            1,
            1,
        );

        encoder.dispatch_threads(thread_count, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let solutions: Vec<u32> = self.read_buffer(&solutions_buffer, num_primes * n);
        let singular_flags: Vec<u32> = self.read_buffer(&singular_buffer, num_primes);

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

        // Create buffers
        // Input buffers: StorageModeManaged (CPU → GPU)
        let augmented_buffer = self.create_buffer(&augmented_data);
        let primes_buffer = self.create_buffer(primes);
        // Output buffers: StorageModeShared (GPU → CPU)
        let solutions_buffer = self.create_empty_buffer(num_primes * n * k * 4);
        let singular_buffer = self.create_empty_buffer(num_primes * 4);
        // Workspace: StorageModePrivate (GPU-only, maximum performance)
        let workspace_buffer = self.create_workspace_buffer(num_primes * aug_stride * 4);

        // Create command buffer
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Choose pipeline based on matrix size
        // For n >= 32, use tiled kernel with threadgroup parallelism
        let use_tiled = n >= 32;

        let n_val = n as u32;
        let k_val = k as u32;
        let num_primes_val = num_primes as u32;

        if use_tiled {
            // Tiled kernel: one threadgroup (16x16 = 256 threads) per prime
            encoder.set_compute_pipeline_state(&self.multi_rhs_tiled_pipeline);
            encoder.set_buffer(0, Some(&augmented_buffer), 0);
            encoder.set_buffer(1, Some(&primes_buffer), 0);
            encoder.set_buffer(2, Some(&solutions_buffer), 0);
            encoder.set_buffer(3, Some(&singular_buffer), 0);

            encoder.set_bytes(4, 4, &n_val as *const u32 as *const _);
            encoder.set_bytes(5, 4, &k_val as *const u32 as *const _);
            encoder.set_bytes(6, 4, &num_primes_val as *const u32 as *const _);
            encoder.set_buffer(7, Some(&workspace_buffer), 0);

            // Dispatch: num_primes threadgroups, each with 16x16 threads
            let threadgroups = MTLSize::new(num_primes as u64, 1, 1);
            let threadgroup_size = MTLSize::new(16, 16, 1);  // 256 threads per threadgroup

            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        } else {
            // Serial kernel: one thread per prime
            encoder.set_compute_pipeline_state(&self.multi_rhs_pipeline);
            encoder.set_buffer(0, Some(&augmented_buffer), 0);
            encoder.set_buffer(1, Some(&primes_buffer), 0);
            encoder.set_buffer(2, Some(&solutions_buffer), 0);
            encoder.set_buffer(3, Some(&singular_buffer), 0);

            encoder.set_bytes(4, 4, &n_val as *const u32 as *const _);
            encoder.set_bytes(5, 4, &k_val as *const u32 as *const _);
            encoder.set_bytes(6, 4, &num_primes_val as *const u32 as *const _);
            encoder.set_buffer(7, Some(&workspace_buffer), 0);

            let thread_count = MTLSize::new(num_primes as u64, 1, 1);
            let threadgroup_size = MTLSize::new(
                (num_primes as u64).min(self.multi_rhs_pipeline.max_total_threads_per_threadgroup()),
                1,
                1,
            );

            encoder.dispatch_threads(thread_count, threadgroup_size);
        }

        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let solutions: Vec<u32> = self.read_buffer(&solutions_buffer, num_primes * n * k);
        let singular_flags: Vec<u32> = self.read_buffer(&singular_buffer, num_primes);

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
    ///
    /// CSR format: row_ptr[n+1], col_idx[nnz], values[nnz]
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

        // Create GPU buffers
        let row_ptr_buf = self.create_buffer(row_ptr);
        let col_idx_buf = self.create_buffer(col_idx);
        let values_buf = self.create_buffer(values);
        let x_buf = self.create_buffer(x);
        let y_buf = self.create_empty_buffer(n * std::mem::size_of::<u32>());
        let p_buf = self.create_buffer(&[p]);
        let n_buf = self.create_buffer(&[n as u32]);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.sparse_matvec_single_pipeline);
        encoder.set_buffer(0, Some(&row_ptr_buf), 0);
        encoder.set_buffer(1, Some(&col_idx_buf), 0);
        encoder.set_buffer(2, Some(&values_buf), 0);
        encoder.set_buffer(3, Some(&x_buf), 0);
        encoder.set_buffer(4, Some(&y_buf), 0);
        encoder.set_buffer(5, Some(&p_buf), 0);
        encoder.set_buffer(6, Some(&n_buf), 0);

        let threads_per_group = 256.min(n as u64);

        encoder.dispatch_threads(
            MTLSize::new(n as u64, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        self.read_buffer(&y_buf, n)
    }

    /// GPU batched sparse matrix-vector multiply: y = A * x mod p for multiple primes
    ///
    /// CSR format is shared across primes (structure), but values differ per prime
    pub fn sparse_matvec_batch(
        &self,
        row_ptr: &[u32],
        col_idx: &[u32],
        values_per_prime: &[Vec<u32>],  // values[prime_idx][nnz_idx]
        x_per_prime: &[Vec<u32>],       // x[prime_idx][n]
        primes: &[u32],
    ) -> Vec<Vec<u32>> {
        let n = row_ptr.len() - 1;
        let nnz = col_idx.len();
        let num_primes = primes.len();

        // Flatten values and x for GPU
        let values_flat: Vec<u32> = values_per_prime.iter().flatten().copied().collect();
        let x_flat: Vec<u32> = x_per_prime.iter().flatten().copied().collect();

        // Create GPU buffers
        let row_ptr_buf = self.create_buffer(row_ptr);
        let col_idx_buf = self.create_buffer(col_idx);
        let values_buf = self.create_buffer(&values_flat);
        let x_buf = self.create_buffer(&x_flat);
        let y_buf = self.create_empty_buffer(num_primes * n * std::mem::size_of::<u32>());
        let primes_buf = self.create_buffer(primes);
        let n_buf = self.create_buffer(&[n as u32]);
        let nnz_buf = self.create_buffer(&[nnz as u32]);
        let num_primes_buf = self.create_buffer(&[num_primes as u32]);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.sparse_matvec_pipeline);
        encoder.set_buffer(0, Some(&row_ptr_buf), 0);
        encoder.set_buffer(1, Some(&col_idx_buf), 0);
        encoder.set_buffer(2, Some(&values_buf), 0);
        encoder.set_buffer(3, Some(&x_buf), 0);
        encoder.set_buffer(4, Some(&y_buf), 0);
        encoder.set_buffer(5, Some(&primes_buf), 0);
        encoder.set_buffer(6, Some(&n_buf), 0);
        encoder.set_buffer(7, Some(&nnz_buf), 0);
        encoder.set_buffer(8, Some(&num_primes_buf), 0);

        let total_threads = (num_primes * n) as u64;
        let threads_per_group = 256u64;

        encoder.dispatch_threads(
            MTLSize::new(total_threads, 1, 1),
            MTLSize::new(threads_per_group.min(total_threads), 1, 1),
        );
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let y_flat: Vec<u32> = self.read_buffer(&y_buf, num_primes * n);

        // Reshape to per-prime vectors
        y_flat
            .chunks(n)
            .map(|chunk| chunk.to_vec())
            .collect()
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

    /// GPU-accelerated batch computation of BigInt mod prime
    ///
    /// Computes result[v] = bigints[v] mod p for all values in parallel.
    /// BigInts are represented as limbs (u32 words, little-endian).
    ///
    /// This is useful for the Garner step in CRT where we need result mod p_i.
    pub fn gpu_batch_bigint_mod(
        &self,
        limbs: &[u32],  // Flattened: num_values x num_limbs
        num_values: usize,
        num_limbs: usize,
        p: u32,
    ) -> Vec<u32> {
        // Precompute 2^(32*i) mod p for each limb position
        let mut pow2_mods = vec![0u32; num_limbs];
        let p64 = p as u64;
        let mut pow = 1u64;
        for i in 0..num_limbs {
            pow2_mods[i] = pow as u32;
            // pow = pow * 2^32 mod p
            pow = (pow << 32) % p64;
        }

        // Create buffers
        let limbs_buf = self.create_buffer(limbs);
        let pow2_mods_buf = self.create_buffer(&pow2_mods);
        let results_buf = self.create_empty_buffer(num_values * std::mem::size_of::<u32>());

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.bigint_mod_prime_pipeline);
        encoder.set_buffer(0, Some(&limbs_buf), 0);
        encoder.set_buffer(1, Some(&pow2_mods_buf), 0);
        encoder.set_buffer(2, Some(&results_buf), 0);

        let num_values_val = num_values as u32;
        let num_limbs_val = num_limbs as u32;
        encoder.set_bytes(3, 4, &num_values_val as *const u32 as *const _);
        encoder.set_bytes(4, 4, &num_limbs_val as *const u32 as *const _);
        encoder.set_bytes(5, 4, &p as *const u32 as *const _);

        let threads_per_group = 256u64.min(num_values as u64);
        encoder.dispatch_threads(
            MTLSize::new(num_values as u64, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        self.read_buffer(&results_buf, num_values)
    }

    /// GPU-accelerated Garner step computation
    ///
    /// Computes t[v] = (residues[v][i] - result_mods[v]) * inv[i] mod primes[i]
    /// for all values in parallel.
    pub fn gpu_garner_step(
        &self,
        residues: &[u32],       // num_values x num_primes (row-major)
        result_mods: &[u32],    // num_values: result mod current prime
        primes: &[u32],         // All primes
        inverses: &[u32],       // Garner inverses (u32)
        num_values: usize,
        num_primes: usize,
        current_prime_idx: usize,
    ) -> Vec<u32> {
        let residues_buf = self.create_buffer(residues);
        let result_mods_buf = self.create_buffer(result_mods);
        let primes_buf = self.create_buffer(primes);
        let inverses_buf = self.create_buffer(inverses);
        let t_values_buf = self.create_empty_buffer(num_values * std::mem::size_of::<u32>());

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.crt_garner_step_pipeline);
        encoder.set_buffer(0, Some(&residues_buf), 0);
        encoder.set_buffer(1, Some(&result_mods_buf), 0);
        encoder.set_buffer(2, Some(&primes_buf), 0);
        encoder.set_buffer(3, Some(&inverses_buf), 0);
        encoder.set_buffer(4, Some(&t_values_buf), 0);

        let num_values_val = num_values as u32;
        let num_primes_val = num_primes as u32;
        let current_prime_idx_val = current_prime_idx as u32;
        encoder.set_bytes(5, 4, &num_values_val as *const u32 as *const _);
        encoder.set_bytes(6, 4, &num_primes_val as *const u32 as *const _);
        encoder.set_bytes(7, 4, &current_prime_idx_val as *const u32 as *const _);

        let threads_per_group = 256u64.min(num_values as u64);
        encoder.dispatch_threads(
            MTLSize::new(num_values as u64, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        self.read_buffer(&t_values_buf, num_values)
    }

    /// GPU-accelerated sign detection for symmetric range conversion
    ///
    /// Returns 1 for each value where bigint > half_product, 0 otherwise.
    pub fn gpu_batch_sign_detect(
        &self,
        limbs: &[u32],          // num_values x num_limbs
        half_limbs: &[u32],     // num_limbs: M/2 as limbs
        num_values: usize,
        num_limbs: usize,
    ) -> Vec<u32> {
        let limbs_buf = self.create_buffer(limbs);
        let half_limbs_buf = self.create_buffer(half_limbs);
        let results_buf = self.create_empty_buffer(num_values * std::mem::size_of::<u32>());

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.bigint_compare_half_pipeline);
        encoder.set_buffer(0, Some(&limbs_buf), 0);
        encoder.set_buffer(1, Some(&half_limbs_buf), 0);
        encoder.set_buffer(2, Some(&results_buf), 0);

        let num_values_val = num_values as u32;
        let num_limbs_val = num_limbs as u32;
        encoder.set_bytes(3, 4, &num_values_val as *const u32 as *const _);
        encoder.set_bytes(4, 4, &num_limbs_val as *const u32 as *const _);

        let threads_per_group = 256u64.min(num_values as u64);
        encoder.dispatch_threads(
            MTLSize::new(num_values as u64, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        self.read_buffer(&results_buf, num_values)
    }
}

impl Backend for MetalBackend {
    fn name(&self) -> &'static str {
        "Metal"
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
    fn test_metal_backend_creation() {
        if let Some(backend) = MetalBackend::new() {
            assert_eq!(backend.name(), "Metal");
        }
        // Skip test if Metal not available
    }

    #[test]
    fn test_metal_determinant() {
        if let Some(backend) = MetalBackend::new() {
            let matrix = vec![1, 2, 3, 4];
            let det = backend.determinant_mod(&matrix, 2, 101);
            assert_eq!(det, 99); // -2 mod 101
        }
    }

    #[test]
    fn test_metal_solve() {
        if let Some(backend) = MetalBackend::new() {
            // Ax = b where A = [[2, 1], [1, 3]], b = [5, 10]
            // Solution: x = [1, 3]
            let matrix = vec![2, 1, 1, 3];
            let b = vec![5, 10];
            let x = backend.solve_mod(&matrix, &b, 2, 101).unwrap();
            assert_eq!(x, vec![1, 3]);
        }
    }

    #[test]
    fn test_metal_batch_determinant() {
        if let Some(backend) = MetalBackend::new() {
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
    fn test_metal_batch_solve() {
        if let Some(backend) = MetalBackend::new() {
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
    fn test_metal_batch_multi_rhs() {
        if let Some(backend) = MetalBackend::new() {
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
    fn test_metal_sparse_matvec_single() {
        if let Some(backend) = MetalBackend::new() {
            // Sparse matrix: [[2, 0], [0, 3]]
            // CSR format:
            //   row_ptr = [0, 1, 2]
            //   col_idx = [0, 1]
            //   values = [2, 3]
            // x = [5, 7]
            // Expected: y = [2*5, 3*7] = [10, 21]
            let row_ptr = vec![0, 1, 2];
            let col_idx = vec![0, 1];
            let values = vec![2, 3];
            let x = vec![5, 7];
            let p = 101;

            let y = backend.sparse_matvec_single(&row_ptr, &col_idx, &values, &x, p);
            assert_eq!(y, vec![10, 21]);
        }
    }

    #[test]
    fn test_metal_sparse_matvec_batch() {
        if let Some(backend) = MetalBackend::new() {
            // Sparse matrix: [[2, 0], [0, 3]]
            let row_ptr = vec![0, 1, 2];
            let col_idx = vec![0, 1];
            let primes = vec![101, 103];

            // Values reduced mod each prime (same values since they're small)
            let values_per_prime = vec![vec![2, 3], vec![2, 3]];
            let x_per_prime = vec![vec![5, 7], vec![5, 7]];

            let y_per_prime = backend.sparse_matvec_batch(
                &row_ptr,
                &col_idx,
                &values_per_prime,
                &x_per_prime,
                &primes,
            );

            // Expected: y = [10, 21] for both primes
            assert_eq!(y_per_prime[0], vec![10, 21]);
            assert_eq!(y_per_prime[1], vec![10, 21]);
        }
    }
}
