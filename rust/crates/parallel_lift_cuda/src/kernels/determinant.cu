// Batched modular determinant kernels for CUDA
// Computes det(A) mod p for multiple primes in parallel

#include <stdint.h>
#include "modular_ops.cuh"

// Serial kernel: 1 thread per prime (n < 32)
extern "C" __global__ void modular_determinant(
    const uint32_t* __restrict__ matrices,  // [num_primes][n][n]
    const uint32_t* __restrict__ primes,    // [num_primes]
    uint32_t* __restrict__ results,         // [num_primes]
    uint32_t* __restrict__ singular_flags,  // [num_primes]
    uint32_t* __restrict__ workspace,       // [num_primes][n][n]
    uint32_t n,
    uint32_t num_primes
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_primes) return;

    uint32_t p = primes[tid];
    uint32_t nn = n * n;

    // Copy matrix to workspace
    const uint32_t* src = matrices + tid * nn;
    uint32_t* A = workspace + tid * nn;
    for (uint32_t i = 0; i < nn; i++) {
        A[i] = src[i];
    }

    uint32_t det = 1;
    bool singular = false;

    // Gaussian elimination with partial pivoting
    for (uint32_t k = 0; k < n && !singular; k++) {
        // Find pivot
        uint32_t pivot_row = k;
        for (uint32_t i = k; i < n; i++) {
            if (A[i * n + k] != 0) {
                pivot_row = i;
                break;
            }
        }

        if (A[pivot_row * n + k] == 0) {
            singular = true;
            det = 0;
            break;
        }

        // Swap rows
        if (pivot_row != k) {
            for (uint32_t j = k; j < n; j++) {
                uint32_t tmp = A[k * n + j];
                A[k * n + j] = A[pivot_row * n + j];
                A[pivot_row * n + j] = tmp;
            }
            det = (p - det) % p;  // Negate for swap
        }

        uint32_t pivot = A[k * n + k];
        det = mod_mul(det, pivot, p);
        uint32_t pivot_inv = mod_inv(pivot, p);

        // Eliminate below pivot
        for (uint32_t i = k + 1; i < n; i++) {
            uint32_t factor = mod_mul(A[i * n + k], pivot_inv, p);
            if (factor != 0) {
                for (uint32_t j = k; j < n; j++) {
                    uint32_t sub = mod_mul(factor, A[k * n + j], p);
                    A[i * n + j] = mod_sub(A[i * n + j], sub, p);
                }
            }
        }
    }

    results[tid] = det;
    singular_flags[tid] = singular ? 1 : 0;
}

// Small matrix kernel: thread-local storage for n <= 16
extern "C" __global__ void modular_determinant_small(
    const uint32_t* __restrict__ matrices,
    const uint32_t* __restrict__ primes,
    uint32_t* __restrict__ results,
    uint32_t* __restrict__ singular_flags,
    uint32_t n,
    uint32_t num_primes
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_primes) return;

    uint32_t p = primes[tid];
    uint32_t nn = n * n;

    // Thread-local storage (max 16x16 = 256 elements)
    uint32_t A[256];

    const uint32_t* src = matrices + tid * nn;
    for (uint32_t i = 0; i < nn; i++) {
        A[i] = src[i];
    }

    uint32_t det = 1;
    bool singular = false;

    for (uint32_t k = 0; k < n && !singular; k++) {
        uint32_t pivot_row = k;
        for (uint32_t i = k; i < n; i++) {
            if (A[i * n + k] != 0) {
                pivot_row = i;
                break;
            }
        }

        if (A[pivot_row * n + k] == 0) {
            singular = true;
            det = 0;
            break;
        }

        if (pivot_row != k) {
            for (uint32_t j = k; j < n; j++) {
                uint32_t tmp = A[k * n + j];
                A[k * n + j] = A[pivot_row * n + j];
                A[pivot_row * n + j] = tmp;
            }
            det = (p - det) % p;
        }

        uint32_t pivot = A[k * n + k];
        det = mod_mul(det, pivot, p);
        uint32_t pivot_inv = mod_inv(pivot, p);

        for (uint32_t i = k + 1; i < n; i++) {
            uint32_t factor = mod_mul(A[i * n + k], pivot_inv, p);
            if (factor != 0) {
                for (uint32_t j = k; j < n; j++) {
                    uint32_t sub = mod_mul(factor, A[k * n + j], p);
                    A[i * n + j] = mod_sub(A[i * n + j], sub, p);
                }
            }
        }
    }

    results[tid] = det;
    singular_flags[tid] = singular ? 1 : 0;
}

#define TILE_SIZE 16  // 16x16 = 256 threads per block

// Tiled kernel: 1 threadblock per prime (n >= 32)
extern "C" __global__ void modular_determinant_tiled(
    const uint32_t* __restrict__ matrices,
    const uint32_t* __restrict__ primes,
    uint32_t* __restrict__ results,
    uint32_t* __restrict__ singular_flags,
    uint32_t* __restrict__ workspace,
    uint32_t n,
    uint32_t num_primes
) {
    // One threadblock per prime
    uint32_t prime_idx = blockIdx.x;
    if (prime_idx >= num_primes) return;

    uint32_t p = primes[prime_idx];
    uint32_t nn = n * n;

    uint32_t tx = threadIdx.x;  // 0-15
    uint32_t ty = threadIdx.y;  // 0-15
    uint32_t local_id = ty * TILE_SIZE + tx;
    uint32_t num_threads = TILE_SIZE * TILE_SIZE;  // 256

    const uint32_t* src = matrices + prime_idx * nn;
    uint32_t* A = workspace + prime_idx * nn;

    // Shared memory for parallel operations
    __shared__ uint32_t pivot_row_shared;
    __shared__ uint32_t pivot_val_shared;
    __shared__ uint32_t pivot_inv_shared;
    __shared__ uint32_t det_sign;
    __shared__ bool is_singular;
    __shared__ uint32_t found_rows[256];
    __shared__ uint32_t found_vals[256];

    // Initialize shared state
    if (local_id == 0) {
        det_sign = 0;
        is_singular = false;
    }
    __syncthreads();

    // Parallel copy to workspace
    for (uint32_t i = local_id; i < nn; i += num_threads) {
        A[i] = src[i];
    }
    __syncthreads();

    // Gaussian elimination with parallel operations
    for (uint32_t k = 0; k < n; k++) {
        __syncthreads();
        if (is_singular) break;

        // Phase 1: Parallel pivot search
        uint32_t search_size = n - k;
        if (local_id < search_size) {
            uint32_t my_row = k + local_id;
            uint32_t val = A[my_row * n + k];
            found_rows[local_id] = (val != 0) ? my_row : n;
            found_vals[local_id] = val;
        } else {
            found_rows[local_id] = n;
            found_vals[local_id] = 0;
        }
        __syncthreads();

        // Tree reduction to find minimum row with non-zero value
        for (uint32_t stride = num_threads / 2; stride > 0; stride /= 2) {
            if (local_id < stride) {
                if (found_rows[local_id + stride] < found_rows[local_id]) {
                    found_rows[local_id] = found_rows[local_id + stride];
                    found_vals[local_id] = found_vals[local_id + stride];
                }
            }
            __syncthreads();
        }

        if (local_id == 0) {
            pivot_row_shared = found_rows[0];
            pivot_val_shared = found_vals[0];
            if (pivot_row_shared >= n || pivot_val_shared == 0) {
                is_singular = true;
            }
        }
        __syncthreads();

        if (is_singular) break;

        // Phase 2: Parallel row swap
        if (pivot_row_shared != k) {
            for (uint32_t j = k + local_id; j < n; j += num_threads) {
                uint32_t tmp = A[k * n + j];
                A[k * n + j] = A[pivot_row_shared * n + j];
                A[pivot_row_shared * n + j] = tmp;
            }
            if (local_id == 0) {
                det_sign ^= 1;
            }
        }
        __syncthreads();

        // Phase 3: Compute pivot inverse
        if (local_id == 0) {
            pivot_inv_shared = mod_inv(pivot_val_shared, p);
        }
        __syncthreads();

        // Phase 4: Parallel row elimination
        uint32_t rows_to_eliminate = n - k - 1;
        uint32_t cols_to_update = n - k;

        for (uint32_t row_off = ty; row_off < rows_to_eliminate; row_off += TILE_SIZE) {
            uint32_t i = k + 1 + row_off;
            uint32_t factor = mod_mul(A[i * n + k], pivot_inv_shared, p);

            if (factor != 0) {
                for (uint32_t col_off = tx; col_off < cols_to_update; col_off += TILE_SIZE) {
                    uint32_t j = k + col_off;
                    uint32_t sub = mod_mul(factor, A[k * n + j], p);
                    A[i * n + j] = mod_sub(A[i * n + j], sub, p);
                }
            }
        }
        __syncthreads();
    }

    // Compute final determinant
    if (local_id == 0) {
        if (is_singular) {
            results[prime_idx] = 0;
            singular_flags[prime_idx] = 1;
        } else {
            uint32_t det = 1;
            for (uint32_t i = 0; i < n; i++) {
                det = mod_mul(det, A[i * n + i], p);
            }
            if (det_sign == 1) {
                det = (p - det) % p;
            }
            results[prime_idx] = det;
            singular_flags[prime_idx] = 0;
        }
    }
}
