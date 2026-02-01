// Batched modular solve kernels for CUDA
// Solves Ax = b mod p for multiple primes in parallel

#include <stdint.h>
#include "modular_ops.cuh"

// Serial single-RHS solve: 1 thread per prime
extern "C" __global__ void modular_solve(
    const uint32_t* __restrict__ augmented,  // [num_primes][n][n+1]
    const uint32_t* __restrict__ primes,
    uint32_t* __restrict__ solutions,        // [num_primes][n]
    uint32_t* __restrict__ singular_flags,
    uint32_t* __restrict__ workspace,        // [num_primes][n][n+1]
    uint32_t n,
    uint32_t num_primes
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_primes) return;

    uint32_t p = primes[tid];
    uint32_t aug_width = n + 1;
    uint32_t stride = n * aug_width;

    // Copy augmented matrix to workspace
    const uint32_t* src = augmented + tid * stride;
    uint32_t* A = workspace + tid * stride;
    for (uint32_t i = 0; i < stride; i++) {
        A[i] = src[i];
    }

    bool singular = false;

    // Forward elimination with Gauss-Jordan (full pivot to identity)
    for (uint32_t col = 0; col < n && !singular; col++) {
        // Find pivot
        uint32_t pivot_row = col;
        for (uint32_t row = col; row < n; row++) {
            if (A[row * aug_width + col] != 0) {
                pivot_row = row;
                break;
            }
        }

        if (A[pivot_row * aug_width + col] == 0) {
            singular = true;
            break;
        }

        // Swap rows
        if (pivot_row != col) {
            for (uint32_t j = col; j < aug_width; j++) {
                uint32_t tmp = A[col * aug_width + j];
                A[col * aug_width + j] = A[pivot_row * aug_width + j];
                A[pivot_row * aug_width + j] = tmp;
            }
        }

        // Scale pivot row
        uint32_t pivot = A[col * aug_width + col];
        uint32_t pivot_inv = mod_inv(pivot, p);

        for (uint32_t j = col; j < aug_width; j++) {
            A[col * aug_width + j] = mod_mul(A[col * aug_width + j], pivot_inv, p);
        }

        // Eliminate in all other rows (Gauss-Jordan)
        for (uint32_t row = 0; row < n; row++) {
            if (row == col) continue;
            uint32_t factor = A[row * aug_width + col];
            if (factor != 0) {
                for (uint32_t j = col; j < aug_width; j++) {
                    uint32_t sub = mod_mul(factor, A[col * aug_width + j], p);
                    A[row * aug_width + j] = mod_sub(A[row * aug_width + j], sub, p);
                }
            }
        }
    }

    singular_flags[tid] = singular ? 1 : 0;

    // Solution is in the last column
    uint32_t* X = solutions + tid * n;
    if (singular) {
        for (uint32_t i = 0; i < n; i++) {
            X[i] = 0;
        }
    } else {
        for (uint32_t i = 0; i < n; i++) {
            X[i] = A[i * aug_width + n];
        }
    }
}

// Serial multi-RHS solve: 1 thread per prime
extern "C" __global__ void modular_solve_multi_rhs(
    const uint32_t* __restrict__ augmented,  // [num_primes][n][n+k]
    const uint32_t* __restrict__ primes,
    uint32_t* __restrict__ solutions,        // [num_primes][k][n] column-major per prime
    uint32_t* __restrict__ singular_flags,
    uint32_t* __restrict__ workspace,        // [num_primes][n][n+k]
    uint32_t n,
    uint32_t k,
    uint32_t num_primes
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_primes) return;

    uint32_t p = primes[tid];
    uint32_t aug_width = n + k;
    uint32_t stride = n * aug_width;

    // Copy augmented matrix to workspace
    const uint32_t* src = augmented + tid * stride;
    uint32_t* A = workspace + tid * stride;
    for (uint32_t i = 0; i < stride; i++) {
        A[i] = src[i];
    }

    bool singular = false;

    // Forward elimination
    for (uint32_t col = 0; col < n && !singular; col++) {
        // Find pivot
        uint32_t pivot_row = col;
        for (uint32_t row = col; row < n; row++) {
            if (A[row * aug_width + col] != 0) {
                pivot_row = row;
                break;
            }
        }

        if (A[pivot_row * aug_width + col] == 0) {
            singular = true;
            break;
        }

        // Swap rows (including all k RHS columns)
        if (pivot_row != col) {
            for (uint32_t j = col; j < aug_width; j++) {
                uint32_t tmp = A[col * aug_width + j];
                A[col * aug_width + j] = A[pivot_row * aug_width + j];
                A[pivot_row * aug_width + j] = tmp;
            }
        }

        uint32_t pivot = A[col * aug_width + col];
        uint32_t pivot_inv = mod_inv(pivot, p);

        // Normalize pivot row
        for (uint32_t j = col; j < aug_width; j++) {
            A[col * aug_width + j] = mod_mul(A[col * aug_width + j], pivot_inv, p);
        }

        // Eliminate below
        for (uint32_t row = col + 1; row < n; row++) {
            uint32_t factor = A[row * aug_width + col];
            if (factor != 0) {
                for (uint32_t j = col; j < aug_width; j++) {
                    uint32_t sub = mod_mul(factor, A[col * aug_width + j], p);
                    A[row * aug_width + j] = mod_sub(A[row * aug_width + j], sub, p);
                }
            }
        }
    }

    singular_flags[tid] = singular ? 1 : 0;

    if (singular) {
        // Zero fill solutions
        for (uint32_t i = 0; i < n * k; i++) {
            solutions[tid * n * k + i] = 0;
        }
        return;
    }

    // Back substitution for all k RHS columns
    uint32_t* X = solutions + tid * n * k;

    for (uint32_t rhs = 0; rhs < k; rhs++) {
        for (int row = n - 1; row >= 0; row--) {
            uint32_t sum = A[row * aug_width + (n + rhs)];
            for (uint32_t j = row + 1; j < n; j++) {
                uint32_t sub = mod_mul(A[row * aug_width + j], X[rhs * n + j], p);
                sum = mod_sub(sum, sub, p);
            }
            X[rhs * n + row] = sum;
        }
    }
}

#define TILE_SIZE 16  // 16x16 = 256 threads per block

// Tiled multi-RHS solve: 1 threadblock per prime (n >= 32)
extern "C" __global__ void modular_solve_multi_rhs_tiled(
    const uint32_t* __restrict__ augmented,
    const uint32_t* __restrict__ primes,
    uint32_t* __restrict__ solutions,
    uint32_t* __restrict__ singular_flags,
    uint32_t* __restrict__ workspace,
    uint32_t n,
    uint32_t k,
    uint32_t num_primes
) {
    uint32_t prime_idx = blockIdx.x;
    if (prime_idx >= num_primes) return;

    uint32_t p = primes[prime_idx];
    uint32_t aug_width = n + k;
    uint32_t stride = n * aug_width;

    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;
    uint32_t local_id = ty * TILE_SIZE + tx;
    uint32_t num_threads = TILE_SIZE * TILE_SIZE;

    const uint32_t* src = augmented + prime_idx * stride;
    uint32_t* A = workspace + prime_idx * stride;

    __shared__ uint32_t pivot_row_shared;
    __shared__ uint32_t pivot_val_shared;
    __shared__ uint32_t pivot_inv_shared;
    __shared__ bool is_singular;
    __shared__ uint32_t found_rows[256];
    __shared__ uint32_t found_vals[256];

    if (local_id == 0) {
        is_singular = false;
    }
    __syncthreads();

    // Parallel copy
    for (uint32_t i = local_id; i < stride; i += num_threads) {
        A[i] = src[i];
    }
    __syncthreads();

    // Forward elimination
    for (uint32_t col = 0; col < n; col++) {
        __syncthreads();
        if (is_singular) break;

        // Parallel pivot search
        uint32_t search_size = n - col;
        if (local_id < search_size) {
            uint32_t my_row = col + local_id;
            uint32_t val = A[my_row * aug_width + col];
            found_rows[local_id] = (val != 0) ? my_row : n;
            found_vals[local_id] = val;
        } else {
            found_rows[local_id] = n;
            found_vals[local_id] = 0;
        }
        __syncthreads();

        for (uint32_t s = num_threads / 2; s > 0; s /= 2) {
            if (local_id < s) {
                if (found_rows[local_id + s] < found_rows[local_id]) {
                    found_rows[local_id] = found_rows[local_id + s];
                    found_vals[local_id] = found_vals[local_id + s];
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

        // Parallel row swap
        if (pivot_row_shared != col) {
            for (uint32_t j = col + local_id; j < aug_width; j += num_threads) {
                uint32_t tmp = A[col * aug_width + j];
                A[col * aug_width + j] = A[pivot_row_shared * aug_width + j];
                A[pivot_row_shared * aug_width + j] = tmp;
            }
        }
        __syncthreads();

        if (local_id == 0) {
            pivot_inv_shared = mod_inv(pivot_val_shared, p);
        }
        __syncthreads();

        // Parallel row normalization
        for (uint32_t j = col + local_id; j < aug_width; j += num_threads) {
            A[col * aug_width + j] = mod_mul(A[col * aug_width + j], pivot_inv_shared, p);
        }
        __syncthreads();

        // Parallel elimination
        uint32_t rows_to_elim = n - col - 1;
        uint32_t cols_to_update = aug_width - col;

        for (uint32_t row_off = ty; row_off < rows_to_elim; row_off += TILE_SIZE) {
            uint32_t i = col + 1 + row_off;
            uint32_t factor = A[i * aug_width + col];

            if (factor != 0) {
                for (uint32_t col_off = tx; col_off < cols_to_update; col_off += TILE_SIZE) {
                    uint32_t j = col + col_off;
                    uint32_t sub = mod_mul(factor, A[col * aug_width + j], p);
                    A[i * aug_width + j] = mod_sub(A[i * aug_width + j], sub, p);
                }
            }
        }
        __syncthreads();
    }

    singular_flags[prime_idx] = is_singular ? 1 : 0;

    if (is_singular) {
        for (uint32_t i = local_id; i < n * k; i += num_threads) {
            solutions[prime_idx * n * k + i] = 0;
        }
        return;
    }

    // Parallel back substitution
    // Each thread handles different RHS columns
    uint32_t* X = solutions + prime_idx * n * k;

    // Back substitution must be done row by row (sequential in row dimension)
    for (int row = n - 1; row >= 0; row--) {
        // Parallel over RHS columns
        for (uint32_t rhs = local_id; rhs < k; rhs += num_threads) {
            uint32_t sum = A[row * aug_width + (n + rhs)];
            for (uint32_t j = row + 1; j < n; j++) {
                uint32_t sub = mod_mul(A[row * aug_width + j], X[rhs * n + j], p);
                sum = mod_sub(sum, sub, p);
            }
            X[rhs * n + row] = sum;
        }
        __syncthreads();
    }
}
