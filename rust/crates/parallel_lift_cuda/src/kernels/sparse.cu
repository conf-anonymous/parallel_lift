// Sparse matrix-vector multiply kernels for CUDA
// CSR format: y = A * x mod p

#include <stdint.h>
#include "modular_ops.cuh"

// CSR SpMV: y = A * x mod p (batched across primes)
extern "C" __global__ void sparse_matvec_csr(
    const uint32_t* __restrict__ row_ptr,    // [n+1]
    const uint32_t* __restrict__ col_idx,    // [nnz]
    const uint32_t* __restrict__ values,     // [num_primes][nnz]
    const uint32_t* __restrict__ x,          // [num_primes][n]
    uint32_t* __restrict__ y,                // [num_primes][n]
    const uint32_t* __restrict__ primes,
    uint32_t n,
    uint32_t nnz,
    uint32_t num_primes
) {
    // Thread assignment: tid = prime_idx * n + row
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t prime_idx = tid / n;
    uint32_t row = tid % n;

    if (prime_idx >= num_primes) return;

    uint32_t p = primes[prime_idx];
    uint32_t row_start = row_ptr[row];
    uint32_t row_end = row_ptr[row + 1];

    const uint32_t* my_values = values + prime_idx * nnz;
    const uint32_t* my_x = x + prime_idx * n;

    uint64_t sum = 0;

    for (uint32_t j = row_start; j < row_end; j++) {
        uint32_t col = col_idx[j];
        sum += (uint64_t)my_values[j] * (uint64_t)my_x[col];

        // Reduce every 512 iterations to prevent overflow
        if ((j - row_start) % 512 == 511) {
            sum %= (uint64_t)p;
        }
    }

    y[prime_idx * n + row] = (uint32_t)(sum % (uint64_t)p);
}

// Single-prime variant (simpler addressing)
extern "C" __global__ void sparse_matvec_csr_single(
    const uint32_t* __restrict__ row_ptr,
    const uint32_t* __restrict__ col_idx,
    const uint32_t* __restrict__ values,
    const uint32_t* __restrict__ x,
    uint32_t* __restrict__ y,
    uint32_t p,
    uint32_t n
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    uint32_t row_start = row_ptr[row];
    uint32_t row_end = row_ptr[row + 1];

    uint64_t sum = 0;

    for (uint32_t j = row_start; j < row_end; j++) {
        uint32_t col = col_idx[j];
        sum += (uint64_t)values[j] * (uint64_t)x[col];

        if ((j - row_start) % 512 == 511) {
            sum %= (uint64_t)p;
        }
    }

    y[row] = (uint32_t)(sum % (uint64_t)p);
}
