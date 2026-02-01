// Combined kernel file for single PTX compilation
// This file includes all kernel implementations to create a single valid PTX output

#include <stdint.h>

// ============================================================================
// Modular operations (inline from modular_ops.cuh)
// ============================================================================

// Modular addition: (a + b) mod p
__device__ __forceinline__ uint32_t mod_add(uint32_t a, uint32_t b, uint32_t p) {
    uint64_t sum = (uint64_t)a + (uint64_t)b;
    return (uint32_t)(sum % (uint64_t)p);
}

// Modular subtraction: (a - b) mod p
__device__ __forceinline__ uint32_t mod_sub(uint32_t a, uint32_t b, uint32_t p) {
    if (a >= b) {
        return a - b;
    } else {
        return p - (b - a);
    }
}

// Modular multiplication: (a * b) mod p
__device__ __forceinline__ uint32_t mod_mul(uint32_t a, uint32_t b, uint32_t p) {
    uint64_t prod = (uint64_t)a * (uint64_t)b;
    return (uint32_t)(prod % (uint64_t)p);
}

// Modular exponentiation: a^exp mod p
__device__ __forceinline__ uint32_t mod_pow(uint32_t a, uint32_t exp, uint32_t p) {
    uint64_t result = 1;
    uint64_t base = a % p;

    while (exp > 0) {
        if (exp & 1) {
            result = (result * base) % p;
        }
        base = (base * base) % p;
        exp >>= 1;
    }

    return (uint32_t)result;
}

// Modular inverse using Fermat's little theorem: a^(-1) = a^(p-2) mod p
// Only works when p is prime
__device__ __forceinline__ uint32_t mod_inv(uint32_t a, uint32_t p) {
    if (a == 0) return 0; // No inverse for 0
    return mod_pow(a, p - 2, p);
}

// ============================================================================
// Determinant kernels
// ============================================================================

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

        // Swap rows if needed
        if (pivot_row != k) {
            for (uint32_t j = 0; j < n; j++) {
                uint32_t tmp = A[k * n + j];
                A[k * n + j] = A[pivot_row * n + j];
                A[pivot_row * n + j] = tmp;
            }
            det = mod_sub(0, det, p); // det = -det mod p
        }

        // Get pivot value and its inverse
        uint32_t pivot_val = A[k * n + k];
        det = mod_mul(det, pivot_val, p);

        uint32_t pivot_inv = mod_inv(pivot_val, p);

        // Eliminate below
        for (uint32_t i = k + 1; i < n; i++) {
            if (A[i * n + k] != 0) {
                uint32_t factor = mod_mul(A[i * n + k], pivot_inv, p);
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

// Small matrix kernel: uses thread-local storage for matrices up to 16x16
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

    // Local storage for small matrices (max 16x16 = 256 elements)
    uint32_t A[256];

    // Copy matrix to local storage
    const uint32_t* src = matrices + tid * nn;
    for (uint32_t i = 0; i < nn; i++) {
        A[i] = src[i];
    }

    uint32_t det = 1;
    bool singular = false;

    // Same elimination as serial kernel
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
            for (uint32_t j = 0; j < n; j++) {
                uint32_t tmp = A[k * n + j];
                A[k * n + j] = A[pivot_row * n + j];
                A[pivot_row * n + j] = tmp;
            }
            det = mod_sub(0, det, p);
        }

        uint32_t pivot_val = A[k * n + k];
        det = mod_mul(det, pivot_val, p);
        uint32_t pivot_inv = mod_inv(pivot_val, p);

        for (uint32_t i = k + 1; i < n; i++) {
            if (A[i * n + k] != 0) {
                uint32_t factor = mod_mul(A[i * n + k], pivot_inv, p);
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

// Tiled kernel: 1 threadblock per prime, 16x16 threads collaborate
extern "C" __global__ void modular_determinant_tiled(
    const uint32_t* __restrict__ matrices,
    const uint32_t* __restrict__ primes,
    uint32_t* __restrict__ results,
    uint32_t* __restrict__ singular_flags,
    uint32_t* __restrict__ workspace,
    uint32_t n,
    uint32_t num_primes
) {
    uint32_t prime_idx = blockIdx.x;
    if (prime_idx >= num_primes) return;

    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;
    uint32_t local_tid = ty * 16 + tx;

    uint32_t p = primes[prime_idx];
    uint32_t nn = n * n;

    // Workspace for this prime
    uint32_t* A = workspace + prime_idx * nn;

    // Copy matrix to workspace (all threads participate)
    for (uint32_t i = local_tid; i < nn; i += 256) {
        A[i] = matrices[prime_idx * nn + i];
    }
    __syncthreads();

    // Shared memory for pivot info
    __shared__ uint32_t pivot_row_shared;
    __shared__ uint32_t pivot_val_shared;
    __shared__ uint32_t pivot_inv_shared;
    __shared__ int det_sign;
    __shared__ bool is_singular;

    if (local_tid == 0) {
        det_sign = 1;
        is_singular = false;
    }
    __syncthreads();

    uint32_t det = 1;

    for (uint32_t k = 0; k < n && !is_singular; k++) {
        // Find pivot (thread 0 does this)
        if (local_tid == 0) {
            uint32_t best_row = k;
            for (uint32_t i = k; i < n; i++) {
                if (A[i * n + k] != 0) {
                    best_row = i;
                    break;
                }
            }
            pivot_row_shared = best_row;

            if (A[best_row * n + k] == 0) {
                is_singular = true;
            } else {
                pivot_val_shared = A[best_row * n + k];
                pivot_inv_shared = mod_inv(pivot_val_shared, p);
            }
        }
        __syncthreads();

        if (is_singular) break;

        // Swap rows (all threads participate)
        if (pivot_row_shared != k) {
            for (uint32_t j = local_tid; j < n; j += 256) {
                uint32_t tmp = A[k * n + j];
                A[k * n + j] = A[pivot_row_shared * n + j];
                A[pivot_row_shared * n + j] = tmp;
            }
            if (local_tid == 0) {
                det_sign = -det_sign;
            }
        }
        __syncthreads();

        // Elimination (each thread handles one element)
        for (uint32_t i = k + 1; i < n; i++) {
            if (A[i * n + k] != 0) {
                uint32_t factor = mod_mul(A[i * n + k], pivot_inv_shared, p);
                for (uint32_t j = k + local_tid; j < n; j += 256) {
                    uint32_t sub = mod_mul(factor, A[k * n + j], p);
                    A[i * n + j] = mod_sub(A[i * n + j], sub, p);
                }
            }
        }
        __syncthreads();

        if (local_tid == 0) {
            det = mod_mul(det, pivot_val_shared, p);
        }
    }

    if (local_tid == 0) {
        if (is_singular) {
            results[prime_idx] = 0;
            singular_flags[prime_idx] = 1;
        } else {
            if (det_sign < 0) {
                det = mod_sub(0, det, p);
            }
            results[prime_idx] = det;
            singular_flags[prime_idx] = 0;
        }
    }
}

// ============================================================================
// Solve kernels
// ============================================================================

// Single RHS solve kernel: solve Ax = b mod p
extern "C" __global__ void modular_solve(
    const uint32_t* __restrict__ augmented,  // [num_primes][n][n+1]
    const uint32_t* __restrict__ primes,
    uint32_t* __restrict__ solutions,        // [num_primes][n]
    uint32_t* __restrict__ singular_flags,
    uint32_t* __restrict__ workspace,
    uint32_t n,
    uint32_t num_primes
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_primes) return;

    uint32_t p = primes[tid];
    uint32_t aug_width = n + 1;
    uint32_t aug_size = n * aug_width;

    // Copy augmented matrix to workspace
    const uint32_t* src = augmented + tid * aug_size;
    uint32_t* A = workspace + tid * aug_size;
    for (uint32_t i = 0; i < aug_size; i++) {
        A[i] = src[i];
    }

    bool singular = false;

    // Forward elimination with partial pivoting
    for (uint32_t k = 0; k < n && !singular; k++) {
        uint32_t pivot_row = k;
        for (uint32_t i = k; i < n; i++) {
            if (A[i * aug_width + k] != 0) {
                pivot_row = i;
                break;
            }
        }

        if (A[pivot_row * aug_width + k] == 0) {
            singular = true;
            break;
        }

        // Swap rows
        if (pivot_row != k) {
            for (uint32_t j = 0; j < aug_width; j++) {
                uint32_t tmp = A[k * aug_width + j];
                A[k * aug_width + j] = A[pivot_row * aug_width + j];
                A[pivot_row * aug_width + j] = tmp;
            }
        }

        uint32_t pivot_val = A[k * aug_width + k];
        uint32_t pivot_inv = mod_inv(pivot_val, p);

        // Scale pivot row
        for (uint32_t j = 0; j < aug_width; j++) {
            A[k * aug_width + j] = mod_mul(A[k * aug_width + j], pivot_inv, p);
        }

        // Eliminate all other rows (Gauss-Jordan)
        for (uint32_t i = 0; i < n; i++) {
            if (i != k && A[i * aug_width + k] != 0) {
                uint32_t factor = A[i * aug_width + k];
                for (uint32_t j = 0; j < aug_width; j++) {
                    uint32_t sub = mod_mul(factor, A[k * aug_width + j], p);
                    A[i * aug_width + j] = mod_sub(A[i * aug_width + j], sub, p);
                }
            }
        }
    }

    // Copy solution (last column)
    for (uint32_t i = 0; i < n; i++) {
        solutions[tid * n + i] = singular ? 0 : A[i * aug_width + n];
    }
    singular_flags[tid] = singular ? 1 : 0;
}

// Multi-RHS solve kernel: solve AX = B mod p where B has k columns
extern "C" __global__ void modular_solve_multi_rhs(
    const uint32_t* __restrict__ augmented,  // [num_primes][n][n+k]
    const uint32_t* __restrict__ primes,
    uint32_t* __restrict__ solutions,        // [num_primes][k][n] (column-major per solution)
    uint32_t* __restrict__ singular_flags,
    uint32_t* __restrict__ workspace,
    uint32_t n,
    uint32_t k,
    uint32_t num_primes
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_primes) return;

    uint32_t p = primes[tid];
    uint32_t aug_width = n + k;
    uint32_t aug_size = n * aug_width;

    // Copy augmented matrix to workspace
    const uint32_t* src = augmented + tid * aug_size;
    uint32_t* A = workspace + tid * aug_size;
    for (uint32_t i = 0; i < aug_size; i++) {
        A[i] = src[i];
    }

    bool singular = false;

    // Forward elimination (same as single RHS)
    for (uint32_t col = 0; col < n && !singular; col++) {
        uint32_t pivot_row = col;
        for (uint32_t i = col; i < n; i++) {
            if (A[i * aug_width + col] != 0) {
                pivot_row = i;
                break;
            }
        }

        if (A[pivot_row * aug_width + col] == 0) {
            singular = true;
            break;
        }

        if (pivot_row != col) {
            for (uint32_t j = 0; j < aug_width; j++) {
                uint32_t tmp = A[col * aug_width + j];
                A[col * aug_width + j] = A[pivot_row * aug_width + j];
                A[pivot_row * aug_width + j] = tmp;
            }
        }

        uint32_t pivot_val = A[col * aug_width + col];
        uint32_t pivot_inv = mod_inv(pivot_val, p);

        for (uint32_t j = 0; j < aug_width; j++) {
            A[col * aug_width + j] = mod_mul(A[col * aug_width + j], pivot_inv, p);
        }

        for (uint32_t i = 0; i < n; i++) {
            if (i != col && A[i * aug_width + col] != 0) {
                uint32_t factor = A[i * aug_width + col];
                for (uint32_t j = 0; j < aug_width; j++) {
                    uint32_t sub = mod_mul(factor, A[col * aug_width + j], p);
                    A[i * aug_width + j] = mod_sub(A[i * aug_width + j], sub, p);
                }
            }
        }
    }

    // Copy solutions (columns n to n+k-1)
    // Output is stored column-major: solutions[tid * n * k + col * n + row]
    uint32_t sol_offset = tid * n * k;
    for (uint32_t col_idx = 0; col_idx < k; col_idx++) {
        for (uint32_t row = 0; row < n; row++) {
            solutions[sol_offset + col_idx * n + row] =
                singular ? 0 : A[row * aug_width + n + col_idx];
        }
    }
    singular_flags[tid] = singular ? 1 : 0;
}

// Tiled multi-RHS solve: 1 threadblock per prime
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

    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;
    uint32_t local_tid = ty * 16 + tx;

    uint32_t p = primes[prime_idx];
    uint32_t aug_width = n + k;
    uint32_t aug_size = n * aug_width;

    uint32_t* A = workspace + prime_idx * aug_size;

    // Copy matrix (all threads)
    for (uint32_t i = local_tid; i < aug_size; i += 256) {
        A[i] = augmented[prime_idx * aug_size + i];
    }
    __syncthreads();

    __shared__ uint32_t pivot_row_shared;
    __shared__ uint32_t pivot_val_shared;
    __shared__ uint32_t pivot_inv_shared;
    __shared__ bool is_singular;

    if (local_tid == 0) {
        is_singular = false;
    }
    __syncthreads();

    for (uint32_t col = 0; col < n && !is_singular; col++) {
        if (local_tid == 0) {
            uint32_t best_row = col;
            for (uint32_t i = col; i < n; i++) {
                if (A[i * aug_width + col] != 0) {
                    best_row = i;
                    break;
                }
            }
            pivot_row_shared = best_row;

            if (A[best_row * aug_width + col] == 0) {
                is_singular = true;
            } else {
                pivot_val_shared = A[best_row * aug_width + col];
                pivot_inv_shared = mod_inv(pivot_val_shared, p);
            }
        }
        __syncthreads();

        if (is_singular) break;

        // Swap rows
        if (pivot_row_shared != col) {
            for (uint32_t j = local_tid; j < aug_width; j += 256) {
                uint32_t tmp = A[col * aug_width + j];
                A[col * aug_width + j] = A[pivot_row_shared * aug_width + j];
                A[pivot_row_shared * aug_width + j] = tmp;
            }
        }
        __syncthreads();

        // Scale pivot row
        for (uint32_t j = local_tid; j < aug_width; j += 256) {
            A[col * aug_width + j] = mod_mul(A[col * aug_width + j], pivot_inv_shared, p);
        }
        __syncthreads();

        // Eliminate
        for (uint32_t i = 0; i < n; i++) {
            if (i != col && A[i * aug_width + col] != 0) {
                uint32_t factor = A[i * aug_width + col];
                for (uint32_t j = col + local_tid; j < aug_width; j += 256) {
                    uint32_t sub = mod_mul(factor, A[col * aug_width + j], p);
                    A[i * aug_width + j] = mod_sub(A[i * aug_width + j], sub, p);
                }
            }
            __syncthreads();
        }
    }

    // Copy solutions
    uint32_t sol_offset = prime_idx * n * k;
    for (uint32_t i = local_tid; i < n * k; i += 256) {
        uint32_t col_idx = i / n;
        uint32_t row = i % n;
        solutions[sol_offset + i] = is_singular ? 0 : A[row * aug_width + n + col_idx];
    }

    if (local_tid == 0) {
        singular_flags[prime_idx] = is_singular ? 1 : 0;
    }
}

// ============================================================================
// Sparse kernels
// ============================================================================

// Batched sparse matrix-vector multiply (CSR format)
// y = A * x mod p for multiple primes
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
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t prime_idx = gid / n;
    uint32_t row = gid - prime_idx * n;

    if (prime_idx >= num_primes) return;

    uint32_t p = primes[prime_idx];
    uint32_t start = row_ptr[row];
    uint32_t end = row_ptr[row + 1];

    uint64_t sum = 0;
    for (uint32_t j = start; j < end; j++) {
        uint32_t col = col_idx[j];
        uint32_t val = values[prime_idx * nnz + j];
        uint32_t xv = x[prime_idx * n + col];
        sum += (uint64_t)val * (uint64_t)xv;
    }

    y[prime_idx * n + row] = (uint32_t)(sum % (uint64_t)p);
}

// Single-prime sparse matvec (for Wiedemann)
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

    uint32_t start = row_ptr[row];
    uint32_t end = row_ptr[row + 1];

    uint64_t sum = 0;
    for (uint32_t j = start; j < end; j++) {
        uint32_t col = col_idx[j];
        sum += (uint64_t)values[j] * (uint64_t)x[col];
    }

    y[row] = (uint32_t)(sum % (uint64_t)p);
}

// ============================================================================
// CRT kernels
// ============================================================================

// Garner's algorithm step for CRT reconstruction
// Used for reconstructing BigInt from residues
extern "C" __global__ void crt_garner_step(
    const uint32_t* __restrict__ residues,      // [n_elements][num_primes]
    const uint32_t* __restrict__ accumulated,   // [n_elements] (current accumulated value mod new_prime)
    const uint32_t* __restrict__ primes,        // [num_primes]
    const uint32_t* __restrict__ products_inv,  // [num_primes] (M_i^-1 mod p_i)
    uint32_t* __restrict__ coefficients,        // [n_elements] (output coefficients)
    uint32_t n_elements,
    uint32_t num_primes,
    uint32_t current_prime_idx
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_elements) return;

    uint32_t p = primes[current_prime_idx];
    uint32_t r = residues[tid * num_primes + current_prime_idx];
    uint32_t acc = accumulated[tid];

    // coeff = (r - acc) * products_inv[i] mod p
    uint32_t diff = mod_sub(r, acc, p);
    uint64_t prod = (uint64_t)diff * (uint64_t)products_inv[current_prime_idx];
    coefficients[tid] = (uint32_t)(prod % (uint64_t)p);
}

// Reduce BigInt to residue mod prime
// Input: BigInt as array of 32-bit limbs (little-endian)
extern "C" __global__ void bigint_mod_prime(
    const uint32_t* __restrict__ limbs,     // [n_elements][num_limbs]
    uint32_t* __restrict__ residues,        // [n_elements]
    uint32_t num_limbs,
    uint32_t n_elements,
    uint32_t p
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_elements) return;

    uint64_t result = 0;
    uint64_t base = 1;
    uint64_t p64 = (uint64_t)p;

    const uint32_t* my_limbs = limbs + tid * num_limbs;

    for (uint32_t i = 0; i < num_limbs; i++) {
        result = (result + (base * (uint64_t)my_limbs[i]) % p64) % p64;
        base = (base * ((1ULL << 32) % p64)) % p64;
    }

    residues[tid] = (uint32_t)result;
}

// Compare accumulated product with half of next prime product
// Returns 1 if accumulated >= half_product, 0 otherwise
extern "C" __global__ void bigint_compare_half(
    const uint32_t* __restrict__ accumulated_limbs,  // [n_elements][num_limbs]
    const uint32_t* __restrict__ half_product_limbs, // [num_limbs]
    uint32_t* __restrict__ needs_negation,           // [n_elements]
    uint32_t num_limbs,
    uint32_t n_elements
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_elements) return;

    const uint32_t* acc = accumulated_limbs + tid * num_limbs;

    // Compare from most significant limb
    int cmp = 0;
    for (int i = (int)num_limbs - 1; i >= 0 && cmp == 0; i--) {
        if (acc[i] > half_product_limbs[i]) {
            cmp = 1;
        } else if (acc[i] < half_product_limbs[i]) {
            cmp = -1;
        }
    }

    needs_negation[tid] = (cmp >= 0) ? 1 : 0;
}

// ============================================================================
// Full CRT Reconstruction Kernels
// ============================================================================

// Modular multiplication helper for u64
__device__ __forceinline__ uint64_t crt_mod_mul_u64(uint64_t a, uint64_t b, uint64_t p) {
    return (a * b) % p;
}

// Full CRT reconstruction using Garner's algorithm
// One thread per value, each thread iterates through all primes
extern "C" __global__ void crt_reconstruct_full(
    const uint32_t* __restrict__ residues,          // [num_values * num_primes] row-major
    const uint32_t* __restrict__ primes,            // [num_primes]
    const uint32_t* __restrict__ inverses,          // [num_primes] garner inverses
    const uint32_t* __restrict__ pp_limbs,          // Packed partial product limbs
    const uint32_t* __restrict__ pp_offsets,        // [num_primes] offset into pp_limbs
    const uint32_t* __restrict__ pp_sizes,          // [num_primes] number of limbs
    const uint64_t* __restrict__ pow2_mod,          // [num_primes * max_acc_limbs] 2^(32j) mod p[i]
    uint32_t* __restrict__ output_limbs,            // [num_values * max_acc_limbs]
    uint32_t* __restrict__ output_sizes,            // [num_values] actual size of each result
    uint32_t num_values,
    uint32_t num_primes,
    uint32_t max_acc_limbs
) {
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_values) return;

    // Working accumulator for this thread (in global memory)
    uint32_t* acc = output_limbs + v * max_acc_limbs;

    // Initialize accumulator to first residue
    acc[0] = residues[v * num_primes + 0];
    for (uint32_t j = 1; j < max_acc_limbs; j++) {
        acc[j] = 0;
    }

    uint32_t current_size = 1;

    // Garner iterations: build up the result incrementally
    for (uint32_t i = 1; i < num_primes; i++) {
        uint64_t p = primes[i];
        uint32_t inv = inverses[i];
        uint32_t r = residues[v * num_primes + i];

        // Compute acc mod p using precomputed powers of 2^32
        const uint64_t* pow2 = pow2_mod + i * max_acc_limbs;
        uint64_t acc_mod_p = 0;

        for (uint32_t j = 0; j < current_size; j++) {
            acc_mod_p += crt_mod_mul_u64(acc[j], pow2[j], p);
            if ((j & 15) == 15) {
                acc_mod_p %= p;
            }
        }
        acc_mod_p %= p;

        // Compute Garner coefficient: t = (r - acc_mod_p) * inv mod p
        uint64_t diff;
        if (r >= acc_mod_p) {
            diff = r - acc_mod_p;
        } else {
            diff = p - acc_mod_p + r;
        }
        uint64_t t = (diff * inv) % p;

        if (t == 0) continue;

        // Load partial product info
        uint32_t pp_offset = pp_offsets[i];
        uint32_t pp_size = pp_sizes[i];

        // acc += partial_products[i] * t
        uint64_t carry = 0;
        for (uint32_t j = 0; j < max_acc_limbs; j++) {
            uint64_t pp_limb = (j < pp_size) ? pp_limbs[pp_offset + j] : 0;
            uint64_t product = pp_limb * t + acc[j] + carry;
            acc[j] = (uint32_t)product;
            carry = product >> 32;
        }

        current_size = (pp_size + 1 > current_size) ? pp_size + 1 : current_size;
        if (current_size > max_acc_limbs) current_size = max_acc_limbs;
    }

    output_sizes[v] = current_size;
}

// Sign detection and conversion to signed representation
extern "C" __global__ void crt_to_signed(
    uint32_t* __restrict__ limbs,
    const uint32_t* __restrict__ half_product,
    const uint32_t* __restrict__ full_product,
    uint32_t* __restrict__ signs,
    uint32_t num_values,
    uint32_t max_limbs,
    uint32_t half_size,
    uint32_t product_size
) {
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_values) return;

    uint32_t* acc = limbs + v * max_limbs;

    // Compare with M/2 (from MSB to LSB)
    int cmp = 0;

    // First check if acc has limbs beyond half_product
    for (int j = (int)max_limbs - 1; j >= (int)half_size && cmp == 0; j--) {
        if (acc[j] > 0) {
            cmp = 1;
            break;
        }
    }

    // Then compare overlapping limbs
    if (cmp == 0) {
        uint32_t check_size = (max_limbs < half_size) ? max_limbs : half_size;
        for (int j = (int)check_size - 1; j >= 0 && cmp == 0; j--) {
            if (acc[j] > half_product[j]) {
                cmp = 1;
            } else if (acc[j] < half_product[j]) {
                cmp = -1;
            }
        }
    }

    if (cmp > 0) {
        signs[v] = 1;
        // Compute M - acc
        uint64_t borrow = 0;
        for (uint32_t j = 0; j < max_limbs; j++) {
            uint64_t p_limb = (j < product_size) ? full_product[j] : 0;
            uint64_t a_limb = acc[j];
            uint64_t diff;
            if (p_limb >= a_limb + borrow) {
                diff = p_limb - a_limb - borrow;
                borrow = 0;
            } else {
                diff = (1ULL << 32) + p_limb - a_limb - borrow;
                borrow = 1;
            }
            acc[j] = (uint32_t)diff;
        }
    } else {
        signs[v] = 0;
    }
}

// ============================================================================
// Gram-Schmidt / LLL kernels for lattice reduction
// ============================================================================

// Batch inner product: compute <b_i, b_j> mod p for all primes in parallel
// One thread per (i, j, prime) triple
// Input: basis vectors b[i] as rows, each vector has m components
// Output: gram[i * n + j] = <b_i, b_j> mod p for each prime
extern "C" __global__ void batch_gram_matrix(
    const uint32_t* __restrict__ basis,      // [num_primes][n][m] - basis vectors as rows
    uint32_t* __restrict__ gram,             // [num_primes][n][n] - output Gram matrix
    const uint32_t* __restrict__ primes,     // [num_primes]
    uint32_t n,                              // number of basis vectors
    uint32_t m,                              // dimension of each vector
    uint32_t num_primes
) {
    // Each thread computes one element of the Gram matrix for one prime
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_elements = num_primes * n * n;
    if (gid >= total_elements) return;

    uint32_t prime_idx = gid / (n * n);
    uint32_t ij = gid % (n * n);
    uint32_t i = ij / n;
    uint32_t j = ij % n;

    // Only compute upper triangle (Gram matrix is symmetric)
    if (j < i) {
        gram[gid] = gram[prime_idx * n * n + j * n + i];  // Copy from (j,i)
        return;
    }

    uint32_t p = primes[prime_idx];
    uint64_t sum = 0;

    // Compute <b_i, b_j> mod p
    const uint32_t* bi = basis + prime_idx * n * m + i * m;
    const uint32_t* bj = basis + prime_idx * n * m + j * m;

    for (uint32_t k = 0; k < m; k++) {
        sum += (uint64_t)bi[k] * (uint64_t)bj[k];
        // Periodic reduction to prevent overflow
        if ((k & 63) == 63) {
            sum %= (uint64_t)p;
        }
    }

    gram[gid] = (uint32_t)(sum % (uint64_t)p);
}

// Batch inner product for Gram-Schmidt: compute <b_i, b*_j> mod p
// This is used during Gram-Schmidt orthogonalization
// b*_j is the j-th Gram-Schmidt orthogonalized vector (stored as numerator)
extern "C" __global__ void batch_inner_product_gs(
    const uint32_t* __restrict__ basis,      // [num_primes][n][m] - original basis
    const uint32_t* __restrict__ b_star,     // [num_primes][n][m] - GS orthogonalized vectors (numerator)
    uint32_t* __restrict__ inner_products,   // [num_primes][n][n] - output <b_i, b*_j>
    const uint32_t* __restrict__ primes,
    uint32_t n,
    uint32_t m,
    uint32_t num_primes
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_elements = num_primes * n * n;
    if (gid >= total_elements) return;

    uint32_t prime_idx = gid / (n * n);
    uint32_t ij = gid % (n * n);
    uint32_t i = ij / n;
    uint32_t j = ij % n;

    uint32_t p = primes[prime_idx];
    uint64_t sum = 0;

    const uint32_t* bi = basis + prime_idx * n * m + i * m;
    const uint32_t* bsj = b_star + prime_idx * n * m + j * m;

    for (uint32_t k = 0; k < m; k++) {
        sum += (uint64_t)bi[k] * (uint64_t)bsj[k];
        if ((k & 63) == 63) {
            sum %= (uint64_t)p;
        }
    }

    inner_products[gid] = (uint32_t)(sum % (uint64_t)p);
}

// Batch size reduction: b_k = b_k - round(mu_kj) * b_j for all primes
// Each thread handles one component of one vector for one prime
extern "C" __global__ void batch_size_reduce(
    uint32_t* __restrict__ basis,            // [num_primes][n][m] - basis vectors (modified in place)
    const uint32_t* __restrict__ mu_rounded, // [num_primes] - rounded mu coefficient (|mu| > 0.5 case)
    uint32_t k,                              // index of vector being reduced
    uint32_t j,                              // index of vector to subtract
    const uint32_t* __restrict__ primes,
    uint32_t n,
    uint32_t m,
    uint32_t num_primes
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_elements = num_primes * m;
    if (gid >= total_elements) return;

    uint32_t prime_idx = gid / m;
    uint32_t comp = gid % m;

    uint32_t p = primes[prime_idx];
    uint32_t mu_r = mu_rounded[prime_idx];

    if (mu_r == 0) return;  // No reduction needed

    uint32_t* bk = basis + prime_idx * n * m + k * m;
    const uint32_t* bj = basis + prime_idx * n * m + j * m;

    // b_k[comp] = b_k[comp] - mu_r * b_j[comp] mod p
    uint64_t sub = ((uint64_t)mu_r * (uint64_t)bj[comp]) % (uint64_t)p;
    if (bk[comp] >= sub) {
        bk[comp] = bk[comp] - (uint32_t)sub;
    } else {
        bk[comp] = p - (uint32_t)sub + bk[comp];
    }
}

// Batch vector swap: swap b_k and b_{k-1} for all primes
extern "C" __global__ void batch_swap_vectors(
    uint32_t* __restrict__ basis,            // [num_primes][n][m]
    uint32_t k,                              // swap b_k with b_{k-1}
    const uint32_t* __restrict__ primes,
    uint32_t n,
    uint32_t m,
    uint32_t num_primes
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_elements = num_primes * m;
    if (gid >= total_elements) return;

    uint32_t prime_idx = gid / m;
    uint32_t comp = gid % m;

    uint32_t* bk = basis + prime_idx * n * m + k * m;
    uint32_t* bk1 = basis + prime_idx * n * m + (k - 1) * m;

    uint32_t tmp = bk[comp];
    bk[comp] = bk1[comp];
    bk1[comp] = tmp;
}

// Compute squared norms: ||b_i||^2 mod p for all vectors and primes
extern "C" __global__ void batch_squared_norms(
    const uint32_t* __restrict__ basis,      // [num_primes][n][m]
    uint32_t* __restrict__ norms_sq,         // [num_primes][n]
    const uint32_t* __restrict__ primes,
    uint32_t n,
    uint32_t m,
    uint32_t num_primes
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_elements = num_primes * n;
    if (gid >= total_elements) return;

    uint32_t prime_idx = gid / n;
    uint32_t i = gid % n;

    uint32_t p = primes[prime_idx];
    const uint32_t* bi = basis + prime_idx * n * m + i * m;

    uint64_t sum = 0;
    for (uint32_t k = 0; k < m; k++) {
        uint64_t val = bi[k];
        sum += val * val;
        if ((k & 63) == 63) {
            sum %= (uint64_t)p;
        }
    }

    norms_sq[gid] = (uint32_t)(sum % (uint64_t)p);
}

// Full Gram-Schmidt orthogonalization for one step
// Given b_0, ..., b_{i-1} already orthogonalized, compute b*_i
// b*_i = b_i - sum_{j<i} mu_{ij} * b*_j where mu_{ij} = <b_i, b*_j> / <b*_j, b*_j>
// This kernel computes the mu coefficients and updates b*_i in place
extern "C" __global__ void gram_schmidt_step(
    const uint32_t* __restrict__ basis,      // [num_primes][n][m] - original basis
    uint32_t* __restrict__ b_star,           // [num_primes][n][m] - GS vectors (modified)
    uint32_t* __restrict__ mu_num,           // [num_primes][n*(n-1)/2] - mu numerators
    uint32_t* __restrict__ mu_den,           // [num_primes][n*(n-1)/2] - mu denominators (||b*_j||^2)
    uint32_t* __restrict__ b_star_norm_sq,   // [num_primes][n] - ||b*_i||^2
    const uint32_t* __restrict__ primes,
    uint32_t i,                              // current vector index
    uint32_t n,
    uint32_t m,
    uint32_t num_primes
) {
    // This is a serial kernel per prime - one thread per prime
    uint32_t prime_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (prime_idx >= num_primes) return;

    uint32_t p = primes[prime_idx];
    uint64_t p64 = (uint64_t)p;

    uint32_t* bs_i = b_star + prime_idx * n * m + i * m;
    const uint32_t* b_i = basis + prime_idx * n * m + i * m;

    // Start with b*_i = b_i
    for (uint32_t k = 0; k < m; k++) {
        bs_i[k] = b_i[k];
    }

    // Subtract projections onto previous b*_j
    for (uint32_t j = 0; j < i; j++) {
        const uint32_t* bs_j = b_star + prime_idx * n * m + j * m;
        uint32_t bs_j_norm_sq = b_star_norm_sq[prime_idx * n + j];

        if (bs_j_norm_sq == 0) continue;  // Skip zero vectors

        // Compute <b_i, b*_j>
        uint64_t inner = 0;
        for (uint32_t k = 0; k < m; k++) {
            inner += (uint64_t)b_i[k] * (uint64_t)bs_j[k];
            if ((k & 63) == 63) inner %= p64;
        }
        inner %= p64;

        // Store mu_{i,j} = inner / bs_j_norm_sq (as fraction)
        uint32_t mu_idx = i * (i - 1) / 2 + j;
        mu_num[prime_idx * (n * (n - 1) / 2) + mu_idx] = (uint32_t)inner;
        mu_den[prime_idx * (n * (n - 1) / 2) + mu_idx] = bs_j_norm_sq;

        // b*_i = b*_i - (inner / bs_j_norm_sq) * b*_j
        // We compute: b*_i * bs_j_norm_sq = b*_i * bs_j_norm_sq - inner * b*_j
        // Then divide by bs_j_norm_sq at the end (or keep as scaled)

        // For now, compute modular: b*_i -= (inner * inv(bs_j_norm_sq)) * b*_j
        uint32_t inv_norm = mod_inv(bs_j_norm_sq, p);
        uint64_t mu_mod = (inner * inv_norm) % p64;

        for (uint32_t k = 0; k < m; k++) {
            uint64_t sub = (mu_mod * bs_j[k]) % p64;
            if (bs_i[k] >= sub) {
                bs_i[k] = bs_i[k] - (uint32_t)sub;
            } else {
                bs_i[k] = p - (uint32_t)sub + bs_i[k];
            }
        }
    }

    // Compute ||b*_i||^2
    uint64_t norm_sq = 0;
    for (uint32_t k = 0; k < m; k++) {
        uint64_t val = bs_i[k];
        norm_sq += val * val;
        if ((k & 63) == 63) norm_sq %= p64;
    }
    b_star_norm_sq[prime_idx * n + i] = (uint32_t)(norm_sq % p64);
}

// ============================================================================
// 64-bit Modular Operations for V2 (using 128-bit intermediate via PTX)
// ============================================================================

// 128-bit unsigned integer for intermediate products
struct uint128_t {
    uint64_t lo;
    uint64_t hi;
};

// Multiply two 64-bit numbers to get 128-bit result
__device__ __forceinline__ uint128_t mul64(uint64_t a, uint64_t b) {
    uint128_t result;
    // Use PTX inline assembly for 64x64->128 multiplication
    asm("mul.lo.u64 %0, %2, %3;\n\t"
        "mul.hi.u64 %1, %2, %3;"
        : "=l"(result.lo), "=l"(result.hi)
        : "l"(a), "l"(b));
    return result;
}

// Add 64-bit value to 128-bit value
__device__ __forceinline__ uint128_t add128_64(uint128_t a, uint64_t b) {
    uint128_t result;
    result.lo = a.lo + b;
    result.hi = a.hi + (result.lo < a.lo ? 1 : 0);  // Carry
    return result;
}

// Modular reduction: 128-bit value mod 64-bit prime
// Uses Barrett reduction for efficiency
__device__ __forceinline__ uint64_t mod128(uint128_t x, uint64_t p) {
    // For primes close to 2^62, we can use a simplified approach
    // x = hi * 2^64 + lo
    // x mod p = ((hi mod p) * (2^64 mod p) + (lo mod p)) mod p

    if (x.hi == 0) {
        return x.lo % p;
    }

    // Compute 2^64 mod p
    // 2^64 = 2^62 * 4 = (p + (2^62 - p)) * 4 for p ~ 2^62
    // Simpler: just do the division
    uint64_t hi_mod = x.hi % p;

    // 2^64 mod p: we need this precomputed but for now calculate it
    // 2^64 = p * q + r where r = 2^64 mod p
    // Since p ~ 2^62, q ~ 4 and r can be computed
    uint128_t pow64;
    pow64.hi = 1;
    pow64.lo = 0;
    uint64_t pow64_mod_p = 0;

    // Compute 2^64 mod p by repeated squaring from 2^32
    // 2^32 mod p
    uint64_t pow32 = (1ULL << 32) % p;
    // 2^64 mod p = (2^32 mod p)^2 mod p
    uint128_t tmp = mul64(pow32, pow32);
    pow64_mod_p = tmp.hi == 0 ? tmp.lo % p : (tmp.lo % p + ((tmp.hi % p) * ((1ULL << 32) % p)) % p * ((1ULL << 32) % p)) % p;

    // Actually let's use a simpler method
    // 2^64 mod p for p ~ 2^62:
    // 2^64 = 4 * 2^62 = 4 * (p + delta) where delta = 2^62 - p (small)
    // 2^64 mod p = 4 * delta mod p = 4 * (2^62 - p) mod p
    uint64_t two62 = 1ULL << 62;
    uint64_t delta = two62 - (p & ((1ULL << 62) - 1));  // Wrap-around if p > 2^62
    if (p >= two62) {
        // p is close to 2^62, delta is small
        pow64_mod_p = (4 * (two62 - p)) % p;
    } else {
        // p is smaller than 2^62
        pow64_mod_p = ((1ULL << 32) % p);
        pow64_mod_p = (pow64_mod_p * pow64_mod_p) % p;
    }

    // x mod p = (hi_mod * pow64_mod_p + lo mod p) mod p
    uint128_t term1 = mul64(hi_mod, pow64_mod_p);
    uint64_t term1_mod = term1.hi == 0 ? term1.lo % p : mod128(term1, p);

    uint64_t lo_mod = x.lo % p;
    uint64_t result = (term1_mod + lo_mod) % p;

    return result;
}

// 64-bit modular addition
__device__ __forceinline__ uint64_t mod_add_64(uint64_t a, uint64_t b, uint64_t p) {
    uint64_t sum = a + b;
    // Check for overflow or >= p
    if (sum < a || sum >= p) {
        sum = (sum >= p) ? sum - p : sum;  // Handle overflow case
    }
    return sum >= p ? sum - p : sum;
}

// 64-bit modular subtraction
__device__ __forceinline__ uint64_t mod_sub_64(uint64_t a, uint64_t b, uint64_t p) {
    if (a >= b) {
        return a - b;
    } else {
        return p - (b - a);
    }
}

// 64-bit modular multiplication using 128-bit intermediate
__device__ __forceinline__ uint64_t mod_mul_64(uint64_t a, uint64_t b, uint64_t p) {
    uint128_t prod = mul64(a, b);
    return mod128(prod, p);
}

// 64-bit modular exponentiation
__device__ __forceinline__ uint64_t mod_pow_64(uint64_t a, uint64_t exp, uint64_t p) {
    uint64_t result = 1;
    uint64_t base = a % p;

    while (exp > 0) {
        if (exp & 1) {
            result = mod_mul_64(result, base, p);
        }
        base = mod_mul_64(base, base, p);
        exp >>= 1;
    }

    return result;
}

// 64-bit modular inverse using Fermat's Little Theorem
__device__ __forceinline__ uint64_t mod_inv_64(uint64_t a, uint64_t p) {
    if (a == 0) return 0;
    return mod_pow_64(a, p - 2, p);
}

// ============================================================================
// V2 Solve Kernels (64-bit primes)
// ============================================================================

// Multi-RHS solve with 64-bit primes: 1 thread per prime
extern "C" __global__ void modular_solve_multi_rhs_64(
    const uint64_t* __restrict__ augmented,  // [num_primes][n][n+k]
    const uint64_t* __restrict__ primes,
    uint64_t* __restrict__ solutions,        // [num_primes][k][n] column-major per prime
    uint32_t* __restrict__ singular_flags,
    uint64_t* __restrict__ workspace,        // [num_primes][n][n+k]
    uint32_t n,
    uint32_t k,
    uint32_t num_primes
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_primes) return;

    uint64_t p = primes[tid];
    uint32_t aug_width = n + k;
    uint32_t aug_size = n * aug_width;

    // Copy augmented matrix to workspace
    const uint64_t* src = augmented + tid * aug_size;
    uint64_t* A = workspace + tid * aug_size;
    for (uint32_t i = 0; i < aug_size; i++) {
        A[i] = src[i];
    }

    bool singular = false;

    // Forward elimination with Gauss-Jordan (to identity)
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
            for (uint32_t j = 0; j < aug_width; j++) {
                uint64_t tmp = A[col * aug_width + j];
                A[col * aug_width + j] = A[pivot_row * aug_width + j];
                A[pivot_row * aug_width + j] = tmp;
            }
        }

        uint64_t pivot_val = A[col * aug_width + col];
        uint64_t pivot_inv = mod_inv_64(pivot_val, p);

        // Normalize pivot row
        for (uint32_t j = 0; j < aug_width; j++) {
            A[col * aug_width + j] = mod_mul_64(A[col * aug_width + j], pivot_inv, p);
        }

        // Eliminate all other rows (Gauss-Jordan)
        for (uint32_t row = 0; row < n; row++) {
            if (row != col && A[row * aug_width + col] != 0) {
                uint64_t factor = A[row * aug_width + col];
                for (uint32_t j = 0; j < aug_width; j++) {
                    uint64_t sub = mod_mul_64(factor, A[col * aug_width + j], p);
                    A[row * aug_width + j] = mod_sub_64(A[row * aug_width + j], sub, p);
                }
            }
        }
    }

    // Copy solutions
    uint32_t sol_offset = tid * n * k;
    for (uint32_t col_idx = 0; col_idx < k; col_idx++) {
        for (uint32_t row = 0; row < n; row++) {
            solutions[sol_offset + col_idx * n + row] =
                singular ? 0 : A[row * aug_width + n + col_idx];
        }
    }
    singular_flags[tid] = singular ? 1 : 0;
}

// Tiled multi-RHS solve with 64-bit primes: 1 threadblock per prime
extern "C" __global__ void modular_solve_multi_rhs_tiled_64(
    const uint64_t* __restrict__ augmented,
    const uint64_t* __restrict__ primes,
    uint64_t* __restrict__ solutions,
    uint32_t* __restrict__ singular_flags,
    uint64_t* __restrict__ workspace,
    uint32_t n,
    uint32_t k,
    uint32_t num_primes
) {
    uint32_t prime_idx = blockIdx.x;
    if (prime_idx >= num_primes) return;

    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;
    uint32_t local_tid = ty * 16 + tx;

    uint64_t p = primes[prime_idx];
    uint32_t aug_width = n + k;
    uint32_t aug_size = n * aug_width;

    uint64_t* A = workspace + prime_idx * aug_size;

    // Copy matrix (all threads)
    for (uint32_t i = local_tid; i < aug_size; i += 256) {
        A[i] = augmented[prime_idx * aug_size + i];
    }
    __syncthreads();

    __shared__ uint32_t pivot_row_shared;
    __shared__ uint64_t pivot_val_shared;
    __shared__ uint64_t pivot_inv_shared;
    __shared__ bool is_singular;

    if (local_tid == 0) {
        is_singular = false;
    }
    __syncthreads();

    for (uint32_t col = 0; col < n && !is_singular; col++) {
        if (local_tid == 0) {
            uint32_t best_row = col;
            for (uint32_t i = col; i < n; i++) {
                if (A[i * aug_width + col] != 0) {
                    best_row = i;
                    break;
                }
            }
            pivot_row_shared = best_row;

            if (A[best_row * aug_width + col] == 0) {
                is_singular = true;
            } else {
                pivot_val_shared = A[best_row * aug_width + col];
                pivot_inv_shared = mod_inv_64(pivot_val_shared, p);
            }
        }
        __syncthreads();

        if (is_singular) break;

        // Swap rows
        if (pivot_row_shared != col) {
            for (uint32_t j = local_tid; j < aug_width; j += 256) {
                uint64_t tmp = A[col * aug_width + j];
                A[col * aug_width + j] = A[pivot_row_shared * aug_width + j];
                A[pivot_row_shared * aug_width + j] = tmp;
            }
        }
        __syncthreads();

        // Scale pivot row
        for (uint32_t j = local_tid; j < aug_width; j += 256) {
            A[col * aug_width + j] = mod_mul_64(A[col * aug_width + j], pivot_inv_shared, p);
        }
        __syncthreads();

        // Eliminate all other rows
        for (uint32_t i = 0; i < n; i++) {
            if (i != col && A[i * aug_width + col] != 0) {
                uint64_t factor = A[i * aug_width + col];
                for (uint32_t j = col + local_tid; j < aug_width; j += 256) {
                    uint64_t sub = mod_mul_64(factor, A[col * aug_width + j], p);
                    A[i * aug_width + j] = mod_sub_64(A[i * aug_width + j], sub, p);
                }
            }
            __syncthreads();
        }
    }

    // Copy solutions
    uint32_t sol_offset = prime_idx * n * k;
    for (uint32_t i = local_tid; i < n * k; i += 256) {
        uint32_t col_idx = i / n;
        uint32_t row = i % n;
        solutions[sol_offset + i] = is_singular ? 0 : A[row * aug_width + n + col_idx];
    }

    if (local_tid == 0) {
        singular_flags[prime_idx] = is_singular ? 1 : 0;
    }
}

// ============================================================================
// V2 CRT Reconstruction Kernels (64-bit primes)
// ============================================================================

// 64-bit modular operations for CRT (using 128-bit intermediate)
__device__ __forceinline__ uint64_t crt_mod_mul_128(uint64_t a, uint64_t b, uint64_t p) {
    uint128_t prod = mul64(a, b);
    return mod128(prod, p);
}

// Full CRT reconstruction for 64-bit primes using Garner's algorithm
// One thread per value, each thread iterates through all primes
extern "C" __global__ void crt_reconstruct_full_64(
    const uint64_t* __restrict__ residues,          // [num_values * num_primes] row-major
    const uint64_t* __restrict__ primes,            // [num_primes]
    const uint64_t* __restrict__ inverses,          // [num_primes] garner inverses
    const uint32_t* __restrict__ pp_limbs,          // Packed partial product limbs (32-bit)
    const uint32_t* __restrict__ pp_offsets,        // [num_primes] offset into pp_limbs
    const uint32_t* __restrict__ pp_sizes,          // [num_primes] number of limbs
    const uint64_t* __restrict__ pow2_mod,          // [num_primes * max_acc_limbs] 2^(32j) mod p[i]
    uint32_t* __restrict__ output_limbs,            // [num_values * max_acc_limbs]
    uint32_t* __restrict__ output_sizes,            // [num_values] actual size of each result
    uint32_t num_values,
    uint32_t num_primes,
    uint32_t max_acc_limbs
) {
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_values) return;

    // Working accumulator for this thread (in global memory)
    uint32_t* acc = output_limbs + v * max_acc_limbs;

    // Initialize accumulator to first residue (64-bit -> split into two 32-bit limbs)
    uint64_t first_res = residues[v * num_primes + 0];
    acc[0] = (uint32_t)(first_res & 0xFFFFFFFF);
    acc[1] = (uint32_t)(first_res >> 32);
    for (uint32_t j = 2; j < max_acc_limbs; j++) {
        acc[j] = 0;
    }

    uint32_t current_size = (first_res > 0xFFFFFFFF) ? 2 : 1;

    // Garner iterations: build up the result incrementally
    for (uint32_t i = 1; i < num_primes; i++) {
        uint64_t p = primes[i];
        uint64_t inv = inverses[i];
        uint64_t r = residues[v * num_primes + i];

        // Compute acc mod p using precomputed powers of 2^32
        const uint64_t* pow2 = pow2_mod + i * max_acc_limbs;
        uint64_t acc_mod_p = 0;

        for (uint32_t j = 0; j < current_size; j++) {
            // acc_mod_p += acc[j] * pow2[j] mod p
            uint64_t term = crt_mod_mul_128(acc[j], pow2[j], p);
            acc_mod_p += term;
            // Periodic reduction to prevent overflow
            if ((j & 7) == 7) {
                acc_mod_p %= p;
            }
        }
        acc_mod_p %= p;

        // Compute Garner coefficient: t = (r - acc_mod_p) * inv mod p
        uint64_t diff;
        if (r >= acc_mod_p) {
            diff = r - acc_mod_p;
        } else {
            diff = p - acc_mod_p + r;
        }
        uint64_t t = crt_mod_mul_128(diff, inv, p);

        if (t == 0) continue;

        // Load partial product info
        uint32_t pp_offset = pp_offsets[i];
        uint32_t pp_size = pp_sizes[i];

        // acc += partial_products[i] * t
        // t is 64-bit, so we need to handle this carefully
        uint32_t t_lo = (uint32_t)(t & 0xFFFFFFFF);
        uint32_t t_hi = (uint32_t)(t >> 32);

        uint64_t carry = 0;
        for (uint32_t j = 0; j < max_acc_limbs; j++) {
            uint64_t pp_limb = (j < pp_size) ? pp_limbs[pp_offset + j] : 0;

            // product = pp_limb * t + acc[j] + carry
            // pp_limb * t = pp_limb * t_lo + pp_limb * t_hi * 2^32
            uint64_t prod_lo = (uint64_t)pp_limb * (uint64_t)t_lo;
            uint64_t prod_hi = (uint64_t)pp_limb * (uint64_t)t_hi;

            uint64_t sum = prod_lo + acc[j] + carry;
            acc[j] = (uint32_t)sum;
            carry = (sum >> 32) + prod_hi;

            // Handle prod_hi overflow into next iteration
            if (j + 1 < max_acc_limbs && prod_hi > 0) {
                // This is handled by the carry
            }
        }

        // Update current size
        uint32_t new_size = pp_size + 2;  // +2 for 64-bit coefficient
        if (new_size > current_size) current_size = new_size;
        if (current_size > max_acc_limbs) current_size = max_acc_limbs;
    }

    // Trim trailing zeros
    while (current_size > 1 && acc[current_size - 1] == 0) {
        current_size--;
    }

    output_sizes[v] = current_size;
}

// Sign detection for 64-bit CRT (same as 32-bit, operates on 32-bit limbs)
extern "C" __global__ void crt_to_signed_64(
    uint32_t* __restrict__ limbs,
    const uint32_t* __restrict__ half_product,
    const uint32_t* __restrict__ full_product,
    uint32_t* __restrict__ signs,
    uint32_t num_values,
    uint32_t max_limbs,
    uint32_t half_size,
    uint32_t product_size
) {
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_values) return;

    uint32_t* acc = limbs + v * max_limbs;

    // Compare with M/2 (from MSB to LSB)
    int cmp = 0;

    // First check if acc has limbs beyond half_product
    for (int j = (int)max_limbs - 1; j >= (int)half_size && cmp == 0; j--) {
        if (acc[j] > 0) {
            cmp = 1;
            break;
        }
    }

    // Then compare overlapping limbs
    if (cmp == 0) {
        uint32_t check_size = (max_limbs < half_size) ? max_limbs : half_size;
        for (int j = (int)check_size - 1; j >= 0 && cmp == 0; j--) {
            if (acc[j] > half_product[j]) {
                cmp = 1;
            } else if (acc[j] < half_product[j]) {
                cmp = -1;
            }
        }
    }

    if (cmp > 0) {
        signs[v] = 1;
        // Compute M - acc (two's complement style)
        uint64_t borrow = 0;
        for (uint32_t j = 0; j < max_limbs; j++) {
            uint64_t p_limb = (j < product_size) ? full_product[j] : 0;
            uint64_t a_limb = acc[j];
            uint64_t diff;
            if (p_limb >= a_limb + borrow) {
                diff = p_limb - a_limb - borrow;
                borrow = 0;
            } else {
                diff = (1ULL << 32) + p_limb - a_limb - borrow;
                borrow = 1;
            }
            acc[j] = (uint32_t)diff;
        }
    } else {
        signs[v] = 0;
    }
}

// ============================================================================
// LLL / Lovász Kernels (continued from above)
// ============================================================================

// Check Lovász condition for all primes in parallel
// Returns 1 if Lovász condition is satisfied, 0 otherwise
// Lovász: delta * ||b*_{k-1}||^2 <= ||b*_k||^2 + mu_{k,k-1}^2 * ||b*_{k-1}||^2
// With delta = delta_num/delta_den (typically 3/4)
extern "C" __global__ void check_lovasz_condition(
    const uint32_t* __restrict__ b_star_norm_sq,  // [num_primes][n]
    const uint32_t* __restrict__ mu_num,          // [num_primes][n*(n-1)/2]
    const uint32_t* __restrict__ mu_den,          // [num_primes][n*(n-1)/2]
    uint32_t* __restrict__ lovasz_satisfied,      // [num_primes] - output: 1 if satisfied
    const uint32_t* __restrict__ primes,
    uint32_t k,                                   // index to check
    uint32_t delta_num,                           // numerator of delta (e.g., 3)
    uint32_t delta_den,                           // denominator of delta (e.g., 4)
    uint32_t n,
    uint32_t num_primes
) {
    uint32_t prime_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (prime_idx >= num_primes) return;

    uint32_t p = primes[prime_idx];
    uint64_t p64 = (uint64_t)p;

    uint32_t norm_k = b_star_norm_sq[prime_idx * n + k];
    uint32_t norm_k1 = b_star_norm_sq[prime_idx * n + k - 1];

    // Get mu_{k, k-1}
    uint32_t mu_idx = k * (k - 1) / 2 + (k - 1);
    uint32_t mu_n = mu_num[prime_idx * (n * (n - 1) / 2) + mu_idx];
    uint32_t mu_d = mu_den[prime_idx * (n * (n - 1) / 2) + mu_idx];

    // Lovász condition (cross-multiplied to avoid division):
    // delta_num * mu_d^2 * norm_k1 <= delta_den * mu_d^2 * norm_k + delta_den * mu_n^2 * norm_k1
    // Simplify:
    // delta_num * mu_d^2 * norm_k1 <= delta_den * (mu_d^2 * norm_k + mu_n^2 * norm_k1)

    uint64_t mu_d_sq = ((uint64_t)mu_d * mu_d) % p64;
    uint64_t mu_n_sq = ((uint64_t)mu_n * mu_n) % p64;

    uint64_t lhs = (delta_num * mu_d_sq % p64) * norm_k1 % p64;
    uint64_t rhs_term1 = (mu_d_sq * norm_k) % p64;
    uint64_t rhs_term2 = (mu_n_sq * norm_k1) % p64;
    uint64_t rhs = (delta_den * ((rhs_term1 + rhs_term2) % p64)) % p64;

    // In modular arithmetic, we can't directly compare with <=
    // This is a fundamental limitation - we need the actual BigInt comparison
    // For now, store both values for CPU comparison
    // A practical workaround: if all primes agree on the ordering, we trust it

    // Simple heuristic: satisfied if lhs <= rhs in the modular representation
    // This is only correct when values don't wrap around the modulus
    lovasz_satisfied[prime_idx] = (lhs <= rhs) ? 1 : 0;
}

// ============================================================================
// Hensel (Dixon) p-adic Lifting Kernels
// ============================================================================
//
// These kernels implement GPU-accelerated Hensel lifting for exact linear
// system solving. The algorithm:
// 1. Compute A⁻¹ mod p once (using matrix_inverse_mod kernel)
// 2. x_0 = A⁻¹ b mod p
// 3. For each iteration i:
//    - residual = (b - A*x_approx) / p^i mod p
//    - correction = A⁻¹ * residual mod p
//    - x_approx = x_approx + p^i * correction
//
// Complexity: O(n³) for inverse + O(iterations × n²) for lifting
// vs CRT: O(num_primes × n³) for independent factorizations

// Compute matrix inverse mod p using Gauss-Jordan elimination
// One thread block per work item, processes the full n×n matrix
// Input: a[n][n] - matrix to invert (row-major)
// Output: a_inv[n][n] - inverse matrix (row-major)
// singular[1] - set to 1 if matrix is singular
extern "C" __global__ void hensel_matrix_inverse(
    const uint32_t* __restrict__ a,      // [n][n] input matrix
    uint32_t* __restrict__ a_inv,        // [n][n] output inverse
    uint32_t* __restrict__ singular,     // [1] singularity flag
    uint32_t p,                          // prime modulus
    uint32_t n
) {
    // This kernel is launched with 1 block, n threads
    uint32_t tid = threadIdx.x;
    uint64_t p64 = (uint64_t)p;

    // Shared memory for augmented matrix [A | I]
    extern __shared__ uint32_t aug[];  // [n][2*n]

    // Initialize augmented matrix: [A | I]
    for (uint32_t j = 0; j < 2 * n; j++) {
        if (j < n) {
            aug[tid * 2 * n + j] = a[tid * n + j];
        } else {
            aug[tid * 2 * n + j] = (j - n == tid) ? 1 : 0;
        }
    }
    __syncthreads();

    // Gauss-Jordan elimination
    for (uint32_t col = 0; col < n; col++) {
        // Thread 0 finds pivot and scales row
        if (tid == 0) {
            // Find pivot (first non-zero in column)
            uint32_t pivot_row = col;
            bool found = false;
            for (uint32_t row = col; row < n; row++) {
                if (aug[row * 2 * n + col] != 0) {
                    pivot_row = row;
                    found = true;
                    break;
                }
            }

            if (!found) {
                *singular = 1;
                return;
            }

            // Swap rows if needed
            if (pivot_row != col) {
                for (uint32_t j = 0; j < 2 * n; j++) {
                    uint32_t tmp = aug[col * 2 * n + j];
                    aug[col * 2 * n + j] = aug[pivot_row * 2 * n + j];
                    aug[pivot_row * 2 * n + j] = tmp;
                }
            }

            // Scale pivot row to make pivot = 1
            uint32_t pivot = aug[col * 2 * n + col];
            uint64_t pivot_inv = mod_pow(pivot, p - 2, p);
            for (uint32_t j = 0; j < 2 * n; j++) {
                aug[col * 2 * n + j] = (aug[col * 2 * n + j] * pivot_inv) % p64;
            }
        }
        __syncthreads();

        // Check if singular
        if (*singular) return;

        // All threads eliminate their row
        if (tid != col && tid < n) {
            uint64_t factor = aug[tid * 2 * n + col];
            if (factor != 0) {
                for (uint32_t j = 0; j < 2 * n; j++) {
                    uint64_t sub = (factor * aug[col * 2 * n + j]) % p64;
                    aug[tid * 2 * n + j] = (aug[tid * 2 * n + j] + p64 - sub) % p64;
                }
            }
        }
        __syncthreads();
    }

    // Extract inverse from right half
    for (uint32_t j = 0; j < n; j++) {
        a_inv[tid * n + j] = aug[tid * 2 * n + n + j];
    }
}

// Batched matrix inverse: compute A⁻¹ mod p for multiple primes
// Launch: grid = num_primes, block = n threads
extern "C" __global__ void hensel_batch_matrix_inverse(
    const uint32_t* __restrict__ matrices,   // [num_primes][n][n] input
    uint32_t* __restrict__ inverses,         // [num_primes][n][n] output
    uint32_t* __restrict__ singular_flags,   // [num_primes] singularity flags
    const uint32_t* __restrict__ primes,     // [num_primes]
    uint32_t n,
    uint32_t num_primes
) {
    uint32_t prime_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (prime_idx >= num_primes || tid >= n) return;

    uint32_t p = primes[prime_idx];
    uint64_t p64 = (uint64_t)p;

    // Shared memory for augmented matrix [A | I]
    extern __shared__ uint32_t aug[];  // [n][2*n]

    const uint32_t* a = matrices + prime_idx * n * n;
    uint32_t* a_inv = inverses + prime_idx * n * n;

    // Initialize augmented matrix
    for (uint32_t j = 0; j < 2 * n; j++) {
        if (j < n) {
            aug[tid * 2 * n + j] = a[tid * n + j] % p;
        } else {
            aug[tid * 2 * n + j] = (j - n == tid) ? 1 : 0;
        }
    }
    __syncthreads();

    // Gauss-Jordan elimination (same as single version)
    for (uint32_t col = 0; col < n; col++) {
        if (tid == 0) {
            uint32_t pivot_row = col;
            bool found = false;
            for (uint32_t row = col; row < n; row++) {
                if (aug[row * 2 * n + col] != 0) {
                    pivot_row = row;
                    found = true;
                    break;
                }
            }

            if (!found) {
                singular_flags[prime_idx] = 1;
            } else {
                if (pivot_row != col) {
                    for (uint32_t j = 0; j < 2 * n; j++) {
                        uint32_t tmp = aug[col * 2 * n + j];
                        aug[col * 2 * n + j] = aug[pivot_row * 2 * n + j];
                        aug[pivot_row * 2 * n + j] = tmp;
                    }
                }

                uint32_t pivot = aug[col * 2 * n + col];
                uint64_t pivot_inv = mod_pow(pivot, p - 2, p);
                for (uint32_t j = 0; j < 2 * n; j++) {
                    aug[col * 2 * n + j] = (aug[col * 2 * n + j] * pivot_inv) % p64;
                }
            }
        }
        __syncthreads();

        if (singular_flags[prime_idx]) return;

        if (tid != col && tid < n) {
            uint64_t factor = aug[tid * 2 * n + col];
            if (factor != 0) {
                for (uint32_t j = 0; j < 2 * n; j++) {
                    uint64_t sub = (factor * aug[col * 2 * n + j]) % p64;
                    aug[tid * 2 * n + j] = (aug[tid * 2 * n + j] + p64 - sub) % p64;
                }
            }
        }
        __syncthreads();
    }

    for (uint32_t j = 0; j < n; j++) {
        a_inv[tid * n + j] = aug[tid * 2 * n + n + j];
    }
}

// Matrix-vector multiply: y = A * x mod p
// For Hensel lifting, used to compute A * x_approx
// Launch: grid = num_primes, block = n
extern "C" __global__ void hensel_matvec(
    const uint32_t* __restrict__ a,      // [num_primes][n][n] matrices
    const uint32_t* __restrict__ x,      // [num_primes][k][n] vectors (k RHS)
    uint32_t* __restrict__ y,            // [num_primes][k][n] output
    const uint32_t* __restrict__ primes, // [num_primes]
    uint32_t n,
    uint32_t k,                          // number of RHS
    uint32_t num_primes
) {
    uint32_t prime_idx = blockIdx.x;
    uint32_t row = threadIdx.x;

    if (prime_idx >= num_primes || row >= n) return;

    uint32_t p = primes[prime_idx];
    uint64_t p64 = (uint64_t)p;

    const uint32_t* a_p = a + prime_idx * n * n;

    for (uint32_t col_idx = 0; col_idx < k; col_idx++) {
        const uint32_t* x_col = x + prime_idx * k * n + col_idx * n;
        uint32_t* y_col = y + prime_idx * k * n + col_idx * n;

        uint64_t sum = 0;
        for (uint32_t j = 0; j < n; j++) {
            sum = (sum + (uint64_t)a_p[row * n + j] * x_col[j]) % p64;
        }
        y_col[row] = (uint32_t)sum;
    }
}

// Compute initial solution: x_0 = A⁻¹ * b mod p
// Launch: grid = num_primes, block = n
extern "C" __global__ void hensel_initial_solve(
    const uint32_t* __restrict__ a_inv,  // [num_primes][n][n] precomputed inverses
    const uint32_t* __restrict__ b,      // [num_primes][k][n] RHS vectors
    uint32_t* __restrict__ x,            // [num_primes][k][n] output solutions
    const uint32_t* __restrict__ primes, // [num_primes]
    uint32_t n,
    uint32_t k,
    uint32_t num_primes
) {
    uint32_t prime_idx = blockIdx.x;
    uint32_t row = threadIdx.x;

    if (prime_idx >= num_primes || row >= n) return;

    uint32_t p = primes[prime_idx];
    uint64_t p64 = (uint64_t)p;

    const uint32_t* a_inv_p = a_inv + prime_idx * n * n;

    for (uint32_t col_idx = 0; col_idx < k; col_idx++) {
        const uint32_t* b_col = b + prime_idx * k * n + col_idx * n;
        uint32_t* x_col = x + prime_idx * k * n + col_idx * n;

        uint64_t sum = 0;
        for (uint32_t j = 0; j < n; j++) {
            sum = (sum + (uint64_t)a_inv_p[row * n + j] * b_col[j]) % p64;
        }
        x_col[row] = (uint32_t)sum;
    }
}

// Hensel lifting iteration kernel
// Computes: residual = (b - A*x) mod p, then correction = A⁻¹ * residual mod p
// Input x is in p-adic form (accumulated digits)
// This version works with accumulated x stored as limbs
//
// Algorithm per iteration i:
// 1. Compute A*x mod p^(i+1) from p-adic digits
// 2. residual = (b - A*x) / p^i mod p
// 3. correction = A⁻¹ * residual mod p
// 4. New p-adic digit = correction
//
// Launch: grid = (num_primes, k), block = n
extern "C" __global__ void hensel_lift_iteration(
    const uint32_t* __restrict__ a,           // [num_primes][n][n] original matrices
    const uint32_t* __restrict__ a_inv,       // [num_primes][n][n] inverses mod p
    const uint32_t* __restrict__ b,           // [num_primes][k][n] RHS
    const uint32_t* __restrict__ x_digits,    // [num_primes][k][iter][n] accumulated digits
    uint32_t* __restrict__ new_digit,         // [num_primes][k][n] output new digit
    const uint32_t* __restrict__ primes,      // [num_primes]
    uint32_t n,
    uint32_t k,
    uint32_t iteration,                       // current iteration (0-indexed, 0 = first lift)
    uint32_t num_primes
) {
    uint32_t prime_idx = blockIdx.x;
    uint32_t rhs_idx = blockIdx.y;
    uint32_t row = threadIdx.x;

    if (prime_idx >= num_primes || rhs_idx >= k || row >= n) return;

    uint32_t p = primes[prime_idx];
    uint64_t p64 = (uint64_t)p;

    const uint32_t* a_p = a + prime_idx * n * n;
    const uint32_t* a_inv_p = a_inv + prime_idx * n * n;

    // Compute A*x where x is sum of p-adic digits
    // We need (A*x) mod p^(iteration+1), then extract coefficient of p^iteration
    //
    // For efficiency, we compute sum over j of: a[row,j] * (sum over iter of: x_digit[iter,j] * p^iter)
    // = sum over iter of: p^iter * (sum over j of: a[row,j] * x_digit[iter,j])
    // = sum over iter of: p^iter * (A * x_digit[iter])[row]
    //
    // We only need the coefficient of p^iteration, which comes from:
    // - Contribution from digit iteration (p^iteration * A * digit[iteration])
    // - Carry from lower digits

    // Shared memory for intermediate computations
    extern __shared__ uint64_t smem[];
    uint64_t* ax_contrib = smem;  // [n] contribution to A*x for this row

    // Compute A*x contributions from each digit level
    // ax_contrib[iteration] = (A * x_digit[iteration])[row]

    // For simplicity, compute (A*x) mod p^(iteration+2) and extract digit at iteration
    // This requires tracking carries properly

    // Simpler approach: compute A*x at each digit level and propagate carries
    uint64_t ax_accum = 0;  // Accumulated value mod p^(iteration+2)
    uint64_t p_power = 1;

    for (uint32_t dig = 0; dig <= iteration; dig++) {
        const uint32_t* x_dig = x_digits + (prime_idx * k * (iteration + 1) + rhs_idx * (iteration + 1) + dig) * n;

        uint64_t dot = 0;
        for (uint32_t j = 0; j < n; j++) {
            dot = (dot + (uint64_t)a_p[row * n + j] * x_dig[j]) % (p64 * p64);  // Prevent overflow
        }

        ax_accum += p_power * dot;
        p_power *= p64;
    }

    // b value for this row
    uint32_t b_val = b[prime_idx * k * n + rhs_idx * n + row];

    // Compute residual = (b - A*x) / p^iteration mod p
    // First compute b mod p^(iteration+1) (just b since b is small)
    uint64_t b_ext = (uint64_t)b_val;

    // residual coefficient of p^iteration in (b - A*x)
    // We need: ((b - A*x) / p^iteration) mod p

    // Compute p^iteration
    p_power = 1;
    for (uint32_t i = 0; i < iteration; i++) {
        p_power *= p64;
    }

    // (b - ax) / p^iteration mod p
    // Handle potential negative values
    int64_t diff = (int64_t)b_ext - (int64_t)(ax_accum % (p_power * p64));
    int64_t quotient = diff / (int64_t)p_power;
    int64_t residual = ((quotient % (int64_t)p64) + (int64_t)p64) % (int64_t)p64;

    // Store residual in shared memory
    ax_contrib[row] = (uint64_t)residual;
    __syncthreads();

    // Compute correction = A⁻¹ * residual mod p
    uint64_t corr_sum = 0;
    for (uint32_t j = 0; j < n; j++) {
        corr_sum = (corr_sum + (uint64_t)a_inv_p[row * n + j] * ax_contrib[j]) % p64;
    }

    // Output new digit
    new_digit[prime_idx * k * n + rhs_idx * n + row] = (uint32_t)corr_sum;
}

// Simplified Hensel lifting kernel using p-adic storage
// Computes one lifting iteration at a time
// Input: x_current - current p-adic approximation (multiple of p^(iteration-1))
// Output: new_digit - the digit for p^iteration
//
// This version stores x as accumulated sum (BigInt-style) rather than separate digits
// Launch: grid = ceil(n*k/256), block = 256
extern "C" __global__ void hensel_lift_simple(
    const uint32_t* __restrict__ a,           // [n][n] matrix (single prime)
    const uint32_t* __restrict__ a_inv,       // [n][n] inverse
    const uint32_t* __restrict__ b,           // [k][n] RHS
    const uint64_t* __restrict__ ax_low,      // [k][n] lower bits of A*x mod p^2
    const uint64_t* __restrict__ ax_high,     // [k][n] higher bits for carries
    uint32_t* __restrict__ new_digit,         // [k][n] output digit
    uint32_t p,
    uint32_t n,
    uint32_t k,
    uint32_t iteration
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = n * k;
    if (idx >= total) return;

    uint32_t rhs_idx = idx / n;
    uint32_t row = idx % n;

    uint64_t p64 = (uint64_t)p;

    // Compute p^iteration
    uint64_t p_power = 1;
    for (uint32_t i = 0; i < iteration; i++) {
        p_power *= p64;
    }

    // Get b - A*x coefficient at p^iteration level
    uint64_t b_val = b[rhs_idx * n + row];
    uint64_t ax_val = ax_low[rhs_idx * n + row];

    // residual = ((b - ax) / p^iteration) mod p
    int64_t diff = (int64_t)b_val - (int64_t)(ax_val % (p_power * p64));
    int64_t quotient = diff / (int64_t)p_power;
    uint32_t residual = (uint32_t)(((quotient % (int64_t)p64) + (int64_t)p64) % (int64_t)p64);

    // correction = A⁻¹ * residual mod p
    // Need to accumulate across row - this requires shared memory or atomic
    // For now, output residual and do A⁻¹ multiply separately
    new_digit[rhs_idx * n + row] = residual;
}

// Full Hensel solve: combines inverse, initial solve, and lifting
// This is the main entry point for GPU Hensel lifting
// Launch: grid = 1, block = n (for small matrices)
// For larger matrices, use the batched versions
extern "C" __global__ void hensel_solve_full(
    const uint32_t* __restrict__ a,           // [n][n] input matrix
    uint32_t* __restrict__ a_inv,             // [n][n] workspace for inverse
    const uint32_t* __restrict__ b,           // [k][n] RHS
    uint32_t* __restrict__ x_digits,          // [max_iter][k][n] p-adic digits output
    uint32_t* __restrict__ singular,          // singularity flag
    uint32_t p,
    uint32_t n,
    uint32_t k,
    uint32_t max_iterations
) {
    // This is a placeholder - the actual implementation should use
    // separate kernel calls for better parallelism and memory management
    // See hensel_batch_matrix_inverse, hensel_initial_solve, hensel_lift_iteration
}

// ============================================================================
// GPU-Native Hensel Lifting (Defect-Based)
// ============================================================================
//
// These kernels implement efficient GPU Hensel lifting using the "defect"
// formulation which keeps all values bounded in 64-bit signed integers.
//
// Key insight: Track d_i = (b - A*x_i) / p^i instead of x_i directly.
// The defect stays bounded: |d_i| ≈ n * max(|A|), not growing with iterations.
//
// Recurrence:
//   digit_i = A⁻¹ * (d_i mod p) mod p
//   d_{i+1} = (d_i - A * digit_i) / p
//
// All operations use 64-bit signed arithmetic - no BigInt needed!

// Initialize defect from original RHS (signed 64-bit)
// Launch: grid = ceil(k*n/256), block = 256
extern "C" __global__ void hensel_gpu_init_defect(
    const int64_t* __restrict__ b,      // [k][n] original RHS (signed)
    int64_t* __restrict__ defect,       // [k][n] output defect
    uint32_t n,
    uint32_t k
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * k) return;
    defect[idx] = b[idx];
}

// Compute p-adic digit from defect: digit = A⁻¹ * (defect mod p) mod p
// Launch: grid = k, block = n (one block per RHS column)
extern "C" __global__ void hensel_gpu_compute_digit(
    const uint32_t* __restrict__ a_inv,   // [n][n] inverse mod p
    const int64_t* __restrict__ defect,   // [k][n] current defect
    uint32_t* __restrict__ digit,         // [k][n] output digit
    uint32_t p,
    uint32_t n,
    uint32_t k
) {
    uint32_t col_idx = blockIdx.x;
    uint32_t row = threadIdx.x;

    if (col_idx >= k || row >= n) return;

    int64_t p64 = (int64_t)p;

    // Compute A⁻¹[row,:] * (defect mod p)
    uint64_t sum = 0;
    for (uint32_t j = 0; j < n; j++) {
        // Extract defect mod p (handle negative values)
        int64_t d = defect[col_idx * n + j];
        uint32_t d_mod_p = (uint32_t)(((d % p64) + p64) % p64);

        sum += (uint64_t)a_inv[row * n + j] * d_mod_p;
    }

    digit[col_idx * n + row] = (uint32_t)(sum % p);
}

// Update defect: d = (d - A * digit) / p
// This is exact division since d ≡ A*digit (mod p) by construction
// Launch: grid = k, block = n
extern "C" __global__ void hensel_gpu_update_defect(
    int64_t* __restrict__ defect,         // [k][n] in/out defect
    const int64_t* __restrict__ a,        // [n][n] original matrix (signed)
    const uint32_t* __restrict__ digit,   // [k][n] current digit
    int64_t p,
    uint32_t n,
    uint32_t k
) {
    uint32_t col_idx = blockIdx.x;
    uint32_t row = threadIdx.x;

    if (col_idx >= k || row >= n) return;

    // Compute A[row,:] * digit
    int64_t ax = 0;
    for (uint32_t j = 0; j < n; j++) {
        ax += a[row * n + j] * (int64_t)digit[col_idx * n + j];
    }

    // d = (d - ax) / p (exact division)
    int64_t d = defect[col_idx * n + row];
    defect[col_idx * n + row] = (d - ax) / p;
}

// Tiled version for larger matrices: compute digit using shared memory
// Launch: grid = (ceil(n/16), k), block = (16, 16)
extern "C" __global__ void hensel_gpu_compute_digit_tiled(
    const uint32_t* __restrict__ a_inv,   // [n][n] inverse mod p
    const int64_t* __restrict__ defect,   // [k][n] current defect
    uint32_t* __restrict__ digit,         // [k][n] output digit
    uint32_t p,
    uint32_t n,
    uint32_t k
) {
    __shared__ uint32_t s_ainv[16][16];   // Tile of A⁻¹
    __shared__ uint32_t s_d[16];          // Tile of defect mod p

    uint32_t col_idx = blockIdx.y;
    uint32_t row_base = blockIdx.x * 16;
    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;
    uint32_t row = row_base + tx;

    if (col_idx >= k) return;

    int64_t p64 = (int64_t)p;
    uint64_t sum = 0;

    // Process in tiles
    for (uint32_t tile = 0; tile < (n + 15) / 16; tile++) {
        uint32_t j = tile * 16 + ty;

        // Load A⁻¹ tile
        if (row < n && j < n) {
            s_ainv[tx][ty] = a_inv[row * n + j];
        } else {
            s_ainv[tx][ty] = 0;
        }

        // Load defect tile (only first row of threads)
        if (tx == 0 && j < n) {
            int64_t d = defect[col_idx * n + j];
            s_d[ty] = (uint32_t)(((d % p64) + p64) % p64);
        }
        __syncthreads();

        // Accumulate
        if (row < n) {
            for (uint32_t i = 0; i < 16 && (tile * 16 + i) < n; i++) {
                sum += (uint64_t)s_ainv[tx][i] * s_d[i];
            }
        }
        __syncthreads();
    }

    if (row < n) {
        digit[col_idx * n + row] = (uint32_t)(sum % p);
    }
}

// Tiled version for update defect
// Launch: grid = (ceil(n/16), k), block = (16, 16)
extern "C" __global__ void hensel_gpu_update_defect_tiled(
    int64_t* __restrict__ defect,         // [k][n] in/out defect
    const int64_t* __restrict__ a,        // [n][n] original matrix (signed)
    const uint32_t* __restrict__ digit,   // [k][n] current digit
    int64_t p,
    uint32_t n,
    uint32_t k
) {
    __shared__ int64_t s_a[16][16];       // Tile of A
    __shared__ uint32_t s_digit[16];      // Tile of digit

    uint32_t col_idx = blockIdx.y;
    uint32_t row_base = blockIdx.x * 16;
    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;
    uint32_t row = row_base + tx;

    if (col_idx >= k) return;

    int64_t ax = 0;

    // Process in tiles
    for (uint32_t tile = 0; tile < (n + 15) / 16; tile++) {
        uint32_t j = tile * 16 + ty;

        // Load A tile
        if (row < n && j < n) {
            s_a[tx][ty] = a[row * n + j];
        } else {
            s_a[tx][ty] = 0;
        }

        // Load digit tile
        if (tx == 0 && j < n) {
            s_digit[ty] = digit[col_idx * n + j];
        }
        __syncthreads();

        // Accumulate
        if (row < n) {
            for (uint32_t i = 0; i < 16 && (tile * 16 + i) < n; i++) {
                ax += s_a[tx][i] * (int64_t)s_digit[i];
            }
        }
        __syncthreads();
    }

    if (row < n) {
        int64_t d = defect[col_idx * n + row];
        defect[col_idx * n + row] = (d - ax) / p;
    }
}

// Combined kernel: compute digit AND update defect in one pass
// More efficient as it avoids global memory round-trip for digit
// Launch: grid = k, block = n (for n <= 1024)
extern "C" __global__ void hensel_gpu_lift_step(
    const uint32_t* __restrict__ a_inv,   // [n][n] inverse mod p
    const int64_t* __restrict__ a,        // [n][n] original matrix (signed)
    int64_t* __restrict__ defect,         // [k][n] in/out defect
    uint32_t* __restrict__ digit,         // [k][n] output digit
    uint32_t p,
    uint32_t n,
    uint32_t k
) {
    extern __shared__ uint32_t shared[];
    uint32_t* s_digit = shared;           // [n] digit values

    uint32_t col_idx = blockIdx.x;
    uint32_t row = threadIdx.x;

    if (col_idx >= k || row >= n) return;

    int64_t p64 = (int64_t)p;

    // Step 1: Each thread computes its digit element
    uint64_t sum = 0;
    for (uint32_t j = 0; j < n; j++) {
        int64_t d = defect[col_idx * n + j];
        uint32_t d_mod_p = (uint32_t)(((d % p64) + p64) % p64);
        sum += (uint64_t)a_inv[row * n + j] * d_mod_p;
    }
    uint32_t my_digit = (uint32_t)(sum % p);
    s_digit[row] = my_digit;
    digit[col_idx * n + row] = my_digit;
    __syncthreads();

    // Step 2: Each thread updates its defect element
    int64_t ax = 0;
    for (uint32_t j = 0; j < n; j++) {
        ax += a[row * n + j] * (int64_t)s_digit[j];
    }

    int64_t d = defect[col_idx * n + row];
    defect[col_idx * n + row] = (d - ax) / p64;
}

// ============================================================================
// Offset variants for all-on-GPU lifting (no per-iteration CPU transfers)
// ============================================================================

// Compute digit with offset into output array
// This allows storing all iteration digits contiguously on GPU
// Launch: grid = k, block = n
extern "C" __global__ void hensel_gpu_compute_digit_offset(
    const uint32_t* __restrict__ a_inv,   // [n][n] inverse mod p
    const int64_t* __restrict__ defect,   // [k][n] current defect
    uint32_t* __restrict__ all_digits,    // [max_iters][k][n] ALL digits storage
    uint32_t digit_offset,                // offset into all_digits for this iteration
    uint32_t p,
    uint32_t n,
    uint32_t k
) {
    uint32_t col_idx = blockIdx.x;
    uint32_t row = threadIdx.x;

    if (col_idx >= k || row >= n) return;

    int64_t p64 = (int64_t)p;

    // Compute A⁻¹[row,:] * (defect mod p)
    uint64_t sum = 0;
    for (uint32_t j = 0; j < n; j++) {
        int64_t d = defect[col_idx * n + j];
        uint32_t d_mod_p = (uint32_t)(((d % p64) + p64) % p64);
        sum += (uint64_t)a_inv[row * n + j] * d_mod_p;
    }

    // Store at offset location
    all_digits[digit_offset + col_idx * n + row] = (uint32_t)(sum % p);
}

// Update defect reading digit from offset location
// Launch: grid = k, block = n
extern "C" __global__ void hensel_gpu_update_defect_offset(
    int64_t* __restrict__ defect,         // [k][n] in/out defect
    const int64_t* __restrict__ a,        // [n][n] original matrix (signed)
    const uint32_t* __restrict__ all_digits, // [max_iters][k][n] ALL digits storage
    uint32_t digit_offset,                // offset into all_digits for this iteration
    int64_t p,
    uint32_t n,
    uint32_t k
) {
    uint32_t col_idx = blockIdx.x;
    uint32_t row = threadIdx.x;

    if (col_idx >= k || row >= n) return;

    // Compute A[row,:] * digit
    int64_t ax = 0;
    for (uint32_t j = 0; j < n; j++) {
        ax += a[row * n + j] * (int64_t)all_digits[digit_offset + col_idx * n + j];
    }

    // d = (d - ax) / p (exact division)
    int64_t d = defect[col_idx * n + row];
    defect[col_idx * n + row] = (d - ax) / p;
}

// ============================================================================
// FULLY FUSED KERNEL: All iterations in a single kernel launch
// This eliminates all kernel launch overhead and inter-iteration synchronization
// ============================================================================

// Fused all-iterations kernel: runs all lifting iterations without returning to host
// Uses shared memory for digit values within each iteration
// Launch: grid = k, block = n (for n <= 1024)
// Shared memory: n * sizeof(uint32_t) for digit storage
extern "C" __global__ void hensel_gpu_lift_all_fused(
    const uint32_t* __restrict__ a_inv,   // [n][n] inverse mod p
    const int64_t* __restrict__ a,        // [n][n] original matrix (signed)
    const int64_t* __restrict__ b,        // [k][n] RHS values (column-major: [col][row])
    uint32_t* __restrict__ all_digits,    // [max_iters][k][n] output ALL digits
    uint32_t p,
    uint32_t n,
    uint32_t k,
    uint32_t max_iterations
) {
    extern __shared__ char shared_mem[];
    uint32_t* s_digit = (uint32_t*)shared_mem;             // [n] digit values
    uint32_t* s_defect_modp = (uint32_t*)(s_digit + n);    // [n] defect mod p

    uint32_t col_idx = blockIdx.x;
    uint32_t row = threadIdx.x;

    if (col_idx >= k || row >= n) return;

    int64_t p64 = (int64_t)p;

    // Initialize local defect from b
    int64_t my_defect = b[col_idx * n + row];

    // Run all iterations
    for (uint32_t iter = 0; iter < max_iterations; iter++) {
        // Step 1: Compute digit = A⁻¹[row,:] * (defect mod p) mod p
        // Each thread contributes its defect mod p
        s_defect_modp[row] = (uint32_t)(((my_defect % p64) + p64) % p64);
        __syncthreads();

        // Now compute A⁻¹[row,:] * s_defect_modp
        uint64_t sum = 0;
        for (uint32_t j = 0; j < n; j++) {
            sum += (uint64_t)a_inv[row * n + j] * s_defect_modp[j];
        }
        uint32_t my_digit = (uint32_t)(sum % p);
        s_digit[row] = my_digit;

        // Store digit to global memory
        uint32_t digit_offset = iter * k * n;
        all_digits[digit_offset + col_idx * n + row] = my_digit;
        __syncthreads();

        // Step 2: Update defect = (defect - A[row,:] * digit) / p
        int64_t ax = 0;
        for (uint32_t j = 0; j < n; j++) {
            ax += a[row * n + j] * (int64_t)s_digit[j];
        }
        my_defect = (my_defect - ax) / p64;
        __syncthreads();
    }
}
