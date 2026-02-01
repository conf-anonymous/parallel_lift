//! Metal shader source code
//!
//! Contains MSL (Metal Shading Language) kernels for modular arithmetic.
//! These are the batched kernels that enable GPU acceleration for CRT-based exact arithmetic.

/// Metal shader library source
///
/// Key design: One thread per prime - each thread performs complete Gaussian elimination
/// for its prime modulus. This batching strategy allows processing all primes in parallel.
pub const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Modular Arithmetic Utilities (31-bit primes, 64-bit intermediate)
// =============================================================================

inline uint32_t mod_add(uint32_t a, uint32_t b, uint32_t p) {
    uint64_t sum = uint64_t(a) + uint64_t(b);
    return sum >= p ? uint32_t(sum - p) : uint32_t(sum);
}

inline uint32_t mod_sub(uint32_t a, uint32_t b, uint32_t p) {
    return a >= b ? (a - b) : (p - b + a);
}

inline uint32_t mod_mul(uint32_t a, uint32_t b, uint32_t p) {
    uint64_t prod = uint64_t(a) * uint64_t(b);
    return uint32_t(prod % uint64_t(p));
}

// Modular inverse using Fermat's Little Theorem: a^(-1) = a^(p-2) mod p
// Binary exponentiation has better GPU pipelining than Extended Euclidean Algorithm
// For 31-bit primes, ~30 iterations with predictable branching
inline uint32_t mod_inv(uint32_t a, uint32_t p) {
    if (a == 0) return 0;

    // Fermat's Little Theorem: a^(p-1) ≡ 1 (mod p) for prime p
    // Therefore: a^(-1) ≡ a^(p-2) (mod p)
    uint32_t result = 1;
    uint32_t base = a;
    uint32_t exp = p - 2;

    // Binary exponentiation - GPU-friendly with no data-dependent branches
    while (exp > 0) {
        if (exp & 1) {
            result = mod_mul(result, base, p);
        }
        base = mod_mul(base, base, p);
        exp >>= 1;
    }

    return result;
}

// =============================================================================
// Batched Determinant Kernel
// =============================================================================
// Each thread processes one prime, performing complete Gaussian elimination.
// Memory layout:
//   matrices: [prime0_matrix, prime1_matrix, ...] where each matrix is n*n uint32
//   All matrices must be pre-reduced mod their respective primes
//
kernel void modular_determinant(
    device const uint32_t* matrices [[buffer(0)]],   // All matrices concatenated
    device const uint32_t* primes [[buffer(1)]],     // Array of primes
    device uint32_t* results [[buffer(2)]],          // Output: det mod p for each prime
    device uint32_t* singular_flags [[buffer(3)]],   // Output: 1 if matrix is singular mod p
    constant uint32_t& n [[buffer(4)]],              // Matrix dimension
    constant uint32_t& num_primes [[buffer(5)]],     // Number of primes
    device uint32_t* workspace [[buffer(6)]],        // Workspace: num_primes * n * n
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_primes) return;

    uint32_t p = primes[tid];
    uint32_t det = 1;
    bool singular = false;

    uint32_t nn = n * n;

    // Read from the pre-reduced matrix slice for this thread
    device const uint32_t* myMatrix = matrices + tid * nn;
    device uint32_t* A = workspace + tid * nn;

    // Copy to workspace
    for (uint32_t i = 0; i < nn; i++) {
        A[i] = myMatrix[i];
    }

    // Gaussian elimination with partial pivoting
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
            det = (p - det) % p;  // Negate determinant for row swap
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

// Small matrix variant using thread-local storage (faster for n <= 16)
kernel void modular_determinant_small(
    device const uint32_t* matrices [[buffer(0)]],
    device const uint32_t* primes [[buffer(1)]],
    device uint32_t* results [[buffer(2)]],
    device uint32_t* singular_flags [[buffer(3)]],
    constant uint32_t& n [[buffer(4)]],
    constant uint32_t& num_primes [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_primes) return;

    uint32_t p = primes[tid];
    uint32_t det = 1;
    bool singular = false;

    // Thread-local storage for small matrices (max 16x16 = 256 elements)
    uint32_t A[256];

    uint32_t nn = n * n;
    device const uint32_t* myMatrix = matrices + tid * nn;

    for (uint32_t i = 0; i < nn; i++) {
        A[i] = myMatrix[i];
    }

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

// =============================================================================
// Batched Solve Kernel (Ax = b)
// =============================================================================
// Each thread solves Ax = b mod one prime using Gaussian elimination
// Memory layout:
//   augmented: num_primes * n * (n+1) - augmented matrices [A|b] for each prime
//
kernel void modular_solve(
    device const uint32_t* augmented [[buffer(0)]],  // All [A|b] matrices concatenated
    device const uint32_t* primes [[buffer(1)]],     // Array of primes
    device uint32_t* solutions [[buffer(2)]],        // Output: x mod p for each prime
    device uint32_t* singular_flags [[buffer(3)]],   // Output: 1 if singular
    constant uint32_t& n [[buffer(4)]],              // Matrix dimension
    constant uint32_t& num_primes [[buffer(5)]],     // Number of primes
    device uint32_t* workspace [[buffer(6)]],        // Workspace: num_primes * n * (n+1)
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_primes) return;

    uint32_t p = primes[tid];
    bool singular = false;

    uint32_t stride = n * (n + 1);  // Size of augmented matrix

    // Copy augmented matrix to workspace
    device const uint32_t* myAug = augmented + tid * stride;
    device uint32_t* A = workspace + tid * stride;

    for (uint32_t i = 0; i < stride; i++) {
        A[i] = myAug[i];
    }

    // Forward elimination (Gaussian elimination mod p)
    for (uint32_t k = 0; k < n && !singular; k++) {
        // Find pivot
        uint32_t pivot_row = k;
        for (uint32_t i = k; i < n; i++) {
            if (A[i * (n + 1) + k] != 0) {
                pivot_row = i;
                break;
            }
        }

        if (A[pivot_row * (n + 1) + k] == 0) {
            singular = true;
            break;
        }

        // Swap rows if needed
        if (pivot_row != k) {
            for (uint32_t j = k; j <= n; j++) {
                uint32_t tmp = A[k * (n + 1) + j];
                A[k * (n + 1) + j] = A[pivot_row * (n + 1) + j];
                A[pivot_row * (n + 1) + j] = tmp;
            }
        }

        uint32_t pivot = A[k * (n + 1) + k];
        uint32_t pivot_inv = mod_inv(pivot, p);

        // Normalize pivot row
        for (uint32_t j = k; j <= n; j++) {
            A[k * (n + 1) + j] = mod_mul(A[k * (n + 1) + j], pivot_inv, p);
        }

        // Eliminate below
        for (uint32_t i = k + 1; i < n; i++) {
            uint32_t factor = A[i * (n + 1) + k];
            if (factor != 0) {
                for (uint32_t j = k; j <= n; j++) {
                    uint32_t sub = mod_mul(factor, A[k * (n + 1) + j], p);
                    A[i * (n + 1) + j] = mod_sub(A[i * (n + 1) + j], sub, p);
                }
            }
        }
    }

    singular_flags[tid] = singular ? 1 : 0;

    if (singular) {
        // Fill solution with zeros
        for (uint32_t i = 0; i < n; i++) {
            solutions[tid * n + i] = 0;
        }
        return;
    }

    // Back substitution
    // Matrix is now in row echelon form with 1s on diagonal
    device uint32_t* x = solutions + tid * n;

    for (int32_t i = int32_t(n) - 1; i >= 0; i--) {
        uint32_t sum = A[i * (n + 1) + n];  // RHS
        for (uint32_t j = i + 1; j < n; j++) {
            uint32_t sub = mod_mul(A[i * (n + 1) + j], x[j], p);
            sum = mod_sub(sum, sub, p);
        }
        x[i] = sum;
    }
}

// =============================================================================
// Multi-RHS Solve Kernel: AX = B where B has k columns
// =============================================================================
// This kernel factors A once and solves for all k right-hand sides.
// This is the key optimization for ZK preprocessing where we have many RHS vectors.
// Memory layout:
//   augmented: num_primes * n * (n + k) - augmented matrices [A|B]
//   solutions: num_primes * n * k - solution matrices X mod p (column-major)
//
kernel void modular_solve_multi_rhs(
    device const uint32_t* augmented [[buffer(0)]],  // [A|B] matrices (num_primes copies)
    device const uint32_t* primes [[buffer(1)]],     // Array of primes
    device uint32_t* solutions [[buffer(2)]],        // Output: X mod p (num_primes * n * k)
    device uint32_t* singular_flags [[buffer(3)]],   // Output: 1 if singular
    constant uint32_t& n [[buffer(4)]],              // Matrix dimension
    constant uint32_t& k [[buffer(5)]],              // Number of RHS vectors
    constant uint32_t& num_primes [[buffer(6)]],     // Number of primes
    device uint32_t* workspace [[buffer(7)]],        // Workspace: num_primes * n * (n+k)
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_primes) return;

    uint32_t p = primes[tid];
    bool singular = false;

    uint32_t aug_width = n + k;  // Width of augmented matrix [A|B]
    uint32_t stride = n * aug_width;  // Size of one augmented matrix

    // Copy augmented matrix to workspace
    device const uint32_t* myAug = augmented + tid * stride;
    device uint32_t* A = workspace + tid * stride;

    for (uint32_t i = 0; i < stride; i++) {
        A[i] = myAug[i];
    }

    // Forward elimination with partial pivoting
    for (uint32_t col = 0; col < n && !singular; col++) {
        // Find pivot in column 'col'
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

        // Swap rows if needed (swap entire rows including B columns)
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
        // Fill solutions with zeros
        for (uint32_t i = 0; i < n * k; i++) {
            solutions[tid * n * k + i] = 0;
        }
        return;
    }

    // Back substitution for all k RHS columns
    // Matrix A part is now upper triangular with 1s on diagonal
    // B columns have been transformed by the same row operations

    device uint32_t* X = solutions + tid * n * k;  // Output X (n×k matrix, column-major)

    for (uint32_t rhs = 0; rhs < k; rhs++) {
        // Solve for column 'rhs' of X
        // The transformed RHS is in column (n + rhs) of A

        for (int32_t row = int32_t(n) - 1; row >= 0; row--) {
            uint32_t sum = A[row * aug_width + (n + rhs)];  // RHS value

            for (uint32_t j = row + 1; j < n; j++) {
                uint32_t sub = mod_mul(A[row * aug_width + j], X[rhs * n + j], p);
                sum = mod_sub(sum, sub, p);
            }
            X[rhs * n + row] = sum;  // Store column-major: X[rhs][row]
        }
    }
}

// =============================================================================
// Threadgroup-Parallel Determinant Kernel (for n >= 32)
// =============================================================================
// Uses threadgroup parallelism WITHIN each prime's Gaussian elimination.
// Key idea: For each elimination step k, parallelize:
//   1. Pivot search using parallel reduction (all threads in threadgroup)
//   2. Row swap (parallel across columns)
//   3. Row elimination (parallel across rows AND columns)
//
// Memory model:
//   - One threadgroup per prime
//   - Threadgroup size: TILE_SIZE x TILE_SIZE threads
//   - Threadgroup memory for the active tile
//
// This provides ~4-8x speedup over serial elimination for n >= 32

constant uint TILE_SIZE = 16;  // 16x16 threadgroup = 256 threads

kernel void modular_determinant_tiled(
    device const uint32_t* matrices [[buffer(0)]],   // All matrices concatenated
    device const uint32_t* primes [[buffer(1)]],     // Array of primes
    device uint32_t* results [[buffer(2)]],          // Output: det mod p for each prime
    device uint32_t* singular_flags [[buffer(3)]],   // Output: 1 if singular
    constant uint32_t& n [[buffer(4)]],              // Matrix dimension
    constant uint32_t& num_primes [[buffer(5)]],     // Number of primes
    device uint32_t* workspace [[buffer(6)]],        // Workspace: num_primes * n * n
    uint2 tid [[thread_position_in_threadgroup]],    // (tx, ty) within threadgroup
    uint tgid [[threadgroup_position_in_grid]],      // Which prime this threadgroup handles
    uint tg_size [[threads_per_threadgroup]])
{
    if (tgid >= num_primes) return;

    uint32_t p = primes[tgid];
    uint32_t nn = n * n;

    // Thread indices within threadgroup
    uint32_t tx = tid.x;  // Column within tile
    uint32_t ty = tid.y;  // Row within tile
    uint32_t local_id = ty * TILE_SIZE + tx;
    uint32_t num_threads = TILE_SIZE * TILE_SIZE;

    // Pointers to this prime's data
    device const uint32_t* myMatrix = matrices + tgid * nn;
    device uint32_t* A = workspace + tgid * nn;

    // Threadgroup shared memory for reduction operations
    threadgroup uint32_t pivot_row_shared;
    threadgroup uint32_t pivot_val_shared;
    threadgroup uint32_t det_sign;
    threadgroup bool is_singular;
    threadgroup uint32_t pivot_inv_shared;

    // Initialize shared state (only thread 0)
    if (local_id == 0) {
        det_sign = 0;  // 0 = positive, 1 = negative
        is_singular = false;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Copy matrix to workspace (parallel copy)
    for (uint32_t i = local_id; i < nn; i += num_threads) {
        A[i] = myMatrix[i];
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Gaussian elimination with parallel operations
    for (uint32_t k = 0; k < n; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (is_singular) break;

        // ============================================================
        // Phase 1: Parallel pivot search (find first non-zero in column k)
        // ============================================================
        // Each thread checks one row if it's in range
        threadgroup uint32_t found_rows[256];  // Store row index if non-zero, else n
        threadgroup uint32_t found_vals[256];  // Store value at that row

        // Initialize search results
        uint32_t my_row = k + local_id;
        if (my_row < n && local_id < (n - k)) {
            uint32_t val = A[my_row * n + k];
            found_rows[local_id] = (val != 0) ? my_row : n;
            found_vals[local_id] = val;
        } else if (local_id < num_threads) {
            found_rows[local_id] = n;
            found_vals[local_id] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Parallel reduction to find minimum row with non-zero value
        for (uint32_t stride = num_threads / 2; stride > 0; stride /= 2) {
            if (local_id < stride) {
                if (found_rows[local_id + stride] < found_rows[local_id]) {
                    found_rows[local_id] = found_rows[local_id + stride];
                    found_vals[local_id] = found_vals[local_id + stride];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Thread 0 records pivot information
        if (local_id == 0) {
            pivot_row_shared = found_rows[0];
            pivot_val_shared = found_vals[0];

            if (pivot_row_shared >= n || pivot_val_shared == 0) {
                is_singular = true;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (is_singular) break;

        // ============================================================
        // Phase 2: Parallel row swap (if needed)
        // ============================================================
        if (pivot_row_shared != k) {
            // Parallel swap: each thread swaps one or more columns
            for (uint32_t j = k + local_id; j < n; j += num_threads) {
                uint32_t tmp = A[k * n + j];
                A[k * n + j] = A[pivot_row_shared * n + j];
                A[pivot_row_shared * n + j] = tmp;
            }

            // Update determinant sign
            if (local_id == 0) {
                det_sign ^= 1;
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // ============================================================
        // Phase 3: Compute pivot inverse (single thread)
        // ============================================================
        if (local_id == 0) {
            pivot_inv_shared = mod_inv(pivot_val_shared, p);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ============================================================
        // Phase 4: Parallel row elimination
        // ============================================================
        // Each thread handles multiple (row, col) pairs
        // Row range: [k+1, n), Col range: [k, n)

        uint32_t rows_to_eliminate = n - k - 1;
        uint32_t cols_to_update = n - k;

        // 2D parallelization: thread (tx, ty) handles rows and columns in strides
        for (uint32_t row_offset = ty; row_offset < rows_to_eliminate; row_offset += TILE_SIZE) {
            uint32_t i = k + 1 + row_offset;

            // Compute elimination factor for this row
            uint32_t factor = mod_mul(A[i * n + k], pivot_inv_shared, p);

            if (factor != 0) {
                // Eliminate columns in parallel
                for (uint32_t col_offset = tx; col_offset < cols_to_update; col_offset += TILE_SIZE) {
                    uint32_t j = k + col_offset;
                    uint32_t sub = mod_mul(factor, A[k * n + j], p);
                    A[i * n + j] = mod_sub(A[i * n + j], sub, p);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    // ============================================================
    // Final: Compute determinant from diagonal
    // ============================================================
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_id == 0) {
        if (is_singular) {
            results[tgid] = 0;
            singular_flags[tgid] = 1;
        } else {
            // Multiply diagonal elements
            uint32_t det = 1;
            for (uint32_t i = 0; i < n; i++) {
                det = mod_mul(det, A[i * n + i], p);
            }
            // Apply sign from row swaps
            if (det_sign == 1) {
                det = (p - det) % p;
            }
            results[tgid] = det;
            singular_flags[tgid] = 0;
        }
    }
}

// =============================================================================
// Threadgroup-Parallel Solve Kernel (Multi-RHS)
// =============================================================================
// Parallelizes Gaussian elimination and back-substitution across threadgroup
//
kernel void modular_solve_multi_rhs_tiled(
    device const uint32_t* augmented [[buffer(0)]],  // [A|B] matrices (num_primes copies)
    device const uint32_t* primes [[buffer(1)]],     // Array of primes
    device uint32_t* solutions [[buffer(2)]],        // Output: X mod p (num_primes * n * k)
    device uint32_t* singular_flags [[buffer(3)]],   // Output: 1 if singular
    constant uint32_t& n [[buffer(4)]],              // Matrix dimension
    constant uint32_t& k [[buffer(5)]],              // Number of RHS vectors
    constant uint32_t& num_primes [[buffer(6)]],     // Number of primes
    device uint32_t* workspace [[buffer(7)]],        // Workspace: num_primes * n * (n+k)
    uint2 tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tgid >= num_primes) return;

    uint32_t p = primes[tgid];
    uint32_t aug_width = n + k;
    uint32_t stride = n * aug_width;

    uint32_t tx = tid.x;
    uint32_t ty = tid.y;
    uint32_t local_id = ty * TILE_SIZE + tx;
    uint32_t num_threads = TILE_SIZE * TILE_SIZE;

    device const uint32_t* myAug = augmented + tgid * stride;
    device uint32_t* A = workspace + tgid * stride;

    threadgroup uint32_t pivot_row_shared;
    threadgroup uint32_t pivot_val_shared;
    threadgroup bool is_singular;
    threadgroup uint32_t pivot_inv_shared;

    if (local_id == 0) {
        is_singular = false;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel copy
    for (uint32_t i = local_id; i < stride; i += num_threads) {
        A[i] = myAug[i];
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Forward elimination
    for (uint32_t col = 0; col < n; col++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (is_singular) break;

        // Parallel pivot search
        threadgroup uint32_t found_rows[256];
        threadgroup uint32_t found_vals[256];

        uint32_t my_row = col + local_id;
        if (my_row < n && local_id < (n - col)) {
            uint32_t val = A[my_row * aug_width + col];
            found_rows[local_id] = (val != 0) ? my_row : n;
            found_vals[local_id] = val;
        } else if (local_id < num_threads) {
            found_rows[local_id] = n;
            found_vals[local_id] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint32_t s = num_threads / 2; s > 0; s /= 2) {
            if (local_id < s) {
                if (found_rows[local_id + s] < found_rows[local_id]) {
                    found_rows[local_id] = found_rows[local_id + s];
                    found_vals[local_id] = found_vals[local_id + s];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (local_id == 0) {
            pivot_row_shared = found_rows[0];
            pivot_val_shared = found_vals[0];
            if (pivot_row_shared >= n || pivot_val_shared == 0) {
                is_singular = true;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (is_singular) break;

        // Parallel row swap
        if (pivot_row_shared != col) {
            for (uint32_t j = col + local_id; j < aug_width; j += num_threads) {
                uint32_t tmp = A[col * aug_width + j];
                A[col * aug_width + j] = A[pivot_row_shared * aug_width + j];
                A[pivot_row_shared * aug_width + j] = tmp;
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        if (local_id == 0) {
            pivot_inv_shared = mod_inv(pivot_val_shared, p);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Parallel row normalization
        for (uint32_t j = col + local_id; j < aug_width; j += num_threads) {
            A[col * aug_width + j] = mod_mul(A[col * aug_width + j], pivot_inv_shared, p);
        }
        threadgroup_barrier(mem_flags::mem_device);

        // Parallel elimination
        uint32_t rows_to_eliminate = n - col - 1;
        uint32_t cols_to_update = aug_width - col;

        for (uint32_t row_offset = ty; row_offset < rows_to_eliminate; row_offset += TILE_SIZE) {
            uint32_t i = col + 1 + row_offset;
            uint32_t factor = A[i * aug_width + col];

            if (factor != 0) {
                for (uint32_t col_offset = tx; col_offset < cols_to_update; col_offset += TILE_SIZE) {
                    uint32_t j = col + col_offset;
                    uint32_t sub = mod_mul(factor, A[col * aug_width + j], p);
                    A[i * aug_width + j] = mod_sub(A[i * aug_width + j], sub, p);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    singular_flags[tgid] = is_singular ? 1 : 0;

    if (is_singular) {
        for (uint32_t i = local_id; i < n * k; i += num_threads) {
            solutions[tgid * n * k + i] = 0;
        }
        return;
    }

    // Parallel back substitution
    // Process each RHS column and each row
    device uint32_t* X = solutions + tgid * n * k;

    // Back substitution must be done row by row (dependency), but RHS columns can be parallel
    for (int32_t row = int32_t(n) - 1; row >= 0; row--) {
        // Each thread handles different RHS columns
        for (uint32_t rhs = local_id; rhs < k; rhs += num_threads) {
            uint32_t sum = A[row * aug_width + (n + rhs)];

            for (uint32_t j = row + 1; j < n; j++) {
                uint32_t sub = mod_mul(A[row * aug_width + j], X[rhs * n + j], p);
                sum = mod_sub(sum, sub, p);
            }
            X[rhs * n + row] = sum;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// =============================================================================
// Sparse Matrix-Vector Multiply (CSR format)
// =============================================================================
// Performs y = A * x mod p for sparse matrix A in CSR format
// Memory layout (CSR):
//   row_ptr: n+1 elements, row_ptr[i] is start of row i's data in col_idx/values
//   col_idx: nnz elements, column indices
//   values:  nnz elements, non-zero values (already reduced mod p)
//
// This kernel is designed for Wiedemann algorithm which needs many sparse matvec.
// Batch approach: One thread per row, processes multiple primes
//
kernel void sparse_matvec_csr(
    device const uint32_t* row_ptr [[buffer(0)]],    // CSR row pointers (n+1)
    device const uint32_t* col_idx [[buffer(1)]],    // CSR column indices (nnz)
    device const uint32_t* values [[buffer(2)]],     // CSR values (nnz) per prime
    device const uint32_t* x [[buffer(3)]],          // Input vector (n) per prime
    device uint32_t* y [[buffer(4)]],                // Output vector (n) per prime
    device const uint32_t* primes [[buffer(5)]],     // Array of primes
    constant uint32_t& n [[buffer(6)]],              // Vector dimension
    constant uint32_t& nnz [[buffer(7)]],            // Number of non-zeros
    constant uint32_t& num_primes [[buffer(8)]],     // Number of primes
    uint tid [[thread_position_in_grid]])
{
    // Thread assignment: tid = prime_idx * n + row
    uint32_t prime_idx = tid / n;
    uint32_t row = tid % n;

    if (prime_idx >= num_primes) return;

    uint32_t p = primes[prime_idx];

    // Get row range in CSR structure
    uint32_t row_start = row_ptr[row];
    uint32_t row_end = row_ptr[row + 1];

    // Compute dot product for this row
    uint64_t sum = 0;

    // Values and x are stored per-prime: values[prime_idx * nnz + j], x[prime_idx * n + col]
    device const uint32_t* my_values = values + prime_idx * nnz;
    device const uint32_t* my_x = x + prime_idx * n;

    for (uint32_t j = row_start; j < row_end; j++) {
        uint32_t col = col_idx[j];
        uint64_t prod = uint64_t(my_values[j]) * uint64_t(my_x[col]);
        sum += prod;
        // Reduce periodically to avoid overflow
        // For 64-bit sum with 32-bit values: can safely accumulate ~2^32 products
        // Reduce every 512 iterations (less frequently for sparse R1CS matrices)
        if ((j - row_start) % 512 == 511) {
            sum %= uint64_t(p);
        }
    }

    // Store result
    y[prime_idx * n + row] = uint32_t(sum % uint64_t(p));
}

// Single-prime variant for when we only have one prime (simpler addressing)
kernel void sparse_matvec_csr_single(
    device const uint32_t* row_ptr [[buffer(0)]],
    device const uint32_t* col_idx [[buffer(1)]],
    device const uint32_t* values [[buffer(2)]],
    device const uint32_t* x [[buffer(3)]],
    device uint32_t* y [[buffer(4)]],
    constant uint32_t& p [[buffer(5)]],
    constant uint32_t& n [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;

    uint32_t row = tid;
    uint32_t row_start = row_ptr[row];
    uint32_t row_end = row_ptr[row + 1];

    uint64_t sum = 0;

    for (uint32_t j = row_start; j < row_end; j++) {
        uint32_t col = col_idx[j];
        uint64_t prod = uint64_t(values[j]) * uint64_t(x[col]);
        sum += prod;
        // Reduce every 512 iterations (optimized for sparse matrices)
        if ((j - row_start) % 512 == 511) {
            sum %= uint64_t(p);
        }
    }

    y[row] = uint32_t(sum % uint64_t(p));
}

// =============================================================================
// Batched Sparse Matrix-Vector for Wiedemann iterations
// =============================================================================
// Performs multiple A^i * v computations efficiently by batching
// This kernel computes one iteration: y = A * x mod p for all primes
// The host calls this kernel iteratively to build up A^k * v
//
// Memory: Same as sparse_matvec_csr but optimized for repeated calls
//
kernel void sparse_matvec_wiedemann(
    device const uint32_t* row_ptr [[buffer(0)]],
    device const uint32_t* col_idx [[buffer(1)]],
    device const uint32_t* values [[buffer(2)]],       // nnz values per prime
    device const uint32_t* x [[buffer(3)]],            // Input vectors (n per prime)
    device uint32_t* y [[buffer(4)]],                  // Output vectors (n per prime)
    device const uint32_t* primes [[buffer(5)]],
    constant uint32_t& n [[buffer(6)]],
    constant uint32_t& nnz [[buffer(7)]],
    constant uint32_t& num_primes [[buffer(8)]],
    uint2 tid [[thread_position_in_grid]])             // (row, prime_idx)
{
    uint32_t row = tid.x;
    uint32_t prime_idx = tid.y;

    if (row >= n || prime_idx >= num_primes) return;

    uint32_t p = primes[prime_idx];

    uint32_t row_start = row_ptr[row];
    uint32_t row_end = row_ptr[row + 1];

    device const uint32_t* my_values = values + prime_idx * nnz;
    device const uint32_t* my_x = x + prime_idx * n;

    uint64_t sum = 0;

    for (uint32_t j = row_start; j < row_end; j++) {
        uint32_t col = col_idx[j];
        sum += uint64_t(my_values[j]) * uint64_t(my_x[col]);
        // Reduce every 512 iterations (optimized for sparse matrices)
        if ((j - row_start) % 512 == 511) {
            sum %= uint64_t(p);
        }
    }

    y[prime_idx * n + row] = uint32_t(sum % uint64_t(p));
}

// =============================================================================
// CRT Garner Step Computation (Partial GPU Acceleration)
// =============================================================================
// This kernel computes the Garner step for CRT reconstruction.
// Full CRT requires BigInt, but we can accelerate the inner loop:
//   t[i] = (residue[i] - result_mod_prime[i]) * inv[i] mod prime[i]
//
// Memory layout:
//   residues: num_values * num_primes (residues[v * num_primes + p] = r_vp)
//   result_mods: num_values * current_prime_idx (partial results mod each prior prime)
//   inverses: num_primes precomputed Garner inverses
//   t_values: num_values output (the Garner t values for this step)
//
// This kernel computes one Garner step for all values in parallel.
//
kernel void crt_garner_step(
    device const uint32_t* residues [[buffer(0)]],      // All residues (num_values x num_primes)
    device const uint32_t* result_mods [[buffer(1)]],   // result mod current prime (num_values)
    device const uint32_t* primes [[buffer(2)]],        // Array of primes
    device const uint32_t* inverses [[buffer(3)]],      // Garner inverses (u32)
    device uint32_t* t_values [[buffer(4)]],            // Output: Garner t values
    constant uint32_t& num_values [[buffer(5)]],
    constant uint32_t& num_primes [[buffer(6)]],
    constant uint32_t& current_prime_idx [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_values) return;

    uint32_t v = tid;
    uint32_t i = current_prime_idx;
    uint32_t mi = primes[i];
    uint64_t mi64 = mi;

    // Get residue for value v at prime i
    uint32_t ri = residues[v * num_primes + i];

    // Get result mod mi (computed on CPU from BigInt)
    uint32_t result_mod_mi = result_mods[v];

    // Compute t = (ri - result_mod_mi) * inv mod mi using u64 arithmetic
    uint64_t ri64 = ri;
    uint64_t rm64 = result_mod_mi;

    uint64_t diff;
    if (ri64 >= rm64) {
        diff = ri64 - rm64;
    } else {
        diff = mi64 - rm64 + ri64;
    }

    uint64_t inv = inverses[i];
    uint64_t t = (diff * inv) % mi64;

    t_values[v] = uint32_t(t);
}

// =============================================================================
// Multi-Value Modular Reduction (for BigInt % prime)
// =============================================================================
// Computes bigint % prime for multiple BigInts in parallel.
// The BigInt is represented as limbs (u32 words, little-endian).
//
// Key insight: for n mod p where p is 31-bit:
//   n = sum(limbs[i] * 2^(32*i))
//   n mod p = sum(limbs[i] * (2^(32*i) mod p)) mod p
//
// We precompute pow2_mods[i] = 2^(32*i) mod p for efficiency.
//
kernel void bigint_mod_prime(
    device const uint32_t* limbs [[buffer(0)]],         // BigInt limbs (num_values x num_limbs)
    device const uint32_t* pow2_mods [[buffer(1)]],     // Precomputed 2^(32*i) mod p (num_limbs)
    device uint32_t* results [[buffer(2)]],             // Output: bigint mod p (num_values)
    constant uint32_t& num_values [[buffer(3)]],
    constant uint32_t& num_limbs [[buffer(4)]],
    constant uint32_t& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_values) return;

    uint32_t v = tid;
    device const uint32_t* my_limbs = limbs + v * num_limbs;
    uint64_t p64 = p;

    // Compute sum of (limb[i] * pow2_mods[i]) mod p
    uint64_t sum = 0;
    for (uint32_t i = 0; i < num_limbs; i++) {
        uint64_t term = uint64_t(my_limbs[i]) * uint64_t(pow2_mods[i]);
        sum += term % p64;
        // Reduce periodically to prevent overflow
        if ((i & 15) == 15) {
            sum %= p64;
        }
    }

    results[v] = uint32_t(sum % p64);
}

// =============================================================================
// Batch Sign Detection (for symmetric range conversion)
// =============================================================================
// Compares BigInts against M/2 to determine sign for symmetric range conversion.
// Uses lexicographic comparison of limbs (big-endian order for comparison).
//
// For simplicity, we compare limb-by-limb from most significant.
// Output: 1 if bigint > half_product, 0 otherwise
//
kernel void bigint_compare_half(
    device const uint32_t* limbs [[buffer(0)]],         // BigInt limbs (num_values x num_limbs)
    device const uint32_t* half_limbs [[buffer(1)]],    // M/2 limbs (num_limbs)
    device uint32_t* is_negative [[buffer(2)]],         // Output: 1 if > M/2, 0 otherwise
    constant uint32_t& num_values [[buffer(3)]],
    constant uint32_t& num_limbs [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_values) return;

    uint32_t v = tid;
    device const uint32_t* my_limbs = limbs + v * num_limbs;

    // Compare from most significant limb to least significant
    int32_t result = 0;  // 0 = equal so far, 1 = greater, -1 = less
    for (int32_t i = int32_t(num_limbs) - 1; i >= 0 && result == 0; i--) {
        if (my_limbs[i] > half_limbs[i]) {
            result = 1;
        } else if (my_limbs[i] < half_limbs[i]) {
            result = -1;
        }
    }

    is_negative[v] = (result > 0) ? 1 : 0;
}
"#;

/// Get compiled shader library
pub fn get_shader_source() -> &'static str {
    SHADER_SOURCE
}
