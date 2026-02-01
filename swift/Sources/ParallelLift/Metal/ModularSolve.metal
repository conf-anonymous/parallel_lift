#include <metal_stdlib>
using namespace metal;

// Modular arithmetic utilities
inline uint32_t solve_mod_sub(uint32_t a, uint32_t b, uint32_t p) {
    return a >= b ? (a - b) : (p - b + a);
}

inline uint32_t solve_mod_mul(uint32_t a, uint32_t b, uint32_t p) {
    uint64_t prod = uint64_t(a) * uint64_t(b);
    return uint32_t(prod % uint64_t(p));
}

inline uint32_t solve_mod_inv(uint32_t a, uint32_t p) {
    if (a == 0) return 0;
    int64_t t = 0, newt = 1;
    int64_t r = int64_t(p), newr = int64_t(a);
    while (newr != 0) {
        int64_t quotient = r / newr;
        int64_t temp_t = t - quotient * newt;
        t = newt; newt = temp_t;
        int64_t temp_r = r - quotient * newr;
        r = newr; newr = temp_r;
    }
    if (t < 0) t += int64_t(p);
    return uint32_t(t);
}

// Batched kernel: Each thread solves Ax = b mod one prime
// Input layout:
//   - augmented: num_primes * n * (n+1) - augmented matrices [A|b] for each prime
//   - primes: array of primes
// Output:
//   - solutions: num_primes * n - solution vectors x mod p
//   - singular_flags: 1 if matrix is singular mod p
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
        uint32_t pivot_inv = solve_mod_inv(pivot, p);

        // Normalize pivot row
        for (uint32_t j = k; j <= n; j++) {
            A[k * (n + 1) + j] = solve_mod_mul(A[k * (n + 1) + j], pivot_inv, p);
        }

        // Eliminate below
        for (uint32_t i = k + 1; i < n; i++) {
            uint32_t factor = A[i * (n + 1) + k];
            if (factor != 0) {
                for (uint32_t j = k; j <= n; j++) {
                    uint32_t sub = solve_mod_mul(factor, A[k * (n + 1) + j], p);
                    A[i * (n + 1) + j] = solve_mod_sub(A[i * (n + 1) + j], sub, p);
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
            uint32_t sub = solve_mod_mul(A[i * (n + 1) + j], x[j], p);
            sum = solve_mod_sub(sum, sub, p);
        }
        x[i] = sum;
    }
}

// ============================================================
// Multi-RHS Solve: AX = B where B has k columns
// ============================================================
// This kernel factors A once and solves for all k right-hand sides
// Memory layout:
//   - augmented: num_primes * n * (n + k) - augmented matrices [A|B]
//   - solutions: num_primes * n * k - solution matrices X mod p

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

    // Array to store pivot row order for applying to RHS
    // (We'll apply row swaps to entire augmented rows including B columns)

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
        uint32_t pivot_inv = solve_mod_inv(pivot, p);

        // Normalize pivot row
        for (uint32_t j = col; j < aug_width; j++) {
            A[col * aug_width + j] = solve_mod_mul(A[col * aug_width + j], pivot_inv, p);
        }

        // Eliminate below
        for (uint32_t row = col + 1; row < n; row++) {
            uint32_t factor = A[row * aug_width + col];
            if (factor != 0) {
                for (uint32_t j = col; j < aug_width; j++) {
                    uint32_t sub = solve_mod_mul(factor, A[col * aug_width + j], p);
                    A[row * aug_width + j] = solve_mod_sub(A[row * aug_width + j], sub, p);
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

    device uint32_t* X = solutions + tid * n * k;  // Output X (n×k matrix, stored column-major)

    for (uint32_t rhs = 0; rhs < k; rhs++) {
        // Solve for column 'rhs' of X
        // The transformed RHS is in column (n + rhs) of A

        for (int32_t row = int32_t(n) - 1; row >= 0; row--) {
            uint32_t sum = A[row * aug_width + (n + rhs)];  // RHS value

            for (uint32_t j = row + 1; j < n; j++) {
                uint32_t sub = solve_mod_mul(A[row * aug_width + j], X[rhs * n + j], p);
                sum = solve_mod_sub(sum, sub, p);
            }
            X[rhs * n + row] = sum;  // Store in column-major: X[rhs][row]
        }
    }
}
