#include <metal_stdlib>
using namespace metal;

// Modular arithmetic utilities for 32-bit primes
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

inline uint32_t mod_inv(uint32_t a, uint32_t p) {
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

// Batched kernel: Each thread processes one prime
// Matrix data is laid out as: [prime0_matrix, prime1_matrix, ...]
// where each matrix is n*n UInt32 elements already reduced mod its prime
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

// Small matrix kernel using thread-local storage (faster for n <= 16)
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
