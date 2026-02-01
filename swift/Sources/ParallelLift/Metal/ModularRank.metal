#include <metal_stdlib>
using namespace metal;

// Modular arithmetic utilities (same as determinant)
inline uint32_t rank_mod_add(uint32_t a, uint32_t b, uint32_t p) {
    uint64_t sum = uint64_t(a) + uint64_t(b);
    return sum >= p ? uint32_t(sum - p) : uint32_t(sum);
}

inline uint32_t rank_mod_sub(uint32_t a, uint32_t b, uint32_t p) {
    return a >= b ? (a - b) : (p - b + a);
}

inline uint32_t rank_mod_mul(uint32_t a, uint32_t b, uint32_t p) {
    uint64_t prod = uint64_t(a) * uint64_t(b);
    return uint32_t(prod % uint64_t(p));
}

inline uint32_t rank_mod_inv(uint32_t a, uint32_t p) {
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

// Batched kernel: Each thread computes rank mod one prime
// Matrix data: [prime0_matrix, prime1_matrix, ...] pre-reduced
kernel void modular_rank(
    device const uint32_t* matrices [[buffer(0)]],   // All matrices concatenated
    device const uint32_t* primes [[buffer(1)]],     // Array of primes
    device uint32_t* results [[buffer(2)]],          // Output: rank mod p for each prime
    constant uint32_t& m [[buffer(3)]],              // Number of rows
    constant uint32_t& n [[buffer(4)]],              // Number of columns
    constant uint32_t& num_primes [[buffer(5)]],     // Number of primes
    device uint32_t* workspace [[buffer(6)]],        // Workspace: num_primes * m * n
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_primes) return;

    uint32_t p = primes[tid];
    uint32_t mn = m * n;

    // Read from the pre-reduced matrix slice for this thread
    device const uint32_t* myMatrix = matrices + tid * mn;
    device uint32_t* A = workspace + tid * mn;

    // Copy to workspace
    for (uint32_t i = 0; i < mn; i++) {
        A[i] = myMatrix[i];
    }

    // Gaussian elimination to find rank
    uint32_t rank = 0;
    uint32_t pivotRow = 0;
    uint32_t pivotCol = 0;

    while (pivotRow < m && pivotCol < n) {
        // Find pivot in current column
        uint32_t maxRow = pivotRow;
        for (uint32_t i = pivotRow; i < m; i++) {
            if (A[i * n + pivotCol] != 0) {
                maxRow = i;
                break;
            }
        }

        if (A[maxRow * n + pivotCol] == 0) {
            // No pivot in this column
            pivotCol++;
            continue;
        }

        // Swap rows if needed
        if (maxRow != pivotRow) {
            for (uint32_t j = pivotCol; j < n; j++) {
                uint32_t tmp = A[pivotRow * n + j];
                A[pivotRow * n + j] = A[maxRow * n + j];
                A[maxRow * n + j] = tmp;
            }
        }

        uint32_t pivot = A[pivotRow * n + pivotCol];
        uint32_t pivot_inv = rank_mod_inv(pivot, p);

        // Eliminate below pivot
        for (uint32_t i = pivotRow + 1; i < m; i++) {
            uint32_t factor = rank_mod_mul(A[i * n + pivotCol], pivot_inv, p);
            if (factor != 0) {
                for (uint32_t j = pivotCol; j < n; j++) {
                    uint32_t sub = rank_mod_mul(factor, A[pivotRow * n + j], p);
                    A[i * n + j] = rank_mod_sub(A[i * n + j], sub, p);
                }
            }
        }

        rank++;
        pivotRow++;
        pivotCol++;
    }

    results[tid] = rank;
}

// Small matrix kernel using thread-local storage (for m,n <= 16)
kernel void modular_rank_small(
    device const uint32_t* matrices [[buffer(0)]],
    device const uint32_t* primes [[buffer(1)]],
    device uint32_t* results [[buffer(2)]],
    constant uint32_t& m [[buffer(3)]],
    constant uint32_t& n [[buffer(4)]],
    constant uint32_t& num_primes [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_primes) return;

    uint32_t p = primes[tid];

    // Thread-local storage for small matrices (max 16x16 = 256 elements)
    uint32_t A[256];

    uint32_t mn = m * n;
    device const uint32_t* myMatrix = matrices + tid * mn;

    for (uint32_t i = 0; i < mn; i++) {
        A[i] = myMatrix[i];
    }

    // Gaussian elimination
    uint32_t rank = 0;
    uint32_t pivotRow = 0;
    uint32_t pivotCol = 0;

    while (pivotRow < m && pivotCol < n) {
        uint32_t maxRow = pivotRow;
        for (uint32_t i = pivotRow; i < m; i++) {
            if (A[i * n + pivotCol] != 0) {
                maxRow = i;
                break;
            }
        }

        if (A[maxRow * n + pivotCol] == 0) {
            pivotCol++;
            continue;
        }

        if (maxRow != pivotRow) {
            for (uint32_t j = pivotCol; j < n; j++) {
                uint32_t tmp = A[pivotRow * n + j];
                A[pivotRow * n + j] = A[maxRow * n + j];
                A[maxRow * n + j] = tmp;
            }
        }

        uint32_t pivot = A[pivotRow * n + pivotCol];
        uint32_t pivot_inv = rank_mod_inv(pivot, p);

        for (uint32_t i = pivotRow + 1; i < m; i++) {
            uint32_t factor = rank_mod_mul(A[i * n + pivotCol], pivot_inv, p);
            if (factor != 0) {
                for (uint32_t j = pivotCol; j < n; j++) {
                    uint32_t sub = rank_mod_mul(factor, A[pivotRow * n + j], p);
                    A[i * n + j] = rank_mod_sub(A[i * n + j], sub, p);
                }
            }
        }

        rank++;
        pivotRow++;
        pivotCol++;
    }

    results[tid] = rank;
}
