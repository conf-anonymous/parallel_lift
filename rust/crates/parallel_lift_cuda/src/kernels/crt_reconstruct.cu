// GPU-accelerated CRT reconstruction using Garner's algorithm
// Each thread reconstructs one value from its residues across all primes

#include <stdint.h>

// Maximum number of limbs for BigInt (supports up to ~32K bits)
#define MAX_LIMBS 1024

// Modular multiplication with 64-bit intermediate
__device__ __forceinline__ uint64_t mod_mul_u64(uint64_t a, uint64_t b, uint64_t p) {
    return (a * b) % p;
}

// Full CRT reconstruction kernel
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
    uint32_t max_acc_limbs                          // Maximum limbs in accumulator
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

    uint32_t current_size = 1;  // Track actual size of accumulator

    // Garner iterations: build up the result incrementally
    for (uint32_t i = 1; i < num_primes; i++) {
        uint64_t p = primes[i];
        uint32_t inv = inverses[i];
        uint32_t r = residues[v * num_primes + i];

        // Compute acc mod p using precomputed powers of 2^32
        // acc mod p = sum(acc[j] * 2^(32j) mod p) mod p
        const uint64_t* pow2 = pow2_mod + i * max_acc_limbs;
        uint64_t acc_mod_p = 0;

        for (uint32_t j = 0; j < current_size; j++) {
            acc_mod_p += mod_mul_u64(acc[j], pow2[j], p);
            // Periodic reduction to avoid overflow
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

        // Skip if coefficient is zero
        if (t == 0) continue;

        // Load partial product info
        uint32_t pp_offset = pp_offsets[i];
        uint32_t pp_size = pp_sizes[i];

        // Multiply partial_product[i] by t and add to accumulator
        // acc += partial_products[i] * t
        uint64_t carry = 0;
        uint32_t new_size = current_size;

        for (uint32_t j = 0; j < max_acc_limbs; j++) {
            uint64_t pp_limb = (j < pp_size) ? pp_limbs[pp_offset + j] : 0;
            uint64_t product = pp_limb * t + acc[j] + carry;
            acc[j] = (uint32_t)product;
            carry = product >> 32;

            // Update size if we wrote to a new position
            if (acc[j] != 0 && j >= new_size) {
                new_size = j + 1;
            }
        }

        // Account for final carry
        if (carry > 0 && new_size < max_acc_limbs) {
            // carry should have been absorbed, but track size
            new_size = (pp_size > current_size ? pp_size : current_size) + 1;
            if (new_size > max_acc_limbs) new_size = max_acc_limbs;
        }

        current_size = (pp_size + 1 > current_size) ? pp_size + 1 : current_size;
        if (current_size > max_acc_limbs) current_size = max_acc_limbs;
    }

    // Store actual size
    output_sizes[v] = current_size;
}

// Optimized kernel for smaller prime counts using shared memory
// Better for num_primes <= 256
extern "C" __global__ void crt_reconstruct_shared(
    const uint32_t* __restrict__ residues,
    const uint32_t* __restrict__ primes,
    const uint32_t* __restrict__ inverses,
    const uint32_t* __restrict__ pp_limbs,
    const uint32_t* __restrict__ pp_offsets,
    const uint32_t* __restrict__ pp_sizes,
    const uint64_t* __restrict__ pow2_mod,
    uint32_t* __restrict__ output_limbs,
    uint32_t* __restrict__ output_sizes,
    uint32_t num_values,
    uint32_t num_primes,
    uint32_t max_acc_limbs
) {
    // Shared memory for primes and inverses
    __shared__ uint32_t s_primes[256];
    __shared__ uint32_t s_inverses[256];

    uint32_t tid = threadIdx.x;
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;

    // Cooperatively load primes and inverses into shared memory
    if (tid < num_primes) {
        s_primes[tid] = primes[tid];
        s_inverses[tid] = inverses[tid];
    }
    __syncthreads();

    if (v >= num_values) return;

    uint32_t* acc = output_limbs + v * max_acc_limbs;

    // Initialize
    acc[0] = residues[v * num_primes + 0];
    for (uint32_t j = 1; j < max_acc_limbs; j++) {
        acc[j] = 0;
    }

    uint32_t current_size = 1;

    // Garner iterations
    for (uint32_t i = 1; i < num_primes; i++) {
        uint64_t p = s_primes[i];
        uint32_t inv = s_inverses[i];
        uint32_t r = residues[v * num_primes + i];

        const uint64_t* pow2 = pow2_mod + i * max_acc_limbs;
        uint64_t acc_mod_p = 0;

        for (uint32_t j = 0; j < current_size; j++) {
            acc_mod_p += mod_mul_u64(acc[j], pow2[j], p);
            if ((j & 15) == 15) acc_mod_p %= p;
        }
        acc_mod_p %= p;

        uint64_t diff = (r >= acc_mod_p) ? (r - acc_mod_p) : (p - acc_mod_p + r);
        uint64_t t = (diff * inv) % p;

        if (t == 0) continue;

        uint32_t pp_offset = pp_offsets[i];
        uint32_t pp_size = pp_sizes[i];

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
// Compares each result against M/2 and subtracts M if needed
extern "C" __global__ void crt_to_signed(
    uint32_t* __restrict__ limbs,                   // [num_values * max_limbs] in/out
    const uint32_t* __restrict__ half_product,      // [half_product_size] M/2 limbs
    const uint32_t* __restrict__ full_product,      // [product_size] M limbs
    uint32_t* __restrict__ signs,                   // [num_values] output: 1 if negative
    uint32_t num_values,
    uint32_t max_limbs,
    uint32_t half_size,
    uint32_t product_size
) {
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_values) return;

    uint32_t* acc = limbs + v * max_limbs;

    // Compare with M/2 (from MSB to LSB)
    int cmp = 0;  // 0: equal, 1: greater, -1: less

    uint32_t check_size = (max_limbs < half_size) ? max_limbs : half_size;

    // First check if acc has limbs beyond half_product
    for (uint32_t j = max_limbs - 1; j >= half_size && cmp == 0; j--) {
        if (acc[j] > 0) {
            cmp = 1;
            break;
        }
        if (j == 0) break;
    }

    // Then compare overlapping limbs
    if (cmp == 0) {
        for (int j = (int)check_size - 1; j >= 0 && cmp == 0; j--) {
            if (acc[j] > half_product[j]) {
                cmp = 1;
            } else if (acc[j] < half_product[j]) {
                cmp = -1;
            }
        }
    }

    // If acc > M/2, subtract M to get negative representation
    // Result will be stored as absolute value with sign flag
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

// Batch kernel that processes multiple values per block with register tiling
// Each warp handles one value, threads within warp parallelize limb operations
extern "C" __global__ void crt_reconstruct_warp(
    const uint32_t* __restrict__ residues,
    const uint32_t* __restrict__ primes,
    const uint32_t* __restrict__ inverses,
    const uint32_t* __restrict__ pp_limbs,
    const uint32_t* __restrict__ pp_offsets,
    const uint32_t* __restrict__ pp_sizes,
    const uint64_t* __restrict__ pow2_mod,
    uint32_t* __restrict__ output_limbs,
    uint32_t* __restrict__ output_sizes,
    uint32_t num_values,
    uint32_t num_primes,
    uint32_t max_acc_limbs
) {
    // One warp (32 threads) per value
    uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    uint32_t lane_id = threadIdx.x % 32;

    if (warp_id >= num_values) return;

    uint32_t v = warp_id;
    uint32_t* acc = output_limbs + v * max_acc_limbs;

    // Initialize (parallelized across warp)
    for (uint32_t j = lane_id; j < max_acc_limbs; j += 32) {
        acc[j] = (j == 0) ? residues[v * num_primes + 0] : 0;
    }
    __syncwarp();

    uint32_t current_size = 1;

    // Garner iterations
    for (uint32_t i = 1; i < num_primes; i++) {
        uint64_t p = primes[i];
        uint32_t inv = inverses[i];
        uint32_t r = residues[v * num_primes + i];

        // Parallel reduction for acc mod p
        const uint64_t* pow2 = pow2_mod + i * max_acc_limbs;
        uint64_t local_sum = 0;

        for (uint32_t j = lane_id; j < current_size; j += 32) {
            local_sum += mod_mul_u64(acc[j], pow2[j], p);
        }

        // Warp reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }

        uint64_t acc_mod_p = local_sum % p;
        acc_mod_p = __shfl_sync(0xFFFFFFFF, acc_mod_p, 0);  // Broadcast from lane 0

        uint64_t diff = (r >= acc_mod_p) ? (r - acc_mod_p) : (p - acc_mod_p + r);
        uint64_t t = (diff * inv) % p;

        if (t == 0) continue;

        uint32_t pp_offset = pp_offsets[i];
        uint32_t pp_size = pp_sizes[i];

        // Parallel multiply-add with carry (sequential carry propagation)
        // This is tricky - use lane 0 for sequential part
        if (lane_id == 0) {
            uint64_t carry = 0;
            for (uint32_t j = 0; j < max_acc_limbs; j++) {
                uint64_t pp_limb = (j < pp_size) ? pp_limbs[pp_offset + j] : 0;
                uint64_t product = pp_limb * t + acc[j] + carry;
                acc[j] = (uint32_t)product;
                carry = product >> 32;
            }
        }
        __syncwarp();

        current_size = (pp_size + 1 > current_size) ? pp_size + 1 : current_size;
        if (current_size > max_acc_limbs) current_size = max_acc_limbs;
    }

    if (lane_id == 0) {
        output_sizes[v] = current_size;
    }
}
