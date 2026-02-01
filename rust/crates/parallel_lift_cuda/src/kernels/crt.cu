// CRT reconstruction helper kernels for CUDA

#include <stdint.h>
#include "modular_ops.cuh"

// Garner step: t = (residue - result_mod) * inv mod prime
extern "C" __global__ void crt_garner_step(
    const uint32_t* __restrict__ residues,     // [num_values][num_primes]
    const uint32_t* __restrict__ result_mods,  // [num_values]
    const uint32_t* __restrict__ primes,
    const uint32_t* __restrict__ inverses,
    uint32_t* __restrict__ t_values,           // [num_values]
    uint32_t num_values,
    uint32_t num_primes,
    uint32_t current_prime_idx
) {
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_values) return;

    uint32_t i = current_prime_idx;
    uint32_t mi = primes[i];
    uint64_t mi64 = mi;

    uint64_t ri = residues[v * num_primes + i];
    uint64_t rm = result_mods[v];

    uint64_t diff = (ri >= rm) ? (ri - rm) : (mi64 - rm + ri);
    uint64_t inv = inverses[i];
    uint64_t t = (diff * inv) % mi64;

    t_values[v] = (uint32_t)t;
}

// BigInt mod prime using precomputed powers
extern "C" __global__ void bigint_mod_prime(
    const uint32_t* __restrict__ limbs,      // [num_values][num_limbs]
    const uint32_t* __restrict__ pow2_mods,  // [num_limbs]: 2^(32i) mod p
    uint32_t* __restrict__ results,
    uint32_t num_values,
    uint32_t num_limbs,
    uint32_t p
) {
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_values) return;

    const uint32_t* my_limbs = limbs + v * num_limbs;
    uint64_t p64 = p;
    uint64_t sum = 0;

    for (uint32_t i = 0; i < num_limbs; i++) {
        sum += ((uint64_t)my_limbs[i] * (uint64_t)pow2_mods[i]) % p64;
        if ((i & 15) == 15) {
            sum %= p64;
        }
    }

    results[v] = (uint32_t)(sum % p64);
}

// Sign detection: compare BigInt against M/2
extern "C" __global__ void bigint_compare_half(
    const uint32_t* __restrict__ limbs,
    const uint32_t* __restrict__ half_limbs,
    uint32_t* __restrict__ is_negative,
    uint32_t num_values,
    uint32_t num_limbs
) {
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_values) return;

    const uint32_t* my_limbs = limbs + v * num_limbs;

    int result = 0;  // 0=equal, 1=greater, -1=less

    // Compare from MSB to LSB
    for (int i = num_limbs - 1; i >= 0 && result == 0; i--) {
        if (my_limbs[i] > half_limbs[i]) {
            result = 1;
        } else if (my_limbs[i] < half_limbs[i]) {
            result = -1;
        }
    }

    is_negative[v] = (result > 0) ? 1 : 0;
}
