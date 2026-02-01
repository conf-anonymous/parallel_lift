// Modular arithmetic utilities for CUDA kernels
// 31-bit primes with 64-bit intermediate arithmetic

#ifndef MODULAR_OPS_CUH
#define MODULAR_OPS_CUH

#include <stdint.h>

// Modular addition: (a + b) mod p
__device__ __forceinline__
uint32_t mod_add(uint32_t a, uint32_t b, uint32_t p) {
    uint64_t sum = (uint64_t)a + (uint64_t)b;
    return sum >= p ? (uint32_t)(sum - p) : (uint32_t)sum;
}

// Modular subtraction: (a - b) mod p
__device__ __forceinline__
uint32_t mod_sub(uint32_t a, uint32_t b, uint32_t p) {
    return a >= b ? (a - b) : (p - b + a);
}

// Modular multiplication: (a * b) mod p
__device__ __forceinline__
uint32_t mod_mul(uint32_t a, uint32_t b, uint32_t p) {
    return (uint32_t)(((uint64_t)a * (uint64_t)b) % (uint64_t)p);
}

// Modular inverse using Fermat's Little Theorem: a^(-1) = a^(p-2) mod p
// Better GPU pipelining than Extended Euclidean Algorithm
__device__ __forceinline__
uint32_t mod_inv(uint32_t a, uint32_t p) {
    if (a == 0) return 0;

    uint32_t result = 1;
    uint32_t base = a;
    uint32_t exp = p - 2;

    // Binary exponentiation (~30 iterations for 31-bit prime)
    while (exp > 0) {
        if (exp & 1) {
            result = mod_mul(result, base, p);
        }
        base = mod_mul(base, base, p);
        exp >>= 1;
    }
    return result;
}

#endif // MODULAR_OPS_CUH
