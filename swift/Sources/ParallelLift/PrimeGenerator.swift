import BigInt
import Foundation

/// Prime number generation utilities for CRT-based computation
struct PrimeGenerator {

    /// Generate a list of 31-bit primes suitable for modular arithmetic
    /// We use primes just under 2^31 to ensure products fit in 64-bit during intermediate calculations
    /// - Parameter count: Number of primes to generate
    /// - Parameter startFrom: Starting point for prime search (default: near 2^31)
    /// - Returns: Array of 31-bit primes in descending order
    static func generate31BitPrimes(count: Int, startFrom: UInt32 = 2_147_483_647) -> [UInt32] {
        var primes: [UInt32] = []
        var candidate = startFrom

        // Make sure we start with an odd number
        if candidate % 2 == 0 {
            candidate -= 1
        }

        while primes.count < count && candidate > 2 {
            if isPrime(candidate) {
                primes.append(candidate)
            }
            candidate -= 2
        }

        return primes
    }

    /// Miller-Rabin primality test for 32-bit numbers
    /// Deterministic for all 32-bit numbers using specific witness set
    static func isPrime(_ n: UInt32) -> Bool {
        if n < 2 { return false }
        if n == 2 || n == 3 { return true }
        if n % 2 == 0 { return false }

        // Write n-1 as 2^r * d
        var d = n - 1
        var r: UInt32 = 0
        while d % 2 == 0 {
            d /= 2
            r += 1
        }

        // Witnesses that guarantee correctness for all 32-bit numbers
        let witnesses: [UInt32] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

        for a in witnesses {
            if a >= n { continue }
            if !millerRabinWitness(a: UInt64(a), d: UInt64(d), n: UInt64(n), r: r) {
                return false
            }
        }
        return true
    }

    private static func millerRabinWitness(a: UInt64, d: UInt64, n: UInt64, r: UInt32) -> Bool {
        var x = modPow(base: a, exp: d, mod: n)

        if x == 1 || x == n - 1 {
            return true
        }

        for _ in 0..<(r - 1) {
            x = (x * x) % n
            if x == n - 1 {
                return true
            }
        }
        return false
    }

    private static func modPow(base: UInt64, exp: UInt64, mod: UInt64) -> UInt64 {
        var result: UInt64 = 1
        var b = base % mod
        var e = exp

        while e > 0 {
            if e & 1 == 1 {
                result = (result * b) % mod
            }
            e >>= 1
            b = (b * b) % mod
        }
        return result
    }

    /// Compute Hadamard bound for an integer matrix
    /// |det(A)| ≤ ∏||row_i||_2 ≤ (√n · B)^n where B is the max absolute entry
    /// - Parameter matrix: The integer matrix (row-major)
    /// - Parameter n: Matrix dimension
    /// - Parameter maxEntry: Maximum absolute value of any entry
    /// - Returns: Upper bound on |det(A)|
    static func hadamardBound(n: Int, maxEntry: BigInt) -> BigInt {
        // Bound: (sqrt(n) * B)^n = n^(n/2) * B^n
        // We compute this as BigInt to handle large values

        // n^(n/2) ≈ n^n then take sqrt, but easier to compute directly
        // Actually: (sqrt(n) * B)^n = sqrt(n)^n * B^n = n^(n/2) * B^n

        let nBig = BigInt(n)
        let bPowN = maxEntry.power(n)

        // n^(n/2) - we compute n^n then take integer sqrt
        // Or more precisely: ceil(sqrt(n^n))
        let nPowN = nBig.power(n)

        // Integer square root approximation (conservative upper bound)
        let nPowNDiv2 = integerSqrt(nPowN) + 1

        return nPowNDiv2 * bPowN
    }

    /// Integer square root (floor)
    private static func integerSqrt(_ n: BigInt) -> BigInt {
        if n < 0 { return 0 }
        if n == 0 { return 0 }
        if n == 1 { return 1 }

        var x = n
        var y = (x + 1) / 2

        while y < x {
            x = y
            y = (x + n / x) / 2
        }
        return x
    }

    /// Calculate how many primes are needed to exceed a given bound
    /// - Parameter bound: The value to exceed (typically 2 * Hadamard bound)
    /// - Parameter primeBits: Bits per prime (default 31)
    /// - Returns: Minimum number of primes needed
    static func primesNeeded(bound: BigInt, primeBits: Int = 31) -> Int {
        // Product of k primes of size ~2^31 is ~2^(31k)
        // We need 2^(31k) > bound
        // So k > log2(bound) / 31

        let boundBits = bound.bitWidth
        return (boundBits + primeBits - 1) / primeBits + 1
    }
}
