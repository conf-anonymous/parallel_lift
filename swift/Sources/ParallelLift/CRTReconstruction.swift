import BigInt
import Foundation

/// Chinese Remainder Theorem reconstruction utilities
struct CRTReconstruction {

    /// Reconstruct an integer from its residues modulo a set of coprime moduli
    /// Uses the incremental (Garner's) algorithm for efficiency
    /// - Parameter residues: Array of (residue, modulus) pairs
    /// - Returns: The unique integer x in [0, M) where M = ∏moduli such that x ≡ rᵢ (mod mᵢ)
    static func reconstruct(residues: [(residue: UInt32, modulus: UInt32)]) -> BigInt {
        guard !residues.isEmpty else { return 0 }

        if residues.count == 1 {
            return BigInt(residues[0].residue)
        }

        // Garner's algorithm for incremental CRT
        // More efficient than the direct formula for many moduli

        var result = BigInt(residues[0].residue)
        var product = BigInt(residues[0].modulus)

        for i in 1..<residues.count {
            let (ri, mi) = residues[i]
            let riBig = BigInt(ri)
            let miBig = BigInt(mi)

            // Find x such that x ≡ result (mod product) and x ≡ ri (mod mi)
            // x = result + product * t where t = (ri - result) * product^(-1) mod mi

            let diff = ((riBig - result) % miBig + miBig) % miBig
            let productInv = modInverse(product % miBig, miBig)!
            let t = (diff * productInv) % miBig

            result = result + product * t
            product = product * miBig
        }

        // Result is in [0, M), but we may need to interpret as signed
        return result
    }

    /// Reconstruct a signed integer from residues
    /// Interprets the result in the symmetric range [-M/2, M/2)
    /// - Parameter residues: Array of (residue, modulus) pairs
    /// - Returns: The integer in symmetric range
    static func reconstructSigned(residues: [(residue: UInt32, modulus: UInt32)]) -> BigInt {
        let unsigned = reconstruct(residues: residues)

        // Compute M = product of all moduli
        var M = BigInt(1)
        for (_, m) in residues {
            M *= BigInt(m)
        }

        // If result > M/2, subtract M to get negative value
        let halfM = M / 2
        if unsigned > halfM {
            return unsigned - M
        }
        return unsigned
    }

    /// Extended Euclidean algorithm for BigInt
    /// Returns (gcd, x, y) such that ax + by = gcd(a,b)
    static func extendedGCD(_ a: BigInt, _ b: BigInt) -> (gcd: BigInt, x: BigInt, y: BigInt) {
        if b == 0 {
            return (a, 1, 0)
        }

        let (g, x1, y1) = extendedGCD(b, a % b)
        let x = y1
        let y = x1 - (a / b) * y1
        return (g, x, y)
    }

    /// Compute modular inverse of a mod m
    /// Returns nil if inverse doesn't exist
    static func modInverse(_ a: BigInt, _ m: BigInt) -> BigInt? {
        let aReduced = ((a % m) + m) % m
        let (g, x, _) = extendedGCD(aReduced, m)

        if g != 1 {
            return nil  // Inverse doesn't exist
        }

        return ((x % m) + m) % m
    }

    /// Batch reconstruct: given residues for multiple values across same primes
    /// More efficient than calling reconstruct multiple times
    static func batchReconstruct(
        allResidues: [[UInt32]],  // allResidues[valueIndex][primeIndex]
        moduli: [UInt32]
    ) -> [BigInt] {
        // Precompute partial products for efficiency
        let k = moduli.count
        guard k > 0 else { return [] }

        // For each value, reconstruct
        return allResidues.map { residues in
            var pairs: [(residue: UInt32, modulus: UInt32)] = []
            for i in 0..<min(residues.count, k) {
                pairs.append((residues[i], moduli[i]))
            }
            return reconstructSigned(residues: pairs)
        }
    }

    /// Optimized batch CRT reconstruction using precomputed inverses
    /// Processes multiple values simultaneously with vectorized Garner's algorithm
    /// - Parameters:
    ///   - allResidues: [valueIndex][primeIndex] - residues for each value at each prime
    ///   - moduli: Array of coprime moduli (same for all values)
    /// - Returns: Array of reconstructed signed BigInts
    static func batchReconstructOptimized(
        allResidues: [[UInt32]],  // allResidues[valueIndex][primeIndex]
        moduli: [UInt32]
    ) -> [BigInt] {
        let numPrimes = moduli.count
        let numValues = allResidues.count
        guard numPrimes > 0, numValues > 0 else { return [] }

        // Edge case: single prime
        if numPrimes == 1 {
            let m = BigInt(moduli[0])
            let halfM = m / 2
            return allResidues.map { residues in
                let val = BigInt(residues[0])
                return val > halfM ? val - m : val
            }
        }

        // === Phase 1: Precompute invariants (done once for all values) ===

        // Partial products: partialProd[i] = m_0 * m_1 * ... * m_{i-1}
        var partialProd = [BigInt](repeating: BigInt(1), count: numPrimes)
        for i in 1..<numPrimes {
            partialProd[i] = partialProd[i-1] * BigInt(moduli[i-1])
        }

        // Total product M = m_0 * m_1 * ... * m_{numPrimes-1}
        let M = partialProd[numPrimes - 1] * BigInt(moduli[numPrimes - 1])
        let halfM = M / 2

        // Precompute (partialProd[i])^{-1} mod m_i for each i > 0
        // These are used in Garner's algorithm: t = (r_i - result) * inv mod m_i
        var partialProdInvModMi = [BigInt](repeating: BigInt(0), count: numPrimes)
        for i in 1..<numPrimes {
            let mi = BigInt(moduli[i])
            partialProdInvModMi[i] = modInverse(partialProd[i] % mi, mi)!
        }

        // === Phase 2: Batch Garner's algorithm across all values ===

        // Initialize results with first residue
        var results = allResidues.map { BigInt($0[0]) }

        // Process primes 1 through numPrimes-1
        for i in 1..<numPrimes {
            let mi = BigInt(moduli[i])
            let inv = partialProdInvModMi[i]
            let prod = partialProd[i]

            // Update all values in parallel (vectorizable by compiler)
            for vIdx in 0..<numValues {
                let ri = BigInt(allResidues[vIdx][i])

                // t = (r_i - result) * inv mod m_i
                // Handle negative (result mod m_i might be > r_i)
                let resultModMi = ((results[vIdx] % mi) + mi) % mi
                let diff = ((ri - resultModMi) % mi + mi) % mi
                let t = (diff * inv) % mi

                // result += partialProd[i] * t
                results[vIdx] = results[vIdx] + prod * t
            }
        }

        // === Phase 3: Convert to signed representation ===
        for vIdx in 0..<numValues {
            if results[vIdx] > halfM {
                results[vIdx] = results[vIdx] - M
            }
        }

        return results
    }

    /// Multi-RHS optimized CRT: processes n×k solution matrix with shared precomputation
    /// This is specifically optimized for the multi-RHS solve case where:
    /// - We have n rows × k columns of values to reconstruct
    /// - All values share the same set of primes
    /// - Memory layout is [primeIdx][rhsIdx * n + rowIdx]
    ///
    /// - Parameters:
    ///   - residueData: Flat array from GPU: residueData[primeIdx * (n * k) + rhsIdx * n + rowIdx]
    ///   - moduli: Array of primes
    ///   - n: Number of rows
    ///   - k: Number of RHS columns
    ///   - detAResidues: det(A) mod each prime (for numerator recovery)
    /// - Returns: Flat array of reconstructed numerators [rhsIdx * n + rowIdx]
    static func batchReconstructMultiRHS(
        residueData: UnsafePointer<UInt32>,
        moduli: [UInt32],
        n: Int,
        k: Int,
        detAResidues: [UInt32]
    ) -> [BigInt] {
        let numPrimes = moduli.count
        let numValues = n * k
        guard numPrimes > 0, numValues > 0 else { return [] }

        // === Phase 1: Precompute invariants (done once) ===

        // Partial products
        var partialProd = [BigInt](repeating: BigInt(1), count: numPrimes)
        for i in 1..<numPrimes {
            partialProd[i] = partialProd[i-1] * BigInt(moduli[i-1])
        }

        let M = partialProd[numPrimes - 1] * BigInt(moduli[numPrimes - 1])
        let halfM = M / 2

        // Precompute inverses
        var partialProdInvModMi = [BigInt](repeating: BigInt(0), count: numPrimes)
        for i in 1..<numPrimes {
            let mi = BigInt(moduli[i])
            partialProdInvModMi[i] = modInverse(partialProd[i] % mi, mi)!
        }

        // === Phase 2: Initialize with first prime's residues (multiplied by detA) ===

        var results = [BigInt](repeating: BigInt(0), count: numValues)
        let p0 = UInt64(moduli[0])
        let detA0 = UInt64(detAResidues[0])

        for vIdx in 0..<numValues {
            let r = UInt64(residueData[vIdx])
            let numResidue = (r * detA0) % p0
            results[vIdx] = BigInt(numResidue)
        }

        // === Phase 3: Batch Garner iterations ===

        for i in 1..<numPrimes {
            let mi = BigInt(moduli[i])
            let miU64 = UInt64(moduli[i])
            let inv = partialProdInvModMi[i]
            let prod = partialProd[i]
            let detAi = UInt64(detAResidues[i])
            let baseOffset = i * numValues

            for vIdx in 0..<numValues {
                // Get residue and multiply by detA mod p
                let r = UInt64(residueData[baseOffset + vIdx])
                let numResidue = (r * detAi) % miU64
                let ri = BigInt(numResidue)

                // Garner step
                let resultModMi = ((results[vIdx] % mi) + mi) % mi
                let diff = ((ri - resultModMi) % mi + mi) % mi
                let t = (diff * inv) % mi

                results[vIdx] = results[vIdx] + prod * t
            }
        }

        // === Phase 4: Convert to signed ===
        for vIdx in 0..<numValues {
            if results[vIdx] > halfM {
                results[vIdx] = results[vIdx] - M
            }
        }

        return results
    }

    /// Rational reconstruction using extended Euclidean algorithm
    /// Given value v and modulus M, find small integers (num, den) such that:
    ///   num ≡ v * den (mod M)
    ///   |num|, |den| ≤ √M (approximately)
    /// This is useful when the original rational had small numerator and denominator
    /// - Parameters:
    ///   - v: The value to reconstruct (in [0, M))
    ///   - M: The modulus
    ///   - bound: Optional bound on numerator/denominator (defaults to √M)
    /// - Returns: (numerator, denominator) or nil if reconstruction fails
    static func rationalReconstruct(_ v: BigInt, modulus M: BigInt, bound: BigInt? = nil) -> (num: BigInt, den: BigInt)? {
        // Use the extended Euclidean algorithm to find rational approximation
        // Algorithm: Run extended GCD on (v, M) until the remainder is small enough

        let B = bound ?? isqrt(M / 2)  // Default bound ≈ √(M/2)

        // Extended GCD iteration
        var r0 = M
        var r1 = v
        var t0 = BigInt(0)
        var t1 = BigInt(1)

        while r1 > B {
            let q = r0 / r1
            let r2 = r0 - q * r1
            let t2 = t0 - q * t1

            r0 = r1
            r1 = r2
            t0 = t1
            t1 = t2
        }

        // At this point: r1 ≈ num, t1 ≈ den
        // We have: r1 ≡ v * t1 (mod M)
        // So if we set num = r1, den = t1, then num ≡ v * den (mod M)

        let num = r1
        var den = t1

        // Ensure denominator is positive
        if den < 0 {
            return (num: -num, den: -den)
        }

        // Verify the bound
        if num.magnitude > B.magnitude || den.magnitude > B.magnitude {
            return nil  // Reconstruction failed - original rational too large
        }

        // Verify: num ≡ v * den (mod M)
        let check = ((v * den) % M + M) % M
        let numMod = ((num % M) + M) % M
        if numMod != check {
            return nil
        }

        return (num: num, den: den)
    }

    /// Integer square root using Newton's method
    private static func isqrt(_ n: BigInt) -> BigInt {
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
}
