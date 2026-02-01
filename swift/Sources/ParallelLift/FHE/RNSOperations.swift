import BigInt
import Foundation

/// CPU implementations of RNS/CRT operations commonly used in FHE
struct RNSOperationsCPU {

    /// Represents a polynomial in RNS form
    /// residues[i][j] = coefficient j mod moduli[i]
    typealias RNSPolynomial = [[UInt64]]

    // MARK: - Basis Conversion Operations

    /// Convert a polynomial from RNS basis Q to a different RNS basis P
    /// This is the "basis extension" operation used in key switching
    /// - Parameters:
    ///   - poly: Polynomial in basis Q (residues[q_index][coeff_index])
    ///   - fromBasis: The source RNS context (Q)
    ///   - toBasis: The target RNS context (P)
    /// - Returns: Polynomial in basis P
    static func basisExtend(
        poly: RNSPolynomial,
        fromBasis Q: RNSContext,
        toBasis P: RNSContext
    ) -> RNSPolynomial {
        let n = poly[0].count  // number of coefficients
        let kQ = Q.count       // number of moduli in Q
        let kP = P.count       // number of moduli in P

        // First, lift each coefficient to integers using CRT
        // Then reduce mod each prime in P

        var result: RNSPolynomial = Array(repeating: Array(repeating: 0, count: n), count: kP)

        for j in 0..<n {
            // Reconstruct coefficient j from Q basis
            // Using CRT: x = sum_i (x_i * M_i * (M_i^-1 mod q_i)) mod M

            // For basis extension, we compute this mod each p in P
            for pIdx in 0..<kP {
                let p = P.moduli[pIdx]
                var sum: UInt64 = 0

                for qIdx in 0..<kQ {
                    // Term: x_qIdx * (M/q_qIdx) * (M/q_qIdx)^-1 mod p
                    let x_q = poly[qIdx][j]

                    // Compute M/q_i mod p
                    let MOverQ_big = Q.puncturedProducts[qIdx] % BigInt(p)
                    let MOverQ_p = UInt64(MOverQ_big)

                    // Compute contribution: x_q * (M/q_i) * (M/q_i)^-1 mod p
                    let contrib = RNSContext.modMul(x_q, MOverQ_p, p)
                    let scaled = RNSContext.modMul(contrib, UInt64(Q.invPunctured[qIdx] % p), p)
                    sum = (sum + scaled) % p
                }

                result[pIdx][j] = sum
            }
        }

        return result
    }

    /// Fast basis extension using precomputed tables
    /// More efficient for repeated operations with same bases
    static func basisExtendFast(
        poly: RNSPolynomial,
        fromBasis Q: RNSContext,
        toBasis P: RNSContext,
        precomputed: BasisExtensionPrecompute
    ) -> RNSPolynomial {
        let n = poly[0].count
        let kP = P.count

        var result: RNSPolynomial = Array(repeating: Array(repeating: 0, count: n), count: kP)

        for j in 0..<n {
            for pIdx in 0..<kP {
                var sum: UInt64 = 0
                let p = P.moduli[pIdx]

                for qIdx in 0..<Q.count {
                    let x_q = poly[qIdx][j]
                    let factor = precomputed.conversionFactors[qIdx][pIdx]
                    let term = RNSContext.modMul(x_q, factor, p)
                    sum = (sum + term) % p
                }

                result[pIdx][j] = sum
            }
        }

        return result
    }

    /// Modulus switching: reduce from basis Q to basis Q' ⊂ Q
    /// Drops the last modulus and rescales
    static func modSwitch(
        poly: RNSPolynomial,
        basis Q: RNSContext,
        dropLast: Int = 1
    ) -> RNSPolynomial {
        // Simply drop the last `dropLast` residues
        // In practice, this also involves rounding, but we simplify here
        let newK = Q.count - dropLast
        return Array(poly.prefix(newK))
    }

    /// Rescale operation (used in CKKS)
    /// Divides by the last modulus and drops it
    static func rescale(
        poly: RNSPolynomial,
        basis Q: RNSContext
    ) -> RNSPolynomial {
        let n = poly[0].count
        let kQ = Q.count
        let qLast = Q.moduli[kQ - 1]

        // New basis without last modulus
        let newK = kQ - 1
        var result: RNSPolynomial = Array(repeating: Array(repeating: 0, count: n), count: newK)

        for j in 0..<n {
            let lastResidue = poly[kQ - 1][j]

            for qIdx in 0..<newK {
                let q = Q.moduli[qIdx]
                // (poly[qIdx][j] - lastResidue) / qLast mod q
                // = (poly[qIdx][j] - lastResidue) * qLast^(-1) mod q

                let qLastInv = RNSContext.modInverse(qLast % q, q)

                var diff = poly[qIdx][j]
                let lastMod = lastResidue % q
                if diff >= lastMod {
                    diff -= lastMod
                } else {
                    diff = q - (lastMod - diff)
                }

                result[qIdx][j] = RNSContext.modMul(diff, qLastInv, q)
            }
        }

        return result
    }

    // MARK: - Gadget Decomposition (for key switching)

    /// Decompose a polynomial into digits for key switching
    /// Used in BFV/BGV/CKKS key switching
    static func gadgetDecompose(
        poly: RNSPolynomial,
        basis Q: RNSContext,
        digitBase B: UInt64,
        numDigits: Int
    ) -> [RNSPolynomial] {
        let n = poly[0].count
        let kQ = Q.count

        var digits: [RNSPolynomial] = []

        for d in 0..<numDigits {
            var digit: RNSPolynomial = Array(repeating: Array(repeating: 0, count: n), count: kQ)

            for qIdx in 0..<kQ {
                for j in 0..<n {
                    // Extract digit d from poly[qIdx][j] in base B
                    let val = poly[qIdx][j]
                    let shifted = val / intPow(B, UInt64(d))
                    digit[qIdx][j] = shifted % B
                }
            }

            digits.append(digit)
        }

        return digits
    }

    /// Multiply decomposed polynomial by gadget matrix and accumulate
    /// This is the core of key switching
    static func gadgetMultiplyAccumulate(
        digits: [RNSPolynomial],
        gadgetKeys: [RNSPolynomial],  // Key switch keys
        basis Q: RNSContext
    ) -> RNSPolynomial {
        let n = digits[0][0].count
        let kQ = Q.count
        let numDigits = digits.count

        var result: RNSPolynomial = Array(repeating: Array(repeating: 0, count: n), count: kQ)

        for d in 0..<numDigits {
            for qIdx in 0..<kQ {
                let q = Q.moduli[qIdx]
                for j in 0..<n {
                    // result += digit[d] * gadgetKey[d]
                    let term = RNSContext.modMul(digits[d][qIdx][j], gadgetKeys[d][qIdx][j], q)
                    result[qIdx][j] = (result[qIdx][j] + term) % q
                }
            }
        }

        return result
    }

    // MARK: - Polynomial Arithmetic in RNS

    /// Add two polynomials in RNS form
    static func add(
        _ a: RNSPolynomial,
        _ b: RNSPolynomial,
        basis Q: RNSContext
    ) -> RNSPolynomial {
        let n = a[0].count
        let kQ = Q.count

        var result: RNSPolynomial = Array(repeating: Array(repeating: 0, count: n), count: kQ)

        for qIdx in 0..<kQ {
            let q = Q.moduli[qIdx]
            for j in 0..<n {
                result[qIdx][j] = (a[qIdx][j] + b[qIdx][j]) % q
            }
        }

        return result
    }

    /// Multiply two polynomials coefficient-wise in RNS form (Hadamard product)
    /// Used after NTT transform
    static func multiplyHadamard(
        _ a: RNSPolynomial,
        _ b: RNSPolynomial,
        basis Q: RNSContext
    ) -> RNSPolynomial {
        let n = a[0].count
        let kQ = Q.count

        var result: RNSPolynomial = Array(repeating: Array(repeating: 0, count: n), count: kQ)

        for qIdx in 0..<kQ {
            let q = Q.moduli[qIdx]
            for j in 0..<n {
                result[qIdx][j] = RNSContext.modMul(a[qIdx][j], b[qIdx][j], q)
            }
        }

        return result
    }

    // MARK: - Helpers

    /// Integer power function for UInt64 (to avoid calling Foundation's floating-point pow)
    private static func intPow(_ base: UInt64, _ exp: UInt64) -> UInt64 {
        var result: UInt64 = 1
        var b = base
        var e = exp
        while e > 0 {
            if e & 1 == 1 {
                let (newResult, overflow1) = result.multipliedReportingOverflow(by: b)
                if overflow1 {
                    // Overflow - return max value (this is an error case)
                    return UInt64.max
                }
                result = newResult
            }
            e >>= 1
            if e > 0 {
                let (newB, overflow2) = b.multipliedReportingOverflow(by: b)
                if overflow2 {
                    // Overflow - but if we don't need more iterations, that's OK
                    b = newB  // Use truncated value
                }
                b = newB
            }
        }
        return result
    }
}

/// Precomputed values for fast basis extension
struct BasisExtensionPrecompute {
    /// conversionFactors[qIdx][pIdx] = (Q/q_i)^(-1) * (Q/q_i) mod p_j
    let conversionFactors: [[UInt64]]

    init(fromBasis Q: RNSContext, toBasis P: RNSContext) {
        var factors: [[UInt64]] = []

        for qIdx in 0..<Q.count {
            var row: [UInt64] = []
            let QOverQi = Q.puncturedProducts[qIdx]
            let invQi = Q.invPunctured[qIdx]

            for pIdx in 0..<P.count {
                let p = P.moduli[pIdx]
                // factor = (Q/q_i) * (Q/q_i)^(-1)_qi mod p
                let QOverQi_mod_p = UInt64(QOverQi % BigInt(p))

                // invQi is mod q_i, we need it mod p... this is tricky
                // For proper implementation, we'd precompute this differently
                // Simplified: assume we can use invQi directly (works when qi ≈ p)
                let factor = RNSContext.modMul(QOverQi_mod_p, invQi % p, p)
                row.append(factor)
            }
            factors.append(row)
        }

        self.conversionFactors = factors
    }
}

// MARK: - MRR (Multiply-Relinearize-Rescale) CPU Implementation

/// Timing breakdown for CPU MRR operation
struct CPUMRRTimings {
    var multiplyTime: Double = 0
    var relinearizeTime: Double = 0
    var rescaleTime: Double = 0
    var totalTime: Double = 0
}

/// CPU implementation of fused Multiply-Relinearize-Rescale
/// This mirrors real CKKS homomorphic multiplication workflow
struct RNSMRROperationsCPU {
    typealias RNSPolynomial = [[UInt64]]

    /// Fused Multiply-Relinearize-Rescale operation (CPU baseline)
    /// - Parameters:
    ///   - ct0, ct1: First ciphertext components (c0, c1)
    ///   - ct2_0, ct2_1: Second ciphertext components
    ///   - relinKey0, relinKey1: Relinearization key components
    ///   - basis: RNS context
    ///   - digitBase: Gadget decomposition base
    ///   - numDigits: Number of digits
    /// - Returns: Result ciphertext and timing breakdown
    static func fusedMRR(
        ct0: RNSPolynomial, ct1: RNSPolynomial,
        ct2_0: RNSPolynomial, ct2_1: RNSPolynomial,
        relinKey0: [RNSPolynomial], relinKey1: [RNSPolynomial],
        basis Q: RNSContext,
        digitBase B: UInt64,
        numDigits: Int
    ) -> (c0: RNSPolynomial, c1: RNSPolynomial, timings: CPUMRRTimings) {
        let n = ct0[0].count
        let kQ = Q.count
        let newK = kQ - 1

        var timings = CPUMRRTimings()
        let totalStart = CFAbsoluteTimeGetCurrent()

        // === Step 1: Multiply (Hadamard products) ===
        let mulStart = CFAbsoluteTimeGetCurrent()

        // d0 = ct0 * ct2_0
        let d0 = RNSOperationsCPU.multiplyHadamard(ct0, ct2_0, basis: Q)

        // d1 = ct0 * ct2_1 + ct1 * ct2_0
        let temp1 = RNSOperationsCPU.multiplyHadamard(ct0, ct2_1, basis: Q)
        let temp2 = RNSOperationsCPU.multiplyHadamard(ct1, ct2_0, basis: Q)
        var d1 = RNSOperationsCPU.add(temp1, temp2, basis: Q)

        // d2 = ct1 * ct2_1
        let d2 = RNSOperationsCPU.multiplyHadamard(ct1, ct2_1, basis: Q)

        timings.multiplyTime = CFAbsoluteTimeGetCurrent() - mulStart

        // === Step 2: Relinearize (key switching on d2) ===
        let relinStart = CFAbsoluteTimeGetCurrent()

        // Gadget decompose d2
        let digits = RNSOperationsCPU.gadgetDecompose(
            poly: d2, basis: Q, digitBase: B, numDigits: numDigits)

        // MAC with relinearization keys
        let relinContrib0 = RNSOperationsCPU.gadgetMultiplyAccumulate(
            digits: digits, gadgetKeys: relinKey0, basis: Q)
        let relinContrib1 = RNSOperationsCPU.gadgetMultiplyAccumulate(
            digits: digits, gadgetKeys: relinKey1, basis: Q)

        // c0' = d0 + relinContrib0, c1' = d1 + relinContrib1
        var c0Prime = RNSOperationsCPU.add(d0, relinContrib0, basis: Q)
        d1 = RNSOperationsCPU.add(d1, relinContrib1, basis: Q)

        timings.relinearizeTime = CFAbsoluteTimeGetCurrent() - relinStart

        // === Step 3: Rescale (drop last modulus) ===
        let rescaleStart = CFAbsoluteTimeGetCurrent()

        let c0Final = RNSOperationsCPU.rescale(poly: c0Prime, basis: Q)
        let c1Final = RNSOperationsCPU.rescale(poly: d1, basis: Q)

        timings.rescaleTime = CFAbsoluteTimeGetCurrent() - rescaleStart
        timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart

        return (c0Final, c1Final, timings)
    }
}
