import BigInt
import Foundation

/// Timing breakdown for GPU solve computation
struct SolveTimings {
    var totalTime: Double = 0
    var detATime: Double = 0
    var detAiTime: Double = 0
    var numPrimes: Int = 0
    var solutionBits: Int = 0  // Max bits in solution numerator/denominator
}

/// GPU-accelerated exact linear system solve using Cramer's rule
/// Solves Ax = b where A is n×n integer matrix, b is integer vector
/// Returns exact rational solution x = (det(A_1)/det(A), ..., det(A_n)/det(A))
/// where A_i has column i replaced by b
class GPUSolve {

    init?() {
        // Verify GPU is available
        guard GPUDeterminant() != nil else {
            return nil
        }
    }

    /// Solve Ax = b exactly using GPU + CRT via Cramer's rule
    /// x_i = det(A_i) / det(A) where A_i has column i replaced by b
    func solve(A: [BigInt], b: [BigInt], n: Int) -> (solution: [Rational]?, timings: SolveTimings)? {
        var timings = SolveTimings()
        let totalStart = CFAbsoluteTimeGetCurrent()

        // Use GPUDeterminant for computing determinants
        guard let gpuDet = GPUDeterminant() else {
            return nil
        }

        // First compute det(A)
        let detAStart = CFAbsoluteTimeGetCurrent()
        guard let (detA, detTimings) = gpuDet.computeDeterminant(matrix: A, n: n) else {
            return nil
        }
        timings.detATime = CFAbsoluteTimeGetCurrent() - detAStart

        if detA == 0 {
            // Singular matrix
            timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart
            return (nil, timings)
        }

        timings.numPrimes = detTimings.numPrimes

        // Compute det(A_i) for each column i using Cramer's rule
        let detAiStart = CFAbsoluteTimeGetCurrent()
        var numerators: [BigInt] = []

        for i in 0..<n {
            // Create A_i: copy A and replace column i with b
            var Ai = A
            for row in 0..<n {
                Ai[row * n + i] = b[row]
            }

            guard let (detAi, _) = gpuDet.computeDeterminant(matrix: Ai, n: n) else {
                return nil
            }
            numerators.append(detAi)
        }

        timings.detAiTime = CFAbsoluteTimeGetCurrent() - detAiStart

        // Build rational solution: x_i = det(A_i) / det(A)
        var solution: [Rational] = []
        for i in 0..<n {
            solution.append(Rational(numerators[i], detA))
        }

        // Compute max bits
        var maxBits = detA.bitWidth
        for num in numerators {
            maxBits = max(maxBits, num.bitWidth)
        }
        timings.solutionBits = maxBits

        timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart
        return (solution, timings)
    }
}
