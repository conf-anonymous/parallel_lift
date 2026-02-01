import BigInt
import Foundation

/// Timing breakdown for GPU inverse computation
struct InverseTimings {
    var totalTime: Double = 0
    var detATime: Double = 0
    var cofactorTime: Double = 0
    var numPrimes: Int = 0
    var determinantBits: Int = 0
}

/// Timing breakdown for batched inverse (via multi-RHS solve)
struct BatchedInverseTimings {
    var totalTime: Double = 0
    var gpuSolveTime: Double = 0      // GPU modular elimination + backsubstitution
    var detATime: Double = 0          // det(A) computation for denominator
    var crtReconstructTime: Double = 0 // CRT reconstruction
    var numPrimes: Int = 0
    var determinantBits: Int = 0
}

/// GPU-accelerated exact matrix inverse using Cramer's rule
/// Computes A^(-1) where A is n×n integer matrix
/// Returns exact rational matrix via CRT reconstruction
class GPUInverse {

    init?() {
        // Verify GPU is available
        guard GPUDeterminant() != nil else {
            return nil
        }
    }

    /// Compute exact inverse of matrix A using GPU + CRT
    /// (A^(-1))_ij = det(A with column i replaced by e_j) / det(A)
    func inverse(A: [BigInt], n: Int) -> (result: InverseResult, timings: InverseTimings)? {
        var timings = InverseTimings()
        let totalStart = CFAbsoluteTimeGetCurrent()

        guard let gpuDet = GPUDeterminant() else {
            return nil
        }

        // Compute det(A)
        let detAStart = CFAbsoluteTimeGetCurrent()
        guard let (detA, detTimings) = gpuDet.computeDeterminant(matrix: A, n: n) else {
            return nil
        }
        timings.detATime = CFAbsoluteTimeGetCurrent() - detAStart
        timings.numPrimes = detTimings.numPrimes
        timings.determinantBits = detA.bitWidth

        if detA == 0 {
            // Singular matrix - compute rank for diagnostic
            let rank: Int
            if let gpuRank = GPURank(),
               let (r, _) = gpuRank.computeRank(matrix: A, m: n, n: n) {
                rank = r
            } else {
                rank = BareissRank.compute(matrix: A, m: n, n: n)
            }
            timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart
            return (.singular(rank: rank), timings)
        }

        // Compute all cofactors using Cramer's rule
        let cofactorStart = CFAbsoluteTimeGetCurrent()
        var inverseMatrix = [Rational](repeating: Rational(0), count: n * n)

        // For each column j of the inverse (corresponding to solving Ax = e_j)
        for j in 0..<n {
            // Create e_j (standard basis vector)
            var ej = [BigInt](repeating: 0, count: n)
            ej[j] = 1

            // For each row i of the inverse
            for i in 0..<n {
                // (A^(-1))_ij = det(A with column i replaced by e_j) / det(A)
                var Aij = A
                for row in 0..<n {
                    Aij[row * n + i] = ej[row]
                }

                guard let (detAij, _) = gpuDet.computeDeterminant(matrix: Aij, n: n) else {
                    return nil
                }
                inverseMatrix[i * n + j] = Rational(detAij, detA)
            }
        }

        timings.cofactorTime = CFAbsoluteTimeGetCurrent() - cofactorStart
        timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart

        return (.success(matrix: inverseMatrix, determinant: detA), timings)
    }
}

/// GPU-accelerated exact matrix inverse using multi-RHS solve
/// Computes A^(-1) by solving A·X = I (identity matrix)
/// This is O(n³) vs O(n² · n³) = O(n⁵) for Cramer's rule approach
class GPUBatchedInverse {
    private let solver: GPUMultiRHSSolve

    init?() {
        guard let solver = GPUMultiRHSSolve() else {
            return nil
        }
        self.solver = solver
    }

    /// Compute exact inverse of matrix A by solving A·X = I
    /// - Parameters:
    ///   - A: Flattened n×n matrix in row-major order
    ///   - n: Matrix dimension
    /// - Returns: Inverse matrix as rationals with timings, or nil on failure
    func inverse(A: [BigInt], n: Int) -> (result: InverseResult, timings: BatchedInverseTimings)? {
        var timings = BatchedInverseTimings()
        let totalStart = CFAbsoluteTimeGetCurrent()

        // Create identity matrix B = I (column-major for multi-RHS format)
        // B[col * n + row] = (row == col) ? 1 : 0
        var B = [BigInt](repeating: BigInt(0), count: n * n)
        for i in 0..<n {
            B[i * n + i] = BigInt(1)  // Column i, row i
        }

        // Solve A·X = I using multi-RHS solver
        guard let (solution, solveTimings) = solver.solve(A: A, B: B, n: n, k: n) else {
            timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart
            return nil
        }

        // Check for singular matrix (solution is nil)
        guard let solutionCols = solution else {
            // Singular matrix - compute rank for diagnostic
            let rank: Int
            if let gpuRank = GPURank(),
               let (r, _) = gpuRank.computeRank(matrix: A, m: n, n: n) {
                rank = r
            } else {
                rank = BareissRank.compute(matrix: A, m: n, n: n)
            }
            timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart
            return (.singular(rank: rank), timings)
        }

        // Convert column-major solution to row-major inverse matrix
        // solutionCols[col][row] -> inverseMatrix[row * n + col]
        var inverseMatrix = [Rational](repeating: Rational(0), count: n * n)
        for col in 0..<n {
            for row in 0..<n {
                inverseMatrix[row * n + col] = solutionCols[col][row]
            }
        }

        // Extract determinant from solution (all entries have same denominator)
        let detA = solutionCols[0][0].denominator

        // Fill in timings
        timings.gpuSolveTime = solveTimings.gpuSolveTime
        timings.detATime = solveTimings.detATime
        timings.crtReconstructTime = solveTimings.crtReconstructTime
        timings.numPrimes = solveTimings.numPrimes
        timings.determinantBits = detA.bitWidth
        timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart

        return (.success(matrix: inverseMatrix, determinant: detA), timings)
    }
}
