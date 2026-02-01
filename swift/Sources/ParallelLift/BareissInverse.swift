import BigInt
import Foundation

/// Result of matrix inverse computation
enum InverseResult {
    case success(matrix: [Rational], determinant: BigInt)
    case singular(rank: Int)  // Matrix is singular, return its rank
}

/// CPU baseline for exact matrix inverse using Cramer's rule
/// Computes A^(-1) where A is n×n integer matrix
/// Returns exact rational matrix A^(-1) where (A^(-1))_ij = cofactor(A)_ji / det(A)
struct BareissInverse {

    /// Compute exact inverse of matrix A
    /// - Parameters:
    ///   - A: Flattened n×n matrix in row-major order
    ///   - n: Dimension
    /// - Returns: InverseResult with either the inverse matrix or singularity info
    static func inverse(A: [BigInt], n: Int) -> InverseResult {
        // Compute det(A)
        let detA = BareissDeterminant.compute(matrix: A, n: n)

        if detA == 0 {
            // Singular matrix - compute rank for diagnostic
            let rank = BareissRank.compute(matrix: A, m: n, n: n)
            return .singular(rank: rank)
        }

        // Compute inverse using Cramer's rule:
        // (A^(-1))_ij = det(A with column j replaced by e_i) / det(A)
        // Equivalently: solve A * x_j = e_j for each standard basis vector e_j
        // Then A^(-1) has columns x_0, x_1, ..., x_{n-1}

        var inverseMatrix = [Rational](repeating: Rational(0), count: n * n)

        for j in 0..<n {
            // Create e_j (standard basis vector)
            var ej = [BigInt](repeating: 0, count: n)
            ej[j] = 1

            // Solve A * x = e_j using Cramer's rule
            for i in 0..<n {
                // (A^(-1))_ij = det(A_i^j) / det(A)
                // where A_i^j has column i of A replaced by e_j

                var Aij = A
                for row in 0..<n {
                    Aij[row * n + i] = ej[row]
                }

                let detAij = BareissDeterminant.compute(matrix: Aij, n: n)
                inverseMatrix[i * n + j] = Rational(detAij, detA)
            }
        }

        return .success(matrix: inverseMatrix, determinant: detA)
    }

    /// Verify inverse by computing A * A^(-1) and checking for identity
    static func verifyInverse(A: [BigInt], inverse: [Rational], n: Int) -> Bool {
        // Compute A * inverse and check if it's identity
        for i in 0..<n {
            for j in 0..<n {
                // Compute (A * inverse)[i,j] = sum_k A[i,k] * inverse[k,j]
                var sumNum: BigInt = 0
                var sumDen: BigInt = 1

                for k in 0..<n {
                    let termNum = A[i * n + k] * inverse[k * n + j].numerator
                    sumNum = sumNum * inverse[k * n + j].denominator + termNum * sumDen
                    sumDen = sumDen * inverse[k * n + j].denominator
                }

                // Expected: 1 if i==j, 0 otherwise
                let expected: BigInt = (i == j) ? 1 : 0
                if sumNum != expected * sumDen {
                    return false
                }
            }
        }
        return true
    }

    /// Compute inverse with timing
    static func inverseTimed(A: [BigInt], n: Int) -> (result: InverseResult, time: Double) {
        let start = CFAbsoluteTimeGetCurrent()
        let result = inverse(A: A, n: n)
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        return (result, elapsed)
    }
}
