import BigInt
import Foundation

/// Bareiss fraction-free determinant algorithm
/// This is the CPU baseline for exact integer determinant computation.
/// The algorithm avoids fractions during elimination by using the "division-free" update formula.
struct BareissDeterminant {

    /// Compute the exact determinant of an integer matrix using Bareiss algorithm
    /// - Parameter matrix: A square matrix of BigInt values (row-major order)
    /// - Parameter n: Dimension of the matrix
    /// - Returns: The exact determinant as a BigInt
    static func compute(matrix: [BigInt], n: Int) -> BigInt {
        precondition(matrix.count == n * n, "Matrix must be n×n")

        // Make a working copy
        var A = matrix
        var sign: BigInt = 1

        // Previous pivot (starts as 1 for the Bareiss formula)
        var prevPivot: BigInt = 1

        for k in 0..<n {
            // Find pivot row
            var pivotRow = -1
            for i in k..<n {
                if A[i * n + k] != 0 {
                    pivotRow = i
                    break
                }
            }

            // If no pivot found, determinant is zero
            if pivotRow == -1 {
                return 0
            }

            // Swap rows if needed
            if pivotRow != k {
                for j in k..<n {
                    let temp = A[k * n + j]
                    A[k * n + j] = A[pivotRow * n + j]
                    A[pivotRow * n + j] = temp
                }
                sign = -sign
            }

            let pivot = A[k * n + k]

            // Bareiss elimination
            for i in (k + 1)..<n {
                for j in (k + 1)..<n {
                    // Bareiss formula: A[i][j] = (A[k][k] * A[i][j] - A[i][k] * A[k][j]) / prevPivot
                    // This division is exact due to the structure of the algorithm
                    let numerator = pivot * A[i * n + j] - A[i * n + k] * A[k * n + j]
                    A[i * n + j] = numerator / prevPivot
                }
                A[i * n + k] = 0  // Zero out the column below pivot
            }

            prevPivot = pivot
        }

        // Determinant is the last diagonal element times the sign
        return sign * A[(n - 1) * n + (n - 1)]
    }

    /// Compute determinant with timing information
    static func computeTimed(matrix: [BigInt], n: Int) -> (result: BigInt, timeSeconds: Double) {
        let start = CFAbsoluteTimeGetCurrent()
        let result = compute(matrix: matrix, n: n)
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        return (result, elapsed)
    }
}
