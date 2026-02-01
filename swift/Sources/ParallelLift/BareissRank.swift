import BigInt
import Foundation

/// CPU baseline for exact rank computation using Bareiss algorithm
/// Computes the rank of an integer matrix exactly using fraction-free elimination
struct BareissRank {

    /// Compute the rank of a matrix using Bareiss fraction-free Gaussian elimination
    /// - Parameters:
    ///   - matrix: Flattened m×n matrix in row-major order
    ///   - m: Number of rows
    ///   - n: Number of columns
    /// - Returns: The exact rank of the matrix
    static func compute(matrix: [BigInt], m: Int, n: Int) -> Int {
        // Create working copy
        var A = matrix

        var rank = 0
        var pivotRow = 0
        var pivotCol = 0
        var lastPivot: BigInt = 1

        while pivotRow < m && pivotCol < n {
            // Find pivot in current column
            var maxRow = pivotRow
            for i in (pivotRow + 1)..<m {
                if A[i * n + pivotCol].magnitude > A[maxRow * n + pivotCol].magnitude {
                    maxRow = i
                }
            }

            if A[maxRow * n + pivotCol] == 0 {
                // No pivot in this column, move to next column
                pivotCol += 1
                continue
            }

            // Swap rows if needed
            if maxRow != pivotRow {
                for j in 0..<n {
                    let temp = A[pivotRow * n + j]
                    A[pivotRow * n + j] = A[maxRow * n + j]
                    A[maxRow * n + j] = temp
                }
            }

            let pivot = A[pivotRow * n + pivotCol]

            // Eliminate below pivot using Bareiss algorithm
            for i in (pivotRow + 1)..<m {
                if A[i * n + pivotCol] != 0 {
                    for j in (pivotCol + 1)..<n {
                        // Bareiss update: avoids fractions
                        let numerator = pivot * A[i * n + j] - A[i * n + pivotCol] * A[pivotRow * n + j]
                        A[i * n + j] = numerator / lastPivot
                    }
                    A[i * n + pivotCol] = 0
                }
            }

            lastPivot = pivot
            rank += 1
            pivotRow += 1
            pivotCol += 1
        }

        return rank
    }

    /// Compute rank with timing
    static func computeTimed(matrix: [BigInt], m: Int, n: Int) -> (rank: Int, time: Double) {
        let start = CFAbsoluteTimeGetCurrent()
        let rank = compute(matrix: matrix, m: m, n: n)
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        return (rank, elapsed)
    }
}
