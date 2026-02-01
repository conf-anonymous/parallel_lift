import BigInt
import Foundation

/// Rational number representation for exact solutions
struct Rational: CustomStringConvertible, Equatable {
    let numerator: BigInt
    let denominator: BigInt

    init(_ numerator: BigInt, _ denominator: BigInt) {
        // Normalize: ensure denominator is positive and reduce
        if denominator == 0 {
            self.numerator = 0
            self.denominator = 1
        } else {
            let g = Rational.gcd(numerator.magnitude, denominator.magnitude)
            let sign: BigInt = denominator < 0 ? -1 : 1
            self.numerator = sign * numerator / BigInt(g)
            self.denominator = sign * denominator / BigInt(g)
        }
    }

    init(_ value: BigInt) {
        self.numerator = value
        self.denominator = 1
    }

    static func gcd(_ a: BigInt.Magnitude, _ b: BigInt.Magnitude) -> BigInt.Magnitude {
        var a = a
        var b = b
        while b != 0 {
            let t = b
            b = a % b
            a = t
        }
        return a
    }

    var description: String {
        if denominator == 1 {
            return "\(numerator)"
        }
        return "\(numerator)/\(denominator)"
    }

    var isZero: Bool {
        return numerator == 0
    }

    static func == (lhs: Rational, rhs: Rational) -> Bool {
        return lhs.numerator * rhs.denominator == rhs.numerator * lhs.denominator
    }

    // Arithmetic operators
    static func + (lhs: Rational, rhs: Rational) -> Rational {
        let num = lhs.numerator * rhs.denominator + rhs.numerator * lhs.denominator
        let den = lhs.denominator * rhs.denominator
        return Rational(num, den)
    }

    static func - (lhs: Rational, rhs: Rational) -> Rational {
        let num = lhs.numerator * rhs.denominator - rhs.numerator * lhs.denominator
        let den = lhs.denominator * rhs.denominator
        return Rational(num, den)
    }

    static func * (lhs: Rational, rhs: Rational) -> Rational {
        return Rational(lhs.numerator * rhs.numerator, lhs.denominator * rhs.denominator)
    }

    static func / (lhs: Rational, rhs: Rational) -> Rational {
        return Rational(lhs.numerator * rhs.denominator, lhs.denominator * rhs.numerator)
    }

    static prefix func - (r: Rational) -> Rational {
        return Rational(-r.numerator, r.denominator)
    }
}

/// CPU baseline for exact linear system solve using Cramer's rule
/// Solves Ax = b where A is n×n integer matrix, b is integer vector
/// Returns exact rational solution x where x_i = det(A_i) / det(A)
struct BareissSolve {

    /// Solve Ax = b exactly using Cramer's rule with Bareiss determinant
    /// - Parameters:
    ///   - A: Flattened n×n matrix in row-major order
    ///   - b: Right-hand side vector of length n
    ///   - n: Dimension
    /// - Returns: Solution vector x as rationals, or nil if no unique solution
    static func solve(A: [BigInt], b: [BigInt], n: Int) -> [Rational]? {
        // Compute det(A)
        let detA = BareissDeterminant.compute(matrix: A, n: n)

        if detA == 0 {
            // Singular matrix - no unique solution
            return nil
        }

        // Compute det(A_i) for each column i using Cramer's rule
        var solution: [Rational] = []

        for i in 0..<n {
            // Create A_i: copy A and replace column i with b
            var Ai = A
            for row in 0..<n {
                Ai[row * n + i] = b[row]
            }

            let detAi = BareissDeterminant.compute(matrix: Ai, n: n)
            solution.append(Rational(detAi, detA))
        }

        return solution
    }

    /// Verify a solution by computing A*x and comparing to b
    static func verifySolution(A: [BigInt], b: [BigInt], x: [Rational], n: Int) -> Bool {
        for i in 0..<n {
            // Compute (A * x)[i] = sum_j A[i,j] * x[j]
            // Using common denominator for accuracy

            var sumNum: BigInt = 0
            var sumDen: BigInt = 1

            for j in 0..<n {
                // Add A[i,j] * x[j].num / x[j].den
                // sumNum/sumDen + A[i,j]*x[j].num / x[j].den
                // = (sumNum * x[j].den + A[i,j] * x[j].num * sumDen) / (sumDen * x[j].den)

                let termNum = A[i * n + j] * x[j].numerator
                sumNum = sumNum * x[j].denominator + termNum * sumDen
                sumDen = sumDen * x[j].denominator
            }

            // Check if sumNum / sumDen == b[i]
            // i.e., sumNum == b[i] * sumDen
            if sumNum != b[i] * sumDen {
                return false
            }
        }
        return true
    }

    /// Solve with timing
    static func solveTimed(A: [BigInt], b: [BigInt], n: Int) -> (solution: [Rational]?, time: Double) {
        let start = CFAbsoluteTimeGetCurrent()
        let solution = solve(A: A, b: b, n: n)
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        return (solution, elapsed)
    }
}
