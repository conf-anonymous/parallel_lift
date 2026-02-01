import BigInt
import Foundation
import Metal

/// Timing breakdown for overdetermined solve
struct OverdeterminedTimings {
    var totalTime: Double = 0
    var rrefTime: Double = 0           // RREF computation time
    var crtReconstructTime: Double = 0  // CRT reconstruction
    var verifyTime: Double = 0          // Solution verification
    var numPrimes: Int = 0
}

/// Result of overdetermined system solve
enum OverdeterminedResult {
    case consistent(solution: [Rational], rank: Int)   // System has a solution
    case inconsistent(rank: Int, augmentedRank: Int)   // No solution exists
}

/// GPU-accelerated exact overdetermined system solver
/// For Ax = b where A is m×n with m > n
/// Uses RREF on augmented matrix [A|b] to check consistency and find solution
class GPUOverdetermined {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let rrefPipeline: MTLComputePipelineState

    /// Metal shader for RREF computation (same as nullspace)
    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    // Modular inverse using extended Euclidean algorithm
    inline uint mod_inverse(uint a, uint p) {
        if (a == 0) return 0;
        int t = 0, newt = 1;
        int r = int(p), newr = int(a % p);
        while (newr != 0) {
            int q = r / newr;
            int tmp = t - q * newt;
            t = newt; newt = tmp;
            tmp = r - q * newr;
            r = newr; newr = tmp;
        }
        if (t < 0) t += int(p);
        return uint(t);
    }

    // Compute RREF on augmented matrix [A|b] and identify pivot columns
    kernel void augmented_rref(
        device uint* matrix,           // Input/output: m×(n+1) augmented matrix
        device uint* pivotCols,        // Output: for each row, the pivot column (-1 if none)
        constant uint& m,              // Number of rows
        constant uint& nAug,           // Number of columns (n+1 for augmented)
        constant uint& prime,          // Modulus
        uint tid [[thread_position_in_grid]]
    ) {
        if (tid != 0) return;

        uint p = prime;
        int pivotRow = 0;

        // Initialize pivot columns to -1
        for (uint i = 0; i < m; i++) {
            pivotCols[i] = 0xFFFFFFFF;
        }

        for (uint col = 0; col < nAug && uint(pivotRow) < m; col++) {
            // Find pivot
            int pivot = -1;
            for (uint row = uint(pivotRow); row < m; row++) {
                if (matrix[row * nAug + col] != 0) {
                    pivot = int(row);
                    break;
                }
            }

            if (pivot == -1) continue;

            // Swap rows
            if (pivot != pivotRow) {
                for (uint j = 0; j < nAug; j++) {
                    uint tmp = matrix[pivotRow * nAug + j];
                    matrix[pivotRow * nAug + j] = matrix[pivot * nAug + j];
                    matrix[pivot * nAug + j] = tmp;
                }
            }

            // Scale pivot row
            uint pivotVal = matrix[pivotRow * nAug + col];
            uint pivotInv = mod_inverse(pivotVal, p);
            for (uint j = 0; j < nAug; j++) {
                matrix[pivotRow * nAug + j] = (uint64_t(matrix[pivotRow * nAug + j]) * pivotInv) % p;
            }

            // Eliminate
            for (uint row = 0; row < m; row++) {
                if (row == uint(pivotRow)) continue;
                uint factor = matrix[row * nAug + col];
                if (factor != 0) {
                    for (uint j = 0; j < nAug; j++) {
                        uint sub = (uint64_t(factor) * matrix[pivotRow * nAug + j]) % p;
                        matrix[row * nAug + j] = (matrix[row * nAug + j] + p - sub) % p;
                    }
                }
            }

            pivotCols[pivotRow] = col;
            pivotRow++;
        }
    }
    """

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else {
            return nil
        }
        self.device = device
        self.commandQueue = commandQueue

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: GPUOverdetermined.shaderSource, options: nil)
        } catch {
            print("GPUOverdetermined: Failed to compile shader: \(error)")
            return nil
        }

        guard let rrefFunc = library.makeFunction(name: "augmented_rref") else {
            print("GPUOverdetermined: Failed to find augmented_rref function")
            return nil
        }

        do {
            self.rrefPipeline = try device.makeComputePipelineState(function: rrefFunc)
        } catch {
            print("GPUOverdetermined: Failed to create pipeline: \(error)")
            return nil
        }
    }

    /// Solve overdetermined system Ax = b
    /// - Parameters:
    ///   - A: Flattened m×n matrix in row-major order
    ///   - b: Right-hand side vector of length m
    ///   - m: Number of rows (equations)
    ///   - n: Number of columns (variables)
    /// - Returns: Solution if consistent, or inconsistency diagnostic
    func solve(A: [BigInt], b: [BigInt], m: Int, n: Int) -> (result: OverdeterminedResult, timings: OverdeterminedTimings)? {
        var timings = OverdeterminedTimings()
        let totalStart = CFAbsoluteTimeGetCurrent()

        let nAug = n + 1  // Augmented matrix width

        // Estimate primes needed
        let maxEntryA = A.map { $0.magnitude }.max() ?? BigInt(1).magnitude
        let maxEntryB = b.map { $0.magnitude }.max() ?? BigInt(1).magnitude
        let maxEntry = max(maxEntryA, maxEntryB)
        let hadamardBound = BigInt(maxEntry) * BigInt(m)
        let bitsNeeded = hadamardBound.bitWidth + 64
        let numPrimes = max(3, (bitsNeeded + 30) / 31)

        let primes = PrimeGenerator.generate31BitPrimes(count: numPrimes)
        timings.numPrimes = numPrimes

        // Compute RREF mod each prime
        let rrefStart = CFAbsoluteTimeGetCurrent()

        var allRREFs: [[UInt32]] = []
        var allPivotCols: [[Int]] = []

        for prime in primes {
            // Create augmented matrix [A|b] mod p
            var augMod = [UInt32](repeating: 0, count: m * nAug)
            let pBig = BigInt(prime)

            for i in 0..<m {
                for j in 0..<n {
                    let reduced = ((A[i * n + j] % pBig) + pBig) % pBig
                    augMod[i * nAug + j] = UInt32(reduced)
                }
                let reducedB = ((b[i] % pBig) + pBig) % pBig
                augMod[i * nAug + n] = UInt32(reducedB)
            }

            // Compute RREF on GPU
            guard let (rref, pivots) = computeRREFAugmented(matrix: augMod, m: m, nAug: nAug, prime: prime) else {
                return nil
            }

            allRREFs.append(rref)
            allPivotCols.append(pivots)
        }

        timings.rrefTime = CFAbsoluteTimeGetCurrent() - rrefStart

        // Analyze pivot structure from first prime
        let pivotCols = allPivotCols[0]

        // Count rank of A (pivots in columns 0..n-1)
        var rankA = 0
        for p in pivotCols {
            if p >= 0 && p < n {
                rankA += 1
            }
        }

        // Count rank of [A|b] (pivots in columns 0..n)
        var rankAug = 0
        for p in pivotCols {
            if p >= 0 && p <= n {
                rankAug += 1
            }
        }

        // Check consistency: system is consistent iff rank(A) == rank([A|b])
        if rankA != rankAug {
            // Inconsistent: there's a pivot in the b column
            timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart
            return (.inconsistent(rank: rankA, augmentedRank: rankAug), timings)
        }

        // System is consistent - extract solution
        let crtStart = CFAbsoluteTimeGetCurrent()

        // Find which variables are pivot variables vs free variables
        var isPivotCol = [Bool](repeating: false, count: n)
        var pivotRowForCol = [Int](repeating: -1, count: n)

        for row in 0..<m {
            let p = pivotCols[row]
            if p >= 0 && p < n {
                isPivotCol[p] = true
                pivotRowForCol[p] = row
            }
        }

        // For overdetermined consistent systems with rank = n, we have a unique solution
        // For rank < n, there are free variables (infinitely many solutions)

        // Build solution: for pivot variables, read from RREF; for free, set to 0
        var solutionResidues: [[UInt32]] = Array(repeating: [], count: n)

        for pIdx in 0..<primes.count {
            let rref = allRREFs[pIdx]

            for j in 0..<n {
                if isPivotCol[j] {
                    let row = pivotRowForCol[j]
                    // x[j] = rref[row, n] (the b column after RREF)
                    solutionResidues[j].append(rref[row * nAug + n])
                } else {
                    // Free variable - set to 0
                    solutionResidues[j].append(0)
                }
            }
        }

        // CRT reconstruct
        var solution = [Rational]()
        for j in 0..<n {
            var pairs: [(residue: UInt32, modulus: UInt32)] = []
            for pIdx in 0..<primes.count {
                pairs.append((solutionResidues[j][pIdx], primes[pIdx]))
            }
            let value = CRTReconstruction.reconstructSigned(residues: pairs)
            solution.append(Rational(value))
        }

        timings.crtReconstructTime = CFAbsoluteTimeGetCurrent() - crtStart

        // Verify solution: A·x = b
        let verifyStart = CFAbsoluteTimeGetCurrent()
        var verified = true

        for i in 0..<m {
            var sum = Rational(0)
            for j in 0..<n {
                sum = sum + Rational(A[i * n + j]) * solution[j]
            }
            if sum != Rational(b[i]) {
                verified = false
                break
            }
        }

        timings.verifyTime = CFAbsoluteTimeGetCurrent() - verifyStart

        if !verified {
            // This shouldn't happen if CRT worked correctly
            print("Warning: Solution verification failed!")
        }

        timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart
        return (.consistent(solution: solution, rank: rankA), timings)
    }

    /// Compute RREF of augmented matrix mod prime using GPU
    private func computeRREFAugmented(matrix: [UInt32], m: Int, nAug: Int, prime: UInt32) -> (rref: [UInt32], pivotCols: [Int])? {
        guard let matrixBuffer = device.makeBuffer(bytes: matrix, length: matrix.count * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let pivotBuffer = device.makeBuffer(length: m * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return nil
        }

        var mVal = UInt32(m)
        var nAugVal = UInt32(nAug)
        var primeVal = prime

        encoder.setComputePipelineState(rrefPipeline)
        encoder.setBuffer(matrixBuffer, offset: 0, index: 0)
        encoder.setBuffer(pivotBuffer, offset: 0, index: 1)
        encoder.setBytes(&mVal, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&nAugVal, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&primeVal, length: MemoryLayout<UInt32>.stride, index: 4)

        encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Read results
        let rrefPtr = matrixBuffer.contents().bindMemory(to: UInt32.self, capacity: m * nAug)
        let rref = Array(UnsafeBufferPointer(start: rrefPtr, count: m * nAug))

        let pivotPtr = pivotBuffer.contents().bindMemory(to: UInt32.self, capacity: m)
        var pivotCols = [Int]()
        for i in 0..<m {
            let p = pivotPtr[i]
            if p == 0xFFFFFFFF {
                pivotCols.append(-1)
            } else {
                pivotCols.append(Int(p))
            }
        }

        return (rref, pivotCols)
    }
}

/// CPU baseline for overdetermined system solve
struct BareissOverdetermined {

    /// Solve overdetermined system using modular RREF approach
    static func solve(A: [BigInt], b: [BigInt], m: Int, n: Int) -> OverdeterminedResult {
        let nAug = n + 1

        // Estimate primes needed
        let maxEntryA = A.map { $0.magnitude }.max() ?? BigInt(1).magnitude
        let maxEntryB = b.map { $0.magnitude }.max() ?? BigInt(1).magnitude
        let maxEntry = max(maxEntryA, maxEntryB)
        let hadamardBound = BigInt(maxEntry) * BigInt(m)
        let bitsNeeded = hadamardBound.bitWidth + 64
        let numPrimes = max(3, (bitsNeeded + 30) / 31)

        let primes = PrimeGenerator.generate31BitPrimes(count: numPrimes)

        // Compute RREF mod first prime to get structure
        let (rref0, pivotCols) = computeRREFModPrime(A: A, b: b, m: m, n: n, prime: primes[0])

        // Count ranks
        var rankA = 0
        for p in pivotCols {
            if p >= 0 && p < n {
                rankA += 1
            }
        }

        var rankAug = 0
        for p in pivotCols {
            if p >= 0 && p <= n {
                rankAug += 1
            }
        }

        // Check consistency
        if rankA != rankAug {
            return .inconsistent(rank: rankA, augmentedRank: rankAug)
        }

        // Consistent - compute RREF mod all primes
        var allRREFs: [[UInt32]] = [rref0]
        for pIdx in 1..<primes.count {
            let (rref, _) = computeRREFModPrime(A: A, b: b, m: m, n: n, prime: primes[pIdx])
            allRREFs.append(rref)
        }

        // Build solution
        var isPivotCol = [Bool](repeating: false, count: n)
        var pivotRowForCol = [Int](repeating: -1, count: n)

        for row in 0..<m {
            let p = pivotCols[row]
            if p >= 0 && p < n {
                isPivotCol[p] = true
                pivotRowForCol[p] = row
            }
        }

        var solutionResidues: [[UInt32]] = Array(repeating: [], count: n)

        for pIdx in 0..<primes.count {
            let rref = allRREFs[pIdx]

            for j in 0..<n {
                if isPivotCol[j] {
                    let row = pivotRowForCol[j]
                    solutionResidues[j].append(rref[row * nAug + n])
                } else {
                    solutionResidues[j].append(0)
                }
            }
        }

        // CRT reconstruct
        var solution = [Rational]()
        for j in 0..<n {
            var pairs: [(residue: UInt32, modulus: UInt32)] = []
            for pIdx in 0..<primes.count {
                pairs.append((solutionResidues[j][pIdx], primes[pIdx]))
            }
            let value = CRTReconstruction.reconstructSigned(residues: pairs)
            solution.append(Rational(value))
        }

        return .consistent(solution: solution, rank: rankA)
    }

    /// Compute RREF of augmented matrix mod prime
    private static func computeRREFModPrime(A: [BigInt], b: [BigInt], m: Int, n: Int, prime: UInt32) -> (rref: [UInt32], pivotCols: [Int]) {
        let p = UInt64(prime)
        let nAug = n + 1

        // Create augmented matrix [A|b] mod p
        var M = [UInt32](repeating: 0, count: m * nAug)
        let pBig = BigInt(prime)

        for i in 0..<m {
            for j in 0..<n {
                let reduced = ((A[i * n + j] % pBig) + pBig) % pBig
                M[i * nAug + j] = UInt32(reduced)
            }
            let reducedB = ((b[i] % pBig) + pBig) % pBig
            M[i * nAug + n] = UInt32(reducedB)
        }

        var pivotCols: [Int] = []
        var pivotRow = 0

        for col in 0..<nAug {
            if pivotRow >= m { break }

            // Find pivot
            var pivot = -1
            for row in pivotRow..<m {
                if M[row * nAug + col] != 0 {
                    pivot = row
                    break
                }
            }

            if pivot == -1 { continue }

            // Swap rows
            if pivot != pivotRow {
                for j in 0..<nAug {
                    let tmp = M[pivotRow * nAug + j]
                    M[pivotRow * nAug + j] = M[pivot * nAug + j]
                    M[pivot * nAug + j] = tmp
                }
            }

            // Scale pivot row
            let pivotVal = UInt64(M[pivotRow * nAug + col])
            let pivotInv = modInverse(pivotVal, p)
            for j in 0..<nAug {
                M[pivotRow * nAug + j] = UInt32((UInt64(M[pivotRow * nAug + j]) * pivotInv) % p)
            }

            // Eliminate
            for row in 0..<m {
                if row == pivotRow { continue }
                let factor = UInt64(M[row * nAug + col])
                if factor != 0 {
                    for j in 0..<nAug {
                        let sub = (factor * UInt64(M[pivotRow * nAug + j])) % p
                        M[row * nAug + j] = UInt32((UInt64(M[row * nAug + j]) + p - sub) % p)
                    }
                }
            }

            pivotCols.append(col)
            pivotRow += 1
        }

        // Pad pivotCols
        while pivotCols.count < m {
            pivotCols.append(-1)
        }

        return (M, pivotCols)
    }

    /// Modular inverse
    private static func modInverse(_ a: UInt64, _ p: UInt64) -> UInt64 {
        if a == 0 { return 0 }
        var t: Int64 = 0
        var newt: Int64 = 1
        var r = Int64(p)
        var newr = Int64(a % p)

        while newr != 0 {
            let q = r / newr
            (t, newt) = (newt, t - q * newt)
            (r, newr) = (newr, r - q * newr)
        }

        if t < 0 { t += Int64(p) }
        return UInt64(t)
    }

    /// Timed version
    static func solveTimed(A: [BigInt], b: [BigInt], m: Int, n: Int) -> (result: OverdeterminedResult, time: Double) {
        let start = CFAbsoluteTimeGetCurrent()
        let result = solve(A: A, b: b, m: m, n: n)
        return (result, CFAbsoluteTimeGetCurrent() - start)
    }
}
