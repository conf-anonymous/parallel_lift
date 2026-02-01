import BigInt
import Foundation
import Metal

/// Timing breakdown for nullspace computation
struct NullspaceTimings {
    var totalTime: Double = 0
    var gpuReduceTime: Double = 0      // GPU modular RREF computation
    var crtReconstructTime: Double = 0  // CRT reconstruction of basis vectors
    var verifyTime: Double = 0          // Verification time
    var numPrimes: Int = 0
    var nullity: Int = 0                // Dimension of nullspace (n - rank)
}

/// Result of nullspace computation
struct NullspaceResult {
    let basis: [[Rational]]       // Each inner array is a basis vector
    let nullity: Int              // dim(ker(A)) = n - rank(A)
    let rank: Int                 // rank(A)
    let verified: Bool            // Whether A·v = 0 was verified for all v
}

/// GPU-accelerated exact nullspace computation
/// Computes a basis for ker(A) = {x : Ax = 0}
class GPUNullspace {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let rrefPipeline: MTLComputePipelineState

    /// Metal shader for RREF computation
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

    // Compute RREF and identify pivot columns
    // Output: rref matrix + pivot column indicators
    kernel void modular_rref(
        device uint* matrix,           // Input/output: m×n matrix (row-major)
        device uint* pivotCols,        // Output: for each row, the pivot column (-1 if none)
        constant uint& m,              // Number of rows
        constant uint& n,              // Number of columns
        constant uint& prime,          // Modulus
        uint tid [[thread_position_in_grid]]
    ) {
        // Each thread handles one prime (for batch processing)
        // For now, single-threaded RREF per prime
        if (tid != 0) return;

        uint p = prime;
        int pivotRow = 0;

        // Initialize pivot columns to -1 (no pivot)
        for (uint i = 0; i < m; i++) {
            pivotCols[i] = 0xFFFFFFFF;  // -1 as unsigned
        }

        for (uint col = 0; col < n && uint(pivotRow) < m; col++) {
            // Find pivot in this column
            int pivot = -1;
            for (uint row = uint(pivotRow); row < m; row++) {
                if (matrix[row * n + col] != 0) {
                    pivot = int(row);
                    break;
                }
            }

            if (pivot == -1) continue;  // No pivot in this column (free variable)

            // Swap rows if needed
            if (pivot != pivotRow) {
                for (uint j = 0; j < n; j++) {
                    uint tmp = matrix[pivotRow * n + j];
                    matrix[pivotRow * n + j] = matrix[pivot * n + j];
                    matrix[pivot * n + j] = tmp;
                }
            }

            // Scale pivot row to make pivot = 1
            uint pivotVal = matrix[pivotRow * n + col];
            uint pivotInv = mod_inverse(pivotVal, p);
            for (uint j = 0; j < n; j++) {
                matrix[pivotRow * n + j] = (uint64_t(matrix[pivotRow * n + j]) * pivotInv) % p;
            }

            // Eliminate all other entries in this column (full RREF)
            for (uint row = 0; row < m; row++) {
                if (row == uint(pivotRow)) continue;
                uint factor = matrix[row * n + col];
                if (factor != 0) {
                    for (uint j = 0; j < n; j++) {
                        uint sub = (uint64_t(factor) * matrix[pivotRow * n + j]) % p;
                        matrix[row * n + j] = (matrix[row * n + j] + p - sub) % p;
                    }
                }
            }

            // Record pivot column for this row
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

        // Compile shader
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: GPUNullspace.shaderSource, options: nil)
        } catch {
            print("GPUNullspace: Failed to compile shader: \(error)")
            return nil
        }

        guard let rrefFunc = library.makeFunction(name: "modular_rref") else {
            print("GPUNullspace: Failed to find modular_rref function")
            return nil
        }

        do {
            self.rrefPipeline = try device.makeComputePipelineState(function: rrefFunc)
        } catch {
            print("GPUNullspace: Failed to create pipeline: \(error)")
            return nil
        }
    }

    /// Compute a basis for the nullspace (kernel) of matrix A
    /// - Parameters:
    ///   - A: Flattened m×n matrix in row-major order
    ///   - m: Number of rows
    ///   - n: Number of columns
    /// - Returns: NullspaceResult with basis vectors, or nil on failure
    func computeNullspace(A: [BigInt], m: Int, n: Int) -> (result: NullspaceResult, timings: NullspaceTimings)? {
        var timings = NullspaceTimings()
        let totalStart = CFAbsoluteTimeGetCurrent()

        // Estimate number of primes needed
        // Nullspace vectors have entries bounded by Hadamard bound
        let maxEntry = A.map { $0.magnitude }.max() ?? BigInt(1).magnitude
        let hadamardBound = BigInt(maxEntry) * BigInt(m)  // Rough estimate
        let bitsNeeded = hadamardBound.bitWidth + 64  // Extra margin
        let numPrimes = max(3, (bitsNeeded + 30) / 31)

        let primes = PrimeGenerator.generate31BitPrimes(count: numPrimes)
        timings.numPrimes = numPrimes

        // Compute RREF modulo each prime
        let gpuStart = CFAbsoluteTimeGetCurrent()

        // We'll use the first prime's RREF to identify structure,
        // then verify/refine with others
        var allRREFs: [[UInt32]] = []
        var allPivotCols: [[Int]] = []

        for prime in primes {
            // Reduce matrix mod p
            var matrixMod = [UInt32](repeating: 0, count: m * n)
            let pBig = BigInt(prime)
            for i in 0..<(m * n) {
                let reduced = ((A[i] % pBig) + pBig) % pBig
                matrixMod[i] = UInt32(reduced)
            }

            // Compute RREF on GPU
            guard let (rref, pivots) = computeRREFModular(matrix: matrixMod, m: m, n: n, prime: prime) else {
                return nil
            }

            allRREFs.append(rref)
            allPivotCols.append(pivots)
        }

        timings.gpuReduceTime = CFAbsoluteTimeGetCurrent() - gpuStart

        // Identify pivot and free columns from first prime (should be consistent)
        let pivotCols = allPivotCols[0]
        var isPivotCol = [Bool](repeating: false, count: n)
        var rank = 0
        for p in pivotCols {
            if p >= 0 && p < n {
                isPivotCol[p] = true
                rank += 1
            }
        }

        let nullity = n - rank
        timings.nullity = nullity

        if nullity == 0 {
            // Full rank - trivial nullspace
            timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart
            return (NullspaceResult(basis: [], nullity: 0, rank: rank, verified: true), timings)
        }

        // Find free columns
        var freeCols: [Int] = []
        for j in 0..<n {
            if !isPivotCol[j] {
                freeCols.append(j)
            }
        }

        // Construct nullspace basis vectors using CRT
        let crtStart = CFAbsoluteTimeGetCurrent()
        var basisVectors: [[Rational]] = []

        for freeCol in freeCols {
            // For each free column, construct a basis vector
            // Set free variable to 1, others to 0, solve for pivot variables

            // Collect residues for each component across primes
            var componentResidues: [[UInt32]] = Array(repeating: [], count: n)

            for pIdx in 0..<primes.count {
                let rref = allRREFs[pIdx]
                let prime = primes[pIdx]

                // Construct basis vector mod p
                var vec = [UInt32](repeating: 0, count: n)
                vec[freeCol] = 1  // Set free variable to 1

                // For each pivot row, solve for the pivot variable
                for row in 0..<rank {
                    let pivotCol = allPivotCols[pIdx][row]
                    if pivotCol < 0 || pivotCol >= n { continue }

                    // From RREF: x[pivotCol] + sum(rref[row,j] * x[j] for free j) = 0
                    // So x[pivotCol] = -rref[row, freeCol] (since x[freeCol] = 1, others = 0)
                    let coeff = rref[row * n + freeCol]
                    vec[pivotCol] = (prime - coeff) % prime  // Negate mod p
                }

                for j in 0..<n {
                    componentResidues[j].append(vec[j])
                }
            }

            // CRT reconstruct each component
            var basisVec = [Rational]()
            for j in 0..<n {
                var pairs: [(residue: UInt32, modulus: UInt32)] = []
                for pIdx in 0..<primes.count {
                    pairs.append((componentResidues[j][pIdx], primes[pIdx]))
                }
                let value = CRTReconstruction.reconstructSigned(residues: pairs)
                basisVec.append(Rational(value))
            }

            basisVectors.append(basisVec)
        }

        timings.crtReconstructTime = CFAbsoluteTimeGetCurrent() - crtStart

        // Verify: A · v = 0 for each basis vector
        let verifyStart = CFAbsoluteTimeGetCurrent()
        var verified = true

        for vec in basisVectors {
            for i in 0..<m {
                // Compute (A · v)[i] = sum_j A[i,j] * v[j]
                var sum = Rational(0)
                for j in 0..<n {
                    let aij = Rational(A[i * n + j])
                    sum = sum + aij * vec[j]
                }
                if !sum.isZero {
                    verified = false
                    break
                }
            }
            if !verified { break }
        }

        timings.verifyTime = CFAbsoluteTimeGetCurrent() - verifyStart
        timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart

        return (NullspaceResult(
            basis: basisVectors,
            nullity: nullity,
            rank: rank,
            verified: verified
        ), timings)
    }

    /// Compute RREF of matrix mod prime using GPU
    private func computeRREFModular(matrix: [UInt32], m: Int, n: Int, prime: UInt32) -> (rref: [UInt32], pivotCols: [Int])? {
        // Create buffers
        guard let matrixBuffer = device.makeBuffer(bytes: matrix, length: matrix.count * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let pivotBuffer = device.makeBuffer(length: m * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return nil
        }

        var mVal = UInt32(m)
        var nVal = UInt32(n)
        var primeVal = prime

        encoder.setComputePipelineState(rrefPipeline)
        encoder.setBuffer(matrixBuffer, offset: 0, index: 0)
        encoder.setBuffer(pivotBuffer, offset: 0, index: 1)
        encoder.setBytes(&mVal, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&primeVal, length: MemoryLayout<UInt32>.stride, index: 4)

        // Single thread for now (RREF is inherently sequential per matrix)
        encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Read results
        let rrefPtr = matrixBuffer.contents().bindMemory(to: UInt32.self, capacity: m * n)
        let rref = Array(UnsafeBufferPointer(start: rrefPtr, count: m * n))

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

/// CPU baseline for nullspace computation using modular arithmetic
/// This is much faster than exact Rational RREF
struct BareissNullspace {

    /// Compute nullspace basis using modular approach (same as GPU but on CPU)
    static func compute(A: [BigInt], m: Int, n: Int) -> NullspaceResult {
        // Use modular arithmetic like GPU does - much faster than exact rationals
        // Compute RREF mod several primes, then CRT reconstruct

        // Estimate primes needed
        let maxEntry = A.map { $0.magnitude }.max() ?? BigInt(1).magnitude
        let hadamardBound = BigInt(maxEntry) * BigInt(m)
        let bitsNeeded = hadamardBound.bitWidth + 64
        let numPrimes = max(3, (bitsNeeded + 30) / 31)

        let primes = PrimeGenerator.generate31BitPrimes(count: numPrimes)

        // Compute RREF mod first prime to get structure
        let (rref0, pivotCols) = computeRREFModPrime(A: A, m: m, n: n, prime: primes[0])

        // Identify pivot and free columns
        var isPivotCol = [Bool](repeating: false, count: n)
        var rank = 0
        for p in pivotCols {
            if p >= 0 && p < n {
                isPivotCol[p] = true
                rank += 1
            }
        }

        let nullity = n - rank

        if nullity == 0 {
            return NullspaceResult(basis: [], nullity: 0, rank: rank, verified: true)
        }

        // Find free columns
        var freeCols: [Int] = []
        for j in 0..<n {
            if !isPivotCol[j] {
                freeCols.append(j)
            }
        }

        // Compute RREF mod all primes for CRT
        var allRREFs: [[UInt32]] = [rref0]
        for pIdx in 1..<primes.count {
            let (rref, _) = computeRREFModPrime(A: A, m: m, n: n, prime: primes[pIdx])
            allRREFs.append(rref)
        }

        // Construct basis vectors using CRT
        var basisVectors: [[Rational]] = []

        for freeCol in freeCols {
            var componentResidues: [[UInt32]] = Array(repeating: [], count: n)

            for pIdx in 0..<primes.count {
                let rref = allRREFs[pIdx]
                let prime = primes[pIdx]

                var vec = [UInt32](repeating: 0, count: n)
                vec[freeCol] = 1

                for row in 0..<rank {
                    let pivotCol = pivotCols[row]
                    if pivotCol < 0 || pivotCol >= n { continue }
                    let coeff = rref[row * n + freeCol]
                    vec[pivotCol] = (prime - coeff) % prime
                }

                for j in 0..<n {
                    componentResidues[j].append(vec[j])
                }
            }

            var basisVec = [Rational]()
            for j in 0..<n {
                var pairs: [(residue: UInt32, modulus: UInt32)] = []
                for pIdx in 0..<primes.count {
                    pairs.append((componentResidues[j][pIdx], primes[pIdx]))
                }
                let value = CRTReconstruction.reconstructSigned(residues: pairs)
                basisVec.append(Rational(value))
            }

            basisVectors.append(basisVec)
        }

        // Verify
        var verified = true
        for vec in basisVectors {
            for i in 0..<m {
                var sum = Rational(0)
                for j in 0..<n {
                    sum = sum + Rational(A[i * n + j]) * vec[j]
                }
                if !sum.isZero {
                    verified = false
                    break
                }
            }
        }

        return NullspaceResult(basis: basisVectors, nullity: nullity, rank: rank, verified: verified)
    }

    /// Compute RREF mod a single prime (CPU version)
    private static func computeRREFModPrime(A: [BigInt], m: Int, n: Int, prime: UInt32) -> (rref: [UInt32], pivotCols: [Int]) {
        let p = UInt64(prime)

        // Reduce matrix mod p
        var M = [UInt32](repeating: 0, count: m * n)
        let pBig = BigInt(prime)
        for i in 0..<(m * n) {
            let reduced = ((A[i] % pBig) + pBig) % pBig
            M[i] = UInt32(reduced)
        }

        var pivotCols: [Int] = []
        var pivotRow = 0

        for col in 0..<n {
            if pivotRow >= m { break }

            // Find pivot
            var pivot = -1
            for row in pivotRow..<m {
                if M[row * n + col] != 0 {
                    pivot = row
                    break
                }
            }

            if pivot == -1 { continue }

            // Swap rows
            if pivot != pivotRow {
                for j in 0..<n {
                    let tmp = M[pivotRow * n + j]
                    M[pivotRow * n + j] = M[pivot * n + j]
                    M[pivot * n + j] = tmp
                }
            }

            // Scale pivot row
            let pivotVal = UInt64(M[pivotRow * n + col])
            let pivotInv = modInverse(pivotVal, p)
            for j in 0..<n {
                M[pivotRow * n + j] = UInt32((UInt64(M[pivotRow * n + j]) * pivotInv) % p)
            }

            // Eliminate (full RREF)
            for row in 0..<m {
                if row == pivotRow { continue }
                let factor = UInt64(M[row * n + col])
                if factor != 0 {
                    for j in 0..<n {
                        let sub = (factor * UInt64(M[pivotRow * n + j])) % p
                        M[row * n + j] = UInt32((UInt64(M[row * n + j]) + p - sub) % p)
                    }
                }
            }

            pivotCols.append(col)
            pivotRow += 1
        }

        // Pad pivotCols to m entries
        while pivotCols.count < m {
            pivotCols.append(-1)
        }

        return (M, pivotCols)
    }

    /// Modular inverse using extended Euclidean algorithm
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
    static func computeTimed(A: [BigInt], m: Int, n: Int) -> (result: NullspaceResult, time: Double) {
        let start = CFAbsoluteTimeGetCurrent()
        let result = compute(A: A, m: m, n: n)
        return (result, CFAbsoluteTimeGetCurrent() - start)
    }
}
