import BigInt
import Foundation
import Metal

/// Timing breakdown for multi-RHS solve
struct MultiRHSSolveTimings {
    var totalTime: Double = 0
    var gpuSolveTime: Double = 0    // Time for GPU modular solve (elim + backsub)
    var detATime: Double = 0        // Time to compute det(A) for denominator
    var crtReconstructTime: Double = 0  // Time for CRT reconstruction
    var numPrimes: Int = 0
    var numRHS: Int = 0

    /// For convenience, report total "factor" time as GPU solve + det(A)
    var factorTime: Double {
        return gpuSolveTime + detATime
    }
}

/// GPU-accelerated multi-RHS exact linear system solve
/// Solves AX = B where A is n×n, B is n×k, X is n×k
/// Factorizes A once and applies to all k right-hand sides
class GPUMultiRHSSolve {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let multiRHSPipeline: MTLComputePipelineState

    /// Metal shader source code for multi-RHS solve
    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    inline uint32_t solve_mod_sub(uint32_t a, uint32_t b, uint32_t p) {
        return a >= b ? (a - b) : (p - b + a);
    }

    inline uint32_t solve_mod_mul(uint32_t a, uint32_t b, uint32_t p) {
        uint64_t prod = uint64_t(a) * uint64_t(b);
        return uint32_t(prod % uint64_t(p));
    }

    inline uint32_t solve_mod_inv(uint32_t a, uint32_t p) {
        if (a == 0) return 0;
        int64_t t = 0, newt = 1;
        int64_t r = int64_t(p), newr = int64_t(a);
        while (newr != 0) {
            int64_t quotient = r / newr;
            int64_t temp_t = t - quotient * newt;
            t = newt; newt = temp_t;
            int64_t temp_r = r - quotient * newr;
            r = newr; newr = temp_r;
        }
        if (t < 0) t += int64_t(p);
        return uint32_t(t);
    }

    kernel void modular_solve_multi_rhs(
        device const uint32_t* augmented [[buffer(0)]],
        device const uint32_t* primes [[buffer(1)]],
        device uint32_t* solutions [[buffer(2)]],
        device uint32_t* singular_flags [[buffer(3)]],
        constant uint32_t& n [[buffer(4)]],
        constant uint32_t& k [[buffer(5)]],
        constant uint32_t& num_primes [[buffer(6)]],
        device uint32_t* workspace [[buffer(7)]],
        uint tid [[thread_position_in_grid]])
    {
        if (tid >= num_primes) return;

        uint32_t p = primes[tid];
        bool singular = false;

        uint32_t aug_width = n + k;
        uint32_t stride = n * aug_width;

        device const uint32_t* myAug = augmented + tid * stride;
        device uint32_t* A = workspace + tid * stride;

        for (uint32_t i = 0; i < stride; i++) {
            A[i] = myAug[i];
        }

        // Forward elimination with partial pivoting
        for (uint32_t col = 0; col < n && !singular; col++) {
            uint32_t pivot_row = col;
            for (uint32_t row = col; row < n; row++) {
                if (A[row * aug_width + col] != 0) {
                    pivot_row = row;
                    break;
                }
            }

            if (A[pivot_row * aug_width + col] == 0) {
                singular = true;
                break;
            }

            if (pivot_row != col) {
                for (uint32_t j = col; j < aug_width; j++) {
                    uint32_t tmp = A[col * aug_width + j];
                    A[col * aug_width + j] = A[pivot_row * aug_width + j];
                    A[pivot_row * aug_width + j] = tmp;
                }
            }

            uint32_t pivot = A[col * aug_width + col];
            uint32_t pivot_inv = solve_mod_inv(pivot, p);

            for (uint32_t j = col; j < aug_width; j++) {
                A[col * aug_width + j] = solve_mod_mul(A[col * aug_width + j], pivot_inv, p);
            }

            for (uint32_t row = col + 1; row < n; row++) {
                uint32_t factor = A[row * aug_width + col];
                if (factor != 0) {
                    for (uint32_t j = col; j < aug_width; j++) {
                        uint32_t sub = solve_mod_mul(factor, A[col * aug_width + j], p);
                        A[row * aug_width + j] = solve_mod_sub(A[row * aug_width + j], sub, p);
                    }
                }
            }
        }

        singular_flags[tid] = singular ? 1 : 0;

        if (singular) {
            for (uint32_t i = 0; i < n * k; i++) {
                solutions[tid * n * k + i] = 0;
            }
            return;
        }

        device uint32_t* X = solutions + tid * n * k;

        for (uint32_t rhs = 0; rhs < k; rhs++) {
            for (int32_t row = int32_t(n) - 1; row >= 0; row--) {
                uint32_t sum = A[row * aug_width + (n + rhs)];

                for (uint32_t j = row + 1; j < n; j++) {
                    uint32_t sub = solve_mod_mul(A[row * aug_width + j], X[rhs * n + j], p);
                    sum = solve_mod_sub(sum, sub, p);
                }
                X[rhs * n + row] = sum;
            }
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

        // Compile shader from source
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: GPUMultiRHSSolve.shaderSource, options: nil)
        } catch {
            print("GPUMultiRHSSolve: Failed to compile Metal shader: \(error)")
            return nil
        }

        guard let multiRHSFunc = library.makeFunction(name: "modular_solve_multi_rhs") else {
            print("GPUMultiRHSSolve: Failed to find modular_solve_multi_rhs kernel")
            return nil
        }

        do {
            self.multiRHSPipeline = try device.makeComputePipelineState(function: multiRHSFunc)
        } catch {
            print("GPUMultiRHSSolve: Failed to create pipeline: \(error)")
            return nil
        }
    }

    /// Solve AX = B exactly using GPU + CRT
    /// - Parameters:
    ///   - A: Flattened n×n matrix in row-major order
    ///   - B: Flattened n×k matrix in column-major order (k columns of n elements each)
    ///   - n: Matrix dimension
    ///   - k: Number of right-hand side vectors
    /// - Returns: Solution matrix X as rationals (n×k), or nil if singular
    func solve(A: [BigInt], B: [BigInt], n: Int, k: Int) -> (solution: [[Rational]]?, timings: MultiRHSSolveTimings)? {
        var timings = MultiRHSSolveTimings()
        timings.numRHS = k
        let totalStart = CFAbsoluteTimeGetCurrent()

        // Estimate number of primes needed
        // Solution entries can be roughly det(A_i)/det(A) which have similar bit growth
        let maxEntry = A.map { $0.magnitude }.max() ?? BigInt.Magnitude(1)
        let entryBits = maxEntry.bitWidth
        let maxBEntry = B.map { $0.magnitude }.max() ?? BigInt.Magnitude(1)
        let bBits = maxBEntry.bitWidth
        let estimatedBits = (n + 1) * max(entryBits, bBits) + 2 * n + 64

        let primes = PrimeGenerator.generate31BitPrimes(count: (estimatedBits + 29) / 30)
        timings.numPrimes = primes.count

        // Create augmented matrices [A|B] for each prime
        // Layout: for each prime p, store n×(n+k) matrix [A mod p | B mod p] in row-major
        let augWidth = n + k
        let augSize = n * augWidth
        var augmentedData = [UInt32](repeating: 0, count: primes.count * augSize)

        for pIdx in 0..<primes.count {
            let p = primes[pIdx]
            let pBig = BigInt(p)
            let offset = pIdx * augSize

            for row in 0..<n {
                // Copy A columns
                for col in 0..<n {
                    let val = A[row * n + col]
                    var reduced = val % pBig
                    if reduced < 0 { reduced += pBig }
                    augmentedData[offset + row * augWidth + col] = UInt32(reduced)
                }
                // Copy B columns
                for col in 0..<k {
                    let val = B[col * n + row]  // B is column-major
                    var reduced = val % pBig
                    if reduced < 0 { reduced += pBig }
                    augmentedData[offset + row * augWidth + (n + col)] = UInt32(reduced)
                }
            }
        }

        // Create Metal buffers
        guard let augBuffer = device.makeBuffer(bytes: augmentedData, length: augmentedData.count * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let primesBuffer = device.makeBuffer(bytes: primes, length: primes.count * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let solutionsBuffer = device.makeBuffer(length: primes.count * n * k * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let singularBuffer = device.makeBuffer(length: primes.count * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let workspaceBuffer = device.makeBuffer(length: primes.count * augSize * MemoryLayout<UInt32>.stride, options: .storageModeShared) else {
            return nil
        }

        var nVal = UInt32(n)
        var kVal = UInt32(k)
        var numPrimesVal = UInt32(primes.count)

        // Execute kernel (GPU modular solve: elimination + back-substitution)
        let gpuSolveStart = CFAbsoluteTimeGetCurrent()

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return nil
        }

        encoder.setComputePipelineState(multiRHSPipeline)
        encoder.setBuffer(augBuffer, offset: 0, index: 0)
        encoder.setBuffer(primesBuffer, offset: 0, index: 1)
        encoder.setBuffer(solutionsBuffer, offset: 0, index: 2)
        encoder.setBuffer(singularBuffer, offset: 0, index: 3)
        encoder.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&numPrimesVal, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBuffer(workspaceBuffer, offset: 0, index: 7)

        let threadgroupSize = MTLSize(width: min(64, multiRHSPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        let gridSize = MTLSize(width: primes.count, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        timings.gpuSolveTime = CFAbsoluteTimeGetCurrent() - gpuSolveStart

        // Check for singular matrices
        let singularPtr = singularBuffer.contents().bindMemory(to: UInt32.self, capacity: primes.count)
        for i in 0..<primes.count {
            if singularPtr[i] != 0 {
                // Matrix is singular mod this prime - might be truly singular
                // For simplicity, we'll consider it singular
                timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart
                return (nil, timings)
            }
        }

        // Compute det(A) - we need this as the common denominator
        let detAStart = CFAbsoluteTimeGetCurrent()
        guard let gpuDet = GPUDeterminant(),
              let (detA, _) = gpuDet.computeDeterminant(matrix: A, n: n) else {
            return nil
        }
        timings.detATime = CFAbsoluteTimeGetCurrent() - detAStart

        if detA == 0 {
            timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart
            return (nil, timings)
        }

        // CRT reconstruction for all elements of X using optimized batch algorithm
        let crtStart = CFAbsoluteTimeGetCurrent()
        let solutionsPtr = solutionsBuffer.contents().bindMemory(to: UInt32.self, capacity: primes.count * n * k)

        // The GPU solver uses Gaussian elimination with pivot normalization.
        // For rational solution x = num/den, the modular result is num * den^(-1) mod p.
        // We know den = det(A), so to recover num, we multiply modular result by det(A):
        //   num ≡ (modular_result) * det(A) (mod p)
        // After CRT reconstruction of (modular_result * det(A)), we get num directly.

        // Precompute det(A) mod each prime
        var detAResidues = [UInt32](repeating: 0, count: primes.count)
        for pIdx in 0..<primes.count {
            let pBig = BigInt(primes[pIdx])
            detAResidues[pIdx] = UInt32(((detA % pBig) + pBig) % pBig)
        }

        // Use optimized batch CRT reconstruction
        let numerators = CRTReconstruction.batchReconstructMultiRHS(
            residueData: solutionsPtr,
            moduli: primes,
            n: n,
            k: k,
            detAResidues: detAResidues
        )

        // Convert to Rational matrix format
        var solutionMatrix: [[Rational]] = Array(repeating: [], count: k)
        for rhsIdx in 0..<k {
            var column = [Rational]()
            column.reserveCapacity(n)
            for rowIdx in 0..<n {
                let num = numerators[rhsIdx * n + rowIdx]
                column.append(Rational(num, detA))
            }
            solutionMatrix[rhsIdx] = column
        }

        timings.crtReconstructTime = CFAbsoluteTimeGetCurrent() - crtStart
        timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart

        return (solutionMatrix, timings)
    }
}

/// CPU baseline for multi-RHS solve using Cramer's rule (naive - O(n^2 * k) determinants)
struct BareissMultiRHSSolve {

    /// Solve AX = B exactly using Cramer's rule
    /// - Parameters:
    ///   - A: Flattened n×n matrix
    ///   - B: Flattened n×k matrix (column-major)
    ///   - n: Matrix dimension
    ///   - k: Number of RHS vectors
    /// - Returns: Solution matrix X (k columns of n Rationals), or nil if singular
    static func solve(A: [BigInt], B: [BigInt], n: Int, k: Int) -> [[Rational]]? {
        // Compute det(A) once
        let detA = BareissDeterminant.compute(matrix: A, n: n)

        if detA == 0 {
            return nil
        }

        var solution: [[Rational]] = []

        // For each RHS column
        for rhsIdx in 0..<k {
            var column: [Rational] = []

            // Extract b vector for this RHS
            var b: [BigInt] = []
            for row in 0..<n {
                b.append(B[rhsIdx * n + row])
            }

            // Compute x_i = det(A_i) / det(A) for each i
            for i in 0..<n {
                // Create A_i: replace column i with b
                var Ai = A
                for row in 0..<n {
                    Ai[row * n + i] = b[row]
                }

                let detAi = BareissDeterminant.compute(matrix: Ai, n: n)
                column.append(Rational(detAi, detA))
            }

            solution.append(column)
        }

        return solution
    }

    /// Solve with timing
    static func solveTimed(A: [BigInt], B: [BigInt], n: Int, k: Int) -> (solution: [[Rational]]?, time: Double) {
        let start = CFAbsoluteTimeGetCurrent()
        let solution = solve(A: A, B: B, n: n, k: k)
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        return (solution, elapsed)
    }
}

/// CPU baseline using exact LU factorization (Bareiss) with reuse across RHS vectors
/// This is the "fair" CPU baseline that amortizes factorization like the GPU does
struct BareissLUMultiRHSSolve {

    /// Timing breakdown for LU-based solve
    struct LUTimings {
        var factorTime: Double = 0
        var solveTime: Double = 0
        var totalTime: Double = 0
    }

    /// Result of LU factorization using Bareiss algorithm
    /// Stores the transformed matrix and pivot information for back-substitution
    struct BareissLUFactorization {
        let U: [BigInt]          // Upper triangular part (row-echelon form)
        let pivots: [Int]        // Pivot row indices (for permutation)
        let detA: BigInt         // Determinant
        let n: Int
    }

    /// Compute exact LU factorization using Bareiss (fraction-free) algorithm
    /// Returns the row-echelon form and determinant
    static func factorize(A: [BigInt], n: Int) -> BareissLUFactorization? {
        var M = A  // Working copy
        var pivots = Array(0..<n)  // Track row permutations
        var det = BigInt(1)
        var prevPivot = BigInt(1)

        for col in 0..<n {
            // Find pivot
            var pivotRow = -1
            for row in col..<n {
                if M[pivots[row] * n + col] != 0 {
                    pivotRow = row
                    break
                }
            }

            if pivotRow == -1 {
                return nil  // Singular
            }

            // Swap pivot rows in tracking array
            if pivotRow != col {
                pivots.swapAt(col, pivotRow)
                det = -det  // Sign change for determinant
            }

            let pivot = M[pivots[col] * n + col]

            // Eliminate below pivot using Bareiss formula
            for row in (col + 1)..<n {
                let rowIdx = pivots[row]
                let factor = M[rowIdx * n + col]

                if factor != 0 {
                    for j in (col + 1)..<n {
                        // Bareiss: M[row][j] = (pivot * M[row][j] - factor * M[pivotRow][j]) / prevPivot
                        let newVal = (pivot * M[rowIdx * n + j] - factor * M[pivots[col] * n + j])
                        M[rowIdx * n + j] = newVal / prevPivot
                    }
                }
                M[rowIdx * n + col] = 0  // Eliminated
            }

            prevPivot = pivot
        }

        // Compute determinant (product of diagonal after Bareiss)
        det = det * M[pivots[n-1] * n + (n-1)]
        for i in 0..<(n-1) {
            det = det * M[pivots[i] * n + i] / (i == 0 ? BigInt(1) : M[pivots[i-1] * n + (i-1)])
        }

        // Actually, for Bareiss the final diagonal element is the determinant
        det = M[pivots[n-1] * n + (n-1)]
        if n > 1 {
            // Sign from permutations already tracked
        }

        return BareissLUFactorization(U: M, pivots: pivots, detA: det, n: n)
    }

    /// Apply the factorization to solve for one RHS vector
    /// Returns numerators (divide by det(A) for final answer)
    static func solveWithFactorization(_ factor: BareissLUFactorization, b: [BigInt]) -> [BigInt]? {
        let n = factor.n

        // Apply same row operations to b as were applied to A
        // We need to re-do the elimination on the augmented [A|b] to get the transformed b
        // This is because Bareiss doesn't store L explicitly

        // Actually, for Bareiss we need to redo the forward elimination on b
        // This still saves work vs Cramer's rule (O(n^2) vs O(n^3) per RHS)

        var bWork = b
        var M = factor.U
        var pivots = factor.pivots

        // Unfortunately, Bareiss doesn't give us a simple L that we can apply
        // We need to store the elimination multipliers or redo partial work
        // For a proper implementation, we'd store the L factors

        // Simpler approach: use the factorized U for back-substitution
        // and compute the transformed b by applying Cramer on the last column

        // For now, let's use a different approach: fraction-free back-sub on augmented system

        return nil  // TODO: implement properly
    }

    /// Solve AX = B using LU factorization (factor once, solve k times)
    /// This implementation uses rational arithmetic after factorization
    static func solve(A: [BigInt], B: [BigInt], n: Int, k: Int) -> (solution: [[Rational]]?, timings: LUTimings) {
        var timings = LUTimings()
        let totalStart = CFAbsoluteTimeGetCurrent()

        // Factorize A using Gaussian elimination with rational arithmetic
        let factorStart = CFAbsoluteTimeGetCurrent()

        // Convert to rational matrix and do LU decomposition
        var L = [[Rational]](repeating: [Rational](repeating: Rational(0), count: n), count: n)
        var U = [[Rational]](repeating: [Rational](repeating: Rational(0), count: n), count: n)
        var P = Array(0..<n)  // Permutation

        // Copy A to U (as rationals)
        for i in 0..<n {
            for j in 0..<n {
                U[i][j] = Rational(A[i * n + j])
            }
            L[i][i] = Rational(1)  // Diagonal of L is 1
        }

        // LU decomposition with partial pivoting
        for col in 0..<n {
            // Find pivot
            var maxRow = col
            var maxVal = U[P[col]][col].numerator.magnitude
            for row in (col + 1)..<n {
                let val = U[P[row]][col].numerator.magnitude
                if val > maxVal {
                    maxVal = val
                    maxRow = row
                }
            }

            if U[P[maxRow]][col].numerator == 0 {
                timings.factorTime = CFAbsoluteTimeGetCurrent() - factorStart
                timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart
                return (nil, timings)  // Singular
            }

            // Swap in permutation
            if maxRow != col {
                P.swapAt(col, maxRow)
            }

            let pivotRow = P[col]
            let pivot = U[pivotRow][col]

            // Eliminate below
            for row in (col + 1)..<n {
                let targetRow = P[row]
                let factor = U[targetRow][col] / pivot
                L[targetRow][col] = factor

                for j in col..<n {
                    U[targetRow][j] = U[targetRow][j] - factor * U[pivotRow][j]
                }
            }
        }

        timings.factorTime = CFAbsoluteTimeGetCurrent() - factorStart

        // Solve for each RHS
        let solveStart = CFAbsoluteTimeGetCurrent()
        var solution: [[Rational]] = []

        for rhsIdx in 0..<k {
            // Extract b vector
            var b = [Rational](repeating: Rational(0), count: n)
            for row in 0..<n {
                b[row] = Rational(B[rhsIdx * n + row])
            }

            // Apply permutation to b
            var pb = [Rational](repeating: Rational(0), count: n)
            for i in 0..<n {
                pb[i] = b[P[i]]
            }

            // Forward substitution: Ly = Pb
            var y = [Rational](repeating: Rational(0), count: n)
            for i in 0..<n {
                var sum = pb[i]
                for j in 0..<i {
                    sum = sum - L[P[i]][j] * y[j]
                }
                y[i] = sum
            }

            // Back substitution: Ux = y
            var x = [Rational](repeating: Rational(0), count: n)
            for i in stride(from: n - 1, through: 0, by: -1) {
                var sum = y[i]
                for j in (i + 1)..<n {
                    sum = sum - U[P[i]][j] * x[j]
                }
                x[i] = sum / U[P[i]][i]
            }

            solution.append(x)
        }

        timings.solveTime = CFAbsoluteTimeGetCurrent() - solveStart
        timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart

        return (solution, timings)
    }
}
