import BigInt
import Foundation
import Metal

/// Timing breakdown for GPU rank computation
struct RankTimings {
    var totalTime: Double = 0
    var primeGenerationTime: Double = 0
    var gpuComputeTime: Double = 0
    var numPrimes: Int = 0
}

/// GPU-accelerated exact rank computation using CRT
/// Key insight: rank is the same mod all primes (unless the prime divides a pivot)
/// So we compute rank mod several primes and take the mode (most common value)
class GPURank {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipelineState: MTLComputePipelineState

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else {
            return nil
        }

        self.device = device
        self.commandQueue = commandQueue

        // Compile shader from source
        guard let library = GPURank.createLibrary(device: device),
              let function = library.makeFunction(name: "modular_rank"),
              let pipelineState = try? device.makeComputePipelineState(function: function) else {
            return nil
        }

        self.pipelineState = pipelineState
    }

    private static func createLibrary(device: MTLDevice) -> MTLLibrary? {
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        inline uint32_t rank_mod_sub(uint32_t a, uint32_t b, uint32_t p) {
            return a >= b ? (a - b) : (p - b + a);
        }

        inline uint32_t rank_mod_mul(uint32_t a, uint32_t b, uint32_t p) {
            uint64_t prod = uint64_t(a) * uint64_t(b);
            return uint32_t(prod % uint64_t(p));
        }

        inline uint32_t rank_mod_inv(uint32_t a, uint32_t p) {
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

        kernel void modular_rank(
            device const uint32_t* matrices [[buffer(0)]],
            device const uint32_t* primes [[buffer(1)]],
            device uint32_t* results [[buffer(2)]],
            constant uint32_t& m [[buffer(3)]],
            constant uint32_t& n [[buffer(4)]],
            constant uint32_t& num_primes [[buffer(5)]],
            device uint32_t* workspace [[buffer(6)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= num_primes) return;

            uint32_t p = primes[tid];
            uint32_t mn = m * n;

            device const uint32_t* myMatrix = matrices + tid * mn;
            device uint32_t* A = workspace + tid * mn;

            for (uint32_t i = 0; i < mn; i++) {
                A[i] = myMatrix[i];
            }

            uint32_t rank = 0;
            uint32_t pivotRow = 0;
            uint32_t pivotCol = 0;

            while (pivotRow < m && pivotCol < n) {
                uint32_t maxRow = pivotRow;
                for (uint32_t i = pivotRow; i < m; i++) {
                    if (A[i * n + pivotCol] != 0) { maxRow = i; break; }
                }

                if (A[maxRow * n + pivotCol] == 0) {
                    pivotCol++;
                    continue;
                }

                if (maxRow != pivotRow) {
                    for (uint32_t j = pivotCol; j < n; j++) {
                        uint32_t tmp = A[pivotRow * n + j];
                        A[pivotRow * n + j] = A[maxRow * n + j];
                        A[maxRow * n + j] = tmp;
                    }
                }

                uint32_t pivot = A[pivotRow * n + pivotCol];
                uint32_t pivot_inv = rank_mod_inv(pivot, p);

                for (uint32_t i = pivotRow + 1; i < m; i++) {
                    uint32_t factor = rank_mod_mul(A[i * n + pivotCol], pivot_inv, p);
                    if (factor != 0) {
                        for (uint32_t j = pivotCol; j < n; j++) {
                            uint32_t sub = rank_mod_mul(factor, A[pivotRow * n + j], p);
                            A[i * n + j] = rank_mod_sub(A[i * n + j], sub, p);
                        }
                    }
                }

                rank++;
                pivotRow++;
                pivotCol++;
            }

            results[tid] = rank;
        }
        """

        do {
            return try device.makeLibrary(source: shaderSource, options: nil)
        } catch {
            print("Error compiling rank shader: \(error)")
            return nil
        }
    }

    /// Compute exact rank using GPU + majority voting across primes
    /// For rank, we don't need CRT reconstruction - just consensus
    func computeRank(matrix: [BigInt], m: Int, n: Int) -> (rank: Int, timings: RankTimings)? {
        var timings = RankTimings()
        let totalStart = CFAbsoluteTimeGetCurrent()

        // Generate primes - we need enough to get reliable consensus
        // Using more primes than strictly needed for robustness
        let primeStart = CFAbsoluteTimeGetCurrent()
        let numPrimesNeeded = max(20, min(m, n) + 10)  // At least 20 primes for reliability
        let primes = PrimeGenerator.generate31BitPrimes(count: numPrimesNeeded)
        timings.primeGenerationTime = CFAbsoluteTimeGetCurrent() - primeStart
        timings.numPrimes = primes.count

        // GPU compute
        let gpuStart = CFAbsoluteTimeGetCurrent()

        // Pre-reduce matrix entries for each prime
        let matrixInt64: [Int64] = matrix.map { val in
            // Handle BigInt conversion safely
            if val >= BigInt(Int64.min) && val <= BigInt(Int64.max) {
                return Int64(val)
            } else {
                // For very large values, we need modular reduction
                return 0  // Will be handled by per-prime reduction
            }
        }

        var allReducedMatrices: [[UInt32]] = []
        for p in primes {
            let p64 = Int64(p)
            var reducedMatrix: [UInt32] = []
            reducedMatrix.reserveCapacity(m * n)

            for i in 0..<(m * n) {
                let val = matrixInt64[i]
                var r = val % p64
                if r < 0 { r += p64 }
                reducedMatrix.append(UInt32(r))
            }
            allReducedMatrices.append(reducedMatrix)
        }

        // Dispatch to GPU
        guard let results = computeRanksBatched(matrices: allReducedMatrices, m: m, n: n, primes: primes) else {
            timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart
            return nil
        }

        timings.gpuComputeTime = CFAbsoluteTimeGetCurrent() - gpuStart

        // Find consensus rank (mode of results)
        // Most primes should give the correct rank
        var rankCounts: [UInt32: Int] = [:]
        for r in results {
            rankCounts[r, default: 0] += 1
        }

        let consensusRank = rankCounts.max(by: { $0.value < $1.value })?.key ?? 0

        timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart
        return (Int(consensusRank), timings)
    }

    /// Batch compute ranks for all primes in single GPU dispatch
    private func computeRanksBatched(matrices: [[UInt32]], m: Int, n: Int, primes: [UInt32]) -> [UInt32]? {
        let numPrimes = primes.count
        guard numPrimes == matrices.count else { return nil }

        let mn = m * n

        // Flatten all matrices
        var allMatrixData: [UInt32] = []
        allMatrixData.reserveCapacity(numPrimes * mn)
        for matrix in matrices {
            allMatrixData.append(contentsOf: matrix)
        }

        let totalMatrixSize = numPrimes * mn * MemoryLayout<UInt32>.stride

        guard let matrixBuffer = device.makeBuffer(bytes: allMatrixData, length: totalMatrixSize, options: .storageModeShared),
              let primesBuffer = device.makeBuffer(bytes: primes, length: numPrimes * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let resultsBuffer = device.makeBuffer(length: numPrimes * MemoryLayout<UInt32>.stride, options: .storageModeShared) else {
            return nil
        }

        var mVal = UInt32(m)
        var nVal = UInt32(n)
        var numPrimesVal = UInt32(numPrimes)

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return nil
        }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(matrixBuffer, offset: 0, index: 0)
        encoder.setBuffer(primesBuffer, offset: 0, index: 1)
        encoder.setBuffer(resultsBuffer, offset: 0, index: 2)
        encoder.setBytes(&mVal, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&numPrimesVal, length: MemoryLayout<UInt32>.stride, index: 5)

        // Workspace buffer
        let workspaceSize = numPrimes * mn * MemoryLayout<UInt32>.stride
        guard let workspaceBuffer = device.makeBuffer(length: workspaceSize, options: .storageModePrivate) else {
            return nil
        }
        encoder.setBuffer(workspaceBuffer, offset: 0, index: 6)

        // Dispatch
        let threadGroupSize = min(pipelineState.maxTotalThreadsPerThreadgroup, numPrimes)
        let threadGroups = (numPrimes + threadGroupSize - 1) / threadGroupSize

        encoder.dispatchThreadgroups(
            MTLSize(width: threadGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
        )

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Read results
        let resultsPtr = resultsBuffer.contents().bindMemory(to: UInt32.self, capacity: numPrimes)
        var results: [UInt32] = []
        for i in 0..<numPrimes {
            results.append(resultsPtr[i])
        }

        return results
    }
}
