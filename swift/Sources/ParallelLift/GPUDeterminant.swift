import Metal
import MetalKit
import BigInt
import Foundation

/// GPU-accelerated determinant computation using CRT
class GPUDeterminant {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipelineState: MTLComputePipelineState
    let smallPipelineState: MTLComputePipelineState

    /// Initialize the GPU compute pipeline
    init?() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Error: No Metal device found")
            return nil
        }
        self.device = device

        guard let commandQueue = device.makeCommandQueue() else {
            print("Error: Could not create command queue")
            return nil
        }
        self.commandQueue = commandQueue

        // Load the Metal shader from source
        guard let library = GPUDeterminant.loadLibraryFromSource(device: device) else {
            print("Error: Could not load Metal library")
            return nil
        }

        guard let function = library.makeFunction(name: "modular_determinant"),
              let smallFunction = library.makeFunction(name: "modular_determinant_small") else {
            print("Error: Could not find kernel functions")
            return nil
        }

        do {
            self.pipelineState = try device.makeComputePipelineState(function: function)
            self.smallPipelineState = try device.makeComputePipelineState(function: smallFunction)
        } catch {
            print("Error creating pipeline state: \(error)")
            return nil
        }
    }

    /// Load Metal library from source code
    private static func loadLibraryFromSource(device: MTLDevice) -> MTLLibrary? {
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        inline uint32_t mod_add(uint32_t a, uint32_t b, uint32_t p) {
            uint64_t sum = uint64_t(a) + uint64_t(b);
            return sum >= p ? uint32_t(sum - p) : uint32_t(sum);
        }

        inline uint32_t mod_sub(uint32_t a, uint32_t b, uint32_t p) {
            return a >= b ? (a - b) : (p - b + a);
        }

        inline uint32_t mod_mul(uint32_t a, uint32_t b, uint32_t p) {
            uint64_t prod = uint64_t(a) * uint64_t(b);
            return uint32_t(prod % uint64_t(p));
        }

        inline uint32_t mod_inv(uint32_t a, uint32_t p) {
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

        // Batched kernel: matrices contains pre-reduced matrices concatenated
        kernel void modular_determinant(
            device const uint32_t* matrices [[buffer(0)]],
            device const uint32_t* primes [[buffer(1)]],
            device uint32_t* results [[buffer(2)]],
            device uint32_t* singular_flags [[buffer(3)]],
            constant uint32_t& n [[buffer(4)]],
            constant uint32_t& num_primes [[buffer(5)]],
            device uint32_t* workspace [[buffer(6)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= num_primes) return;

            uint32_t p = primes[tid];
            uint32_t det = 1;
            bool singular = false;

            uint32_t nn = n * n;
            device const uint32_t* myMatrix = matrices + tid * nn;
            device uint32_t* A = workspace + tid * nn;

            for (uint32_t i = 0; i < nn; i++) {
                A[i] = myMatrix[i];
            }

            for (uint32_t k = 0; k < n && !singular; k++) {
                uint32_t pivot_row = k;
                for (uint32_t i = k; i < n; i++) {
                    if (A[i * n + k] != 0) { pivot_row = i; break; }
                }
                if (A[pivot_row * n + k] == 0) {
                    singular = true; det = 0; break;
                }
                if (pivot_row != k) {
                    for (uint32_t j = k; j < n; j++) {
                        uint32_t tmp = A[k * n + j];
                        A[k * n + j] = A[pivot_row * n + j];
                        A[pivot_row * n + j] = tmp;
                    }
                    det = (p - det) % p;
                }
                uint32_t pivot = A[k * n + k];
                det = mod_mul(det, pivot, p);
                uint32_t pivot_inv = mod_inv(pivot, p);
                for (uint32_t i = k + 1; i < n; i++) {
                    uint32_t factor = mod_mul(A[i * n + k], pivot_inv, p);
                    if (factor != 0) {
                        for (uint32_t j = k; j < n; j++) {
                            uint32_t sub = mod_mul(factor, A[k * n + j], p);
                            A[i * n + j] = mod_sub(A[i * n + j], sub, p);
                        }
                    }
                }
            }
            results[tid] = det;
            singular_flags[tid] = singular ? 1 : 0;
        }

        // Small matrix kernel with thread-local storage
        kernel void modular_determinant_small(
            device const uint32_t* matrices [[buffer(0)]],
            device const uint32_t* primes [[buffer(1)]],
            device uint32_t* results [[buffer(2)]],
            device uint32_t* singular_flags [[buffer(3)]],
            constant uint32_t& n [[buffer(4)]],
            constant uint32_t& num_primes [[buffer(5)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= num_primes) return;

            uint32_t p = primes[tid];
            uint32_t det = 1;
            bool singular = false;

            uint32_t A[256];

            uint32_t nn = n * n;
            device const uint32_t* myMatrix = matrices + tid * nn;

            for (uint32_t i = 0; i < nn; i++) {
                A[i] = myMatrix[i];
            }

            for (uint32_t k = 0; k < n && !singular; k++) {
                uint32_t pivot_row = k;
                for (uint32_t i = k; i < n; i++) {
                    if (A[i * n + k] != 0) { pivot_row = i; break; }
                }
                if (A[pivot_row * n + k] == 0) {
                    singular = true; det = 0; break;
                }
                if (pivot_row != k) {
                    for (uint32_t j = k; j < n; j++) {
                        uint32_t tmp = A[k * n + j];
                        A[k * n + j] = A[pivot_row * n + j];
                        A[pivot_row * n + j] = tmp;
                    }
                    det = (p - det) % p;
                }
                uint32_t pivot = A[k * n + k];
                det = mod_mul(det, pivot, p);
                uint32_t pivot_inv = mod_inv(pivot, p);
                for (uint32_t i = k + 1; i < n; i++) {
                    uint32_t factor = mod_mul(A[i * n + k], pivot_inv, p);
                    if (factor != 0) {
                        for (uint32_t j = k; j < n; j++) {
                            uint32_t sub = mod_mul(factor, A[k * n + j], p);
                            A[i * n + j] = mod_sub(A[i * n + j], sub, p);
                        }
                    }
                }
            }
            results[tid] = det;
            singular_flags[tid] = singular ? 1 : 0;
        }
        """

        do {
            return try device.makeLibrary(source: shaderSource, options: nil)
        } catch {
            print("Error compiling shader: \(error)")
            return nil
        }
    }

    /// Compute determinant residues for a single prime (legacy interface)
    func computeResidues(matrix: [UInt32], n: Int, primes: [UInt32]) -> [(residue: UInt32, singular: Bool)]? {
        // For a single prime, use the batched interface with one matrix
        return computeResiduesBatched(matrices: [matrix], n: n, primes: primes)
    }

    /// Full determinant computation using GPU + CRT
    /// Batches all primes into single GPU dispatch for maximum parallelism
    func computeDeterminant(matrix: [BigInt], n: Int) -> (result: BigInt, timings: DeterminantTimings)? {
        var timings = DeterminantTimings()
        let totalStart = CFAbsoluteTimeGetCurrent()

        // Find max absolute value
        var maxAbsValue: BigInt = 0
        for val in matrix {
            let absVal = BigInt(val.magnitude)
            if absVal > maxAbsValue {
                maxAbsValue = absVal
            }
        }

        // Compute Hadamard bound
        let boundStart = CFAbsoluteTimeGetCurrent()
        let hadamardBound = PrimeGenerator.hadamardBound(n: n, maxEntry: maxAbsValue)
        let reconstructionBound = 2 * hadamardBound + 1
        timings.boundComputationTime = CFAbsoluteTimeGetCurrent() - boundStart

        // Generate primes
        let primeStart = CFAbsoluteTimeGetCurrent()
        let numPrimesNeeded = PrimeGenerator.primesNeeded(bound: reconstructionBound)
        let primes = PrimeGenerator.generate31BitPrimes(count: numPrimesNeeded + 10)
        timings.primeGenerationTime = CFAbsoluteTimeGetCurrent() - primeStart
        timings.numPrimes = primes.count

        // GPU compute - batch all primes together
        let gpuStart = CFAbsoluteTimeGetCurrent()

        // Pre-reduce matrix entries to fit in Int64 for efficient mod operations
        // Then compute all primes in a single batch
        let matrixInt64: [Int64] = matrix.map { Int64($0) }

        // For truly batched execution, we pre-compute all reduced matrices
        // and dispatch a single GPU call with all primes
        var allReducedMatrices: [[UInt32]] = []

        for p in primes {
            let p64 = Int64(p)
            let reducedMatrix: [UInt32] = matrixInt64.map { val in
                var r = val % p64
                if r < 0 { r += p64 }
                return UInt32(r)
            }
            allReducedMatrices.append(reducedMatrix)
        }

        // Now dispatch all primes at once using a batched approach
        guard let results = computeResiduesBatched(matrices: allReducedMatrices, n: n, primes: primes) else {
            // Fall back to sequential if batch fails
            var allResults: [(residue: UInt32, singular: Bool)] = []
            for (i, p) in primes.enumerated() {
                if let res = computeResidues(matrix: allReducedMatrices[i], n: n, primes: [p]) {
                    allResults.append(contentsOf: res)
                }
            }
            timings.gpuComputeTime = CFAbsoluteTimeGetCurrent() - gpuStart

            let validResults = allResults.filter { !$0.singular }
            if validResults.isEmpty {
                timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart
                return (BigInt(0), timings)
            }

            let crtStart = CFAbsoluteTimeGetCurrent()
            let residuePairs: [(residue: UInt32, modulus: UInt32)] = zip(validResults, primes).map {
                ($0.0.residue, $0.1)
            }
            let result = CRTReconstruction.reconstructSigned(residues: residuePairs)
            timings.crtReconstructionTime = CFAbsoluteTimeGetCurrent() - crtStart
            timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart

            var M = BigInt(1)
            for p in primes.prefix(validResults.count) {
                M *= BigInt(p)
            }
            timings.finalModulusBits = M.bitWidth
            return (result, timings)
        }

        timings.gpuComputeTime = CFAbsoluteTimeGetCurrent() - gpuStart

        // Check for singular matrices
        let allSingular = results.allSatisfy { $0.singular }
        if allSingular {
            timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart
            return (BigInt(0), timings)
        }

        // Filter valid results
        let validResults = results.filter { !$0.singular }

        // CRT reconstruction
        let crtStart = CFAbsoluteTimeGetCurrent()
        let residuePairs: [(residue: UInt32, modulus: UInt32)] = zip(validResults, primes).map {
            ($0.0.residue, $0.1)
        }
        let result = CRTReconstruction.reconstructSigned(residues: residuePairs)
        timings.crtReconstructionTime = CFAbsoluteTimeGetCurrent() - crtStart

        timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart

        // Compute final modulus size in bits
        var M = BigInt(1)
        for p in primes.prefix(validResults.count) {
            M *= BigInt(p)
        }
        timings.finalModulusBits = M.bitWidth

        return (result, timings)
    }

    /// Compute residues for multiple pre-reduced matrices in a single GPU dispatch
    func computeResiduesBatched(matrices: [[UInt32]], n: Int, primes: [UInt32]) -> [(residue: UInt32, singular: Bool)]? {
        let numPrimes = primes.count
        guard numPrimes == matrices.count else { return nil }

        let nn = n * n

        // Flatten all matrices into a single buffer
        var allMatrixData: [UInt32] = []
        allMatrixData.reserveCapacity(numPrimes * nn)
        for m in matrices {
            allMatrixData.append(contentsOf: m)
        }

        let totalMatrixSize = numPrimes * nn * MemoryLayout<UInt32>.stride

        guard let matrixBuffer = device.makeBuffer(bytes: allMatrixData, length: totalMatrixSize, options: .storageModeShared),
              let primesBuffer = device.makeBuffer(bytes: primes, length: numPrimes * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let resultsBuffer = device.makeBuffer(length: numPrimes * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let singularBuffer = device.makeBuffer(length: numPrimes * MemoryLayout<UInt32>.stride, options: .storageModeShared) else {
            return nil
        }

        var nVal = UInt32(n)
        var numPrimesVal = UInt32(numPrimes)

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return nil
        }

        // Use the batched kernel that reads from per-prime matrix slices
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(matrixBuffer, offset: 0, index: 0)
        encoder.setBuffer(primesBuffer, offset: 0, index: 1)
        encoder.setBuffer(resultsBuffer, offset: 0, index: 2)
        encoder.setBuffer(singularBuffer, offset: 0, index: 3)
        encoder.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&numPrimesVal, length: MemoryLayout<UInt32>.stride, index: 5)

        // Create workspace buffer for the kernel
        let workspaceSize = numPrimes * nn * MemoryLayout<UInt32>.stride
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
        let singularPtr = singularBuffer.contents().bindMemory(to: UInt32.self, capacity: numPrimes)

        var results: [(residue: UInt32, singular: Bool)] = []
        for i in 0..<numPrimes {
            results.append((resultsPtr[i], singularPtr[i] != 0))
        }

        return results
    }

    /// Batch GPU computation - compute all primes at once for better parallelism
    func computeDeterminantBatched(matrix: [BigInt], n: Int) -> (result: BigInt, timings: DeterminantTimings)? {
        var timings = DeterminantTimings()
        let totalStart = CFAbsoluteTimeGetCurrent()

        // Find max absolute value
        var maxAbsValue: BigInt = 0
        for val in matrix {
            let absVal = BigInt(val.magnitude)
            if absVal > maxAbsValue {
                maxAbsValue = absVal
            }
        }

        // Compute Hadamard bound
        let boundStart = CFAbsoluteTimeGetCurrent()
        let hadamardBound = PrimeGenerator.hadamardBound(n: n, maxEntry: maxAbsValue)
        let reconstructionBound = 2 * hadamardBound + 1
        timings.boundComputationTime = CFAbsoluteTimeGetCurrent() - boundStart

        // Generate primes
        let primeStart = CFAbsoluteTimeGetCurrent()
        let numPrimesNeeded = PrimeGenerator.primesNeeded(bound: reconstructionBound)
        let primes = PrimeGenerator.generate31BitPrimes(count: numPrimesNeeded + 10)
        timings.primeGenerationTime = CFAbsoluteTimeGetCurrent() - primeStart
        timings.numPrimes = primes.count

        // Pre-reduce matrix for all primes and batch the GPU call
        let gpuStart = CFAbsoluteTimeGetCurrent()

        // For truly parallel GPU execution, we process all primes at once
        // But we still need to reduce mod each prime on CPU first
        var allResults: [(residue: UInt32, singular: Bool)] = []

        // Check if entries fit in int32 for fast path
        let maxFits = maxAbsValue < BigInt(1) << 30

        if maxFits {
            // Fast path: entries fit in int32, can batch GPU calls more efficiently
            // Still need per-prime reduction unfortunately
            for p in primes {
                let pBig = BigInt(p)
                let reducedMatrix: [UInt32] = matrix.map { val in
                    var r = val % pBig
                    if r < 0 { r += pBig }
                    return UInt32(r)
                }

                if let results = computeResidues(matrix: reducedMatrix, n: n, primes: [p]) {
                    allResults.append(contentsOf: results)
                }
            }
        } else {
            // Large entries - same approach but note it's necessary
            for p in primes {
                let pBig = BigInt(p)
                let reducedMatrix: [UInt32] = matrix.map { val in
                    var r = val % pBig
                    if r < 0 { r += pBig }
                    return UInt32(r)
                }

                if let results = computeResidues(matrix: reducedMatrix, n: n, primes: [p]) {
                    allResults.append(contentsOf: results)
                }
            }
        }

        timings.gpuComputeTime = CFAbsoluteTimeGetCurrent() - gpuStart

        // Check for singular matrices
        let allSingular = allResults.allSatisfy { $0.singular }
        if allSingular {
            timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart
            return (BigInt(0), timings)
        }

        let validResults = allResults.filter { !$0.singular }

        // CRT reconstruction
        let crtStart = CFAbsoluteTimeGetCurrent()
        let residuePairs: [(residue: UInt32, modulus: UInt32)] = zip(validResults, primes).map {
            ($0.0.residue, $0.1)
        }
        let result = CRTReconstruction.reconstructSigned(residues: residuePairs)
        timings.crtReconstructionTime = CFAbsoluteTimeGetCurrent() - crtStart

        timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart

        var M = BigInt(1)
        for p in primes.prefix(validResults.count) {
            M *= BigInt(p)
        }
        timings.finalModulusBits = M.bitWidth

        return (result, timings)
    }
}

/// Timing breakdown for determinant computation
struct DeterminantTimings {
    var boundComputationTime: Double = 0
    var primeGenerationTime: Double = 0
    var gpuComputeTime: Double = 0
    var crtReconstructionTime: Double = 0
    var totalTime: Double = 0
    var numPrimes: Int = 0
    var finalModulusBits: Int = 0

    func printReport() {
        print("=== GPU-CRT Determinant Timing Breakdown ===")
        print(String(format: "Bound computation:    %8.4f s", boundComputationTime))
        print(String(format: "Prime generation:     %8.4f s", primeGenerationTime))
        print(String(format: "GPU compute:          %8.4f s", gpuComputeTime))
        print(String(format: "CRT reconstruction:   %8.4f s", crtReconstructionTime))
        print(String(format: "Total:                %8.4f s", totalTime))
        print("Number of primes used: \(numPrimes)")
        print("Final modulus size: \(finalModulusBits) bits")
    }
}
