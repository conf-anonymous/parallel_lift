import BigInt
import Foundation
import Metal

/// Timing breakdown for GPU RNS operations
struct RNSTimings {
    var basisExtendTime: Double = 0
    var hadamardTime: Double = 0
    var rescaleTime: Double = 0
    var gadgetDecomposeTime: Double = 0
    var gadgetMACTime: Double = 0
    var totalTime: Double = 0
}

/// GPU-accelerated RNS/CRT operations for FHE
class GPURNSOperations {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    // Compiled pipelines
    private let basisExtendPipeline: MTLComputePipelineState
    private let hadamardPipeline: MTLComputePipelineState
    private let addPipeline: MTLComputePipelineState
    private let subPipeline: MTLComputePipelineState
    private let rescalePipeline: MTLComputePipelineState
    private let gadgetDecomposePipeline: MTLComputePipelineState
    private let gadgetMACPipeline: MTLComputePipelineState

    typealias RNSPolynomial = [[UInt64]]

    /// Metal shader source code embedded in Swift
    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    // Modular multiplication using 128-bit intermediate
    // Since Metal doesn't support double, we use iterative subtraction for the high case
    inline uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t mod) {
        // For typical 50-60 bit moduli with 50-60 bit inputs,
        // the product fits in 128 bits but may overflow 64 bits

        uint64_t aLo = a & 0xFFFFFFFF;
        uint64_t aHi = a >> 32;
        uint64_t bLo = b & 0xFFFFFFFF;
        uint64_t bHi = b >> 32;

        uint64_t p0 = aLo * bLo;
        uint64_t p1 = aLo * bHi;
        uint64_t p2 = aHi * bLo;
        uint64_t p3 = aHi * bHi;

        uint64_t mid = p1 + p2;
        uint64_t midCarry = (mid < p1) ? 1ULL : 0ULL;

        uint64_t low = p0 + (mid << 32);
        uint64_t lowCarry = (low < p0) ? 1ULL : 0ULL;

        uint64_t high = p3 + (mid >> 32) + (midCarry << 32) + lowCarry;

        // If no high bits, simple modulo
        if (high == 0) {
            return low % mod;
        }

        // Use Barrett-like reduction with float approximation
        // Since float has 23 bits mantissa, this is approximate but good for starting point
        float approx = (float(high) * 18446744073709551616.0f + float(low)) / float(mod);
        uint64_t q = uint64_t(approx);

        // Correct the quotient - might be off by 1-2 due to float precision
        uint64_t qm_lo = q * mod;

        // Compute remainder: low - q*mod (handling underflow)
        uint64_t r;
        if (low >= qm_lo) {
            r = low - qm_lo;
        } else {
            // Underflow - quotient was too high
            r = mod - (qm_lo - low);
            if (r >= mod) r -= mod;
        }

        // Final correction - ensure r < mod
        while (r >= mod) {
            r -= mod;
        }
        return r;
    }

    // Basis extension kernel
    kernel void basis_extend(
        device const uint64_t* poly,
        device uint64_t* result,
        device const uint64_t* Q_moduli,
        device const uint64_t* P_moduli,
        device const uint64_t* conv_factors,
        constant uint& n,
        constant uint& kQ,
        constant uint& kP,
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint j = gid.x;
        uint pIdx = gid.y;

        if (j >= n || pIdx >= kP) return;

        uint64_t p = P_moduli[pIdx];
        uint64_t sum = 0;

        for (uint qIdx = 0; qIdx < kQ; qIdx++) {
            uint64_t x_q = poly[qIdx * n + j];
            uint64_t factor = conv_factors[qIdx * kP + pIdx];
            uint64_t term = mod_mul(x_q, factor, p);
            sum = (sum + term) % p;
        }

        result[pIdx * n + j] = sum;
    }

    // Hadamard product kernel
    kernel void hadamard_multiply(
        device const uint64_t* a,
        device const uint64_t* b,
        device uint64_t* result,
        device const uint64_t* moduli,
        constant uint& n,
        constant uint& kQ,
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint j = gid.x;
        uint qIdx = gid.y;

        if (j >= n || qIdx >= kQ) return;

        uint64_t q = moduli[qIdx];
        uint64_t idx = qIdx * n + j;

        result[idx] = mod_mul(a[idx], b[idx], q);
    }

    // RNS addition kernel (1D dispatch)
    kernel void rns_add(
        device const uint64_t* a,
        device const uint64_t* b,
        device uint64_t* result,
        device const uint64_t* moduli,
        constant uint& n,
        constant uint& kQ,
        uint gid [[thread_position_in_grid]]
    ) {
        uint totalSize = n * kQ;
        if (gid >= totalSize) return;

        uint qIdx = gid / n;
        uint j = gid % n;

        uint64_t q = moduli[qIdx];
        uint64_t idx = qIdx * n + j;

        uint64_t sum = a[idx] + b[idx];
        result[idx] = (sum >= q) ? (sum - q) : sum;
    }

    // RNS subtraction kernel
    kernel void rns_sub(
        device const uint64_t* a,
        device const uint64_t* b,
        device uint64_t* result,
        device const uint64_t* moduli,
        constant uint& n,
        constant uint& kQ,
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint j = gid.x;
        uint qIdx = gid.y;

        if (j >= n || qIdx >= kQ) return;

        uint64_t q = moduli[qIdx];
        uint64_t idx = qIdx * n + j;

        result[idx] = (a[idx] >= b[idx]) ? (a[idx] - b[idx]) : (q - (b[idx] - a[idx]));
    }

    // Rescale kernel (CKKS-style)
    kernel void rescale(
        device const uint64_t* poly,
        device uint64_t* result,
        device const uint64_t* moduli,
        device const uint64_t* qLastInv,
        constant uint& n,
        constant uint& kQ,
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint j = gid.x;
        uint qIdx = gid.y;

        if (j >= n || qIdx >= (kQ - 1)) return;

        uint64_t q = moduli[qIdx];
        uint64_t lastResidue = poly[(kQ - 1) * n + j];
        uint64_t lastMod = lastResidue % q;

        uint64_t val = poly[qIdx * n + j];
        uint64_t diff;

        if (val >= lastMod) {
            diff = val - lastMod;
        } else {
            diff = q - (lastMod - val);
        }

        result[qIdx * n + j] = mod_mul(diff, qLastInv[qIdx], q);
    }

    // Gadget decomposition kernel
    kernel void gadget_decompose(
        device const uint64_t* poly,
        device uint64_t* digit,
        constant uint64_t& B,
        constant uint& d,
        constant uint& n,
        constant uint& kQ,
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint j = gid.x;
        uint qIdx = gid.y;

        if (j >= n || qIdx >= kQ) return;

        uint64_t idx = qIdx * n + j;
        uint64_t val = poly[idx];

        // Compute B^d
        uint64_t power = 1;
        for (uint i = 0; i < d; i++) {
            power *= B;
        }

        digit[idx] = (val / power) % B;
    }

    // Gadget multiply-accumulate kernel
    kernel void gadget_mac(
        device const uint64_t* digit,
        device const uint64_t* key,
        device uint64_t* accum,
        device const uint64_t* moduli,
        constant uint& n,
        constant uint& kQ,
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint j = gid.x;
        uint qIdx = gid.y;

        if (j >= n || qIdx >= kQ) return;

        uint64_t q = moduli[qIdx];
        uint64_t idx = qIdx * n + j;

        uint64_t term = mod_mul(digit[idx], key[idx], q);
        accum[idx] = (accum[idx] + term) % q;
    }
    """

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else {
            print("GPURNSOperations: Failed to create Metal device or command queue")
            return nil
        }
        self.device = device
        self.commandQueue = commandQueue

        // Compile shader from source
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: GPURNSOperations.shaderSource, options: nil)
        } catch {
            print("GPURNSOperations: Failed to compile Metal shader: \(error)")
            return nil
        }

        // Create pipelines
        guard let basisExtend = library.makeFunction(name: "basis_extend"),
              let hadamard = library.makeFunction(name: "hadamard_multiply"),
              let add = library.makeFunction(name: "rns_add"),
              let sub = library.makeFunction(name: "rns_sub"),
              let rescale = library.makeFunction(name: "rescale"),
              let gadgetDecompose = library.makeFunction(name: "gadget_decompose"),
              let gadgetMAC = library.makeFunction(name: "gadget_mac") else {
            print("GPURNSOperations: Failed to find kernel functions")
            return nil
        }

        do {
            self.basisExtendPipeline = try device.makeComputePipelineState(function: basisExtend)
            self.hadamardPipeline = try device.makeComputePipelineState(function: hadamard)
            self.addPipeline = try device.makeComputePipelineState(function: add)
            self.subPipeline = try device.makeComputePipelineState(function: sub)
            self.rescalePipeline = try device.makeComputePipelineState(function: rescale)
            self.gadgetDecomposePipeline = try device.makeComputePipelineState(function: gadgetDecompose)
            self.gadgetMACPipeline = try device.makeComputePipelineState(function: gadgetMAC)
        } catch {
            print("GPURNSOperations: Failed to create pipeline state: \(error)")
            return nil
        }
    }

    // MARK: - Helper: Flatten/Unflatten RNS Polynomial

    private func flatten(_ poly: RNSPolynomial) -> [UInt64] {
        return poly.flatMap { $0 }
    }

    private func unflatten(_ data: [UInt64], kQ: Int, n: Int) -> RNSPolynomial {
        var result: RNSPolynomial = []
        for qIdx in 0..<kQ {
            let start = qIdx * n
            let end = start + n
            result.append(Array(data[start..<end]))
        }
        return result
    }

    // MARK: - Basis Extension

    /// Convert polynomial from RNS basis Q to basis P using GPU
    func basisExtend(
        poly: RNSPolynomial,
        fromBasis Q: RNSContext,
        toBasis P: RNSContext,
        precomputed: BasisExtensionPrecompute
    ) -> RNSPolynomial? {
        let n = poly[0].count
        let kQ = Q.count
        let kP = P.count

        // Flatten input
        let flatPoly = flatten(poly)
        let flatFactors = precomputed.conversionFactors.flatMap { $0 }

        // Create buffers
        guard let polyBuffer = device.makeBuffer(bytes: flatPoly, length: flatPoly.count * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let resultBuffer = device.makeBuffer(length: kP * n * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let qModuliBuffer = device.makeBuffer(bytes: Q.moduli, length: Q.moduli.count * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let pModuliBuffer = device.makeBuffer(bytes: P.moduli, length: P.moduli.count * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let factorsBuffer = device.makeBuffer(bytes: flatFactors, length: flatFactors.count * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return nil
        }

        var nVal = UInt32(n)
        var kQVal = UInt32(kQ)
        var kPVal = UInt32(kP)

        encoder.setComputePipelineState(basisExtendPipeline)
        encoder.setBuffer(polyBuffer, offset: 0, index: 0)
        encoder.setBuffer(resultBuffer, offset: 0, index: 1)
        encoder.setBuffer(qModuliBuffer, offset: 0, index: 2)
        encoder.setBuffer(pModuliBuffer, offset: 0, index: 3)
        encoder.setBuffer(factorsBuffer, offset: 0, index: 4)
        encoder.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&kQVal, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&kPVal, length: MemoryLayout<UInt32>.stride, index: 7)

        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(width: (n + 15) / 16 * 16, height: (kP + 15) / 16 * 16, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Read result
        let resultPtr = resultBuffer.contents().bindMemory(to: UInt64.self, capacity: kP * n)
        let resultArray = Array(UnsafeBufferPointer(start: resultPtr, count: kP * n))

        return unflatten(resultArray, kQ: kP, n: n)
    }

    // MARK: - Hadamard Product

    /// Element-wise multiplication of two polynomials in RNS form
    func multiplyHadamard(
        _ a: RNSPolynomial,
        _ b: RNSPolynomial,
        basis Q: RNSContext
    ) -> RNSPolynomial? {
        let n = a[0].count
        let kQ = Q.count

        let flatA = flatten(a)
        let flatB = flatten(b)

        guard let aBuffer = device.makeBuffer(bytes: flatA, length: flatA.count * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let bBuffer = device.makeBuffer(bytes: flatB, length: flatB.count * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let resultBuffer = device.makeBuffer(length: kQ * n * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let moduliBuffer = device.makeBuffer(bytes: Q.moduli, length: Q.moduli.count * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return nil
        }

        var nVal = UInt32(n)
        var kQVal = UInt32(kQ)

        encoder.setComputePipelineState(hadamardPipeline)
        encoder.setBuffer(aBuffer, offset: 0, index: 0)
        encoder.setBuffer(bBuffer, offset: 0, index: 1)
        encoder.setBuffer(resultBuffer, offset: 0, index: 2)
        encoder.setBuffer(moduliBuffer, offset: 0, index: 3)
        encoder.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&kQVal, length: MemoryLayout<UInt32>.stride, index: 5)

        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(width: (n + 15) / 16 * 16, height: (kQ + 15) / 16 * 16, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let resultPtr = resultBuffer.contents().bindMemory(to: UInt64.self, capacity: kQ * n)
        let resultArray = Array(UnsafeBufferPointer(start: resultPtr, count: kQ * n))

        return unflatten(resultArray, kQ: kQ, n: n)
    }

    // MARK: - Add

    /// Add two polynomials in RNS form
    func add(
        _ a: RNSPolynomial,
        _ b: RNSPolynomial,
        basis Q: RNSContext
    ) -> RNSPolynomial? {
        let n = a[0].count
        let kQ = Q.count

        // Validate input sizes
        guard a.count == kQ, b.count == kQ else {
            print("GPURNSOperations.add: Invalid polynomial size")
            return nil
        }
        for i in 0..<kQ {
            guard a[i].count == n, b[i].count == n else {
                print("GPURNSOperations.add: Inconsistent coefficient count")
                return nil
            }
        }

        let flatA = flatten(a)
        let flatB = flatten(b)

        guard let aBuffer = device.makeBuffer(bytes: flatA, length: flatA.count * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let bBuffer = device.makeBuffer(bytes: flatB, length: flatB.count * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let resultBuffer = device.makeBuffer(length: kQ * n * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let moduliBuffer = device.makeBuffer(bytes: Q.moduli, length: Q.moduli.count * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return nil
        }

        var nVal = UInt32(n)
        var kQVal = UInt32(kQ)

        encoder.setComputePipelineState(addPipeline)
        encoder.setBuffer(aBuffer, offset: 0, index: 0)
        encoder.setBuffer(bBuffer, offset: 0, index: 1)
        encoder.setBuffer(resultBuffer, offset: 0, index: 2)
        encoder.setBuffer(moduliBuffer, offset: 0, index: 3)
        encoder.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&kQVal, length: MemoryLayout<UInt32>.stride, index: 5)

        // Use dispatchThreadgroups for more predictable behavior
        let threadgroupWidth = min(256, addPipeline.maxTotalThreadsPerThreadgroup)
        let threadgroupSize = MTLSize(width: threadgroupWidth, height: 1, depth: 1)
        let totalThreads = n * kQ
        let threadgroupsPerGrid = MTLSize(width: (totalThreads + threadgroupWidth - 1) / threadgroupWidth, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadgroupSize)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Check for GPU errors
        if let error = commandBuffer.error {
            print("GPURNSOperations.add: GPU error: \(error)")
            return nil
        }

        let resultPtr = resultBuffer.contents().bindMemory(to: UInt64.self, capacity: kQ * n)
        let resultArray = Array(UnsafeBufferPointer(start: resultPtr, count: kQ * n))

        return unflatten(resultArray, kQ: kQ, n: n)
    }

    // MARK: - Rescale (CKKS)

    /// Rescale polynomial by dividing by last modulus
    func rescale(
        poly: RNSPolynomial,
        basis Q: RNSContext
    ) -> RNSPolynomial? {
        let n = poly[0].count
        let kQ = Q.count
        let newK = kQ - 1

        if newK == 0 { return nil }

        let flatPoly = flatten(poly)

        // Precompute qLast^(-1) mod q_i for each i < kQ-1
        let qLast = Q.moduli[kQ - 1]
        var qLastInv: [UInt64] = []
        for i in 0..<newK {
            let qi = Q.moduli[i]
            qLastInv.append(RNSContext.modInverse(qLast % qi, qi))
        }

        guard let polyBuffer = device.makeBuffer(bytes: flatPoly, length: flatPoly.count * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let resultBuffer = device.makeBuffer(length: newK * n * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let moduliBuffer = device.makeBuffer(bytes: Q.moduli, length: Q.moduli.count * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let qLastInvBuffer = device.makeBuffer(bytes: qLastInv, length: qLastInv.count * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return nil
        }

        var nVal = UInt32(n)
        var kQVal = UInt32(kQ)

        encoder.setComputePipelineState(rescalePipeline)
        encoder.setBuffer(polyBuffer, offset: 0, index: 0)
        encoder.setBuffer(resultBuffer, offset: 0, index: 1)
        encoder.setBuffer(moduliBuffer, offset: 0, index: 2)
        encoder.setBuffer(qLastInvBuffer, offset: 0, index: 3)
        encoder.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&kQVal, length: MemoryLayout<UInt32>.stride, index: 5)

        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(width: (n + 15) / 16 * 16, height: (newK + 15) / 16 * 16, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let resultPtr = resultBuffer.contents().bindMemory(to: UInt64.self, capacity: newK * n)
        let resultArray = Array(UnsafeBufferPointer(start: resultPtr, count: newK * n))

        return unflatten(resultArray, kQ: newK, n: n)
    }

    // MARK: - Gadget Decomposition + MAC (Key Switching Core)

    /// Perform gadget decomposition and multiply-accumulate
    /// This is the core of key switching in BFV/BGV/CKKS
    func gadgetDecomposeAndMAC(
        poly: RNSPolynomial,
        gadgetKeys: [RNSPolynomial],
        basis Q: RNSContext,
        digitBase B: UInt64,
        numDigits: Int
    ) -> RNSPolynomial? {
        let n = poly[0].count
        let kQ = Q.count

        let flatPoly = flatten(poly)

        // Allocate result (accumulator)
        let accumulator = [UInt64](repeating: 0, count: kQ * n)

        guard let polyBuffer = device.makeBuffer(bytes: flatPoly, length: flatPoly.count * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let accumBuffer = device.makeBuffer(bytes: accumulator, length: accumulator.count * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let digitBuffer = device.makeBuffer(length: kQ * n * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let moduliBuffer = device.makeBuffer(bytes: Q.moduli, length: Q.moduli.count * MemoryLayout<UInt64>.stride, options: .storageModeShared) else {
            return nil
        }

        var nVal = UInt32(n)
        var kQVal = UInt32(kQ)
        var BVal = B

        for d in 0..<numDigits {
            var dVal = UInt32(d)

            // Gadget decompose: extract digit d
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                return nil
            }

            encoder.setComputePipelineState(gadgetDecomposePipeline)
            encoder.setBuffer(polyBuffer, offset: 0, index: 0)
            encoder.setBuffer(digitBuffer, offset: 0, index: 1)
            encoder.setBytes(&BVal, length: MemoryLayout<UInt64>.stride, index: 2)
            encoder.setBytes(&dVal, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBytes(&kQVal, length: MemoryLayout<UInt32>.stride, index: 5)

            let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
            let gridSize = MTLSize(width: (n + 15) / 16 * 16, height: (kQ + 15) / 16 * 16, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            // MAC: accumulator += digit * gadgetKeys[d]
            let flatKey = flatten(gadgetKeys[d])
            guard let keyBuffer = device.makeBuffer(bytes: flatKey, length: flatKey.count * MemoryLayout<UInt64>.stride, options: .storageModeShared),
                  let macBuffer = commandQueue.makeCommandBuffer(),
                  let macEncoder = macBuffer.makeComputeCommandEncoder() else {
                return nil
            }

            macEncoder.setComputePipelineState(gadgetMACPipeline)
            macEncoder.setBuffer(digitBuffer, offset: 0, index: 0)
            macEncoder.setBuffer(keyBuffer, offset: 0, index: 1)
            macEncoder.setBuffer(accumBuffer, offset: 0, index: 2)
            macEncoder.setBuffer(moduliBuffer, offset: 0, index: 3)
            macEncoder.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 4)
            macEncoder.setBytes(&kQVal, length: MemoryLayout<UInt32>.stride, index: 5)

            macEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            macEncoder.endEncoding()
            macBuffer.commit()
            macBuffer.waitUntilCompleted()
        }

        // Read accumulator result
        let resultPtr = accumBuffer.contents().bindMemory(to: UInt64.self, capacity: kQ * n)
        let resultArray = Array(UnsafeBufferPointer(start: resultPtr, count: kQ * n))

        return unflatten(resultArray, kQ: kQ, n: n)
    }

    // MARK: - Fused MRR (Multiply-Relinearize-Rescale) Macro-Kernel

    /// Timing breakdown for MRR operation
    struct MRRTimings {
        var multiplyTime: Double = 0
        var relinearizeTime: Double = 0
        var rescaleTime: Double = 0
        var totalTime: Double = 0
    }

    /// Fused Multiply-Relinearize-Rescale operation
    /// This is the hot path in CKKS homomorphic multiplication
    /// - Parameters:
    ///   - ct0, ct1: First ciphertext components (c0, c1)
    ///   - ct2_0, ct2_1: Second ciphertext components
    ///   - relinKey0, relinKey1: Relinearization key components (gadget decomposed)
    ///   - basis: RNS context for moduli
    ///   - digitBase: Gadget decomposition base
    ///   - numDigits: Number of digits in decomposition
    /// - Returns: Result ciphertext (c0', c1') after MRR, or nil on failure
    func fusedMRR(
        ct0: RNSPolynomial, ct1: RNSPolynomial,
        ct2_0: RNSPolynomial, ct2_1: RNSPolynomial,
        relinKey0: [RNSPolynomial], relinKey1: [RNSPolynomial],
        basis Q: RNSContext,
        digitBase B: UInt64,
        numDigits: Int
    ) -> (c0: RNSPolynomial, c1: RNSPolynomial, timings: MRRTimings)? {
        let n = ct0[0].count
        let kQ = Q.count
        let newK = kQ - 1  // After rescale

        if newK <= 0 { return nil }

        var timings = MRRTimings()
        let totalStart = CFAbsoluteTimeGetCurrent()

        // === Step 1: Hadamard multiply (ct0*ct2_0, ct0*ct2_1 + ct1*ct2_0, ct1*ct2_1) ===
        let mulStart = CFAbsoluteTimeGetCurrent()

        let flat_ct0 = flatten(ct0)
        let flat_ct1 = flatten(ct1)
        let flat_ct2_0 = flatten(ct2_0)
        let flat_ct2_1 = flatten(ct2_1)

        guard let buf_ct0 = device.makeBuffer(bytes: flat_ct0, length: flat_ct0.count * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let buf_ct1 = device.makeBuffer(bytes: flat_ct1, length: flat_ct1.count * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let buf_ct2_0 = device.makeBuffer(bytes: flat_ct2_0, length: flat_ct2_0.count * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let buf_ct2_1 = device.makeBuffer(bytes: flat_ct2_1, length: flat_ct2_1.count * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let buf_d0 = device.makeBuffer(length: kQ * n * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let buf_d1 = device.makeBuffer(length: kQ * n * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let buf_d2 = device.makeBuffer(length: kQ * n * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let buf_temp = device.makeBuffer(length: kQ * n * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let moduliBuffer = device.makeBuffer(bytes: Q.moduli, length: Q.moduli.count * MemoryLayout<UInt64>.stride, options: .storageModeShared) else {
            return nil
        }

        var nVal = UInt32(n)
        var kQVal = UInt32(kQ)

        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(width: (n + 15) / 16 * 16, height: (kQ + 15) / 16 * 16, depth: 1)

        // Single command buffer for multiply phase
        guard let mulCmdBuf = commandQueue.makeCommandBuffer() else { return nil }

        // d0 = ct0 * ct2_0
        if let enc = mulCmdBuf.makeComputeCommandEncoder() {
            enc.setComputePipelineState(hadamardPipeline)
            enc.setBuffer(buf_ct0, offset: 0, index: 0)
            enc.setBuffer(buf_ct2_0, offset: 0, index: 1)
            enc.setBuffer(buf_d0, offset: 0, index: 2)
            enc.setBuffer(moduliBuffer, offset: 0, index: 3)
            enc.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 4)
            enc.setBytes(&kQVal, length: MemoryLayout<UInt32>.stride, index: 5)
            enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            enc.endEncoding()
        }

        // d1 = ct0 * ct2_1
        if let enc = mulCmdBuf.makeComputeCommandEncoder() {
            enc.setComputePipelineState(hadamardPipeline)
            enc.setBuffer(buf_ct0, offset: 0, index: 0)
            enc.setBuffer(buf_ct2_1, offset: 0, index: 1)
            enc.setBuffer(buf_d1, offset: 0, index: 2)
            enc.setBuffer(moduliBuffer, offset: 0, index: 3)
            enc.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 4)
            enc.setBytes(&kQVal, length: MemoryLayout<UInt32>.stride, index: 5)
            enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            enc.endEncoding()
        }

        // temp = ct1 * ct2_0
        if let enc = mulCmdBuf.makeComputeCommandEncoder() {
            enc.setComputePipelineState(hadamardPipeline)
            enc.setBuffer(buf_ct1, offset: 0, index: 0)
            enc.setBuffer(buf_ct2_0, offset: 0, index: 1)
            enc.setBuffer(buf_temp, offset: 0, index: 2)
            enc.setBuffer(moduliBuffer, offset: 0, index: 3)
            enc.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 4)
            enc.setBytes(&kQVal, length: MemoryLayout<UInt32>.stride, index: 5)
            enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            enc.endEncoding()
        }

        // d2 = ct1 * ct2_1
        if let enc = mulCmdBuf.makeComputeCommandEncoder() {
            enc.setComputePipelineState(hadamardPipeline)
            enc.setBuffer(buf_ct1, offset: 0, index: 0)
            enc.setBuffer(buf_ct2_1, offset: 0, index: 1)
            enc.setBuffer(buf_d2, offset: 0, index: 2)
            enc.setBuffer(moduliBuffer, offset: 0, index: 3)
            enc.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 4)
            enc.setBytes(&kQVal, length: MemoryLayout<UInt32>.stride, index: 5)
            enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            enc.endEncoding()
        }

        mulCmdBuf.commit()
        mulCmdBuf.waitUntilCompleted()

        // d1 += temp (using 1D add kernel)
        let addThreadgroupWidth = min(256, addPipeline.maxTotalThreadsPerThreadgroup)
        let addThreadgroupSize = MTLSize(width: addThreadgroupWidth, height: 1, depth: 1)
        let totalThreads = n * kQ
        let addThreadgroupsPerGrid = MTLSize(width: (totalThreads + addThreadgroupWidth - 1) / addThreadgroupWidth, height: 1, depth: 1)

        guard let addCmdBuf = commandQueue.makeCommandBuffer(),
              let addEnc = addCmdBuf.makeComputeCommandEncoder() else { return nil }

        addEnc.setComputePipelineState(addPipeline)
        addEnc.setBuffer(buf_d1, offset: 0, index: 0)
        addEnc.setBuffer(buf_temp, offset: 0, index: 1)
        addEnc.setBuffer(buf_d1, offset: 0, index: 2)  // in-place
        addEnc.setBuffer(moduliBuffer, offset: 0, index: 3)
        addEnc.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 4)
        addEnc.setBytes(&kQVal, length: MemoryLayout<UInt32>.stride, index: 5)
        addEnc.dispatchThreadgroups(addThreadgroupsPerGrid, threadsPerThreadgroup: addThreadgroupSize)
        addEnc.endEncoding()
        addCmdBuf.commit()
        addCmdBuf.waitUntilCompleted()

        timings.multiplyTime = CFAbsoluteTimeGetCurrent() - mulStart

        // === Step 2: Relinearization (key switching on d2) ===
        let relinStart = CFAbsoluteTimeGetCurrent()

        // Gadget decompose d2 and MAC with relinearization keys
        guard let accumBuf0 = device.makeBuffer(length: kQ * n * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let accumBuf1 = device.makeBuffer(length: kQ * n * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let digitBuffer = device.makeBuffer(length: kQ * n * MemoryLayout<UInt64>.stride, options: .storageModeShared) else {
            return nil
        }

        // Zero accumulators
        memset(accumBuf0.contents(), 0, kQ * n * MemoryLayout<UInt64>.stride)
        memset(accumBuf1.contents(), 0, kQ * n * MemoryLayout<UInt64>.stride)

        var BVal = B

        for d in 0..<numDigits {
            var dVal = UInt32(d)

            // Single command buffer per digit (fused decompose + 2x MAC)
            guard let digitCmdBuf = commandQueue.makeCommandBuffer() else { return nil }

            // Decompose d2 -> digit
            if let enc = digitCmdBuf.makeComputeCommandEncoder() {
                enc.setComputePipelineState(gadgetDecomposePipeline)
                enc.setBuffer(buf_d2, offset: 0, index: 0)
                enc.setBuffer(digitBuffer, offset: 0, index: 1)
                enc.setBytes(&BVal, length: MemoryLayout<UInt64>.stride, index: 2)
                enc.setBytes(&dVal, length: MemoryLayout<UInt32>.stride, index: 3)
                enc.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 4)
                enc.setBytes(&kQVal, length: MemoryLayout<UInt32>.stride, index: 5)
                enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
                enc.endEncoding()
            }

            digitCmdBuf.commit()
            digitCmdBuf.waitUntilCompleted()

            // MAC with relinKey0[d]
            let flatKey0 = flatten(relinKey0[d])
            guard let keyBuf0 = device.makeBuffer(bytes: flatKey0, length: flatKey0.count * MemoryLayout<UInt64>.stride, options: .storageModeShared),
                  let mac0CmdBuf = commandQueue.makeCommandBuffer(),
                  let mac0Enc = mac0CmdBuf.makeComputeCommandEncoder() else { return nil }

            mac0Enc.setComputePipelineState(gadgetMACPipeline)
            mac0Enc.setBuffer(digitBuffer, offset: 0, index: 0)
            mac0Enc.setBuffer(keyBuf0, offset: 0, index: 1)
            mac0Enc.setBuffer(accumBuf0, offset: 0, index: 2)
            mac0Enc.setBuffer(moduliBuffer, offset: 0, index: 3)
            mac0Enc.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 4)
            mac0Enc.setBytes(&kQVal, length: MemoryLayout<UInt32>.stride, index: 5)
            mac0Enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            mac0Enc.endEncoding()
            mac0CmdBuf.commit()
            mac0CmdBuf.waitUntilCompleted()

            // MAC with relinKey1[d]
            let flatKey1 = flatten(relinKey1[d])
            guard let keyBuf1 = device.makeBuffer(bytes: flatKey1, length: flatKey1.count * MemoryLayout<UInt64>.stride, options: .storageModeShared),
                  let mac1CmdBuf = commandQueue.makeCommandBuffer(),
                  let mac1Enc = mac1CmdBuf.makeComputeCommandEncoder() else { return nil }

            mac1Enc.setComputePipelineState(gadgetMACPipeline)
            mac1Enc.setBuffer(digitBuffer, offset: 0, index: 0)
            mac1Enc.setBuffer(keyBuf1, offset: 0, index: 1)
            mac1Enc.setBuffer(accumBuf1, offset: 0, index: 2)
            mac1Enc.setBuffer(moduliBuffer, offset: 0, index: 3)
            mac1Enc.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 4)
            mac1Enc.setBytes(&kQVal, length: MemoryLayout<UInt32>.stride, index: 5)
            mac1Enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            mac1Enc.endEncoding()
            mac1CmdBuf.commit()
            mac1CmdBuf.waitUntilCompleted()
        }

        // c0' = d0 + accum0, c1' = d1 + accum1
        guard let finalAddCmdBuf = commandQueue.makeCommandBuffer() else { return nil }

        // c0' = d0 + accum0
        if let enc = finalAddCmdBuf.makeComputeCommandEncoder() {
            enc.setComputePipelineState(addPipeline)
            enc.setBuffer(buf_d0, offset: 0, index: 0)
            enc.setBuffer(accumBuf0, offset: 0, index: 1)
            enc.setBuffer(buf_d0, offset: 0, index: 2)  // in-place
            enc.setBuffer(moduliBuffer, offset: 0, index: 3)
            enc.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 4)
            enc.setBytes(&kQVal, length: MemoryLayout<UInt32>.stride, index: 5)
            enc.dispatchThreadgroups(addThreadgroupsPerGrid, threadsPerThreadgroup: addThreadgroupSize)
            enc.endEncoding()
        }

        // c1' = d1 + accum1
        if let enc = finalAddCmdBuf.makeComputeCommandEncoder() {
            enc.setComputePipelineState(addPipeline)
            enc.setBuffer(buf_d1, offset: 0, index: 0)
            enc.setBuffer(accumBuf1, offset: 0, index: 1)
            enc.setBuffer(buf_d1, offset: 0, index: 2)  // in-place
            enc.setBuffer(moduliBuffer, offset: 0, index: 3)
            enc.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 4)
            enc.setBytes(&kQVal, length: MemoryLayout<UInt32>.stride, index: 5)
            enc.dispatchThreadgroups(addThreadgroupsPerGrid, threadsPerThreadgroup: addThreadgroupSize)
            enc.endEncoding()
        }

        finalAddCmdBuf.commit()
        finalAddCmdBuf.waitUntilCompleted()

        timings.relinearizeTime = CFAbsoluteTimeGetCurrent() - relinStart

        // === Step 3: Rescale (drop last modulus) ===
        let rescaleStart = CFAbsoluteTimeGetCurrent()

        let qLast = Q.moduli[kQ - 1]
        var qLastInv: [UInt64] = []
        for i in 0..<newK {
            let qi = Q.moduli[i]
            qLastInv.append(RNSContext.modInverse(qLast % qi, qi))
        }

        guard let qLastInvBuffer = device.makeBuffer(bytes: qLastInv, length: qLastInv.count * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let rescaleOut0 = device.makeBuffer(length: newK * n * MemoryLayout<UInt64>.stride, options: .storageModeShared),
              let rescaleOut1 = device.makeBuffer(length: newK * n * MemoryLayout<UInt64>.stride, options: .storageModeShared) else {
            return nil
        }

        let rescaleGridSize = MTLSize(width: (n + 15) / 16 * 16, height: (newK + 15) / 16 * 16, depth: 1)

        guard let rescaleCmdBuf = commandQueue.makeCommandBuffer() else { return nil }

        // Rescale c0'
        if let enc = rescaleCmdBuf.makeComputeCommandEncoder() {
            enc.setComputePipelineState(rescalePipeline)
            enc.setBuffer(buf_d0, offset: 0, index: 0)
            enc.setBuffer(rescaleOut0, offset: 0, index: 1)
            enc.setBuffer(moduliBuffer, offset: 0, index: 2)
            enc.setBuffer(qLastInvBuffer, offset: 0, index: 3)
            enc.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 4)
            enc.setBytes(&kQVal, length: MemoryLayout<UInt32>.stride, index: 5)
            enc.dispatchThreads(rescaleGridSize, threadsPerThreadgroup: threadgroupSize)
            enc.endEncoding()
        }

        // Rescale c1'
        if let enc = rescaleCmdBuf.makeComputeCommandEncoder() {
            enc.setComputePipelineState(rescalePipeline)
            enc.setBuffer(buf_d1, offset: 0, index: 0)
            enc.setBuffer(rescaleOut1, offset: 0, index: 1)
            enc.setBuffer(moduliBuffer, offset: 0, index: 2)
            enc.setBuffer(qLastInvBuffer, offset: 0, index: 3)
            enc.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 4)
            enc.setBytes(&kQVal, length: MemoryLayout<UInt32>.stride, index: 5)
            enc.dispatchThreads(rescaleGridSize, threadsPerThreadgroup: threadgroupSize)
            enc.endEncoding()
        }

        rescaleCmdBuf.commit()
        rescaleCmdBuf.waitUntilCompleted()

        timings.rescaleTime = CFAbsoluteTimeGetCurrent() - rescaleStart
        timings.totalTime = CFAbsoluteTimeGetCurrent() - totalStart

        // Read results
        let out0Ptr = rescaleOut0.contents().bindMemory(to: UInt64.self, capacity: newK * n)
        let out0Array = Array(UnsafeBufferPointer(start: out0Ptr, count: newK * n))

        let out1Ptr = rescaleOut1.contents().bindMemory(to: UInt64.self, capacity: newK * n)
        let out1Array = Array(UnsafeBufferPointer(start: out1Ptr, count: newK * n))

        return (unflatten(out0Array, kQ: newK, n: n), unflatten(out1Array, kQ: newK, n: n), timings)
    }
}
