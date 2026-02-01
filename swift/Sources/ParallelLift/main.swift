import BigInt
import Foundation

// MARK: - Matrix Generation

/// Generate a random integer matrix with entries in [-bound, bound]
func generateRandomMatrix(n: Int, bound: Int64) -> [BigInt] {
    var matrix: [BigInt] = []
    matrix.reserveCapacity(n * n)

    for _ in 0..<(n * n) {
        let value = Int64.random(in: -bound...bound)
        matrix.append(BigInt(value))
    }

    return matrix
}

/// Generate a matrix with known determinant for testing
func generateTestMatrix(n: Int) -> (matrix: [BigInt], expectedDet: BigInt) {
    // Create a triangular matrix with known determinant
    var matrix = [BigInt](repeating: 0, count: n * n)

    // Set diagonal elements
    var det: BigInt = 1
    for i in 0..<n {
        let diagVal = BigInt(Int.random(in: 1...10))
        matrix[i * n + i] = diagVal
        det *= diagVal

        // Add some off-diagonal elements (upper triangular)
        for j in (i+1)..<n {
            matrix[i * n + j] = BigInt(Int.random(in: -5...5))
        }
    }

    return (matrix, det)
}

// MARK: - Benchmark Runner

struct BenchmarkResult {
    let matrixSize: Int
    let entryBits: Int
    let cpuTime: Double
    let gpuTotalTime: Double
    let gpuComputeTime: Double
    let crtTime: Double
    let speedup: Double
    let numPrimes: Int
    let resultsMatch: Bool
    let determinantBits: Int

    // Statistics from multiple runs
    var cpuTimeStdDev: Double = 0
    var gpuTimeStdDev: Double = 0
    var cpuTimeMin: Double = 0
    var cpuTimeMax: Double = 0
    var gpuTimeMin: Double = 0
    var gpuTimeMax: Double = 0
    var numRuns: Int = 1
}

struct StabilityResult {
    let matrixSize: Int
    let entryBits: Int
    let cpuMean: Double
    let cpuStdDev: Double
    let cpuMin: Double
    let cpuMax: Double
    let gpuMean: Double
    let gpuStdDev: Double
    let gpuMin: Double
    let gpuMax: Double
    let speedupMean: Double
    let numPrimes: Int
    let determinantBits: Int
    let numRuns: Int
}

func runBenchmark(n: Int, entryBound: Int64, verbose: Bool = true) -> BenchmarkResult? {
    if verbose {
        print("\n" + String(repeating: "=", count: 60))
        print("Benchmark: \(n)×\(n) matrix, entries in [-\(entryBound), \(entryBound)]")
        print(String(repeating: "=", count: 60))
    }

    // Generate random matrix
    let matrix = generateRandomMatrix(n: n, bound: entryBound)

    // CPU Baseline (Bareiss)
    if verbose { print("\nRunning CPU baseline (Bareiss algorithm)...") }
    let (cpuResult, cpuTime) = BareissDeterminant.computeTimed(matrix: matrix, n: n)
    if verbose { print(String(format: "CPU time: %.4f s", cpuTime)) }

    // GPU + CRT
    if verbose { print("\nRunning GPU + CRT computation...") }
    guard let gpu = GPUDeterminant() else {
        print("Error: Could not initialize GPU")
        return nil
    }

    guard let (gpuResult, timings) = gpu.computeDeterminant(matrix: matrix, n: n) else {
        print("Error: GPU computation failed")
        return nil
    }

    if verbose {
        timings.printReport()
    }

    // Verify results match
    let resultsMatch = cpuResult == gpuResult

    if verbose {
        print("\n=== Comparison ===")
        print(String(format: "CPU result: %@ (%d bits)", String(cpuResult.description.prefix(50)) + (cpuResult.description.count > 50 ? "..." : ""), cpuResult.bitWidth))
        print(String(format: "GPU result: %@ (%d bits)", String(gpuResult.description.prefix(50)) + (gpuResult.description.count > 50 ? "..." : ""), gpuResult.bitWidth))
        print("Results match: \(resultsMatch ? "✓ YES" : "✗ NO")")

        let speedup = cpuTime / timings.totalTime
        print(String(format: "\nSpeedup: %.2fx", speedup))

        if speedup > 1 {
            print("→ GPU+CRT is faster!")
        } else {
            print("→ CPU is faster (matrix may be too small for GPU overhead)")
        }
    }

    let speedup = cpuTime / timings.totalTime

    return BenchmarkResult(
        matrixSize: n,
        entryBits: Int(log2(Double(entryBound)) + 1),
        cpuTime: cpuTime,
        gpuTotalTime: timings.totalTime,
        gpuComputeTime: timings.gpuComputeTime,
        crtTime: timings.crtReconstructionTime,
        speedup: speedup,
        numPrimes: timings.numPrimes,
        resultsMatch: resultsMatch,
        determinantBits: cpuResult.bitWidth
    )
}

func runCorrectnessTest() {
    print("\n" + String(repeating: "=", count: 60))
    print("Running Correctness Tests")
    print(String(repeating: "=", count: 60))

    // Test 1: Small matrix with known result
    print("\nTest 1: 3×3 matrix with known determinant")
    let (testMatrix, expectedDet) = generateTestMatrix(n: 3)
    let computedDet = BareissDeterminant.compute(matrix: testMatrix, n: 3)
    print("Expected: \(expectedDet)")
    print("Computed: \(computedDet)")
    print("Match: \(expectedDet == computedDet ? "✓" : "✗")")

    // Test 2: Verify GPU matches CPU for small random matrix
    print("\nTest 2: 8×8 random matrix - CPU vs GPU")
    let smallMatrix = generateRandomMatrix(n: 8, bound: 100)
    let cpuSmall = BareissDeterminant.compute(matrix: smallMatrix, n: 8)

    if let gpu = GPUDeterminant(),
       let (gpuSmall, _) = gpu.computeDeterminant(matrix: smallMatrix, n: 8) {
        print("CPU result: \(cpuSmall)")
        print("GPU result: \(gpuSmall)")
        print("Match: \(cpuSmall == gpuSmall ? "✓" : "✗")")
    } else {
        print("GPU initialization failed")
    }

    // Test 3: Zero determinant (singular matrix)
    print("\nTest 3: Singular matrix (should give det = 0)")
    var singularMatrix = [BigInt](repeating: 0, count: 9)
    singularMatrix[0] = 1; singularMatrix[1] = 2; singularMatrix[2] = 3
    singularMatrix[3] = 4; singularMatrix[4] = 5; singularMatrix[5] = 6
    singularMatrix[6] = 5; singularMatrix[7] = 7; singularMatrix[8] = 9  // row3 = row1 + row2
    let singularDet = BareissDeterminant.compute(matrix: singularMatrix, n: 3)
    print("Determinant: \(singularDet)")
    print("Is zero: \(singularDet == 0 ? "✓" : "✗")")

    // Test 4: CRT reconstruction test
    print("\nTest 4: CRT reconstruction")
    // x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7) => x = 23
    let residues: [(residue: UInt32, modulus: UInt32)] = [(2, 3), (3, 5), (2, 7)]
    let reconstructed = CRTReconstruction.reconstruct(residues: residues)
    print("Reconstructed: \(reconstructed)")
    print("Expected: 23")
    print("Match: \(reconstructed == 23 ? "✓" : "✗")")

    // Test 5: Prime generation
    print("\nTest 5: Prime generation")
    let primes = PrimeGenerator.generate31BitPrimes(count: 10)
    print("First 10 31-bit primes: \(primes)")
    let allPrime = primes.allSatisfy { PrimeGenerator.isPrime($0) }
    print("All are prime: \(allPrime ? "✓" : "✗")")
}

func runScalingSuite() {
    print("\n" + String(repeating: "=", count: 60))
    print("SCALING BENCHMARK SUITE")
    print("Testing hypothesis: GPU-CRT beats CPU BigInt at scale")
    print(String(repeating: "=", count: 60))
    fflush(stdout)

    var results: [BenchmarkResult] = []

    // Test different matrix sizes
    // Focus on larger matrices where GPU shines
    let sizes = [48, 64, 80, 96, 112, 128, 160, 192]
    let entryBound: Int64 = 1_000_000  // ~20 bit entries

    for n in sizes {
        print("Starting benchmark for n=\(n)...")
        fflush(stdout)
        autoreleasepool {
            if let result = runBenchmark(n: n, entryBound: entryBound, verbose: true) {
                results.append(result)
            }
        }
        print("Completed benchmark for n=\(n)")
        fflush(stdout)
    }

    // Print summary table
    print("\n" + String(repeating: "=", count: 80))
    fflush(stdout)
    print("SUMMARY TABLE")
    fflush(stdout)
    print(String(repeating: "=", count: 80))
    fflush(stdout)

    print("Size   | CPU (s)    | GPU (s)    | Speedup    | Primes   | Match  | DetBits")
    print(String(repeating: "-", count: 80))
    fflush(stdout)

    for r in results {
        let matchStr = r.resultsMatch ? "✓" : "✗"
        print("\(r.matrixSize)      | \(String(format: "%.4f", r.cpuTime))     | \(String(format: "%.4f", r.gpuTotalTime))     | \(String(format: "%.2f", r.speedup))x      | \(r.numPrimes)       | \(matchStr)      | \(r.determinantBits)")
        fflush(stdout)
    }

    // Find crossover point
    if let firstFaster = results.first(where: { $0.speedup > 1.0 }) {
        print("\n→ GPU becomes faster at \(firstFaster.matrixSize)×\(firstFaster.matrixSize) matrices")
    } else if (results.last?.speedup ?? 0) > 0.5 {
        print("\n→ GPU approaching parity, larger matrices likely to show speedup")
    } else {
        print("\n→ CPU faster for all tested sizes (try larger matrices or entries)")
    }
    fflush(stdout)

    // Export to CSV
    exportToCSV(results: results, filename: "scaling_results.csv")
}

func exportToCSV(results: [BenchmarkResult], filename: String) {
    var csv = "matrix_size,entry_bits,cpu_time_s,gpu_time_s,speedup,num_primes,results_match,determinant_bits\n"
    for r in results {
        csv += "\(r.matrixSize),\(r.entryBits),\(r.cpuTime),\(r.gpuTotalTime),\(r.speedup),\(r.numPrimes),\(r.resultsMatch),\(r.determinantBits)\n"
    }
    do {
        try csv.write(toFile: filename, atomically: true, encoding: .utf8)
        print("\n✓ Results exported to \(filename)")
    } catch {
        print("Error exporting CSV: \(error)")
    }
}

func runLargeEntryTest() {
    print("\n" + String(repeating: "=", count: 60))
    print("LARGE ENTRY TEST")
    print("Testing with larger integer entries (more BigInt pressure)")
    print("Fixed matrix size n=96, varying entry magnitude")
    print(String(repeating: "=", count: 60))
    fflush(stdout)

    // Test with increasingly large entries at fixed matrix size
    // This isolates the effect of BigInt growth
    let n = 96
    let bounds: [(Int64, String)] = [
        (1_000, "10-bit"),
        (100_000, "17-bit"),
        (10_000_000, "24-bit"),
        (1_000_000_000, "30-bit"),
    ]

    var results: [BenchmarkResult] = []

    for (bound, label) in bounds {
        autoreleasepool {
            print("\nEntry magnitude: \(label) (±\(bound))")
            fflush(stdout)
            if let result = runBenchmark(n: n, entryBound: bound, verbose: false) {
                results.append(result)
                print(String(format: "  CPU: %.4fs, GPU: %.4fs, Speedup: %.2fx, Primes: %d, DetBits: %d",
                             result.cpuTime, result.gpuTotalTime, result.speedup, result.numPrimes, result.determinantBits))
                fflush(stdout)
            }
        }
    }

    // Summary
    print("\n" + String(repeating: "=", count: 70))
    print("ENTRY MAGNITUDE SCALING (n=\(n) fixed)")
    print(String(repeating: "=", count: 70))
    print("EntryBits | CPU (s)    | GPU (s)    | Speedup  | DetBits")
    print(String(repeating: "-", count: 70))
    for r in results {
        print(String(format: "%-9d | %-10.4f | %-10.4f | %-8.2fx | %d",
                     r.entryBits, r.cpuTime, r.gpuTotalTime, r.speedup, r.determinantBits))
    }
    fflush(stdout)

    // Export
    exportToCSV(results: results, filename: "entry_magnitude_results.csv")
}

func runStabilityTest() {
    print("\n" + String(repeating: "=", count: 70))
    print("STABILITY TEST")
    print("Multiple runs to measure variance (5 runs per configuration)")
    print(String(repeating: "=", count: 70))
    fflush(stdout)

    let numRuns = 5
    let testConfigs: [(n: Int, bound: Int64)] = [
        (64, 1_000_000),
        (96, 1_000_000),
        (128, 1_000_000),
    ]

    var stabilityResults: [StabilityResult] = []

    for (n, bound) in testConfigs {
        print("\nTesting \(n)×\(n) matrix with ±\(bound) entries (\(numRuns) runs)...")
        fflush(stdout)

        var cpuTimes: [Double] = []
        var gpuTimes: [Double] = []
        var numPrimes = 0
        var detBits = 0

        for run in 1...numRuns {
            autoreleasepool {
                let matrix = generateRandomMatrix(n: n, bound: bound)

                // CPU timing
                let (cpuResult, cpuTime) = BareissDeterminant.computeTimed(matrix: matrix, n: n)
                cpuTimes.append(cpuTime)

                // GPU timing
                if let gpu = GPUDeterminant(),
                   let (gpuResult, timings) = gpu.computeDeterminant(matrix: matrix, n: n) {
                    gpuTimes.append(timings.totalTime)
                    numPrimes = timings.numPrimes

                    // Verify correctness
                    if cpuResult != gpuResult {
                        print("  ⚠️ Run \(run): Results mismatch!")
                    }
                    detBits = cpuResult.bitWidth
                }

                print("  Run \(run): CPU=\(String(format: "%.4f", cpuTime))s, GPU=\(String(format: "%.4f", gpuTimes.last ?? 0))s")
                fflush(stdout)
            }
        }

        // Compute statistics
        let cpuMean = cpuTimes.reduce(0, +) / Double(cpuTimes.count)
        let gpuMean = gpuTimes.reduce(0, +) / Double(gpuTimes.count)

        let cpuVariance = cpuTimes.map { pow($0 - cpuMean, 2) }.reduce(0, +) / Double(cpuTimes.count)
        let gpuVariance = gpuTimes.map { pow($0 - gpuMean, 2) }.reduce(0, +) / Double(gpuTimes.count)

        let cpuStdDev = sqrt(cpuVariance)
        let gpuStdDev = sqrt(gpuVariance)

        let result = StabilityResult(
            matrixSize: n,
            entryBits: Int(log2(Double(bound)) + 1),
            cpuMean: cpuMean,
            cpuStdDev: cpuStdDev,
            cpuMin: cpuTimes.min() ?? 0,
            cpuMax: cpuTimes.max() ?? 0,
            gpuMean: gpuMean,
            gpuStdDev: gpuStdDev,
            gpuMin: gpuTimes.min() ?? 0,
            gpuMax: gpuTimes.max() ?? 0,
            speedupMean: cpuMean / gpuMean,
            numPrimes: numPrimes,
            determinantBits: detBits,
            numRuns: numRuns
        )
        stabilityResults.append(result)
    }

    // Print summary
    print("\n" + String(repeating: "=", count: 100))
    print("STABILITY RESULTS SUMMARY")
    print(String(repeating: "=", count: 100))
    print("Size | CPU Mean±StdDev (s)  | CPU Min/Max      | GPU Mean±StdDev (s)  | GPU Min/Max      | Speedup")
    print(String(repeating: "-", count: 100))

    for r in stabilityResults {
        print(String(format: "%-4d | %.4f ± %.4f      | %.4f / %.4f  | %.4f ± %.4f      | %.4f / %.4f  | %.2fx",
                     r.matrixSize,
                     r.cpuMean, r.cpuStdDev, r.cpuMin, r.cpuMax,
                     r.gpuMean, r.gpuStdDev, r.gpuMin, r.gpuMax,
                     r.speedupMean))
    }
    fflush(stdout)

    // Export
    var csv = "matrix_size,entry_bits,cpu_mean,cpu_stddev,cpu_min,cpu_max,gpu_mean,gpu_stddev,gpu_min,gpu_max,speedup_mean,num_primes,det_bits,num_runs\n"
    for r in stabilityResults {
        csv += "\(r.matrixSize),\(r.entryBits),\(r.cpuMean),\(r.cpuStdDev),\(r.cpuMin),\(r.cpuMax),\(r.gpuMean),\(r.gpuStdDev),\(r.gpuMin),\(r.gpuMax),\(r.speedupMean),\(r.numPrimes),\(r.determinantBits),\(r.numRuns)\n"
    }
    do {
        try csv.write(toFile: "stability_results.csv", atomically: true, encoding: .utf8)
        print("\n✓ Stability results exported to stability_results.csv")
    } catch {
        print("Error exporting CSV: \(error)")
    }
}

// MARK: - Rank Benchmark

struct RankBenchmarkResult {
    let rows: Int
    let cols: Int
    let entryBits: Int
    let cpuTime: Double
    let gpuTime: Double
    let speedup: Double
    let cpuRank: Int
    let gpuRank: Int
    let resultsMatch: Bool
    let numPrimes: Int
}

/// Generate a random matrix with specified rank
func generateRandomMatrixWithRank(m: Int, n: Int, targetRank: Int, bound: Int64) -> [BigInt] {
    // Create a random m×targetRank matrix A and targetRank×n matrix B
    // Their product A*B will have rank at most targetRank
    // Then add small perturbations to get exactly targetRank

    var result = [BigInt](repeating: 0, count: m * n)

    // For simplicity, create a matrix where first targetRank rows are random
    // and remaining rows are linear combinations
    for i in 0..<targetRank {
        for j in 0..<n {
            result[i * n + j] = BigInt(Int64.random(in: -bound...bound))
        }
    }

    // Remaining rows are linear combinations of first targetRank rows
    for i in targetRank..<m {
        // Random coefficients for linear combination
        var coeffs: [Int64] = []
        for _ in 0..<targetRank {
            coeffs.append(Int64.random(in: -3...3))
        }

        for j in 0..<n {
            var sum: BigInt = 0
            for k in 0..<targetRank {
                sum += BigInt(coeffs[k]) * result[k * n + j]
            }
            result[i * n + j] = sum
        }
    }

    return result
}

func runRankBenchmark(m: Int, n: Int, entryBound: Int64, verbose: Bool = true) -> RankBenchmarkResult? {
    if verbose {
        print("\n" + String(repeating: "=", count: 60))
        print("Rank Benchmark: \(m)×\(n) matrix, entries in [-\(entryBound), \(entryBound)]")
        print(String(repeating: "=", count: 60))
    }

    // Generate random full-rank matrix (for square) or near-full-rank
    let matrix = generateRandomMatrix(n: m, bound: entryBound)
    // Reshape to m×n (use first m*n elements, works for square matrices)
    let matrixMN: [BigInt]
    if m == n {
        matrixMN = matrix
    } else {
        matrixMN = Array(matrix.prefix(m * n))
    }

    // CPU Baseline
    if verbose { print("\nRunning CPU baseline (Bareiss rank)...") }
    let (cpuRank, cpuTime) = BareissRank.computeTimed(matrix: matrixMN, m: m, n: n)
    if verbose { print(String(format: "CPU time: %.4f s, Rank: %d", cpuTime, cpuRank)) }

    // GPU + CRT
    if verbose { print("\nRunning GPU + consensus rank...") }
    guard let gpuRank = GPURank() else {
        print("Error: Could not initialize GPU for rank")
        return nil
    }

    guard let (gpuResult, timings) = gpuRank.computeRank(matrix: matrixMN, m: m, n: n) else {
        print("Error: GPU rank computation failed")
        return nil
    }

    if verbose {
        print(String(format: "GPU time: %.4f s, Rank: %d", timings.totalTime, gpuResult))
    }

    let speedup = cpuTime / timings.totalTime
    let match = cpuRank == gpuResult

    if verbose {
        print(String(format: "\nSpeedup: %.2fx", speedup))
        print("Results match: \(match ? "✓" : "✗ (CPU: \(cpuRank), GPU: \(gpuResult))")")
    }

    return RankBenchmarkResult(
        rows: m,
        cols: n,
        entryBits: Int(log2(Double(entryBound)) + 1),
        cpuTime: cpuTime,
        gpuTime: timings.totalTime,
        speedup: speedup,
        cpuRank: cpuRank,
        gpuRank: gpuResult,
        resultsMatch: match,
        numPrimes: timings.numPrimes
    )
}

func runRankScalingSuite() {
    print("\n" + String(repeating: "=", count: 70))
    print("RANK SCALING BENCHMARK SUITE")
    print("Testing exact rank computation: GPU consensus vs CPU Bareiss")
    print(String(repeating: "=", count: 70))
    fflush(stdout)

    var results: [RankBenchmarkResult] = []

    // Test different matrix sizes (square matrices)
    let sizes = [48, 64, 80, 96, 112, 128, 160, 192]
    let entryBound: Int64 = 1_000_000

    for n in sizes {
        print("\nStarting rank benchmark for n=\(n)...")
        fflush(stdout)
        autoreleasepool {
            if let result = runRankBenchmark(m: n, n: n, entryBound: entryBound, verbose: true) {
                results.append(result)
            }
        }
        print("Completed rank benchmark for n=\(n)")
        fflush(stdout)
    }

    // Print summary
    print("\n" + String(repeating: "=", count: 80))
    fflush(stdout)
    print("RANK BENCHMARK SUMMARY")
    fflush(stdout)
    print(String(repeating: "=", count: 80))
    fflush(stdout)

    print("Size   | CPU (s)    | GPU (s)    | Speedup    | Primes   | Match  | Rank")
    print(String(repeating: "-", count: 80))
    fflush(stdout)

    for r in results {
        let matchStr = r.resultsMatch ? "✓" : "✗"
        print(String(format: "%-6d | %-10.4f | %-10.4f | %-10.2fx | %-8d | %-6s | %d",
                     r.rows, r.cpuTime, r.gpuTime, r.speedup, r.numPrimes, matchStr, r.cpuRank))
        fflush(stdout)
    }

    // Find crossover
    if let firstFaster = results.first(where: { $0.speedup > 1.0 }) {
        print("\n→ GPU becomes faster at \(firstFaster.rows)×\(firstFaster.cols) matrices")
    }
    fflush(stdout)

    // Export CSV
    var csv = "rows,cols,entry_bits,cpu_time_s,gpu_time_s,speedup,cpu_rank,gpu_rank,match,num_primes\n"
    for r in results {
        csv += "\(r.rows),\(r.cols),\(r.entryBits),\(r.cpuTime),\(r.gpuTime),\(r.speedup),\(r.cpuRank),\(r.gpuRank),\(r.resultsMatch),\(r.numPrimes)\n"
    }
    do {
        try csv.write(toFile: "rank_scaling_results.csv", atomically: true, encoding: .utf8)
        print("\n✓ Rank results exported to rank_scaling_results.csv")
    } catch {
        print("Error exporting CSV: \(error)")
    }
}

func runRankDeficientTest() {
    print("\n" + String(repeating: "=", count: 70))
    print("RANK-DEFICIENT MATRIX TEST")
    print("Testing matrices with known rank < min(m,n)")
    print(String(repeating: "=", count: 70))
    fflush(stdout)

    let n = 96
    let entryBound: Int64 = 1_000_000

    // Test matrices with various ranks
    let targetRanks = [10, 30, 50, 70, 90, 96]

    for targetRank in targetRanks {
        print("\n--- Target rank: \(targetRank) (matrix \(n)×\(n)) ---")
        fflush(stdout)

        autoreleasepool {
            let matrix = generateRandomMatrixWithRank(m: n, n: n, targetRank: targetRank, bound: entryBound)

            let (cpuRank, cpuTime) = BareissRank.computeTimed(matrix: matrix, m: n, n: n)

            if let gpuRankEngine = GPURank(),
               let (gpuRank, timings) = gpuRankEngine.computeRank(matrix: matrix, m: n, n: n) {
                let speedup = cpuTime / timings.totalTime
                let match = cpuRank == gpuRank

                print(String(format: "  CPU: rank=%d, time=%.4fs", cpuRank, cpuTime))
                print(String(format: "  GPU: rank=%d, time=%.4fs, speedup=%.2fx", gpuRank, timings.totalTime, speedup))
                print("  Match: \(match ? "✓" : "✗")")
                fflush(stdout)
            }
        }
    }
}

// MARK: - Solve Benchmark

struct SolveBenchmarkResult {
    let matrixSize: Int
    let entryBits: Int
    let cpuTime: Double
    let gpuTime: Double
    let speedup: Double
    let resultsMatch: Bool
    let numPrimes: Int
    let solutionBits: Int
}

/// Generate a random vector with entries in [-bound, bound]
func generateRandomVector(n: Int, bound: Int64) -> [BigInt] {
    var vector: [BigInt] = []
    vector.reserveCapacity(n)
    for _ in 0..<n {
        vector.append(BigInt(Int64.random(in: -bound...bound)))
    }
    return vector
}

/// Verify that two rational solutions match
func solutionsMatch(_ a: [Rational]?, _ b: [Rational]?) -> Bool {
    guard let a = a, let b = b, a.count == b.count else {
        return a == nil && b == nil
    }
    for i in 0..<a.count {
        // Compare a[i] and b[i] as rationals
        // a.num/a.den == b.num/b.den  =>  a.num * b.den == b.num * a.den
        if a[i].numerator * b[i].denominator != b[i].numerator * a[i].denominator {
            return false
        }
    }
    return true
}

func runSolveBenchmark(n: Int, entryBound: Int64, verbose: Bool = true) -> SolveBenchmarkResult? {
    if verbose {
        print("\n" + String(repeating: "=", count: 60))
        print("Solve Benchmark: \(n)×\(n) system, entries in [-\(entryBound), \(entryBound)]")
        print(String(repeating: "=", count: 60))
    }

    // Generate random matrix A and vector b
    let A = generateRandomMatrix(n: n, bound: entryBound)
    let b = generateRandomVector(n: n, bound: entryBound)

    // CPU Baseline
    if verbose { print("\nRunning CPU baseline (Bareiss solve)...") }
    let (cpuSolution, cpuTime) = BareissSolve.solveTimed(A: A, b: b, n: n)
    if verbose {
        if cpuSolution != nil {
            print(String(format: "CPU time: %.4f s", cpuTime))
        } else {
            print("CPU: Matrix is singular")
        }
    }

    // GPU + CRT
    if verbose { print("\nRunning GPU + CRT solve...") }
    guard let gpuSolver = GPUSolve() else {
        print("Error: Could not initialize GPU for solve")
        return nil
    }

    guard let (gpuSolution, timings) = gpuSolver.solve(A: A, b: b, n: n) else {
        print("Error: GPU solve computation failed")
        return nil
    }

    if verbose {
        if gpuSolution != nil {
            print(String(format: "GPU time: %.4f s", timings.totalTime))
        } else {
            print("GPU: Matrix is singular")
        }
    }

    let speedup = cpuTime / timings.totalTime
    let match = solutionsMatch(cpuSolution, gpuSolution)

    if verbose {
        print(String(format: "\nSpeedup: %.2fx", speedup))
        print("Results match: \(match ? "✓" : "✗")")
        if !match && cpuSolution != nil && gpuSolution != nil {
            print("  CPU x[0]: \(cpuSolution![0])")
            print("  GPU x[0]: \(gpuSolution![0])")
        }
    }

    return SolveBenchmarkResult(
        matrixSize: n,
        entryBits: Int(log2(Double(entryBound)) + 1),
        cpuTime: cpuTime,
        gpuTime: timings.totalTime,
        speedup: speedup,
        resultsMatch: match,
        numPrimes: timings.numPrimes,
        solutionBits: timings.solutionBits
    )
}

func runSolveScalingSuite() {
    print("\n" + String(repeating: "=", count: 70))
    print("SOLVE SCALING BENCHMARK SUITE")
    print("Testing exact solve Ax = b: GPU+CRT vs CPU Bareiss (Cramer's rule)")
    print("Note: Solve requires n+1 determinant computations per system")
    print(String(repeating: "=", count: 70))
    fflush(stdout)

    var results: [SolveBenchmarkResult] = []

    // Test different matrix sizes (smaller than det/rank since solve is O(n) more expensive)
    let sizes = [24, 32, 48, 64, 80, 96]
    let entryBound: Int64 = 1_000_000

    for n in sizes {
        print("\nStarting solve benchmark for n=\(n)...")
        fflush(stdout)
        autoreleasepool {
            if let result = runSolveBenchmark(n: n, entryBound: entryBound, verbose: true) {
                results.append(result)
            }
        }
        print("Completed solve benchmark for n=\(n)")
        fflush(stdout)
    }

    // Print summary
    print("\n" + String(repeating: "=", count: 85))
    fflush(stdout)
    print("SOLVE BENCHMARK SUMMARY")
    fflush(stdout)
    print(String(repeating: "=", count: 85))
    fflush(stdout)

    print("Size   | CPU (s)    | GPU (s)    | Speedup    | Primes   | Match  | SolBits")
    print(String(repeating: "-", count: 85))
    fflush(stdout)

    for r in results {
        let matchStr = r.resultsMatch ? "✓" : "✗"
        print(String(format: "%-6d | %-10.4f | %-10.4f | %-10.2fx | %-8d | %-6s | %d",
                     r.matrixSize, r.cpuTime, r.gpuTime, r.speedup, r.numPrimes, matchStr, r.solutionBits))
        fflush(stdout)
    }

    // Find crossover
    if let firstFaster = results.first(where: { $0.speedup > 1.0 }) {
        print("\n→ GPU becomes faster at \(firstFaster.matrixSize)×\(firstFaster.matrixSize) systems")
    }
    fflush(stdout)

    // Export CSV
    var csv = "matrix_size,entry_bits,cpu_time_s,gpu_time_s,speedup,num_primes,match,solution_bits\n"
    for r in results {
        csv += "\(r.matrixSize),\(r.entryBits),\(r.cpuTime),\(r.gpuTime),\(r.speedup),\(r.numPrimes),\(r.resultsMatch),\(r.solutionBits)\n"
    }
    do {
        try csv.write(toFile: "solve_scaling_results.csv", atomically: true, encoding: .utf8)
        print("\n✓ Solve results exported to solve_scaling_results.csv")
    } catch {
        print("Error exporting CSV: \(error)")
    }
}

func runSolveCorrectnessTest() {
    print("\n" + String(repeating: "=", count: 70))
    print("SOLVE CORRECTNESS TEST")
    print("Testing exact solve with verification")
    print(String(repeating: "=", count: 70))
    fflush(stdout)

    let testCases = [
        (n: 4, bound: Int64(10)),
        (n: 8, bound: Int64(100)),
        (n: 16, bound: Int64(1000)),
        (n: 32, bound: Int64(10000)),
    ]

    for (n, bound) in testCases {
        print("\nTesting \(n)×\(n) system with ±\(bound) entries...")
        fflush(stdout)

        autoreleasepool {
            let A = generateRandomMatrix(n: n, bound: bound)
            let b = generateRandomVector(n: n, bound: bound)

            // CPU solve
            let (cpuSol, cpuTime) = BareissSolve.solveTimed(A: A, b: b, n: n)

            // GPU solve
            if let gpuSolver = GPUSolve(),
               let (gpuSol, timings) = gpuSolver.solve(A: A, b: b, n: n) {

                let match = solutionsMatch(cpuSol, gpuSol)

                // Verify CPU solution
                var cpuVerified = false
                if let sol = cpuSol {
                    cpuVerified = BareissSolve.verifySolution(A: A, b: b, x: sol, n: n)
                }

                print(String(format: "  CPU: %.4fs, verified: %@", cpuTime, cpuVerified ? "✓" : "✗"))
                print(String(format: "  GPU: %.4fs, primes: %d", timings.totalTime, timings.numPrimes))
                print("  Match: \(match ? "✓" : "✗")")

                if !match && cpuSol != nil && gpuSol != nil {
                    print("  CPU x[0] = \(cpuSol![0])")
                    print("  GPU x[0] = \(gpuSol![0])")
                }
                fflush(stdout)
            }
        }
    }
}

// MARK: - Multi-RHS Solve Benchmark (AX = B)

struct MultiRHSBenchmarkResult {
    let matrixSize: Int
    let numRHS: Int
    let cpuCramerTime: Double      // CPU using Cramer's rule (naive)
    let cpuLUTime: Double          // CPU using LU factorization (fair comparison)
    let cpuLUFactorTime: Double    // Just the factorization part
    let cpuLUSolveTime: Double     // Just the solve part
    let gpuTime: Double
    let gpuSolveTime: Double       // GPU modular solve (elim + backsub)
    let gpuDetATime: Double        // GPU det(A) computation
    let gpuCRTTime: Double         // CRT reconstruction
    let speedupVsCramer: Double
    let speedupVsLU: Double
    let resultsMatch: Bool
    let maxSolutionBits: Int       // Max bit-length of solution numerators
    let avgSolutionBits: Double    // Avg bit-length of solution numerators

    // Per-RHS metrics for amortization analysis
    var cpuLUPerRHS: Double { cpuLUSolveTime / Double(numRHS) }        // Asymptotic cost per RHS
    var gpuPerRHS: Double { gpuTime / Double(numRHS) }                  // Total GPU time per RHS
    var gpuSolvePerRHS: Double { gpuSolveTime / Double(numRHS) }        // GPU solve per RHS
    var gpuCRTPerRHS: Double { gpuCRTTime / Double(numRHS) }            // CRT per RHS
    var gpuFixedOverhead: Double { gpuDetATime }                        // Fixed cost (det(A))
}

func runMultiRHSBenchmark() {
    print("\n" + String(repeating: "=", count: 80))
    print("MULTI-RHS SOLVE BENCHMARK (AX = B)")
    print("Testing how speedup scales with number of right-hand sides k")
    print("Comparing: CPU Cramer (naive) vs CPU LU (factor once) vs GPU+CRT")
    print(String(repeating: "=", count: 80))
    fflush(stdout)

    var results: [MultiRHSBenchmarkResult] = []

    // Test configurations: (matrix size, entry bound)
    let matrixConfigs = [
        (n: 24, bound: Int64(1000)),
        (n: 32, bound: Int64(1000)),
        (n: 48, bound: Int64(1000)),
        (n: 64, bound: Int64(1000)),  // Added larger size
    ]

    // Number of RHS to test
    let kValues = [1, 2, 4, 8, 16, 32]

    for (n, entryBound) in matrixConfigs {
        print("\n" + String(repeating: "-", count: 70))
        print("Matrix size: \(n)×\(n), entries in [-\(entryBound), \(entryBound)]")
        print(String(repeating: "-", count: 70))
        fflush(stdout)

        // Generate one random matrix A for all k values
        let A = generateRandomMatrix(n: n, bound: entryBound)

        for k in kValues {
            // Skip expensive combinations for n=64
            if n == 64 && k > 16 {
                continue
            }

            print("\n  k = \(k) RHS vectors:")
            fflush(stdout)

            autoreleasepool {
                // Generate k random RHS vectors (stored column-major in B)
                var B: [BigInt] = []
                for _ in 0..<k {
                    for _ in 0..<n {
                        B.append(BigInt(Int64.random(in: -entryBound...entryBound)))
                    }
                }

                // === CPU Cramer baseline (naive) ===
                let cpuCramerStart = CFAbsoluteTimeGetCurrent()
                let cpuCramerSolution = BareissMultiRHSSolve.solve(A: A, B: B, n: n, k: k)
                let cpuCramerTime = CFAbsoluteTimeGetCurrent() - cpuCramerStart

                print(String(format: "    CPU Cramer: %.4fs (%.4fs/RHS) - O(n*k) determinants",
                            cpuCramerTime, cpuCramerTime / Double(k)))
                fflush(stdout)

                // === CPU LU baseline (factor once) ===
                let (cpuLUSolution, luTimings) = BareissLUMultiRHSSolve.solve(A: A, B: B, n: n, k: k)

                print(String(format: "    CPU LU:     %.4fs (Factor: %.4fs, Solve: %.4fs)",
                            luTimings.totalTime, luTimings.factorTime, luTimings.solveTime))
                fflush(stdout)

                // === GPU ===
                var gpuTime = cpuCramerTime
                var gpuSolution: [[Rational]]? = nil
                var timings = MultiRHSSolveTimings()

                if let gpuSolver = GPUMultiRHSSolve() {
                    if let (sol, gpuTimings) = gpuSolver.solve(A: A, B: B, n: n, k: k) {
                        gpuSolution = sol
                        gpuTime = gpuTimings.totalTime
                        timings = gpuTimings

                        print(String(format: "    GPU:        %.4fs (Solve: %.4fs, DetA: %.4fs, CRT: %.4fs), primes: %d",
                                    gpuTime, timings.gpuSolveTime, timings.detATime, timings.crtReconstructTime, timings.numPrimes))
                    } else {
                        print("    GPU: Solve failed (singular?)")
                    }
                } else {
                    print("    GPU: Not available")
                }
                fflush(stdout)

                // Compute solution bit-length statistics
                var maxBits = 0
                var totalBits = 0
                var count = 0
                if let sol = gpuSolution {
                    for col in sol {
                        for r in col {
                            let bits = r.numerator.magnitude.bitWidth
                            maxBits = max(maxBits, bits)
                            totalBits += bits
                            count += 1
                        }
                    }
                }
                let avgBits = count > 0 ? Double(totalBits) / Double(count) : 0.0

                // Verify results match (check GPU vs CPU Cramer)
                var match = false
                if let cramer = cpuCramerSolution, let gpu = gpuSolution {
                    match = true
                    outer: for col in 0..<k {
                        for row in 0..<n {
                            if cramer[col][row] != gpu[col][row] {
                                match = false
                                break outer
                            }
                        }
                    }
                } else if cpuCramerSolution == nil && gpuSolution == nil {
                    match = true  // Both report singular
                }

                // Also verify LU matches Cramer
                var luMatch = false
                if let cramer = cpuCramerSolution, let lu = cpuLUSolution {
                    luMatch = true
                    outer2: for col in 0..<k {
                        for row in 0..<n {
                            if cramer[col][row] != lu[col][row] {
                                luMatch = false
                                break outer2
                            }
                        }
                    }
                }

                let speedupVsCramer = cpuCramerTime / gpuTime
                let speedupVsLU = luTimings.totalTime / gpuTime
                print(String(format: "    Speedup vs Cramer: %.2fx, vs LU: %.2fx, Match: %@ (LU match: %@)",
                            speedupVsCramer, speedupVsLU, match ? "✓" : "✗", luMatch ? "✓" : "✗"))
                print(String(format: "    Solution bits: max=%d, avg=%.1f", maxBits, avgBits))

                results.append(MultiRHSBenchmarkResult(
                    matrixSize: n,
                    numRHS: k,
                    cpuCramerTime: cpuCramerTime,
                    cpuLUTime: luTimings.totalTime,
                    cpuLUFactorTime: luTimings.factorTime,
                    cpuLUSolveTime: luTimings.solveTime,
                    gpuTime: gpuTime,
                    gpuSolveTime: timings.gpuSolveTime,
                    gpuDetATime: timings.detATime,
                    gpuCRTTime: timings.crtReconstructTime,
                    speedupVsCramer: speedupVsCramer,
                    speedupVsLU: speedupVsLU,
                    resultsMatch: match,
                    maxSolutionBits: maxBits,
                    avgSolutionBits: avgBits
                ))
            }
        }
    }

    // Print summary
    print("\n" + String(repeating: "=", count: 140))
    print("MULTI-RHS SOLVE SUMMARY")
    print(String(repeating: "=", count: 140))
    print("Size | k   | CPU Cramer | CPU LU    | GPU       | vs Cramer | vs LU  | GPU Breakdown (S/D/C)     | Sol Bits")
    print(String(repeating: "-", count: 140))

    for r in results {
        let sizePad = String(r.matrixSize).padding(toLength: 4, withPad: " ", startingAt: 0)
        let kPad = String(r.numRHS).padding(toLength: 3, withPad: " ", startingAt: 0)
        let cramerStr = String(format: "%.4fs", r.cpuCramerTime).padding(toLength: 10, withPad: " ", startingAt: 0)
        let luStr = String(format: "%.4fs", r.cpuLUTime).padding(toLength: 9, withPad: " ", startingAt: 0)
        let gpuStr = String(format: "%.4fs", r.gpuTime).padding(toLength: 9, withPad: " ", startingAt: 0)
        let speedupCramer = String(format: "%.1fx", r.speedupVsCramer).padding(toLength: 9, withPad: " ", startingAt: 0)
        let speedupLU = String(format: "%.1fx", r.speedupVsLU).padding(toLength: 6, withPad: " ", startingAt: 0)
        // S=Solve, D=DetA, C=CRT
        let breakdown = String(format: "S:%.3f D:%.3f C:%.3f", r.gpuSolveTime, r.gpuDetATime, r.gpuCRTTime)
        let bitsStr = String(format: "%d/%.0f", r.maxSolutionBits, r.avgSolutionBits)
        print("\(sizePad) | \(kPad) | \(cramerStr) | \(luStr) | \(gpuStr) | \(speedupCramer) | \(speedupLU) | \(breakdown) | \(bitsStr)")
    }
    fflush(stdout)

    // Print amortization analysis (per-RHS breakdown)
    print("\n" + String(repeating: "=", count: 100))
    print("AMORTIZATION ANALYSIS: Per-RHS Cost vs k")
    print("Shows how GPU cost per RHS decreases as k increases (fixed overhead amortized)")
    print(String(repeating: "=", count: 100))
    print("Size | k   | CPU LU/RHS  | GPU/RHS    | GPU Solve/RHS | GPU CRT/RHS | Fixed (DetA)")
    print(String(repeating: "-", count: 100))

    for r in results {
        let sizePad = String(r.matrixSize).padding(toLength: 4, withPad: " ", startingAt: 0)
        let kPad = String(r.numRHS).padding(toLength: 3, withPad: " ", startingAt: 0)
        let cpuPerRHS = String(format: "%.4fs", r.cpuLUPerRHS).padding(toLength: 11, withPad: " ", startingAt: 0)
        let gpuPerRHS = String(format: "%.4fs", r.gpuPerRHS).padding(toLength: 10, withPad: " ", startingAt: 0)
        let gpuSolvePerRHS = String(format: "%.4fs", r.gpuSolvePerRHS).padding(toLength: 13, withPad: " ", startingAt: 0)
        let gpuCRTPerRHS = String(format: "%.4fs", r.gpuCRTPerRHS).padding(toLength: 11, withPad: " ", startingAt: 0)
        let fixedOverhead = String(format: "%.4fs", r.gpuFixedOverhead)
        print("\(sizePad) | \(kPad) | \(cpuPerRHS) | \(gpuPerRHS) | \(gpuSolvePerRHS) | \(gpuCRTPerRHS) | \(fixedOverhead)")
    }

    // Show amortization factor per matrix size
    print("\n--- Amortization Summary (k=1 vs k=max) ---")
    let sizes = Set(results.map { $0.matrixSize }).sorted()
    for size in sizes {
        let sizeResults = results.filter { $0.matrixSize == size }
        if let first = sizeResults.first, let last = sizeResults.last, first.numRHS != last.numRHS {
            let amortFactor = first.gpuPerRHS / last.gpuPerRHS
            print(String(format: "  n=%d: GPU/RHS drops %.1fx from k=%d (%.4fs) to k=%d (%.4fs)",
                        size, amortFactor, first.numRHS, first.gpuPerRHS, last.numRHS, last.gpuPerRHS))
        }
    }
    fflush(stdout)

    // Export CSV
    var csv = "matrix_size,num_rhs,cpu_cramer_time,cpu_lu_time,cpu_lu_factor,cpu_lu_solve,"
    csv += "gpu_time,gpu_solve,gpu_det_a,gpu_crt,speedup_vs_cramer,speedup_vs_lu,match,max_sol_bits,avg_sol_bits,"
    csv += "cpu_lu_per_rhs,gpu_per_rhs,gpu_solve_per_rhs,gpu_crt_per_rhs\n"
    for r in results {
        csv += "\(r.matrixSize),\(r.numRHS),\(r.cpuCramerTime),\(r.cpuLUTime),\(r.cpuLUFactorTime),\(r.cpuLUSolveTime),"
        csv += "\(r.gpuTime),\(r.gpuSolveTime),\(r.gpuDetATime),\(r.gpuCRTTime),"
        csv += "\(r.speedupVsCramer),\(r.speedupVsLU),\(r.resultsMatch),\(r.maxSolutionBits),\(r.avgSolutionBits),"
        csv += "\(r.cpuLUPerRHS),\(r.gpuPerRHS),\(r.gpuSolvePerRHS),\(r.gpuCRTPerRHS)\n"
    }
    do {
        try csv.write(toFile: "multi_rhs_results.csv", atomically: true, encoding: .utf8)
        print("\n✓ Multi-RHS results exported to multi_rhs_results.csv")
    } catch {
        print("Error exporting CSV: \(error)")
    }
}

// MARK: - Inverse Benchmark

struct InverseBenchmarkResult {
    let matrixSize: Int
    let entryBits: Int
    let cpuTime: Double
    let gpuTime: Double
    let speedup: Double
    let resultsMatch: Bool
    let numPrimes: Int
    let determinantBits: Int
}

func runInverseBenchmark(n: Int, entryBound: Int64, verbose: Bool = true) -> InverseBenchmarkResult? {
    if verbose {
        print("\n" + String(repeating: "=", count: 60))
        print("Inverse Benchmark: \(n)×\(n) matrix, entries in [-\(entryBound), \(entryBound)]")
        print(String(repeating: "=", count: 60))
    }

    // Generate random matrix
    let A = generateRandomMatrix(n: n, bound: entryBound)

    // CPU Baseline
    if verbose { print("\nRunning CPU baseline (Bareiss inverse)...") }
    let (cpuResult, cpuTime) = BareissInverse.inverseTimed(A: A, n: n)

    var cpuInverse: [Rational]? = nil
    var detBits = 0

    switch cpuResult {
    case .success(let matrix, let det):
        cpuInverse = matrix
        detBits = det.bitWidth
        if verbose { print(String(format: "CPU time: %.4f s", cpuTime)) }
    case .singular(let rank):
        if verbose { print("CPU: Matrix is singular (rank = \(rank))") }
        return nil
    }

    // GPU + CRT
    if verbose { print("\nRunning GPU + CRT inverse...") }
    guard let gpuInv = GPUInverse() else {
        print("Error: Could not initialize GPU for inverse")
        return nil
    }

    guard let (gpuResult, timings) = gpuInv.inverse(A: A, n: n) else {
        print("Error: GPU inverse computation failed")
        return nil
    }

    var gpuInverse: [Rational]? = nil

    switch gpuResult {
    case .success(let matrix, _):
        gpuInverse = matrix
        if verbose { print(String(format: "GPU time: %.4f s", timings.totalTime)) }
    case .singular(let rank):
        if verbose { print("GPU: Matrix is singular (rank = \(rank))") }
        return nil
    }

    let speedup = cpuTime / timings.totalTime

    // Verify results match
    var match = false
    if let cpuInv = cpuInverse, let gpuInv = gpuInverse, cpuInv.count == gpuInv.count {
        match = true
        for i in 0..<cpuInv.count {
            if cpuInv[i] != gpuInv[i] {
                match = false
                break
            }
        }
    }

    if verbose {
        print(String(format: "\nSpeedup: %.2fx", speedup))
        print("Results match: \(match ? "✓" : "✗")")
    }

    return InverseBenchmarkResult(
        matrixSize: n,
        entryBits: Int(log2(Double(entryBound)) + 1),
        cpuTime: cpuTime,
        gpuTime: timings.totalTime,
        speedup: speedup,
        resultsMatch: match,
        numPrimes: timings.numPrimes,
        determinantBits: detBits
    )
}

func runInverseScalingSuite() {
    print("\n" + String(repeating: "=", count: 70))
    print("INVERSE SCALING BENCHMARK SUITE")
    print("Testing exact matrix inverse: GPU+CRT vs CPU Bareiss")
    print("Note: Inverse requires n² determinant computations")
    print(String(repeating: "=", count: 70))
    fflush(stdout)

    var results: [InverseBenchmarkResult] = []

    // Smaller sizes since inverse is O(n²) more expensive than det
    let sizes = [16, 24, 32, 48]
    let entryBound: Int64 = 1_000_000

    for n in sizes {
        print("\nStarting inverse benchmark for n=\(n)...")
        fflush(stdout)
        autoreleasepool {
            if let result = runInverseBenchmark(n: n, entryBound: entryBound, verbose: true) {
                results.append(result)
            }
        }
        print("Completed inverse benchmark for n=\(n)")
        fflush(stdout)
    }

    // Print summary
    print("\n" + String(repeating: "=", count: 85))
    fflush(stdout)
    print("INVERSE BENCHMARK SUMMARY")
    fflush(stdout)
    print(String(repeating: "=", count: 85))
    fflush(stdout)

    print("Size   | CPU (s)    | GPU (s)    | Speedup    | Primes   | Match  | DetBits")
    print(String(repeating: "-", count: 85))
    fflush(stdout)

    for r in results {
        let matchStr = r.resultsMatch ? "✓" : "✗"
        print(String(format: "%-6d | %-10.4f | %-10.4f | %-10.2fx | %-8d | %-6s | %d",
                     r.matrixSize, r.cpuTime, r.gpuTime, r.speedup, r.numPrimes, matchStr, r.determinantBits))
        fflush(stdout)
    }

    if let firstFaster = results.first(where: { $0.speedup > 1.0 }) {
        print("\n→ GPU becomes faster at \(firstFaster.matrixSize)×\(firstFaster.matrixSize) matrices")
    }
    fflush(stdout)

    // Export CSV
    var csv = "matrix_size,entry_bits,cpu_time_s,gpu_time_s,speedup,num_primes,match,det_bits\n"
    for r in results {
        csv += "\(r.matrixSize),\(r.entryBits),\(r.cpuTime),\(r.gpuTime),\(r.speedup),\(r.numPrimes),\(r.resultsMatch),\(r.determinantBits)\n"
    }
    do {
        try csv.write(toFile: "inverse_scaling_results.csv", atomically: true, encoding: .utf8)
        print("\n✓ Inverse results exported to inverse_scaling_results.csv")
    } catch {
        print("Error exporting CSV: \(error)")
    }
}

// MARK: - Batched Inverse Benchmark (via Multi-RHS Solve)

struct BatchedInverseBenchmarkResult {
    let matrixSize: Int
    let entryBits: Int
    let cpuCramerTime: Double      // CPU using n² minors (Cramer's rule)
    let gpuCramerTime: Double      // GPU using n² minors (Cramer's rule)
    let gpuBatchedTime: Double     // GPU using multi-RHS solve (A·X = I)
    let speedupVsCPU: Double
    let speedupVsGPUCramer: Double
    let batchedResultsMatch: Bool
    let numPrimes: Int
    let determinantBits: Int
    // Timing breakdown for batched approach
    let gpuSolveTime: Double
    let gpuDetATime: Double
    let gpuCRTTime: Double
}

func runBatchedInverseBenchmark() {
    print("\n" + String(repeating: "=", count: 90))
    print("BATCHED INVERSE BENCHMARK (A·X = I via Multi-RHS)")
    print("Comparing: CPU Cramer (n² minors) vs GPU Cramer vs GPU Batched Solve")
    print("GPU Batched is O(n³) vs O(n⁵) for Cramer's rule")
    print(String(repeating: "=", count: 90))
    fflush(stdout)

    var results: [BatchedInverseBenchmarkResult] = []

    // Test sizes - batched approach can handle larger matrices efficiently
    let sizes = [16, 24, 32, 48, 64]
    let entryBound: Int64 = 1_000

    for n in sizes {
        print("\n" + String(repeating: "-", count: 70))
        print("Matrix size: \(n)×\(n), entries in [-\(entryBound), \(entryBound)]")
        print(String(repeating: "-", count: 70))
        fflush(stdout)

        autoreleasepool {
            let A = generateRandomMatrix(n: n, bound: entryBound)

            // === CPU Cramer baseline (n² minors) ===
            print("  Running CPU Cramer (n² determinants)...")
            fflush(stdout)
            let (cpuResult, cpuTime) = BareissInverse.inverseTimed(A: A, n: n)

            var cpuInverse: [Rational]? = nil
            var detBits = 0

            switch cpuResult {
            case .success(let matrix, let det):
                cpuInverse = matrix
                detBits = det.bitWidth
                print(String(format: "    CPU Cramer: %.4fs (det bits: %d)", cpuTime, detBits))
            case .singular(let rank):
                print("    CPU: Matrix is singular (rank = \(rank))")
                return
            }
            fflush(stdout)

            // === GPU Cramer (n² minors) ===
            var gpuCramerTime = cpuTime
            var gpuCramerInverse: [Rational]? = nil

            // Skip GPU Cramer for larger matrices (too slow)
            if n <= 32 {
                print("  Running GPU Cramer (n² determinants)...")
                fflush(stdout)
                if let gpuInv = GPUInverse(),
                   let (gpuResult, gpuTimings) = gpuInv.inverse(A: A, n: n) {
                    switch gpuResult {
                    case .success(let matrix, _):
                        gpuCramerInverse = matrix
                        gpuCramerTime = gpuTimings.totalTime
                        print(String(format: "    GPU Cramer: %.4fs", gpuCramerTime))
                    case .singular:
                        print("    GPU Cramer: Matrix is singular")
                    }
                }
            } else {
                print("    GPU Cramer: Skipped (n > 32, too slow)")
            }
            fflush(stdout)

            // === GPU Batched Solve (A·X = I) ===
            print("  Running GPU Batched Solve (A·X = I)...")
            fflush(stdout)
            var gpuBatchedTime = cpuTime
            var gpuBatchedInverse: [Rational]? = nil
            var batchedTimings = BatchedInverseTimings()
            var numPrimes = 0

            if let batchedInv = GPUBatchedInverse(),
               let (batchedResult, timings) = batchedInv.inverse(A: A, n: n) {
                batchedTimings = timings
                numPrimes = timings.numPrimes
                switch batchedResult {
                case .success(let matrix, _):
                    gpuBatchedInverse = matrix
                    gpuBatchedTime = timings.totalTime
                    print(String(format: "    GPU Batched: %.4fs (Solve: %.4f, DetA: %.4f, CRT: %.4f)",
                                gpuBatchedTime, timings.gpuSolveTime, timings.detATime, timings.crtReconstructTime))
                case .singular:
                    print("    GPU Batched: Matrix is singular")
                }
            }
            fflush(stdout)

            // Verify batched result matches CPU
            var batchedMatch = false
            if let cpuInv = cpuInverse, let batchedInv = gpuBatchedInverse, cpuInv.count == batchedInv.count {
                batchedMatch = true
                for i in 0..<cpuInv.count {
                    if cpuInv[i] != batchedInv[i] {
                        batchedMatch = false
                        break
                    }
                }
            }

            let speedupVsCPU = cpuTime / gpuBatchedTime
            let speedupVsGPUCramer = gpuCramerTime / gpuBatchedTime

            print(String(format: "    Speedup vs CPU Cramer: %.1fx, vs GPU Cramer: %.1fx, Match: %@",
                        speedupVsCPU, speedupVsGPUCramer, batchedMatch ? "✓" : "✗"))
            fflush(stdout)

            results.append(BatchedInverseBenchmarkResult(
                matrixSize: n,
                entryBits: Int(log2(Double(entryBound)) + 1),
                cpuCramerTime: cpuTime,
                gpuCramerTime: gpuCramerTime,
                gpuBatchedTime: gpuBatchedTime,
                speedupVsCPU: speedupVsCPU,
                speedupVsGPUCramer: speedupVsGPUCramer,
                batchedResultsMatch: batchedMatch,
                numPrimes: numPrimes,
                determinantBits: detBits,
                gpuSolveTime: batchedTimings.gpuSolveTime,
                gpuDetATime: batchedTimings.detATime,
                gpuCRTTime: batchedTimings.crtReconstructTime
            ))
        }
    }

    // Print summary
    print("\n" + String(repeating: "=", count: 120))
    print("BATCHED INVERSE SUMMARY")
    print(String(repeating: "=", count: 120))
    print("Size | CPU Cramer  | GPU Cramer  | GPU Batched | vs CPU   | vs Cramer | GPU Breakdown (S/D/C)      | Match")
    print(String(repeating: "-", count: 120))

    for r in results {
        let sizePad = String(r.matrixSize).padding(toLength: 4, withPad: " ", startingAt: 0)
        let cpuStr = String(format: "%.4fs", r.cpuCramerTime).padding(toLength: 11, withPad: " ", startingAt: 0)
        let gpuCramerStr = String(format: "%.4fs", r.gpuCramerTime).padding(toLength: 11, withPad: " ", startingAt: 0)
        let gpuBatchedStr = String(format: "%.4fs", r.gpuBatchedTime).padding(toLength: 11, withPad: " ", startingAt: 0)
        let speedupCPU = String(format: "%.1fx", r.speedupVsCPU).padding(toLength: 8, withPad: " ", startingAt: 0)
        let speedupCramer = String(format: "%.1fx", r.speedupVsGPUCramer).padding(toLength: 9, withPad: " ", startingAt: 0)
        let breakdown = String(format: "S:%.3f D:%.3f C:%.3f", r.gpuSolveTime, r.gpuDetATime, r.gpuCRTTime)
        let matchStr = r.batchedResultsMatch ? "✓" : "✗"
        print("\(sizePad) | \(cpuStr) | \(gpuCramerStr) | \(gpuBatchedStr) | \(speedupCPU) | \(speedupCramer) | \(breakdown) | \(matchStr)")
    }
    fflush(stdout)

    // Export CSV
    var csv = "matrix_size,entry_bits,cpu_cramer_time,gpu_cramer_time,gpu_batched_time,"
    csv += "speedup_vs_cpu,speedup_vs_gpu_cramer,match,num_primes,det_bits,"
    csv += "gpu_solve,gpu_det_a,gpu_crt\n"
    for r in results {
        csv += "\(r.matrixSize),\(r.entryBits),\(r.cpuCramerTime),\(r.gpuCramerTime),\(r.gpuBatchedTime),"
        csv += "\(r.speedupVsCPU),\(r.speedupVsGPUCramer),\(r.batchedResultsMatch),\(r.numPrimes),\(r.determinantBits),"
        csv += "\(r.gpuSolveTime),\(r.gpuDetATime),\(r.gpuCRTTime)\n"
    }
    do {
        try csv.write(toFile: "batched_inverse_results.csv", atomically: true, encoding: .utf8)
        print("\n✓ Batched inverse results exported to batched_inverse_results.csv")
    } catch {
        print("Error exporting CSV: \(error)")
    }
}

// MARK: - Nullspace (Kernel) Benchmark

struct NullspaceBenchmarkResult {
    let rows: Int
    let cols: Int
    let targetNullity: Int
    let cpuTime: Double
    let gpuTime: Double
    let speedup: Double
    let cpuNullity: Int
    let gpuNullity: Int
    let cpuVerified: Bool
    let gpuVerified: Bool
    let numPrimes: Int
}

/// Generate a matrix with specified nullspace dimension
func generateMatrixWithNullspace(m: Int, n: Int, nullity: Int, bound: Int64) -> [BigInt] {
    // Create a matrix with column rank = n - nullity (so nullspace has dimension = nullity)
    // Strategy: Create a matrix where the last `nullity` columns are linear combinations
    // of the first (n - nullity) columns
    let targetRank = n - nullity
    guard targetRank > 0 && targetRank <= n else {
        return [BigInt](repeating: 0, count: m * n)
    }

    var matrix = [BigInt](repeating: 0, count: m * n)

    // First targetRank columns are random (linearly independent with high probability)
    for j in 0..<targetRank {
        for i in 0..<m {
            matrix[i * n + j] = BigInt(Int64.random(in: -bound...bound))
        }
    }

    // Remaining columns are linear combinations of the first targetRank columns
    // This creates a matrix where the nullspace has basis vectors of the form:
    // [c1, c2, ..., c_r, 0, ..., -1, 0, ...] (with -1 in position of free column)
    for j in targetRank..<n {
        // Generate random coefficients
        var coeffs = [Int64]()
        for _ in 0..<targetRank {
            coeffs.append(Int64.random(in: -3...3))
        }
        // Column j = sum of coeffs[k] * column[k]
        for i in 0..<m {
            var sum: BigInt = 0
            for k in 0..<targetRank {
                sum += BigInt(coeffs[k]) * matrix[i * n + k]
            }
            matrix[i * n + j] = sum
        }
    }

    return matrix
}

func runNullspaceBenchmark() {
    print("\n" + String(repeating: "=", count: 90))
    print("NULLSPACE (KERNEL) BENCHMARK")
    print("Computing exact basis for ker(A) = {x : Ax = 0}")
    print("Useful for: constraint systems, ZK verification, symbolic checks")
    print(String(repeating: "=", count: 90))
    fflush(stdout)

    var results: [NullspaceBenchmarkResult] = []

    // Test configurations: (m, n, nullity, bound)
    // Nullity = dimension of nullspace = n - rank
    let configs: [(Int, Int, Int, Int64)] = [
        (32, 32, 4, 100),    // Square, small nullspace
        (32, 32, 8, 100),    // Square, medium nullspace
        (32, 32, 16, 100),   // Square, large nullspace
        (48, 64, 16, 100),   // Wide matrix (m < n), some free variables
        (64, 48, 8, 100),    // Tall matrix (m > n), some free variables
        (64, 64, 4, 100),    // Larger square
    ]

    for (m, n, targetNullity, entryBound) in configs {
        print("\n" + String(repeating: "-", count: 70))
        print("Matrix: \(m)×\(n), target nullity: \(targetNullity), entries in [-\(entryBound), \(entryBound)]")
        print(String(repeating: "-", count: 70))
        fflush(stdout)

        autoreleasepool {
            let A = generateMatrixWithNullspace(m: m, n: n, nullity: targetNullity, bound: entryBound)

            // === CPU Baseline ===
            print("  Running CPU nullspace computation...")
            fflush(stdout)
            let (cpuResult, cpuTime) = BareissNullspace.computeTimed(A: A, m: m, n: n)
            print(String(format: "    CPU: %.4fs, nullity=%d, rank=%d, verified=%@",
                        cpuTime, cpuResult.nullity, cpuResult.rank, cpuResult.verified ? "✓" : "✗"))
            fflush(stdout)

            // === GPU ===
            var gpuTime = cpuTime
            var gpuNullity = cpuResult.nullity
            var gpuVerified = cpuResult.verified
            var numPrimes = 0

            if let gpuNS = GPUNullspace(),
               let (gpuResult, gpuTimings) = gpuNS.computeNullspace(A: A, m: m, n: n) {
                gpuTime = gpuTimings.totalTime
                gpuNullity = gpuResult.nullity
                gpuVerified = gpuResult.verified
                numPrimes = gpuTimings.numPrimes
                print(String(format: "    GPU: %.4fs, nullity=%d, rank=%d, verified=%@, primes=%d",
                            gpuTime, gpuResult.nullity, gpuResult.rank, gpuResult.verified ? "✓" : "✗", numPrimes))
                print(String(format: "         (RREF: %.4fs, CRT: %.4fs, Verify: %.4fs)",
                            gpuTimings.gpuReduceTime, gpuTimings.crtReconstructTime, gpuTimings.verifyTime))
            } else {
                print("    GPU: Failed or not available")
            }
            fflush(stdout)

            let speedup = cpuTime / gpuTime
            let nullityMatch = cpuResult.nullity == gpuNullity
            print(String(format: "    Speedup: %.2fx, Nullity match: %@",
                        speedup, nullityMatch ? "✓" : "✗"))

            results.append(NullspaceBenchmarkResult(
                rows: m,
                cols: n,
                targetNullity: targetNullity,
                cpuTime: cpuTime,
                gpuTime: gpuTime,
                speedup: speedup,
                cpuNullity: cpuResult.nullity,
                gpuNullity: gpuNullity,
                cpuVerified: cpuResult.verified,
                gpuVerified: gpuVerified,
                numPrimes: numPrimes
            ))
        }
    }

    // Print summary
    print("\n" + String(repeating: "=", count: 110))
    print("NULLSPACE BENCHMARK SUMMARY")
    print(String(repeating: "=", count: 110))
    print("Size (m×n) | Nullity | CPU (s)    | GPU (s)    | Speedup  | CPU Verified | GPU Verified | Match")
    print(String(repeating: "-", count: 110))

    for r in results {
        let sizeStr = "\(r.rows)×\(r.cols)".padding(toLength: 10, withPad: " ", startingAt: 0)
        let nullStr = String(r.cpuNullity).padding(toLength: 7, withPad: " ", startingAt: 0)
        let cpuStr = String(format: "%.4fs", r.cpuTime).padding(toLength: 10, withPad: " ", startingAt: 0)
        let gpuStr = String(format: "%.4fs", r.gpuTime).padding(toLength: 10, withPad: " ", startingAt: 0)
        let speedupStr = String(format: "%.2fx", r.speedup).padding(toLength: 8, withPad: " ", startingAt: 0)
        let cpuVerStr = r.cpuVerified ? "✓" : "✗"
        let gpuVerStr = r.gpuVerified ? "✓" : "✗"
        let matchStr = (r.cpuNullity == r.gpuNullity) ? "✓" : "✗"
        print("\(sizeStr) | \(nullStr) | \(cpuStr) | \(gpuStr) | \(speedupStr) | \(cpuVerStr.padding(toLength: 12, withPad: " ", startingAt: 0)) | \(gpuVerStr.padding(toLength: 12, withPad: " ", startingAt: 0)) | \(matchStr)")
    }
    fflush(stdout)

    // Export CSV
    var csv = "rows,cols,target_nullity,cpu_time,gpu_time,speedup,cpu_nullity,gpu_nullity,cpu_verified,gpu_verified,num_primes\n"
    for r in results {
        csv += "\(r.rows),\(r.cols),\(r.targetNullity),\(r.cpuTime),\(r.gpuTime),\(r.speedup),"
        csv += "\(r.cpuNullity),\(r.gpuNullity),\(r.cpuVerified),\(r.gpuVerified),\(r.numPrimes)\n"
    }
    do {
        try csv.write(toFile: "nullspace_results.csv", atomically: true, encoding: .utf8)
        print("\n✓ Nullspace results exported to nullspace_results.csv")
    } catch {
        print("Error exporting CSV: \(error)")
    }
}

// MARK: - Singular System Handling Test

func runSingularSystemTest() {
    print("\n" + String(repeating: "=", count: 70))
    print("SINGULAR SYSTEM HANDLING TEST")
    print("Testing detection and handling of singular matrices")
    print(String(repeating: "=", count: 70))
    fflush(stdout)

    let n = 32
    let entryBound: Int64 = 1_000_000

    // Test 1: Full rank matrix (should succeed)
    print("\n--- Test 1: Full-rank matrix ---")
    fflush(stdout)
    autoreleasepool {
        let A = generateRandomMatrix(n: n, bound: entryBound)
        let b = generateRandomVector(n: n, bound: entryBound)

        // CPU
        let (cpuSol, cpuTime) = BareissSolve.solveTimed(A: A, b: b, n: n)
        print(String(format: "  CPU: %@, time=%.4fs",
                     cpuSol != nil ? "solution found" : "singular", cpuTime))

        // GPU
        if let gpuSolver = GPUSolve(),
           let (gpuSol, timings) = gpuSolver.solve(A: A, b: b, n: n) {
            print(String(format: "  GPU: %@, time=%.4fs",
                         gpuSol != nil ? "solution found" : "singular", timings.totalTime))
        }
        fflush(stdout)
    }

    // Test 2: Rank-deficient matrix (should detect singular)
    print("\n--- Test 2: Rank-deficient matrix (rank = n/2) ---")
    fflush(stdout)
    autoreleasepool {
        let targetRank = n / 2
        let A = generateRandomMatrixWithRank(m: n, n: n, targetRank: targetRank, bound: entryBound)
        let b = generateRandomVector(n: n, bound: entryBound)

        // CPU
        let (cpuSol, cpuTime) = BareissSolve.solveTimed(A: A, b: b, n: n)
        let cpuRank = BareissRank.compute(matrix: A, m: n, n: n)
        print(String(format: "  CPU: %@, rank=%d, time=%.4fs",
                     cpuSol != nil ? "solution found" : "singular (no unique solution)", cpuRank, cpuTime))

        // GPU
        if let gpuSolver = GPUSolve(),
           let (gpuSol, timings) = gpuSolver.solve(A: A, b: b, n: n) {
            if let gpuRank = GPURank(),
               let (rank, _) = gpuRank.computeRank(matrix: A, m: n, n: n) {
                print(String(format: "  GPU: %@, rank=%d, time=%.4fs",
                             gpuSol != nil ? "solution found" : "singular (no unique solution)", rank, timings.totalTime))
            }
        }
        fflush(stdout)
    }

    // Test 3: Zero matrix (rank = 0)
    print("\n--- Test 3: Zero matrix (rank = 0) ---")
    fflush(stdout)
    autoreleasepool {
        let A = [BigInt](repeating: 0, count: n * n)
        let b = generateRandomVector(n: n, bound: entryBound)

        // CPU
        let (cpuSol, _) = BareissSolve.solveTimed(A: A, b: b, n: n)
        let cpuRank = BareissRank.compute(matrix: A, m: n, n: n)
        print("  CPU: \(cpuSol != nil ? "solution found" : "singular"), rank=\(cpuRank)")

        // GPU
        if let gpuSolver = GPUSolve(),
           let (gpuSol, _) = gpuSolver.solve(A: A, b: b, n: n) {
            if let gpuRank = GPURank(),
               let (rank, _) = gpuRank.computeRank(matrix: A, m: n, n: n) {
                print("  GPU: \(gpuSol != nil ? "solution found" : "singular"), rank=\(rank)")
            }
        }
        fflush(stdout)
    }

    // Test 4: Matrix with one zero row
    print("\n--- Test 4: Matrix with one zero row ---")
    fflush(stdout)
    autoreleasepool {
        var A = generateRandomMatrix(n: n, bound: entryBound)
        // Zero out the last row
        for j in 0..<n {
            A[(n-1) * n + j] = 0
        }
        let b = generateRandomVector(n: n, bound: entryBound)

        let cpuRank = BareissRank.compute(matrix: A, m: n, n: n)
        let (cpuSol, _) = BareissSolve.solveTimed(A: A, b: b, n: n)
        print("  CPU: \(cpuSol != nil ? "solution found" : "singular"), rank=\(cpuRank)")

        if let gpuRank = GPURank(),
           let (rank, _) = gpuRank.computeRank(matrix: A, m: n, n: n) {
            print("  GPU: rank=\(rank)")
        }
        fflush(stdout)
    }

    // Test 5: Inverse of singular matrix
    print("\n--- Test 5: Inverse of singular matrix ---")
    fflush(stdout)
    autoreleasepool {
        let targetRank = n / 2
        let A = generateRandomMatrixWithRank(m: n, n: n, targetRank: targetRank, bound: entryBound)

        // CPU
        let (cpuResult, cpuTime) = BareissInverse.inverseTimed(A: A, n: n)
        switch cpuResult {
        case .success(_, let det):
            print(String(format: "  CPU: inverse found (det bits=%d), time=%.4fs", det.bitWidth, cpuTime))
        case .singular(let rank):
            print(String(format: "  CPU: singular detected, rank=%d, time=%.4fs", rank, cpuTime))
        }

        // GPU
        if let gpuInv = GPUInverse(),
           let (gpuResult, timings) = gpuInv.inverse(A: A, n: n) {
            switch gpuResult {
            case .success(_, let det):
                print(String(format: "  GPU: inverse found (det bits=%d), time=%.4fs", det.bitWidth, timings.totalTime))
            case .singular(let rank):
                print(String(format: "  GPU: singular detected, rank=%d, time=%.4fs", rank, timings.totalTime))
            }
        }
        fflush(stdout)
    }

    print("\n✓ Singular system handling test complete")
}

// MARK: - FHE Primitives Benchmark

struct FHEBenchmarkResult {
    let operation: String
    let polyDegree: Int
    let numModuli: Int
    let cpuTime: Double
    let gpuTime: Double
    let speedup: Double
    let resultsMatch: Bool
}

func runFHEBenchmark() {
    print("\n" + String(repeating: "=", count: 70))
    print("FHE PRIMITIVES BENCHMARK")
    print("Testing RNS/CRT operations common in homomorphic encryption")
    print(String(repeating: "=", count: 70))
    fflush(stdout)

    // Check GPU availability
    print("Checking GPU RNS operations availability...")
    fflush(stdout)
    if let _ = GPURNSOperations() {
        print("  GPU RNS operations: available")
    } else {
        print("  GPU RNS operations: NOT available (will use CPU-only)")
    }
    fflush(stdout)

    var results: [FHEBenchmarkResult] = []

    // Test configurations (polyDegree, numModuliQ, numModuliP)
    let configs: [(Int, Int, Int)] = [
        (1024, 3, 2),
        (2048, 4, 3),
        (4096, 5, 4),
        (8192, 6, 5),
        (16384, 8, 6),
    ]

    // Create a single GPU instance to reuse across all tests
    let gpuRNS = GPURNSOperations()

    for (polyDegree, kQ, kP) in configs {
        print("\n--- Polynomial degree N=\(polyDegree), Q-basis=\(kQ) moduli, P-basis=\(kP) moduli ---")
        fflush(stdout)

        autoreleasepool {
            // Generate RNS contexts with 31-bit primes (faster to find and compute with)
            let Q = RNSContext.generate(count: kQ, bits: 31)
            let P = RNSContext.generate(count: kP, bits: 31)

            // Generate random polynomial in RNS form
            var poly: RNSOperationsCPU.RNSPolynomial = []
            for qIdx in 0..<kQ {
                var residues: [UInt64] = []
                for _ in 0..<polyDegree {
                    residues.append(UInt64.random(in: 0..<Q.moduli[qIdx]))
                }
                poly.append(residues)
            }

            // Generate second polynomial for Hadamard product
            var poly2: RNSOperationsCPU.RNSPolynomial = []
            for qIdx in 0..<kQ {
                var residues: [UInt64] = []
                for _ in 0..<polyDegree {
                    residues.append(UInt64.random(in: 0..<Q.moduli[qIdx]))
                }
                poly2.append(residues)
            }

            let precompute = BasisExtensionPrecompute(fromBasis: Q, toBasis: P)

            // ==================== Basis Extension ====================
            print("  Basis Extension Q→P...")
            fflush(stdout)

            // CPU
            let cpuBasisStart = CFAbsoluteTimeGetCurrent()
            let cpuBasisResult = RNSOperationsCPU.basisExtendFast(poly: poly, fromBasis: Q, toBasis: P, precomputed: precompute)
            let cpuBasisTime = CFAbsoluteTimeGetCurrent() - cpuBasisStart

            // GPU (if available)
            var gpuBasisTime = cpuBasisTime
            var basisMatch = true

            if let gpu = gpuRNS,
               let gpuBasisResult = gpu.basisExtend(poly: poly, fromBasis: Q, toBasis: P, precomputed: precompute) {
                let gpuStart = CFAbsoluteTimeGetCurrent()
                _ = gpu.basisExtend(poly: poly, fromBasis: Q, toBasis: P, precomputed: precompute)
                gpuBasisTime = CFAbsoluteTimeGetCurrent() - gpuStart

                // Verify
                basisMatch = (gpuBasisResult == cpuBasisResult)
            }

            let basisSpeedup = cpuBasisTime / gpuBasisTime
            print(String(format: "    CPU: %.6fs, GPU: %.6fs, Speedup: %.2fx, Match: %@",
                        cpuBasisTime, gpuBasisTime, basisSpeedup, basisMatch ? "✓" : "✗"))

            results.append(FHEBenchmarkResult(
                operation: "BasisExtend",
                polyDegree: polyDegree,
                numModuli: kQ,
                cpuTime: cpuBasisTime,
                gpuTime: gpuBasisTime,
                speedup: basisSpeedup,
                resultsMatch: basisMatch
            ))

            // ==================== Hadamard Product ====================
            print("  Hadamard Multiply...")
            fflush(stdout)

            // CPU
            let cpuHadamardStart = CFAbsoluteTimeGetCurrent()
            let cpuHadamardResult = RNSOperationsCPU.multiplyHadamard(poly, poly2, basis: Q)
            let cpuHadamardTime = CFAbsoluteTimeGetCurrent() - cpuHadamardStart

            // GPU
            var gpuHadamardTime = cpuHadamardTime
            var hadamardMatch = true

            if let gpu = gpuRNS,
               let gpuHadamardResult = gpu.multiplyHadamard(poly, poly2, basis: Q) {
                let gpuStart = CFAbsoluteTimeGetCurrent()
                _ = gpu.multiplyHadamard(poly, poly2, basis: Q)
                gpuHadamardTime = CFAbsoluteTimeGetCurrent() - gpuStart

                hadamardMatch = (gpuHadamardResult == cpuHadamardResult)
            }

            let hadamardSpeedup = cpuHadamardTime / gpuHadamardTime
            print(String(format: "    CPU: %.6fs, GPU: %.6fs, Speedup: %.2fx, Match: %@",
                        cpuHadamardTime, gpuHadamardTime, hadamardSpeedup, hadamardMatch ? "✓" : "✗"))

            results.append(FHEBenchmarkResult(
                operation: "Hadamard",
                polyDegree: polyDegree,
                numModuli: kQ,
                cpuTime: cpuHadamardTime,
                gpuTime: gpuHadamardTime,
                speedup: hadamardSpeedup,
                resultsMatch: hadamardMatch
            ))

            // ==================== Rescale ====================
            if kQ > 1 {
                print("  Rescale (CKKS-style)...")
                fflush(stdout)

                // CPU
                let cpuRescaleStart = CFAbsoluteTimeGetCurrent()
                let cpuRescaleResult = RNSOperationsCPU.rescale(poly: poly, basis: Q)
                let cpuRescaleTime = CFAbsoluteTimeGetCurrent() - cpuRescaleStart

                // GPU
                var gpuRescaleTime = cpuRescaleTime
                var rescaleMatch = true

                if let gpu = gpuRNS,
                   let gpuRescaleResult = gpu.rescale(poly: poly, basis: Q) {
                    let gpuStart = CFAbsoluteTimeGetCurrent()
                    _ = gpu.rescale(poly: poly, basis: Q)
                    gpuRescaleTime = CFAbsoluteTimeGetCurrent() - gpuStart

                    rescaleMatch = (gpuRescaleResult == cpuRescaleResult)
                }

                let rescaleSpeedup = cpuRescaleTime / gpuRescaleTime
                print(String(format: "    CPU: %.6fs, GPU: %.6fs, Speedup: %.2fx, Match: %@",
                            cpuRescaleTime, gpuRescaleTime, rescaleSpeedup, rescaleMatch ? "✓" : "✗"))

                results.append(FHEBenchmarkResult(
                    operation: "Rescale",
                    polyDegree: polyDegree,
                    numModuli: kQ,
                    cpuTime: cpuRescaleTime,
                    gpuTime: gpuRescaleTime,
                    speedup: rescaleSpeedup,
                    resultsMatch: rescaleMatch
                ))
            }

            // ==================== RNS Add ====================
            print("  RNS Add...")
            fflush(stdout)

            // CPU
            let cpuAddStart = CFAbsoluteTimeGetCurrent()
            let cpuAddResult = RNSOperationsCPU.add(poly, poly2, basis: Q)
            let cpuAddTime = CFAbsoluteTimeGetCurrent() - cpuAddStart

            // GPU
            var gpuAddTime = cpuAddTime
            var addMatch = true

            if let gpu = gpuRNS,
               let gpuAddResult = gpu.add(poly, poly2, basis: Q) {
                let gpuStart = CFAbsoluteTimeGetCurrent()
                _ = gpu.add(poly, poly2, basis: Q)
                gpuAddTime = CFAbsoluteTimeGetCurrent() - gpuStart

                addMatch = (gpuAddResult == cpuAddResult)
            }

            let addSpeedup = cpuAddTime / gpuAddTime
            let addMatchStr = addMatch ? "✓" : "✗"
            print(String(format: "    CPU: %.6fs, GPU: %.6fs, Speedup: %.2fx, Match: ", cpuAddTime, gpuAddTime, addSpeedup) + addMatchStr)

            results.append(FHEBenchmarkResult(
                operation: "Add",
                polyDegree: polyDegree,
                numModuli: kQ,
                cpuTime: cpuAddTime,
                gpuTime: gpuAddTime,
                speedup: addSpeedup,
                resultsMatch: addMatch
            ))
        }
    }

    // Print summary
    print("\n" + String(repeating: "=", count: 90))
    print("FHE PRIMITIVES SUMMARY")
    print(String(repeating: "=", count: 90))
    print("Operation     | PolyDeg | Moduli | CPU (s)    | GPU (s)    | Speedup | Match")
    print(String(repeating: "-", count: 90))

    for r in results {
        let matchStr = r.resultsMatch ? "Y" : "N"
        // Use Swift string padding instead of C format specifiers to avoid memory corruption
        let opPadded = r.operation.padding(toLength: 13, withPad: " ", startingAt: 0)
        let polyPadded = String(r.polyDegree).padding(toLength: 7, withPad: " ", startingAt: 0)
        let moduliPadded = String(r.numModuli).padding(toLength: 6, withPad: " ", startingAt: 0)
        let cpuStr = String(format: "%.6f", r.cpuTime).padding(toLength: 10, withPad: " ", startingAt: 0)
        let gpuStr = String(format: "%.6f", r.gpuTime).padding(toLength: 10, withPad: " ", startingAt: 0)
        let speedupStr = String(format: "%.2fx", r.speedup).padding(toLength: 8, withPad: " ", startingAt: 0)
        print("\(opPadded) | \(polyPadded) | \(moduliPadded) | \(cpuStr) | \(gpuStr) | \(speedupStr) | \(matchStr)")
    }
    fflush(stdout)

    // Export CSV
    var csv = "operation,poly_degree,num_moduli,cpu_time_s,gpu_time_s,speedup,match\n"
    for r in results {
        csv += "\(r.operation),\(r.polyDegree),\(r.numModuli),\(r.cpuTime),\(r.gpuTime),\(r.speedup),\(r.resultsMatch)\n"
    }
    do {
        try csv.write(toFile: "fhe_primitives_results.csv", atomically: true, encoding: .utf8)
        print("\n✓ FHE results exported to fhe_primitives_results.csv")
    } catch {
        print("Error exporting CSV: \(error)")
    }
}

func runFHEKeySwitchBenchmark() {
    print("\n" + String(repeating: "=", count: 70))
    print("FHE KEY SWITCHING BENCHMARK")
    print("Testing gadget decomposition + multiply-accumulate")
    print(String(repeating: "=", count: 70))
    fflush(stdout)

    // Test configurations
    let configs: [(Int, Int, UInt64, Int)] = [
        // (polyDegree, numModuli, digitBase, numDigits)
        (2048, 4, 1 << 10, 5),
        (4096, 5, 1 << 12, 4),
        (8192, 6, 1 << 15, 4),
    ]

    for (polyDegree, kQ, digitBase, numDigits) in configs {
        print("\n--- N=\(polyDegree), kQ=\(kQ), B=2^\(Int(log2(Double(digitBase)))), digits=\(numDigits) ---")
        fflush(stdout)

        autoreleasepool {
            // Use 31-bit primes for GPU compatibility
            let Q = RNSContext.generate(count: kQ, bits: 31)

            // Generate random polynomial
            var poly: RNSOperationsCPU.RNSPolynomial = []
            for qIdx in 0..<kQ {
                var residues: [UInt64] = []
                for _ in 0..<polyDegree {
                    residues.append(UInt64.random(in: 0..<Q.moduli[qIdx]))
                }
                poly.append(residues)
            }

            // Generate random "gadget keys" (simulated)
            var gadgetKeys: [RNSOperationsCPU.RNSPolynomial] = []
            for _ in 0..<numDigits {
                var key: RNSOperationsCPU.RNSPolynomial = []
                for qIdx in 0..<kQ {
                    var residues: [UInt64] = []
                    for _ in 0..<polyDegree {
                        residues.append(UInt64.random(in: 0..<Q.moduli[qIdx]))
                    }
                    key.append(residues)
                }
                gadgetKeys.append(key)
            }

            // CPU: Gadget decompose + MAC
            let cpuStart = CFAbsoluteTimeGetCurrent()
            let cpuDigits = RNSOperationsCPU.gadgetDecompose(
                poly: poly, basis: Q, digitBase: digitBase, numDigits: numDigits)
            let cpuResult = RNSOperationsCPU.gadgetMultiplyAccumulate(
                digits: cpuDigits, gadgetKeys: gadgetKeys, basis: Q)
            let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart

            // GPU
            var gpuTime = cpuTime
            var match = true

            if let gpuRNS = GPURNSOperations(),
               let gpuResult = gpuRNS.gadgetDecomposeAndMAC(
                    poly: poly, gadgetKeys: gadgetKeys, basis: Q,
                    digitBase: digitBase, numDigits: numDigits) {
                let gpuStart = CFAbsoluteTimeGetCurrent()
                _ = gpuRNS.gadgetDecomposeAndMAC(
                    poly: poly, gadgetKeys: gadgetKeys, basis: Q,
                    digitBase: digitBase, numDigits: numDigits)
                gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart

                match = (gpuResult == cpuResult)
            }

            let speedup = cpuTime / gpuTime
            let matchStr = match ? "✓" : "✗"
            print(String(format: "  CPU: %.6fs, GPU: %.6fs, Speedup: %.2fx, Match: ", cpuTime, gpuTime, speedup) + matchStr)
        }
    }
}

// MARK: - MRR (Multiply-Relinearize-Rescale) Macro-Benchmark

struct MRRBenchmarkResult {
    let polyDegree: Int
    let numModuli: Int
    let numDigits: Int
    let cpuTotal: Double
    let cpuMultiply: Double
    let cpuRelin: Double
    let cpuRescale: Double
    let gpuTotal: Double
    let gpuMultiply: Double
    let gpuRelin: Double
    let gpuRescale: Double
    let speedup: Double
}

func runMRRBenchmark() {
    print("\n" + String(repeating: "=", count: 80))
    print("MRR (MULTIPLY-RELINEARIZE-RESCALE) MACRO-BENCHMARK")
    print("This is the hot path in CKKS homomorphic multiplication")
    print(String(repeating: "=", count: 80))
    fflush(stdout)

    var results: [MRRBenchmarkResult] = []

    // Test configurations: (polyDegree, numModuli, digitBase, numDigits)
    // These mirror realistic CKKS parameters
    let configs: [(Int, Int, UInt64, Int)] = [
        (2048, 4, 1 << 10, 4),   // Small
        (4096, 5, 1 << 12, 4),   // Medium
        (8192, 6, 1 << 12, 4),   // Large
        (16384, 8, 1 << 15, 4),  // Production-scale
    ]

    // Reuse GPU instance
    let gpuRNS = GPURNSOperations()

    for (polyDegree, kQ, digitBase, numDigits) in configs {
        print("\n--- N=\(polyDegree), kQ=\(kQ), B=2^\(Int(log2(Double(digitBase)))), digits=\(numDigits) ---")
        fflush(stdout)

        autoreleasepool {
            // Generate RNS context (use 31-bit primes for speed)
            let Q = RNSContext.generate(count: kQ, bits: 31)

            // Generate random "ciphertext" components (ct0, ct1) and (ct2_0, ct2_1)
            func randomPoly() -> RNSOperationsCPU.RNSPolynomial {
                var poly: RNSOperationsCPU.RNSPolynomial = []
                for qIdx in 0..<kQ {
                    var residues: [UInt64] = []
                    for _ in 0..<polyDegree {
                        residues.append(UInt64.random(in: 0..<Q.moduli[qIdx]))
                    }
                    poly.append(residues)
                }
                return poly
            }

            let ct0 = randomPoly()
            let ct1 = randomPoly()
            let ct2_0 = randomPoly()
            let ct2_1 = randomPoly()

            // Generate random "relinearization keys" (simulated)
            var relinKey0: [RNSOperationsCPU.RNSPolynomial] = []
            var relinKey1: [RNSOperationsCPU.RNSPolynomial] = []
            for _ in 0..<numDigits {
                relinKey0.append(randomPoly())
                relinKey1.append(randomPoly())
            }

            // === CPU Baseline ===
            print("  Running CPU MRR...")
            fflush(stdout)

            let cpuResult = RNSMRROperationsCPU.fusedMRR(
                ct0: ct0, ct1: ct1,
                ct2_0: ct2_0, ct2_1: ct2_1,
                relinKey0: relinKey0, relinKey1: relinKey1,
                basis: Q,
                digitBase: digitBase,
                numDigits: numDigits
            )

            print(String(format: "    CPU Total: %.6fs (Mul: %.6fs, Relin: %.6fs, Rescale: %.6fs)",
                        cpuResult.timings.totalTime,
                        cpuResult.timings.multiplyTime,
                        cpuResult.timings.relinearizeTime,
                        cpuResult.timings.rescaleTime))
            fflush(stdout)

            // === GPU ===
            var gpuTimings = GPURNSOperations.MRRTimings()
            gpuTimings.totalTime = cpuResult.timings.totalTime  // Default to CPU if GPU fails

            if let gpu = gpuRNS {
                print("  Running GPU MRR...")
                fflush(stdout)

                if let gpuResult = gpu.fusedMRR(
                    ct0: ct0, ct1: ct1,
                    ct2_0: ct2_0, ct2_1: ct2_1,
                    relinKey0: relinKey0, relinKey1: relinKey1,
                    basis: Q,
                    digitBase: digitBase,
                    numDigits: numDigits
                ) {
                    gpuTimings = gpuResult.timings

                    print(String(format: "    GPU Total: %.6fs (Mul: %.6fs, Relin: %.6fs, Rescale: %.6fs)",
                                gpuTimings.totalTime,
                                gpuTimings.multiplyTime,
                                gpuTimings.relinearizeTime,
                                gpuTimings.rescaleTime))
                } else {
                    print("    GPU MRR failed")
                }
            } else {
                print("    GPU not available")
            }
            fflush(stdout)

            let speedup = cpuResult.timings.totalTime / gpuTimings.totalTime
            print(String(format: "  => Speedup: %.2fx", speedup))

            results.append(MRRBenchmarkResult(
                polyDegree: polyDegree,
                numModuli: kQ,
                numDigits: numDigits,
                cpuTotal: cpuResult.timings.totalTime,
                cpuMultiply: cpuResult.timings.multiplyTime,
                cpuRelin: cpuResult.timings.relinearizeTime,
                cpuRescale: cpuResult.timings.rescaleTime,
                gpuTotal: gpuTimings.totalTime,
                gpuMultiply: gpuTimings.multiplyTime,
                gpuRelin: gpuTimings.relinearizeTime,
                gpuRescale: gpuTimings.rescaleTime,
                speedup: speedup
            ))
        }
    }

    // Print summary
    print("\n" + String(repeating: "=", count: 100))
    print("MRR MACRO-BENCHMARK SUMMARY")
    print(String(repeating: "=", count: 100))
    print("PolyDeg | Moduli | Digits | CPU Total  | GPU Total  | Speedup | GPU Breakdown (Mul/Relin/Rescale)")
    print(String(repeating: "-", count: 100))

    for r in results {
        let polyPadded = String(r.polyDegree).padding(toLength: 7, withPad: " ", startingAt: 0)
        let moduliPadded = String(r.numModuli).padding(toLength: 6, withPad: " ", startingAt: 0)
        let digitsPadded = String(r.numDigits).padding(toLength: 6, withPad: " ", startingAt: 0)
        let cpuStr = String(format: "%.6f", r.cpuTotal).padding(toLength: 10, withPad: " ", startingAt: 0)
        let gpuStr = String(format: "%.6f", r.gpuTotal).padding(toLength: 10, withPad: " ", startingAt: 0)
        let speedupStr = String(format: "%.2fx", r.speedup).padding(toLength: 7, withPad: " ", startingAt: 0)
        let breakdown = String(format: "%.3f/%.3f/%.3f", r.gpuMultiply * 1000, r.gpuRelin * 1000, r.gpuRescale * 1000)
        print("\(polyPadded) | \(moduliPadded) | \(digitsPadded) | \(cpuStr) | \(gpuStr) | \(speedupStr) | \(breakdown) ms")
    }
    fflush(stdout)

    // Export CSV
    var csv = "poly_degree,num_moduli,num_digits,cpu_total,cpu_mul,cpu_relin,cpu_rescale,gpu_total,gpu_mul,gpu_relin,gpu_rescale,speedup\n"
    for r in results {
        csv += "\(r.polyDegree),\(r.numModuli),\(r.numDigits),"
        csv += "\(r.cpuTotal),\(r.cpuMultiply),\(r.cpuRelin),\(r.cpuRescale),"
        csv += "\(r.gpuTotal),\(r.gpuMultiply),\(r.gpuRelin),\(r.gpuRescale),\(r.speedup)\n"
    }
    do {
        try csv.write(toFile: "mrr_benchmark_results.csv", atomically: true, encoding: .utf8)
        print("\n✓ MRR results exported to mrr_benchmark_results.csv")
    } catch {
        print("Error exporting CSV: \(error)")
    }
}

// MARK: - Overdetermined System Benchmark

struct OverdeterminedBenchmarkResult {
    let rows: Int
    let cols: Int
    let isConsistent: Bool
    let rank: Int
    let cpuTime: Double
    let gpuTime: Double
    let speedup: Double
    let cpuVerified: Bool
    let gpuVerified: Bool
    let numPrimes: Int
}

/// Generate an overdetermined system Ax = b with known solution
/// Creates m×n matrix (m > n) and compatible RHS
func generateOverdeterminedSystem(m: Int, n: Int, bound: Int64, consistent: Bool) -> (A: [BigInt], b: [BigInt], trueSolution: [BigInt]?) {
    var A = [BigInt](repeating: 0, count: m * n)

    // Generate random coefficient matrix
    for i in 0..<m {
        for j in 0..<n {
            A[i * n + j] = BigInt(Int64.random(in: -bound...bound))
        }
    }

    if consistent {
        // Generate a true solution
        var x = [BigInt](repeating: 0, count: n)
        for j in 0..<n {
            x[j] = BigInt(Int64.random(in: -bound/10...bound/10))
        }

        // Compute b = A·x (this makes the system consistent)
        var b = [BigInt](repeating: 0, count: m)
        for i in 0..<m {
            for j in 0..<n {
                b[i] += A[i * n + j] * x[j]
            }
        }

        return (A, b, x)
    } else {
        // Make system inconsistent by generating random b
        // that is not in the column space of A
        var b = [BigInt](repeating: 0, count: m)
        for i in 0..<m {
            b[i] = BigInt(Int64.random(in: -bound...bound))
        }

        // Add a component that can't be achieved
        // (for a random system, this is almost certainly inconsistent)
        return (A, b, nil)
    }
}

func runOverdeterminedBenchmark() {
    print("\n" + String(repeating: "=", count: 90))
    print("OVERDETERMINED SYSTEM BENCHMARK")
    print("Solving Ax = b where A is m×n with m > n")
    print("Checks consistency and finds exact solution if exists")
    print(String(repeating: "=", count: 90))
    fflush(stdout)

    var results: [OverdeterminedBenchmarkResult] = []

    // Test configurations: (m, n, consistent, bound)
    let configs: [(Int, Int, Bool, Int64)] = [
        (48, 32, true, 100),     // Overdetermined, consistent
        (48, 32, false, 100),    // Overdetermined, inconsistent
        (64, 32, true, 100),     // More equations, consistent
        (64, 32, false, 100),    // More equations, inconsistent
        (96, 48, true, 100),     // Larger system, consistent
        (96, 48, false, 100),    // Larger system, inconsistent
        (128, 64, true, 100),    // Even larger, consistent
    ]

    for (m, n, consistent, entryBound) in configs {
        print("\n" + String(repeating: "-", count: 70))
        print("Matrix: \(m)×\(n), \(consistent ? "CONSISTENT" : "INCONSISTENT"), entries in [-\(entryBound), \(entryBound)]")
        print(String(repeating: "-", count: 70))
        fflush(stdout)

        autoreleasepool {
            let (A, b, trueSolution) = generateOverdeterminedSystem(m: m, n: n, bound: entryBound, consistent: consistent)

            // === CPU Baseline ===
            print("  Running CPU overdetermined solve...")
            fflush(stdout)
            let cpuStart = CFAbsoluteTimeGetCurrent()
            let cpuResult = BareissOverdetermined.solve(A: A, b: b, m: m, n: n)
            let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart

            var cpuConsistent = false
            var cpuRank = 0
            var cpuVerified = false

            switch cpuResult {
            case .consistent(let solution, let rank):
                cpuConsistent = true
                cpuRank = rank
                // Verify solution
                cpuVerified = true
                for i in 0..<m {
                    var sum = Rational(0)
                    for j in 0..<n {
                        sum = sum + Rational(A[i * n + j]) * solution[j]
                    }
                    if sum != Rational(b[i]) {
                        cpuVerified = false
                        break
                    }
                }
                print(String(format: "    CPU: consistent, rank=%d, verified=%@, time=%.4fs",
                            rank, cpuVerified ? "✓" : "✗", cpuTime))
            case .inconsistent(let rank, let augRank):
                cpuRank = rank
                cpuVerified = true  // Detecting inconsistency is correct behavior
                print(String(format: "    CPU: INCONSISTENT (rank(A)=%d, rank([A|b])=%d), time=%.4fs",
                            rank, augRank, cpuTime))
            }
            fflush(stdout)

            // === GPU ===
            var gpuTime = cpuTime
            var gpuConsistent = cpuConsistent
            var gpuRank = cpuRank
            var gpuVerified = cpuVerified
            var numPrimes = 0

            if let gpuSolver = GPUOverdetermined(),
               let (gpuResult, gpuTimings) = gpuSolver.solve(A: A, b: b, m: m, n: n) {
                gpuTime = gpuTimings.totalTime
                numPrimes = gpuTimings.numPrimes

                switch gpuResult {
                case .consistent(let solution, let rank):
                    gpuConsistent = true
                    gpuRank = rank
                    // Verify solution
                    gpuVerified = true
                    for i in 0..<m {
                        var sum = Rational(0)
                        for j in 0..<n {
                            sum = sum + Rational(A[i * n + j]) * solution[j]
                        }
                        if sum != Rational(b[i]) {
                            gpuVerified = false
                            break
                        }
                    }
                    print(String(format: "    GPU: consistent, rank=%d, verified=%@, primes=%d, time=%.4fs",
                                rank, gpuVerified ? "✓" : "✗", numPrimes, gpuTime))
                    print(String(format: "         (RREF: %.4fs, CRT: %.4fs, Verify: %.4fs)",
                                gpuTimings.rrefTime, gpuTimings.crtReconstructTime, gpuTimings.verifyTime))
                case .inconsistent(let rank, let augRank):
                    gpuConsistent = false
                    gpuRank = rank
                    gpuVerified = true
                    print(String(format: "    GPU: INCONSISTENT (rank(A)=%d, rank([A|b])=%d), primes=%d, time=%.4fs",
                                rank, augRank, numPrimes, gpuTime))
                }
            } else {
                print("    GPU: Failed or not available")
            }
            fflush(stdout)

            let speedup = cpuTime / gpuTime
            let resultMatch = (cpuConsistent == gpuConsistent) && (cpuRank == gpuRank)
            print(String(format: "    Speedup: %.2fx, Results match: %@",
                        speedup, resultMatch ? "✓" : "✗"))

            results.append(OverdeterminedBenchmarkResult(
                rows: m,
                cols: n,
                isConsistent: cpuConsistent,
                rank: cpuRank,
                cpuTime: cpuTime,
                gpuTime: gpuTime,
                speedup: speedup,
                cpuVerified: cpuVerified,
                gpuVerified: gpuVerified,
                numPrimes: numPrimes
            ))
        }
    }

    // Print summary
    print("\n" + String(repeating: "=", count: 110))
    print("OVERDETERMINED BENCHMARK SUMMARY")
    print(String(repeating: "=", count: 110))
    print("Size (m×n) | Consistent | Rank | CPU (s)    | GPU (s)    | Speedup  | CPU Ver | GPU Ver | Match")
    print(String(repeating: "-", count: 110))

    for r in results {
        let sizeStr = "\(r.rows)×\(r.cols)".padding(toLength: 10, withPad: " ", startingAt: 0)
        let consStr = (r.isConsistent ? "Yes" : "No").padding(toLength: 10, withPad: " ", startingAt: 0)
        let rankStr = String(r.rank).padding(toLength: 4, withPad: " ", startingAt: 0)
        let cpuStr = String(format: "%.4fs", r.cpuTime).padding(toLength: 10, withPad: " ", startingAt: 0)
        let gpuStr = String(format: "%.4fs", r.gpuTime).padding(toLength: 10, withPad: " ", startingAt: 0)
        let speedupStr = String(format: "%.2fx", r.speedup).padding(toLength: 8, withPad: " ", startingAt: 0)
        let cpuVerStr = r.cpuVerified ? "✓" : "✗"
        let gpuVerStr = r.gpuVerified ? "✓" : "✗"
        let matchStr = (r.isConsistent == r.isConsistent && r.cpuVerified && r.gpuVerified) ? "✓" : "✗"
        print("\(sizeStr) | \(consStr) | \(rankStr) | \(cpuStr) | \(gpuStr) | \(speedupStr) | \(cpuVerStr.padding(toLength: 7, withPad: " ", startingAt: 0)) | \(gpuVerStr.padding(toLength: 7, withPad: " ", startingAt: 0)) | \(matchStr)")
    }
    fflush(stdout)

    // Export CSV
    var csv = "rows,cols,consistent,rank,cpu_time,gpu_time,speedup,cpu_verified,gpu_verified,num_primes\n"
    for r in results {
        csv += "\(r.rows),\(r.cols),\(r.isConsistent),\(r.rank),\(r.cpuTime),\(r.gpuTime),\(r.speedup),"
        csv += "\(r.cpuVerified),\(r.gpuVerified),\(r.numPrimes)\n"
    }
    do {
        try csv.write(toFile: "overdetermined_results.csv", atomically: true, encoding: .utf8)
        print("\n✓ Overdetermined results exported to overdetermined_results.csv")
    } catch {
        print("Error exporting CSV: \(error)")
    }
}

// MARK: - Main

print("""

╔═══════════════════════════════════════════════════════════════╗
║           PARALLEL LIFT: CRT-GPU Exact Arithmetic             ║
║     Converting Precision into Parallelism on Apple Silicon    ║
╚═══════════════════════════════════════════════════════════════╝

This experiment tests whether exact integer determinant computation
can be accelerated by using the Chinese Remainder Theorem to convert
big-integer arithmetic into parallel modular arithmetic on GPU.

Hypothesis: GPU-CRT path < CPU BigInt path for sufficiently large matrices

Platform: Apple Silicon (Metal)
""")

// Get command line arguments
let args = CommandLine.arguments

if args.contains("--test") {
    runCorrectnessTest()
} else if args.contains("--large-entries") {
    runLargeEntryTest()
} else if args.contains("--quick") {
    // Quick single benchmark
    _ = runBenchmark(n: 24, entryBound: 100_000, verbose: true)
} else if args.contains("--full") {
    // Full comprehensive suite
    runScalingSuite()
    runLargeEntryTest()
} else if args.contains("--stability") {
    // Stability test: multiple runs per configuration
    runStabilityTest()
} else if args.contains("--rank") {
    // Rank benchmark suite
    runRankScalingSuite()
} else if args.contains("--rank-deficient") {
    // Test rank-deficient matrices
    runRankDeficientTest()
} else if args.contains("--solve") {
    // Solve benchmark suite
    runSolveScalingSuite()
} else if args.contains("--solve-test") {
    // Solve correctness test
    runSolveCorrectnessTest()
} else if args.contains("--multi-rhs") {
    // Multi-RHS solve benchmark (AX = B)
    runMultiRHSBenchmark()
} else if args.contains("--inverse") {
    // Inverse benchmark suite
    runInverseScalingSuite()
} else if args.contains("--batched-inverse") {
    // Batched inverse benchmark (A·X = I via multi-RHS)
    runBatchedInverseBenchmark()
} else if args.contains("--nullspace") {
    // Nullspace (kernel) computation benchmark
    runNullspaceBenchmark()
} else if args.contains("--overdetermined") {
    // Overdetermined system solve benchmark (m > n)
    runOverdeterminedBenchmark()
} else if args.contains("--singular") {
    // Singular system handling test
    runSingularSystemTest()
} else if args.contains("--fhe") {
    // FHE primitives benchmark
    runFHEBenchmark()
} else if args.contains("--fhe-keyswitch") {
    // FHE key switching benchmark
    runFHEKeySwitchBenchmark()
} else if args.contains("--fhe-all") {
    // All FHE benchmarks
    runFHEBenchmark()
    runFHEKeySwitchBenchmark()
} else if args.contains("--mrr") {
    // MRR macro-benchmark (CKKS multiply-relinearize-rescale)
    runMRRBenchmark()
} else if args.contains("--fhe-full") {
    // All FHE benchmarks including MRR
    runFHEBenchmark()
    runMRRBenchmark()
} else {
    // Default: scaling suite
    runScalingSuite()
}

print("\n✓ Benchmark complete")
