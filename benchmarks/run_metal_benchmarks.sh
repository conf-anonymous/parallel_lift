#!/bin/bash
# =============================================================================
# Parallel Lift - Metal Benchmark Suite
# Run on MacBook Pro with Apple M3 Max
# =============================================================================

set -e

echo "=========================================="
echo "Parallel Lift Metal Benchmark Suite"
echo "Paper Results"
echo "=========================================="
echo ""

# Configuration
RESULTS_DIR="./paper_results_metal"
TRIALS=5
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$RESULTS_DIR"

# System info
echo "System Information:" | tee "$RESULTS_DIR/system_info.txt"
echo "==================" | tee -a "$RESULTS_DIR/system_info.txt"
system_profiler SPHardwareDataType | tee -a "$RESULTS_DIR/system_info.txt"
echo "" | tee -a "$RESULTS_DIR/system_info.txt"
system_profiler SPDisplaysDataType | tee -a "$RESULTS_DIR/system_info.txt"
echo "" | tee -a "$RESULTS_DIR/system_info.txt"
sw_vers | tee -a "$RESULTS_DIR/system_info.txt"

# Build Swift version
echo ""
echo "Building Parallel Lift Swift/Metal..."
cd swift
swift build -c release
cd ..

BENCH_CMD="./swift/.build/release/parallel-lift"

# =============================================================================
# 1. Determinant - Scaling with Matrix Size (Table 9)
# =============================================================================
echo ""
echo "=========================================="
echo "Benchmark 1: Determinant - Metal Backend"
echo "Configuration: 20-bit entries"
echo "=========================================="

OUTPUT="$RESULTS_DIR/determinant_metal.csv"
echo "n,primes,cpu_s,gpu_s,speedup,verified" > "$OUTPUT"

for n in 48 64 80 96 112 128 160 192; do
    echo "Running n=$n..."
    for trial in $(seq 1 $TRIALS); do
        $BENCH_CMD benchmark \
            --operation det \
            --size $n \
            --entry-bits 20 \
            --gpu >> "$OUTPUT" 2>&1 || true
    done
done

echo "Results saved to $OUTPUT"

# =============================================================================
# 2. Multi-RHS Solve - Metal Backend
# =============================================================================
echo ""
echo "=========================================="
echo "Benchmark 2: Multi-RHS Solve - Metal Backend"
echo "Configuration: 20-bit entries"
echo "=========================================="

OUTPUT="$RESULTS_DIR/multi_rhs_metal.csv"
echo "n,k,primes,cpu_s,gpu_s,speedup,verified" > "$OUTPUT"

for n in 32 48 64 96 128; do
    for k in 1 4 8 16; do
        echo "Running n=$n, k=$k..."
        for trial in $(seq 1 $TRIALS); do
            $BENCH_CMD benchmark \
                --operation solve \
                --size $n \
                --rhs $k \
                --entry-bits 20 \
                --gpu >> "$OUTPUT" 2>&1 || true
        done
    done
done

echo "Results saved to $OUTPUT"

# =============================================================================
# 3. Rank Computation - Metal Backend
# =============================================================================
echo ""
echo "=========================================="
echo "Benchmark 3: Rank Computation - Metal Backend"
echo "=========================================="

OUTPUT="$RESULTS_DIR/rank_metal.csv"
echo "n,primes,cpu_s,gpu_s,speedup,verified" > "$OUTPUT"

for n in 32 64 96 128; do
    echo "Running n=$n..."
    for trial in $(seq 1 $TRIALS); do
        $BENCH_CMD benchmark \
            --operation rank \
            --size $n \
            --entry-bits 20 \
            --gpu >> "$OUTPUT" 2>&1 || true
    done
done

echo "Results saved to $OUTPUT"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=========================================="
echo "Metal Benchmark Suite Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
