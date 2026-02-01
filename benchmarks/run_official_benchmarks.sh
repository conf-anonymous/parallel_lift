#!/bin/bash
# =============================================================================
# Parallel Lift - Official Benchmark Suite
# =============================================================================
#
# This script runs ALL benchmarks required for the paper.
# Run on RunPod with RTX 5090.
#
# Usage:
#   ./run_official_benchmarks.sh           # Run all benchmarks
#   ./run_official_benchmarks.sh --quick   # Quick test (1 trial, fewer sizes)
#
# =============================================================================

set -e

# Configuration
RESULTS_DIR="./paper_results"
TRIALS=5
QUICK_MODE=false

if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
    TRIALS=1
    echo "=== QUICK MODE: 1 trial, reduced sizes ==="
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${RESULTS_DIR}_${TIMESTAMP}"

mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "Parallel Lift Official Benchmark Suite"
echo "Paper Results"
echo "=============================================="
echo "Results directory: $RESULTS_DIR"
echo "Trials per config: $TRIALS"
echo "Timestamp: $TIMESTAMP"
echo ""

# =============================================================================
# System Information
# =============================================================================
echo "[1/12] Collecting system information..."

{
    echo "=== System Information ==="
    echo "Date: $(date)"
    echo ""
    echo "=== GPU Information ==="
    nvidia-smi
    echo ""
    echo "=== CUDA Version ==="
    nvcc --version 2>/dev/null || echo "nvcc not in PATH"
    echo ""
    echo "=== CPU Information ==="
    lscpu | grep -E "Model name|Socket|Core|Thread|CPU MHz" || cat /proc/cpuinfo | grep "model name" | head -1
    echo ""
    echo "=== Memory ==="
    free -h
    echo ""
    echo "=== OS ==="
    uname -a
    cat /etc/os-release 2>/dev/null | head -5 || true
    echo ""
    echo "=== Rust Version ==="
    rustc --version
    cargo --version
} > "$RESULTS_DIR/system_info.txt" 2>&1

echo "System info saved to $RESULTS_DIR/system_info.txt"

# =============================================================================
# Build
# =============================================================================
echo ""
echo "[2/12] Building Parallel Lift..."

cd rust
cargo build --release --features cuda 2>&1 | tail -5
cd ..

BENCH="./rust/target/release/parallel-lift"

if [[ ! -f "$BENCH" ]]; then
    echo "ERROR: Binary not found at $BENCH"
    exit 1
fi

echo "Build successful."

# =============================================================================
# Helper function
# =============================================================================
run_benchmark() {
    local name="$1"
    local cmd="$2"
    local output="$3"

    echo "  Running: $cmd"
    for trial in $(seq 1 $TRIALS); do
        echo -n "    Trial $trial/$TRIALS... "
        eval "$cmd" >> "$output" 2>&1 && echo "done" || echo "FAILED"
    done
}

# =============================================================================
# Benchmark 10: Full Pipeline (CRITICAL - Run First)
# =============================================================================
echo ""
echo "[3/12] Benchmark 10: Full GPU Pipeline (CRITICAL)"
echo "=============================================="

OUTPUT="$RESULTS_DIR/full_pipeline.csv"
echo "n,k,primes,entry_bits,cpu_total_ms,gpu_cpu_crt_ms,gpu_gpu_crt_ms,speedup_cpu_crt,speedup_gpu_crt,verified,trial" > "$OUTPUT"

if $QUICK_MODE; then
    SIZES="64 128"
else
    SIZES="32 64 128 256"
fi

for n in $SIZES; do
    echo "  Size n=$n, k=16:"
    for trial in $(seq 1 $TRIALS); do
        echo -n "    Trial $trial/$TRIALS... "
        $BENCH gpu-full-bench --size $n --rhs 16 --entry-bits 48 --output-csv 2>/dev/null | tail -1 >> "$OUTPUT" && echo "done" || echo "FAILED"
    done
done

echo "Results: $OUTPUT"

# =============================================================================
# Benchmark 1: Multi-RHS Scaling with n (CRITICAL)
# =============================================================================
echo ""
echo "[4/12] Benchmark 1: Multi-RHS Scaling with Matrix Size"
echo "=============================================="

OUTPUT="$RESULTS_DIR/multi_rhs_scaling_n.csv"
echo "n,k,primes,entry_bits,backend,total_ms,solve_ms,crt_ms,verified,trial" > "$OUTPUT"

if $QUICK_MODE; then
    SIZES="32 64 128"
else
    SIZES="32 64 128 256 512"
fi

for n in $SIZES; do
    echo "  Size n=$n, k=16:"

    # CPU baseline
    echo "    CPU:"
    for trial in $(seq 1 $TRIALS); do
        echo -n "      Trial $trial/$TRIALS... "
        $BENCH benchmark --bench multi-rhs --size $n --rhs 16 --entry-bits 48 --backend cpu --output-csv 2>/dev/null | tail -1 >> "$OUTPUT" && echo "done" || echo "FAILED"
    done

    # GPU
    echo "    GPU:"
    for trial in $(seq 1 $TRIALS); do
        echo -n "      Trial $trial/$TRIALS... "
        $BENCH benchmark --bench multi-rhs --size $n --rhs 16 --entry-bits 48 --backend cuda --output-csv 2>/dev/null | tail -1 >> "$OUTPUT" && echo "done" || echo "FAILED"
    done
done

echo "Results: $OUTPUT"

# =============================================================================
# Benchmark 5: CRT Reconstruction (CRITICAL)
# =============================================================================
echo ""
echo "[5/12] Benchmark 5: CRT Reconstruction"
echo "=============================================="

OUTPUT="$RESULTS_DIR/crt_reconstruction.csv"
echo "primes,values,cpu_ms,gpu_ms,speedup,verified,trial" > "$OUTPUT"

if $QUICK_MODE; then
    CONFIGS="128,4096 256,4096"
else
    CONFIGS="64,1024 128,4096 256,4096 512,4096"
fi

for config in $CONFIGS; do
    IFS=',' read -r primes values <<< "$config"
    echo "  Primes=$primes, Values=$values:"
    for trial in $(seq 1 $TRIALS); do
        echo -n "    Trial $trial/$TRIALS... "
        $BENCH gpu-crt-bench --num-primes $primes --num-values $values --output-csv 2>/dev/null | tail -1 >> "$OUTPUT" && echo "done" || echo "FAILED"
    done
done

echo "Results: $OUTPUT"

# =============================================================================
# Benchmark 4: Matrix Inverse (CRITICAL)
# =============================================================================
echo ""
echo "[6/12] Benchmark 4: Matrix Inverse"
echo "=============================================="

OUTPUT="$RESULTS_DIR/inverse.csv"
echo "n,primes,entry_bits,backend,total_ms,solve_ms,crt_ms,verified,trial" > "$OUTPUT"

if $QUICK_MODE; then
    SIZES="32 64"
else
    SIZES="32 64 128"
fi

for n in $SIZES; do
    echo "  Size n=$n (k=$n):"

    # CPU
    echo "    CPU:"
    for trial in $(seq 1 $TRIALS); do
        echo -n "      Trial $trial/$TRIALS... "
        $BENCH benchmark --bench inverse --size $n --entry-bits 48 --backend cpu --output-csv 2>/dev/null | tail -1 >> "$OUTPUT" && echo "done" || echo "FAILED"
    done

    # GPU
    echo "    GPU:"
    for trial in $(seq 1 $TRIALS); do
        echo -n "      Trial $trial/$TRIALS... "
        $BENCH benchmark --bench inverse --size $n --entry-bits 48 --backend cuda --output-csv 2>/dev/null | tail -1 >> "$OUTPUT" && echo "done" || echo "FAILED"
    done
done

echo "Results: $OUTPUT"

# =============================================================================
# Benchmark 2: Multi-RHS Scaling with k (CRITICAL)
# =============================================================================
echo ""
echo "[7/12] Benchmark 2: Multi-RHS Scaling with k"
echo "=============================================="

OUTPUT="$RESULTS_DIR/multi_rhs_scaling_k.csv"
echo "n,k,primes,entry_bits,backend,total_ms,solve_ms,crt_ms,verified,trial" > "$OUTPUT"

if $QUICK_MODE; then
    K_VALUES="1 8 16"
else
    K_VALUES="1 2 4 8 16 32 64"
fi

for k in $K_VALUES; do
    echo "  n=128, k=$k:"

    # CPU
    echo "    CPU:"
    for trial in $(seq 1 $TRIALS); do
        echo -n "      Trial $trial/$TRIALS... "
        $BENCH benchmark --bench multi-rhs --size 128 --rhs $k --entry-bits 48 --backend cpu --output-csv 2>/dev/null | tail -1 >> "$OUTPUT" && echo "done" || echo "FAILED"
    done

    # GPU
    echo "    GPU:"
    for trial in $(seq 1 $TRIALS); do
        echo -n "      Trial $trial/$TRIALS... "
        $BENCH benchmark --bench multi-rhs --size 128 --rhs $k --entry-bits 48 --backend cuda --output-csv 2>/dev/null | tail -1 >> "$OUTPUT" && echo "done" || echo "FAILED"
    done
done

echo "Results: $OUTPUT"

# =============================================================================
# Benchmark 6: ZK Merkle Ledger
# =============================================================================
echo ""
echo "[8/12] Benchmark 6: ZK Merkle Ledger"
echo "=============================================="

OUTPUT="$RESULTS_DIR/zk_merkle.csv"
echo "scenario,n,k,primes,backend,total_ms,residue_ms,solve_ms,crt_ms,verified,trial" > "$OUTPUT"

echo "  Merkle Ledger (n=128, k=16):"

# CPU
echo "    CPU:"
for trial in $(seq 1 $TRIALS); do
    echo -n "      Trial $trial/$TRIALS... "
    $BENCH zk-preprocess --scenario merkle --size 128 --k 16 --backend cpu --output-csv 2>/dev/null | tail -1 >> "$OUTPUT" && echo "done" || echo "FAILED"
done

# GPU
echo "    GPU:"
for trial in $(seq 1 $TRIALS); do
    echo -n "      Trial $trial/$TRIALS... "
    $BENCH zk-preprocess --scenario merkle --size 128 --k 16 --backend cuda --output-csv 2>/dev/null | tail -1 >> "$OUTPUT" && echo "done" || echo "FAILED"
done

echo "Results: $OUTPUT"

# =============================================================================
# Benchmark 11: Time Breakdown
# =============================================================================
echo ""
echo "[9/12] Benchmark 11: Time Breakdown"
echo "=============================================="

OUTPUT="$RESULTS_DIR/time_breakdown.csv"
echo "n,k,primes,mode,total_ms,residue_pct,solve_pct,crt_pct,trial" > "$OUTPUT"

for n in 128 256; do
    echo "  n=$n, k=16:"
    for trial in $(seq 1 $TRIALS); do
        echo -n "    Trial $trial/$TRIALS... "
        $BENCH gpu-full-bench --size $n --rhs 16 --entry-bits 48 --breakdown --output-csv 2>/dev/null | tail -3 >> "$OUTPUT" && echo "done" || echo "FAILED"
    done
done

echo "Results: $OUTPUT"

# =============================================================================
# Benchmark 3: Determinant
# =============================================================================
echo ""
echo "[10/12] Benchmark 3: Determinant"
echo "=============================================="

OUTPUT="$RESULTS_DIR/determinant.csv"
echo "n,primes,entry_bits,backend,total_ms,verified,trial" > "$OUTPUT"

if $QUICK_MODE; then
    SIZES="64 128"
else
    SIZES="64 128 256 512"
fi

for n in $SIZES; do
    echo "  Size n=$n:"

    # CPU
    echo "    CPU:"
    for trial in $(seq 1 $TRIALS); do
        echo -n "      Trial $trial/$TRIALS... "
        $BENCH benchmark --bench det --size $n --entry-bits 48 --backend cpu --output-csv 2>/dev/null | tail -1 >> "$OUTPUT" && echo "done" || echo "FAILED"
    done

    # GPU
    echo "    GPU:"
    for trial in $(seq 1 $TRIALS); do
        echo -n "      Trial $trial/$TRIALS... "
        $BENCH benchmark --bench det --size $n --entry-bits 48 --backend cuda --output-csv 2>/dev/null | tail -1 >> "$OUTPUT" && echo "done" || echo "FAILED"
    done
done

echo "Results: $OUTPUT"

# =============================================================================
# Benchmark 7: ZK Range Proof
# =============================================================================
echo ""
echo "[11/12] Benchmark 7: ZK Range Proof"
echo "=============================================="

OUTPUT="$RESULTS_DIR/zk_range.csv"
echo "scenario,n,k,primes,total_ms,solve_pct,crt_pct,trial" > "$OUTPUT"

echo "  Range Proof (n=128, k=16):"
for trial in $(seq 1 $TRIALS); do
    echo -n "    Trial $trial/$TRIALS... "
    $BENCH zk-preprocess --scenario range --size 128 --k 16 --backend cuda --output-csv 2>/dev/null | tail -1 >> "$OUTPUT" && echo "done" || echo "FAILED"
done

echo "Results: $OUTPUT"

# =============================================================================
# Benchmark 8: Sparse vs Dense
# =============================================================================
echo ""
echo "[12/12] Benchmark 8: Sparse vs Dense"
echo "=============================================="

OUTPUT="$RESULTS_DIR/sparse_dense.csv"
echo "n,nnz,sparsity,dense_ms,sparse_ms,speedup,trial" > "$OUTPUT"

if $QUICK_MODE; then
    SIZES="32 128"
else
    SIZES="32 128 256 512"
fi

for n in $SIZES; do
    echo "  Size n=$n (5 nnz/row):"
    for trial in $(seq 1 $TRIALS); do
        echo -n "    Trial $trial/$TRIALS... "
        $BENCH benchmark --bench sparse --size $n --nnz-per-row 5 --backend cuda --output-csv 2>/dev/null | tail -1 >> "$OUTPUT" && echo "done" || echo "FAILED"
    done
done

echo "Results: $OUTPUT"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Benchmark Suite Complete!"
echo "=============================================="
echo ""
echo "Results saved to: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"

echo ""
echo "Total files:"
wc -l "$RESULTS_DIR"/*.csv 2>/dev/null || true

echo ""
echo "Next steps:"
echo "1. Review results in $RESULTS_DIR/"
echo "2. Run Metal benchmarks on MacBook (Benchmark 9)"
echo "3. Use generate_latex_tables.py to create LaTeX"
echo "4. Update paper with official numbers"

echo ""
echo "To copy results to paper directory:"
echo "  cp -r $RESULTS_DIR/* paper/paper_results/"
