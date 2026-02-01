#!/bin/bash
# =============================================================================
# Parallel Lift - CUDA Benchmark Suite
# Run on RunPod.io with NVIDIA RTX 4090
# =============================================================================

set -e

echo "=========================================="
echo "Parallel Lift CUDA Benchmark Suite"
echo "Paper Results"
echo "=========================================="
echo ""

# Configuration
RESULTS_DIR="./paper_results_cuda"
TRIALS=5
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$RESULTS_DIR"

# System info
echo "System Information:" | tee "$RESULTS_DIR/system_info.txt"
echo "==================" | tee -a "$RESULTS_DIR/system_info.txt"
nvidia-smi | tee -a "$RESULTS_DIR/system_info.txt"
echo "" | tee -a "$RESULTS_DIR/system_info.txt"
nvcc --version | tee -a "$RESULTS_DIR/system_info.txt"
echo "" | tee -a "$RESULTS_DIR/system_info.txt"
uname -a | tee -a "$RESULTS_DIR/system_info.txt"
echo "" | tee -a "$RESULTS_DIR/system_info.txt"
cat /proc/cpuinfo | grep "model name" | head -1 | tee -a "$RESULTS_DIR/system_info.txt"
free -h | tee -a "$RESULTS_DIR/system_info.txt"

# Build
echo ""
echo "Building Parallel Lift with CUDA..."
cd rust
cargo build --release --features cuda
cd ..

BENCH_CMD="./rust/target/release/parallel-lift"

# =============================================================================
# 1. Multi-RHS Solve - Scaling with Matrix Size (Table 2)
# =============================================================================
echo ""
echo "=========================================="
echo "Benchmark 1: Multi-RHS Solve - Scaling with n"
echo "Configuration: k=16 RHS vectors, 48-bit entries"
echo "=========================================="

OUTPUT="$RESULTS_DIR/multi_rhs_scaling_n.csv"
echo "n,k,primes,cpu_ms,gpu_ms,speedup,verified" > "$OUTPUT"

for n in 32 64 128 256 512; do
    echo "Running n=$n..."
    for trial in $(seq 1 $TRIALS); do
        $BENCH_CMD benchmark --bench multi-rhs \
            --size $n \
            --rhs 16 \
            --entry-bits 48 \
            --backend cuda \
            --output-csv >> "$OUTPUT.tmp" 2>/dev/null
    done
done

# Process and compute medians (using Python)
python3 << 'EOF'
import pandas as pd
import sys

try:
    df = pd.read_csv("$RESULTS_DIR/multi_rhs_scaling_n.csv.tmp")
    grouped = df.groupby(['n', 'k', 'primes']).median().reset_index()
    grouped.to_csv("$RESULTS_DIR/multi_rhs_scaling_n.csv", index=False)
except Exception as e:
    print(f"Error processing: {e}")
EOF

echo "Results saved to $OUTPUT"

# =============================================================================
# 2. Multi-RHS Solve - Scaling with k (Table 3)
# =============================================================================
echo ""
echo "=========================================="
echo "Benchmark 2: Multi-RHS Solve - Scaling with k"
echo "Configuration: n=128, 48-bit entries"
echo "=========================================="

OUTPUT="$RESULTS_DIR/multi_rhs_scaling_k.csv"
echo "n,k,primes,cpu_ms,gpu_ms,speedup,verified" > "$OUTPUT"

for k in 1 2 4 8 16 32 64; do
    echo "Running k=$k..."
    for trial in $(seq 1 $TRIALS); do
        $BENCH_CMD benchmark --bench multi-rhs \
            --size 128 \
            --rhs $k \
            --entry-bits 48 \
            --backend cuda \
            --output-csv >> "$OUTPUT" 2>/dev/null
    done
done

echo "Results saved to $OUTPUT"

# =============================================================================
# 3. Determinant Computation (Table 4)
# =============================================================================
echo ""
echo "=========================================="
echo "Benchmark 3: Determinant Computation"
echo "Configuration: 48-bit entries"
echo "=========================================="

OUTPUT="$RESULTS_DIR/determinant.csv"
echo "n,primes,cpu_ms,gpu_ms,speedup,verified" > "$OUTPUT"

for n in 32 64 128 256 512; do
    echo "Running n=$n..."
    for trial in $(seq 1 $TRIALS); do
        $BENCH_CMD benchmark --bench det \
            --size $n \
            --entry-bits 48 \
            --backend cuda \
            --output-csv >> "$OUTPUT" 2>/dev/null

        $BENCH_CMD benchmark --bench det \
            --size $n \
            --entry-bits 48 \
            --backend cpu \
            --output-csv >> "$OUTPUT" 2>/dev/null
    done
done

echo "Results saved to $OUTPUT"

# =============================================================================
# 4. Matrix Inverse (Table 5)
# =============================================================================
echo ""
echo "=========================================="
echo "Benchmark 4: Matrix Inverse"
echo "Configuration: k=n RHS vectors, 48-bit entries"
echo "=========================================="

OUTPUT="$RESULTS_DIR/inverse.csv"
echo "n,primes,cpu_total_ms,gpu_total_ms,total_speedup,cpu_solve_ms,gpu_solve_ms,solve_speedup,verified" > "$OUTPUT"

for n in 32 64 128; do
    echo "Running n=$n..."
    for trial in $(seq 1 $TRIALS); do
        $BENCH_CMD benchmark --bench inverse \
            --size $n \
            --entry-bits 48 \
            --backend cuda \
            --output-csv >> "$OUTPUT" 2>/dev/null
    done
done

echo "Results saved to $OUTPUT"

# =============================================================================
# 5. CRT Reconstruction (Table 6)
# =============================================================================
echo ""
echo "=========================================="
echo "Benchmark 5: CRT Reconstruction (Garner's Algorithm)"
echo "=========================================="

OUTPUT="$RESULTS_DIR/crt_reconstruction.csv"
echo "primes,values,cpu_ms,gpu_ms,speedup,verified" > "$OUTPUT"

for primes in 64 128 256 512; do
    for values in 256 1024 4096; do
        echo "Running primes=$primes, values=$values..."
        for trial in $(seq 1 $TRIALS); do
            $BENCH_CMD gpu-crt-bench \
                --num-primes $primes \
                --num-values $values \
                --output-csv >> "$OUTPUT" 2>/dev/null
        done
    done
done

echo "Results saved to $OUTPUT"

# =============================================================================
# 6. ZK Preprocessing Scenarios (Table 7)
# =============================================================================
echo ""
echo "=========================================="
echo "Benchmark 6: ZK Preprocessing Scenarios"
echo "=========================================="

OUTPUT="$RESULTS_DIR/zk_scenarios.csv"
echo "scenario,n,k,primes,cpu_ms,gpu_ms,speedup,solve_pct,crt_pct" > "$OUTPUT"

# Merkle Ledger (n=128, k=16)
echo "Running Merkle Ledger scenario..."
for trial in $(seq 1 $TRIALS); do
    $BENCH_CMD benchmark --bench zk-merkle \
        --output-csv >> "$OUTPUT" 2>/dev/null
done

# Range Proof (n=128, k=16)
echo "Running Range Proof scenario..."
for trial in $(seq 1 $TRIALS); do
    $BENCH_CMD benchmark --bench zk-range \
        --output-csv >> "$OUTPUT" 2>/dev/null
done

# Sparse R1CS (n=256, k=16)
echo "Running Sparse R1CS scenario..."
for trial in $(seq 1 $TRIALS); do
    $BENCH_CMD benchmark --bench zk-sparse \
        --output-csv >> "$OUTPUT" 2>/dev/null
done

echo "Results saved to $OUTPUT"

# =============================================================================
# 7. Sparse vs Dense Comparison (Table 8)
# =============================================================================
echo ""
echo "=========================================="
echo "Benchmark 7: Sparse vs Dense Solver"
echo "Configuration: R1CS-like matrices (5 nnz/row)"
echo "=========================================="

OUTPUT="$RESULTS_DIR/sparse_comparison.csv"
echo "n,nnz,sparsity,dense_ms,sparse_ms,speedup" > "$OUTPUT"

for n in 32 64 128 256 512; do
    echo "Running n=$n..."
    for trial in $(seq 1 $TRIALS); do
        $BENCH_CMD benchmark --bench sparse \
            --size $n \
            --nnz-per-row 5 \
            --backend cuda \
            --output-csv >> "$OUTPUT" 2>/dev/null
    done
done

echo "Results saved to $OUTPUT"

# =============================================================================
# 8. Full Pipeline with GPU CRT (Key Result)
# =============================================================================
echo ""
echo "=========================================="
echo "Benchmark 8: Full GPU Pipeline (Solve + GPU CRT)"
echo "This is the key result for peak speedup"
echo "=========================================="

OUTPUT="$RESULTS_DIR/full_pipeline.csv"
echo "n,k,primes,cpu_total_ms,gpu_cpu_crt_ms,gpu_gpu_crt_ms,speedup_cpu_crt,speedup_gpu_crt" > "$OUTPUT"

for n in 32 64 128 256; do
    echo "Running n=$n..."
    for trial in $(seq 1 $TRIALS); do
        $BENCH_CMD gpu-full-bench \
            --size $n \
            --rhs 16 \
            --entry-bits 48 \
            --output-csv >> "$OUTPUT" 2>/dev/null
    done
done

echo "Results saved to $OUTPUT"

# =============================================================================
# 9. Time Breakdown Analysis
# =============================================================================
echo ""
echo "=========================================="
echo "Benchmark 9: Time Breakdown"
echo "=========================================="

OUTPUT="$RESULTS_DIR/time_breakdown.csv"
echo "n,k,primes,total_ms,residue_pct,solve_pct,crt_pct" > "$OUTPUT"

for n in 32 64 128 192 256; do
    echo "Running n=$n..."
    $BENCH_CMD benchmark --bench breakdown \
        --size $n \
        --rhs 16 \
        --backend cuda \
        --output-csv >> "$OUTPUT" 2>/dev/null
done

echo "Results saved to $OUTPUT"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=========================================="
echo "Benchmark Suite Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"

echo ""
echo "To generate LaTeX tables, run:"
echo "  python3 paper/benchmarks/generate_latex_tables.py $RESULTS_DIR"
