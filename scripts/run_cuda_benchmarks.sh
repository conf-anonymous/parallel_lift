#!/bin/bash
# =============================================================================
# Parallel Lift - CUDA Benchmark Runner
# =============================================================================
# Runs comprehensive benchmarks on RunPod RTX 4090 and generates results.
#
# Usage:
#   ./scripts/run_cuda_benchmarks.sh [--quick|--full|--compare]
#
# Options:
#   --quick    Run quick validation benchmarks (5 min)
#   --full     Run full benchmark suite (30+ min)
#   --compare  Run side-by-side CPU vs CUDA comparison
#   (default)  Run standard benchmark suite (15 min)
#
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_section() { echo -e "\n${CYAN}=== $1 ===${NC}\n"; }

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUST_DIR="$PROJECT_DIR/rust"
RESULTS_DIR="$PROJECT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$RESULTS_DIR/cuda_benchmark_${TIMESTAMP}.txt"
CSV_FILE="$RESULTS_DIR/cuda_benchmark_${TIMESTAMP}.csv"

# Benchmark configuration
ITERATIONS=5
WARMUP=2

# Parse arguments
MODE="standard"
case "${1:-}" in
    --quick)  MODE="quick" ;;
    --full)   MODE="full" ;;
    --compare) MODE="compare" ;;
    --help|-h)
        echo "Usage: $0 [--quick|--full|--compare]"
        echo ""
        echo "Options:"
        echo "  --quick    Quick validation (5 min)"
        echo "  --full     Full suite (30+ min)"
        echo "  --compare  CPU vs CUDA comparison"
        echo "  (default)  Standard suite (15 min)"
        exit 0
        ;;
esac

# =============================================================================
# Setup
# =============================================================================
log_section "Parallel Lift CUDA Benchmarks"

echo "Mode: $MODE"
echo "Project: $PROJECT_DIR"
echo "Results: $RESULTS_FILE"
echo "Timestamp: $TIMESTAMP"
echo ""

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

# Verify CUDA
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. Not a GPU instance?"
    exit 1
fi

# Log GPU info
log_info "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Build project
log_info "Building project..."
cd "$RUST_DIR"
cargo build --release 2>&1 | tail -5

# Check if CUDA backend is available
# (For now, we'll use the existing Metal backend structure as reference)
BACKEND="cpu"  # Default to CPU until CUDA is implemented
if cargo run --release --bin parallel-lift -- gpu-info 2>/dev/null | grep -q "CUDA"; then
    BACKEND="cuda"
    log_success "CUDA backend detected"
else
    log_warn "CUDA backend not available, using CPU for comparison baseline"
fi

# =============================================================================
# Benchmark Functions
# =============================================================================

run_determinant_bench() {
    local size=$1
    local backend=$2
    local iters=${3:-$ITERATIONS}

    log_info "Determinant n=$size (backend=$backend, iters=$iters)"
    cargo run --release --bin parallel-lift -- benchmark \
        --bench det \
        --size "$size" \
        --backend "$backend" \
        --iterations "$iters" \
        2>&1 | tee -a "$RESULTS_FILE"
}

run_multi_rhs_bench() {
    local size=$1
    local k=$2
    local backend=$3
    local iters=${4:-$ITERATIONS}

    log_info "Multi-RHS n=$size, k=$k (backend=$backend, iters=$iters)"
    cargo run --release --bin parallel-lift -- benchmark \
        --bench multi-rhs \
        --size "$size" \
        --rhs "$k" \
        --backend "$backend" \
        --iterations "$iters" \
        2>&1 | tee -a "$RESULTS_FILE"
}

run_sweep_bench() {
    local bench_type=$1
    local backend=$2

    log_info "Running $bench_type sweep (backend=$backend)"
    cargo run --release --bin parallel-lift -- sweep \
        --bench "$bench_type" \
        --backend "$backend" \
        --export "$RESULTS_DIR/" \
        2>&1 | tee -a "$RESULTS_FILE"
}

# =============================================================================
# Quick Benchmarks
# =============================================================================
run_quick_benchmarks() {
    log_section "Quick Validation Benchmarks"

    # Small sanity checks
    for n in 32 64 128; do
        run_determinant_bench "$n" "$BACKEND" 3
    done

    # Multi-RHS quick check
    run_multi_rhs_bench 64 16 "$BACKEND" 3
}

# =============================================================================
# Standard Benchmarks
# =============================================================================
run_standard_benchmarks() {
    log_section "Determinant Scaling (varying n)"

    for n in 32 64 96 128 160 192; do
        run_determinant_bench "$n" "$BACKEND"
    done

    log_section "Multi-RHS Scaling (varying k, n=128)"

    for k in 1 2 4 8 16 32 64; do
        run_multi_rhs_bench 128 "$k" "$BACKEND"
    done

    log_section "Multi-RHS Scaling (varying n, k=16)"

    for n in 32 64 96 128; do
        run_multi_rhs_bench "$n" 16 "$BACKEND"
    done
}

# =============================================================================
# Full Benchmarks
# =============================================================================
run_full_benchmarks() {
    log_section "Full Determinant Sweep"

    for n in 32 48 64 80 96 112 128 144 160 176 192 208 224 256; do
        run_determinant_bench "$n" "$BACKEND" 10
    done

    log_section "Full Multi-RHS Sweep (varying k)"

    for k in 1 2 4 8 16 32 64 128; do
        run_multi_rhs_bench 128 "$k" "$BACKEND" 10
    done

    log_section "Full Multi-RHS Sweep (varying n)"

    for n in 32 48 64 80 96 112 128 160 192 224 256; do
        run_multi_rhs_bench "$n" 16 "$BACKEND" 10
    done

    log_section "Large Matrix Tests"

    # Push to larger sizes
    for n in 256 320 384; do
        run_determinant_bench "$n" "$BACKEND" 5
    done
}

# =============================================================================
# Comparison Benchmarks (CPU vs GPU)
# =============================================================================
run_comparison_benchmarks() {
    log_section "CPU vs GPU Comparison"

    echo "matrix_size,k,backend,time_ms,speedup" > "$CSV_FILE"

    # Determinant comparison
    log_section "Determinant: CPU vs GPU"

    for n in 64 128 192 256; do
        log_info "n=$n"

        # CPU timing
        cpu_time=$(cargo run --release --bin parallel-lift -- benchmark \
            --bench det --size "$n" --backend cpu --iterations 3 2>&1 | \
            grep -oP 'time:\s*\K[\d.]+' | head -1)

        # GPU timing
        gpu_time=$(cargo run --release --bin parallel-lift -- benchmark \
            --bench det --size "$n" --backend "$BACKEND" --iterations 3 2>&1 | \
            grep -oP 'time:\s*\K[\d.]+' | head -1)

        if [ -n "$cpu_time" ] && [ -n "$gpu_time" ]; then
            speedup=$(echo "scale=2; $cpu_time / $gpu_time" | bc 2>/dev/null || echo "N/A")
            echo "$n,1,cpu,$cpu_time,1.00" >> "$CSV_FILE"
            echo "$n,1,$BACKEND,$gpu_time,$speedup" >> "$CSV_FILE"
            log_success "n=$n: CPU=${cpu_time}ms, GPU=${gpu_time}ms, Speedup=${speedup}x"
        fi
    done

    # Multi-RHS comparison
    log_section "Multi-RHS: CPU vs GPU"

    for k in 1 8 32 64; do
        n=64
        log_info "n=$n, k=$k"

        cpu_time=$(cargo run --release --bin parallel-lift -- benchmark \
            --bench multi-rhs --size "$n" --rhs "$k" --backend cpu --iterations 3 2>&1 | \
            grep -oP 'time:\s*\K[\d.]+' | head -1)

        gpu_time=$(cargo run --release --bin parallel-lift -- benchmark \
            --bench multi-rhs --size "$n" --rhs "$k" --backend "$BACKEND" --iterations 3 2>&1 | \
            grep -oP 'time:\s*\K[\d.]+' | head -1)

        if [ -n "$cpu_time" ] && [ -n "$gpu_time" ]; then
            speedup=$(echo "scale=2; $cpu_time / $gpu_time" | bc 2>/dev/null || echo "N/A")
            echo "$n,$k,cpu,$cpu_time,1.00" >> "$CSV_FILE"
            echo "$n,$k,$BACKEND,$gpu_time,$speedup" >> "$CSV_FILE"
            log_success "n=$n, k=$k: CPU=${cpu_time}ms, GPU=${gpu_time}ms, Speedup=${speedup}x"
        fi
    done

    log_info "CSV results written to: $CSV_FILE"
}

# =============================================================================
# Main
# =============================================================================
START_TIME=$(date +%s)

# Write header to results file
cat >> "$RESULTS_FILE" << EOF
================================================================================
Parallel Lift CUDA Benchmark Results
================================================================================
Date: $(date)
Mode: $MODE
Backend: $BACKEND
Iterations: $ITERATIONS

--------------------------------------------------------------------------------
EOF

case "$MODE" in
    quick)
        run_quick_benchmarks
        ;;
    full)
        run_full_benchmarks
        ;;
    compare)
        run_comparison_benchmarks
        ;;
    *)
        run_standard_benchmarks
        ;;
esac

# =============================================================================
# Summary
# =============================================================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

log_section "Benchmark Complete"

echo "" | tee -a "$RESULTS_FILE"
echo "--------------------------------------------------------------------------------" | tee -a "$RESULTS_FILE"
echo "Total time: ${DURATION}s" | tee -a "$RESULTS_FILE"
echo "Results saved to:" | tee -a "$RESULTS_FILE"
echo "  - $RESULTS_FILE" | tee -a "$RESULTS_FILE"
if [ -f "$CSV_FILE" ]; then
    echo "  - $CSV_FILE" | tee -a "$RESULTS_FILE"
fi
echo "" | tee -a "$RESULTS_FILE"

# Final GPU stats
log_info "Final GPU Memory State:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee -a "$RESULTS_FILE"

log_success "Benchmarks complete!"
