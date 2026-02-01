#!/bin/bash
# =============================================================================
# Parallel Lift - RunPod Setup Script
# =============================================================================
# This script sets up a RunPod machine (RTX 4090) for CUDA development and
# benchmarking of the Parallel Lift project.
#
# Usage:
#   1. Start a RunPod instance with RTX 4090 and Ubuntu 22.04 + CUDA template
#   2. SSH into the instance or use the web terminal
#   3. Run: curl -sSL https://raw.githubusercontent.com/conf-anonymous/parallel_lift/main/scripts/setup_runpod.sh | bash
#   Or clone first, then: bash scripts/setup_runpod.sh
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# 1. System Verification
# =============================================================================
echo ""
echo "=============================================="
echo "  Parallel Lift - RunPod Setup"
echo "  Target: RTX 4090 CUDA Development"
echo "=============================================="
echo ""

log_info "Verifying system requirements..."

# Check if running on Linux
if [[ "$(uname)" != "Linux" ]]; then
    log_error "This script is designed for Linux (RunPod). Detected: $(uname)"
    exit 1
fi

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. Is this a GPU instance?"
    exit 1
fi

log_info "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Check CUDA installation
if ! command -v nvcc &> /dev/null; then
    log_warn "nvcc not found in PATH. Checking common locations..."

    # Try common CUDA paths
    for cuda_path in /usr/local/cuda /usr/local/cuda-12.* /usr/local/cuda-11.*; do
        if [ -f "$cuda_path/bin/nvcc" ]; then
            export PATH="$cuda_path/bin:$PATH"
            export LD_LIBRARY_PATH="$cuda_path/lib64:$LD_LIBRARY_PATH"
            log_success "Found CUDA at $cuda_path"
            break
        fi
    done
fi

if command -v nvcc &> /dev/null; then
    log_info "CUDA Version:"
    nvcc --version | grep "release"
else
    log_error "CUDA toolkit not found. Please use a CUDA-enabled RunPod template."
    exit 1
fi

echo ""

# =============================================================================
# 2. Install System Dependencies
# =============================================================================
log_info "Installing system dependencies..."

# Update package list
sudo apt-get update -qq

# Install essential build tools
sudo apt-get install -y -qq \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    curl \
    wget \
    htop \
    tmux \
    vim \
    tree \
    2>/dev/null

log_success "System dependencies installed"

# =============================================================================
# 3. Install Rust
# =============================================================================
log_info "Setting up Rust..."

if command -v rustc &> /dev/null; then
    log_info "Rust already installed: $(rustc --version)"
else
    log_info "Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source "$HOME/.cargo/env"
    log_success "Rust installed: $(rustc --version)"
fi

# Ensure cargo is in PATH for this session
source "$HOME/.cargo/env" 2>/dev/null || true

# Verify Rust installation
if ! command -v cargo &> /dev/null; then
    log_error "Cargo not found after Rust installation"
    exit 1
fi

log_success "Rust ready: $(cargo --version)"
echo ""

# =============================================================================
# 4. Clone/Update Repository
# =============================================================================
WORKSPACE_DIR="/workspace"
PROJECT_DIR="$WORKSPACE_DIR/parallel-lift"

log_info "Setting up project directory..."

# RunPod typically uses /workspace for persistent storage
if [ ! -d "$WORKSPACE_DIR" ]; then
    WORKSPACE_DIR="$HOME"
    PROJECT_DIR="$WORKSPACE_DIR/parallel-lift"
fi

cd "$WORKSPACE_DIR"

if [ -d "$PROJECT_DIR" ]; then
    log_info "Project exists, pulling latest changes..."
    cd "$PROJECT_DIR"
    git pull origin main 2>/dev/null || log_warn "Could not pull (might be local changes)"
else
    log_info "Cloning Parallel Lift repository..."
    git clone https://github.com/conf-anonymous/parallel_lift.git "$PROJECT_DIR" 2>/dev/null || {
        log_warn "Could not clone from GitHub. Creating local project structure..."
        mkdir -p "$PROJECT_DIR"
    }
    cd "$PROJECT_DIR"
fi

log_success "Project directory: $PROJECT_DIR"
echo ""

# =============================================================================
# 5. Setup Environment Variables
# =============================================================================
log_info "Configuring environment variables..."

# Create/update .bashrc additions
BASHRC_ADDITIONS="$HOME/.bashrc_parallel_lift"

cat > "$BASHRC_ADDITIONS" << 'EOF'
# =============================================================================
# Parallel Lift Environment Configuration
# =============================================================================

# CUDA paths
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Rust
source "$HOME/.cargo/env" 2>/dev/null || true

# Project directory
export PARALLEL_LIFT_DIR="/workspace/parallel-lift"
if [ ! -d "$PARALLEL_LIFT_DIR" ]; then
    export PARALLEL_LIFT_DIR="$HOME/parallel-lift"
fi

# Aliases for common operations
alias pl='cd $PARALLEL_LIFT_DIR'
alias plr='cd $PARALLEL_LIFT_DIR/rust'
alias plbuild='cd $PARALLEL_LIFT_DIR/rust && cargo build --release'
alias pltest='cd $PARALLEL_LIFT_DIR/rust && cargo test'
alias plbench='cd $PARALLEL_LIFT_DIR/rust && cargo run --release --bin parallel-lift -- benchmark'

# GPU monitoring alias
alias gpuwatch='watch -n 1 nvidia-smi'

# Quick benchmark functions
pldet() {
    local size=${1:-128}
    cd $PARALLEL_LIFT_DIR/rust
    cargo run --release --bin parallel-lift -- benchmark --bench det --size "$size" --backend "${2:-metal}"
}

plmulti() {
    local size=${1:-64}
    local k=${2:-16}
    cd $PARALLEL_LIFT_DIR/rust
    cargo run --release --bin parallel-lift -- benchmark --bench multi-rhs --size "$size" --rhs "$k" --backend "${3:-metal}"
}

echo "Parallel Lift environment loaded. Commands: pl, plr, plbuild, pltest, plbench, pldet, plmulti, gpuwatch"
EOF

# Add to .bashrc if not already present
if ! grep -q "bashrc_parallel_lift" "$HOME/.bashrc" 2>/dev/null; then
    echo "" >> "$HOME/.bashrc"
    echo "# Parallel Lift environment" >> "$HOME/.bashrc"
    echo "source $BASHRC_ADDITIONS" >> "$HOME/.bashrc"
fi

# Source for current session
source "$BASHRC_ADDITIONS"

log_success "Environment configured"
echo ""

# =============================================================================
# 6. Create CUDA Backend Crate Structure
# =============================================================================
log_info "Setting up CUDA backend crate structure..."

CUDA_CRATE_DIR="$PROJECT_DIR/rust/crates/parallel_lift_cuda"

if [ ! -d "$CUDA_CRATE_DIR" ]; then
    mkdir -p "$CUDA_CRATE_DIR/src/kernels"

    # Create Cargo.toml
    cat > "$CUDA_CRATE_DIR/Cargo.toml" << 'EOF'
[package]
name = "parallel_lift_cuda"
version = "0.1.0"
edition = "2021"
authors = ["Anonymous"]
license = "Apache-2.0"
description = "CUDA backend for Parallel Lift GPU-accelerated exact arithmetic"

[dependencies]
parallel_lift_core = { path = "../parallel_lift_core" }
cudarc = "0.11"
thiserror = "1.0"

[build-dependencies]
cc = "1.0"

[features]
default = []
# Enable for compile-time stub when CUDA is not available
stub = []
EOF

    # Create lib.rs
    cat > "$CUDA_CRATE_DIR/src/lib.rs" << 'EOF'
//! CUDA backend for Parallel Lift
//!
//! GPU-accelerated modular arithmetic using NVIDIA CUDA.

#[cfg(not(feature = "stub"))]
mod cuda_backend;
#[cfg(not(feature = "stub"))]
mod memory;
#[cfg(not(feature = "stub"))]
mod launch;
mod error;

#[cfg(not(feature = "stub"))]
pub use cuda_backend::CudaBackend;

pub use error::CudaError;

/// Check if CUDA is available on this system
pub fn is_cuda_available() -> bool {
    #[cfg(feature = "stub")]
    return false;

    #[cfg(not(feature = "stub"))]
    {
        cudarc::driver::CudaDevice::new(0).is_ok()
    }
}
EOF

    # Create error.rs
    cat > "$CUDA_CRATE_DIR/src/error.rs" << 'EOF'
//! CUDA error types

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaError {
    #[error("CUDA device not found")]
    DeviceNotFound,

    #[error("CUDA driver error: {0}")]
    DriverError(String),

    #[error("Kernel compilation error: {0}")]
    CompilationError(String),

    #[error("Kernel launch error: {0}")]
    LaunchError(String),

    #[error("Memory allocation error: {0}")]
    MemoryError(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}
EOF

    # Create placeholder kernel files
    cat > "$CUDA_CRATE_DIR/src/kernels/mod.rs" << 'EOF'
//! CUDA kernel module
//!
//! Kernel source is compiled via build.rs using nvcc.

/// Embedded PTX source (populated at build time)
pub const KERNELS_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
EOF

    # Create modular_ops.cu
    cat > "$CUDA_CRATE_DIR/src/kernels/modular_ops.cu" << 'EOF'
// =============================================================================
// Modular Arithmetic Utilities for 31-bit primes with 64-bit intermediate
// =============================================================================

#ifndef MODULAR_OPS_CU
#define MODULAR_OPS_CU

__device__ __forceinline__
uint32_t mod_add(uint32_t a, uint32_t b, uint32_t p) {
    uint64_t sum = (uint64_t)a + (uint64_t)b;
    return sum >= p ? (uint32_t)(sum - p) : (uint32_t)sum;
}

__device__ __forceinline__
uint32_t mod_sub(uint32_t a, uint32_t b, uint32_t p) {
    return a >= b ? (a - b) : (p - b + a);
}

__device__ __forceinline__
uint32_t mod_mul(uint32_t a, uint32_t b, uint32_t p) {
    return (uint32_t)(((uint64_t)a * (uint64_t)b) % (uint64_t)p);
}

// Fermat's Little Theorem: a^(-1) = a^(p-2) mod p
__device__ __forceinline__
uint32_t mod_inv(uint32_t a, uint32_t p) {
    if (a == 0) return 0;

    uint32_t result = 1;
    uint32_t base = a;
    uint32_t exp = p - 2;

    while (exp > 0) {
        if (exp & 1) {
            result = mod_mul(result, base, p);
        }
        base = mod_mul(base, base, p);
        exp >>= 1;
    }
    return result;
}

#endif // MODULAR_OPS_CU
EOF

    # Create build.rs
    cat > "$CUDA_CRATE_DIR/build.rs" << 'EOF'
//! Build script for CUDA kernel compilation

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels/");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernels_dir = manifest_dir.join("src/kernels");

    // Find nvcc
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());

    // Check if nvcc is available
    let nvcc_check = Command::new(&nvcc).arg("--version").output();

    if nvcc_check.is_err() {
        println!("cargo:warning=nvcc not found, creating stub PTX");
        // Create empty PTX for stub builds
        std::fs::write(out_dir.join("kernels.ptx"), "// Stub PTX - CUDA not available\n").unwrap();
        return;
    }

    // Detect GPU architecture (default to sm_89 for RTX 4090)
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_89".to_string());

    // Collect kernel source files
    let kernel_files: Vec<_> = std::fs::read_dir(&kernels_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "cu"))
        .map(|e| e.path())
        .collect();

    if kernel_files.is_empty() {
        println!("cargo:warning=No .cu files found, creating stub PTX");
        std::fs::write(out_dir.join("kernels.ptx"), "// Stub PTX - no kernels\n").unwrap();
        return;
    }

    // Compile to PTX
    let mut cmd = Command::new(&nvcc);
    cmd.args([
        "-ptx",
        &format!("-arch={}", arch),
        "-O3",
        "--use_fast_math",
        "-lineinfo",
        "-o",
    ])
    .arg(out_dir.join("kernels.ptx"));

    for kernel in &kernel_files {
        cmd.arg(kernel);
    }

    let output = cmd.output().expect("Failed to run nvcc");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("CUDA kernel compilation failed:\n{}", stderr);
    }

    // Link CUDA runtime
    if let Ok(cuda_path) = env::var("CUDA_HOME") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    } else {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }
    println!("cargo:rustc-link-lib=cudart");
}
EOF

    log_success "CUDA crate structure created at $CUDA_CRATE_DIR"
else
    log_info "CUDA crate already exists"
fi

echo ""

# =============================================================================
# 7. Update Workspace Cargo.toml
# =============================================================================
log_info "Updating workspace configuration..."

WORKSPACE_CARGO="$PROJECT_DIR/rust/Cargo.toml"

if [ -f "$WORKSPACE_CARGO" ]; then
    # Check if cuda crate is already in workspace
    if ! grep -q "parallel_lift_cuda" "$WORKSPACE_CARGO" 2>/dev/null; then
        log_info "Adding parallel_lift_cuda to workspace members..."
        # This is a simple sed replacement - may need manual adjustment
        sed -i 's/members = \[/members = [\n    "crates\/parallel_lift_cuda",/' "$WORKSPACE_CARGO" 2>/dev/null || {
            log_warn "Could not auto-update Cargo.toml. Please add 'crates/parallel_lift_cuda' to workspace members manually."
        }
    fi

    # Add cudarc to workspace dependencies if not present
    if ! grep -q "cudarc" "$WORKSPACE_CARGO" 2>/dev/null; then
        log_info "Adding cudarc to workspace dependencies..."
        # Append to dependencies section
        echo "" >> "$WORKSPACE_CARGO"
        echo "# CUDA" >> "$WORKSPACE_CARGO"
        echo 'cudarc = "0.11"' >> "$WORKSPACE_CARGO"
    fi
fi

echo ""

# =============================================================================
# 8. Build Project
# =============================================================================
log_info "Building Rust project..."

cd "$PROJECT_DIR/rust"

# Try to build (may fail if CUDA crate has issues, which is OK for initial setup)
if cargo build --release 2>/dev/null; then
    log_success "Project built successfully"
else
    log_warn "Build had warnings/errors. This is expected if CUDA backend is incomplete."
    log_info "Core crates should still work. Trying core-only build..."
    cargo build --release -p parallel_lift_core -p parallel_lift_cli 2>/dev/null && \
        log_success "Core crates built successfully"
fi

echo ""

# =============================================================================
# 9. Verify Setup
# =============================================================================
log_info "Verifying setup..."

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""

echo "System Information:"
echo "  - OS: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'"' -f2)"
echo "  - Kernel: $(uname -r)"
echo "  - CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' | tr -d ',')"
echo "  - Rust: $(rustc --version 2>/dev/null | awk '{print $2}')"
echo "  - Cargo: $(cargo --version 2>/dev/null | awk '{print $2}')"
echo ""

echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

echo "Project Structure:"
echo "  - Project: $PROJECT_DIR"
echo "  - Rust crates: $PROJECT_DIR/rust/crates/"
echo "  - CUDA backend: $PROJECT_DIR/rust/crates/parallel_lift_cuda/"
echo "  - Results: $PROJECT_DIR/results/"
echo ""

echo "Quick Commands (after reloading shell or sourcing ~/.bashrc):"
echo "  pl          - Go to project directory"
echo "  plbuild     - Build Rust project (release)"
echo "  pltest      - Run tests"
echo "  plbench     - Run benchmarks"
echo "  pldet 128   - Run determinant benchmark (n=128)"
echo "  plmulti 64 16 - Run multi-RHS benchmark (n=64, k=16)"
echo "  gpuwatch    - Monitor GPU usage"
echo ""

echo "Next Steps:"
echo "  1. Reload shell: source ~/.bashrc"
echo "  2. Navigate to project: pl"
echo "  3. Implement CUDA kernels in: rust/crates/parallel_lift_cuda/src/kernels/"
echo "  4. Build and test: plbuild && pltest"
echo "  5. Run benchmarks: plbench --backend cuda"
echo ""

log_success "RunPod setup complete!"
