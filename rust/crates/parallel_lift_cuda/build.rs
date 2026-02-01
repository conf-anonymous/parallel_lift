//! Build script for CUDA kernel compilation
//!
//! Compiles .cu kernel files to PTX using nvcc at build time.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels/");

    // Check if we're building with the stub feature
    if env::var("CARGO_FEATURE_STUB").is_ok() {
        println!("cargo:warning=Building with stub feature, skipping CUDA compilation");
        create_stub_ptx();
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernels_dir = manifest_dir.join("src").join("kernels");

    // Check if nvcc is available
    let nvcc_check = Command::new("nvcc").arg("--version").output();

    if nvcc_check.is_err() {
        println!("cargo:warning=nvcc not found, creating stub PTX");
        create_stub_ptx();
        return;
    }

    // Determine GPU architecture
    // Default to sm_89 for RTX 4090 (Ada Lovelace)
    // Can be overridden with CUDA_ARCH environment variable
    let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_89".to_string());

    // Use the combined kernel file for a single valid PTX output
    let combined_kernel = kernels_dir.join("all_kernels.cu");
    let ptx_output = out_dir.join("kernels.ptx");

    // Build nvcc command for single file compilation
    let output = Command::new("nvcc")
        .args([
            "-ptx",
            &format!("-arch={}", cuda_arch),
            "-O3",
            "--use_fast_math",
            "-lineinfo",
        ])
        .arg("-o")
        .arg(ptx_output.as_os_str())
        .arg(combined_kernel.as_os_str())
        .output()
        .expect("Failed to run nvcc");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!(
            "CUDA kernel compilation failed!\nstderr: {}\nstdout: {}",
            stderr, stdout
        );
    }

    println!("cargo:warning=CUDA kernels compiled successfully to {}", ptx_output.display());

    // Link CUDA runtime (optional, for runtime loading)
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    } else {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }
}

fn create_stub_ptx() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_output = out_dir.join("kernels.ptx");

    // Create an empty PTX file as a stub
    let stub_ptx = r#"
// Stub PTX file - CUDA not available
// This file is a placeholder when building without CUDA support
.version 7.0
.target sm_89
.address_size 64
"#;

    std::fs::write(&ptx_output, stub_ptx).expect("Failed to write stub PTX");
}
