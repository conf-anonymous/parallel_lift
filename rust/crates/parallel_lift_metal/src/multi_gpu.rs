//! Multi-GPU dispatch for Metal
//!
//! Distributes work across multiple Metal devices (GPUs) when available.
//! Falls back to single-device mode on systems with only one GPU.
//!
//! # Architecture
//!
//! On Apple Silicon Macs, there's typically one GPU (the integrated Apple GPU).
//! On Mac Pro or external GPU setups, multiple GPUs may be available.
//!
//! Work distribution strategy:
//! - Partition primes across available GPUs
//! - Each GPU processes its subset independently
//! - Combine results after all GPUs complete

use metal::Device;
use std::sync::Arc;
use crate::MetalBackend;

/// Information about a Metal GPU device
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// Device name
    pub name: String,
    /// Whether this is the default device
    pub is_default: bool,
    /// Supports Metal
    pub supports_metal: bool,
    /// Device index
    pub index: usize,
}

/// Multi-GPU manager
///
/// Manages multiple Metal backends and distributes work across them.
pub struct MultiGpuManager {
    /// Available backends, one per GPU
    backends: Vec<Arc<MetalBackend>>,
    /// Device info for each GPU
    device_info: Vec<GpuDeviceInfo>,
}

impl MultiGpuManager {
    /// Create a new multi-GPU manager
    ///
    /// Enumerates all available Metal devices and creates backends for each.
    pub fn new() -> Option<Self> {
        let all_devices = Device::all();

        if all_devices.is_empty() {
            return None;
        }

        let mut backends = Vec::new();
        let mut device_info = Vec::new();

        let default_device = Device::system_default();
        let default_name = default_device.as_ref().map(|d| d.name().to_string());

        for (index, device) in all_devices.into_iter().enumerate() {
            let name = device.name().to_string();
            let is_default = default_name.as_ref() == Some(&name);

            // Try to create a backend for this device
            if let Some(backend) = Self::create_backend_for_device(device) {
                backends.push(Arc::new(backend));
                device_info.push(GpuDeviceInfo {
                    name,
                    is_default,
                    supports_metal: true,
                    index,
                });
            }
        }

        if backends.is_empty() {
            return None;
        }

        Some(Self {
            backends,
            device_info,
        })
    }

    /// Create a backend for a specific device
    fn create_backend_for_device(_device: Device) -> Option<MetalBackend> {
        // For now, use the standard MetalBackend::new() which uses system_default
        // In a full implementation, we'd pass the device to the backend
        MetalBackend::new()
    }

    /// Get the number of available GPUs
    pub fn num_gpus(&self) -> usize {
        self.backends.len()
    }

    /// Get device info for all GPUs
    pub fn device_info(&self) -> &[GpuDeviceInfo] {
        &self.device_info
    }

    /// Get a reference to the primary (default) backend
    pub fn primary_backend(&self) -> &MetalBackend {
        &self.backends[0]
    }

    /// Get a reference to a specific backend by index
    pub fn backend(&self, index: usize) -> Option<&MetalBackend> {
        self.backends.get(index).map(|b| b.as_ref())
    }

    /// Partition work indices across GPUs
    ///
    /// Given N items, returns N vectors where each vector contains
    /// the indices that should be processed by that GPU.
    pub fn partition_work(&self, total_items: usize) -> Vec<Vec<usize>> {
        let num_gpus = self.backends.len();
        let mut partitions: Vec<Vec<usize>> = vec![Vec::new(); num_gpus];

        for i in 0..total_items {
            partitions[i % num_gpus].push(i);
        }

        partitions
    }

    /// Print device information
    pub fn print_info(&self) {
        println!("Multi-GPU Configuration:");
        println!("  Available GPUs: {}", self.backends.len());
        for info in &self.device_info {
            println!("    [{}] {} {}",
                info.index,
                info.name,
                if info.is_default { "(default)" } else { "" }
            );
        }
    }
}

impl Default for MultiGpuManager {
    fn default() -> Self {
        Self::new().expect("No Metal devices available")
    }
}

/// Statistics for multi-GPU execution
#[derive(Debug, Clone, Default)]
pub struct MultiGpuStats {
    /// Time per GPU
    pub gpu_times_ms: Vec<f64>,
    /// Work items per GPU
    pub work_per_gpu: Vec<usize>,
    /// Total elapsed time
    pub total_time_ms: f64,
    /// Efficiency (parallel speedup / num_gpus)
    pub efficiency: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_gpu_manager() {
        if let Some(mgr) = MultiGpuManager::new() {
            println!("Found {} GPU(s)", mgr.num_gpus());
            mgr.print_info();

            // Test work partitioning
            let partitions = mgr.partition_work(16);
            let total: usize = partitions.iter().map(|p| p.len()).sum();
            assert_eq!(total, 16);

            // Each GPU should have work
            for (i, partition) in partitions.iter().enumerate() {
                println!("GPU {}: {} items", i, partition.len());
            }
        } else {
            println!("No Metal devices available");
        }
    }
}
