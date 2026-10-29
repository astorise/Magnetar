//! Hardware capability evidence (A03). Every capability carries three
//! independent stages -- discovery, executed compute, and production
//! validation -- so finding an instruction set or a Device never reads as
//! having run or qualified anything on it. Absent hardware is an explicit
//! skip with a reason, never a pass.

use std::{fmt, fs, hint::black_box, time::Instant};

use serde::{Deserialize, Serialize};

/// A capability the qualification matrix distinguishes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardwareCapability {
    Avx2,
    Avx512F,
    Neon,
    CudaComputeCapability,
    CudaGraphCapture,
    Fp16Compute,
    Bf16Compute,
    MemoryBandwidth,
    Hip,
    Wgpu,
    /// Real-checkpoint F32 generation on a Provider's Device.
    F32Generation,
}

impl fmt::Display for HardwareCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = serde_json::to_value(self).map_err(|_| fmt::Error)?;
        f.write_str(value.as_str().ok_or(fmt::Error)?)
    }
}

/// The outcome of one evidence stage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum StageOutcome {
    Passed { detail: String },
    Failed { reason: String },
    Skipped { reason: String },
}

impl StageOutcome {
    pub fn passed(detail: impl Into<String>) -> Self {
        Self::Passed {
            detail: detail.into(),
        }
    }

    pub fn failed(reason: impl Into<String>) -> Self {
        Self::Failed {
            reason: reason.into(),
        }
    }

    pub fn skipped(reason: impl Into<String>) -> Self {
        Self::Skipped {
            reason: reason.into(),
        }
    }

    pub fn is_passed(&self) -> bool {
        matches!(self, Self::Passed { .. })
    }
}

/// Evidence for one capability on one (named) Device.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapabilityEvidence {
    pub capability: HardwareCapability,
    /// The tested Device, or `None` when no Device was found.
    pub device: Option<String>,
    pub discovery: StageOutcome,
    pub compute: StageOutcome,
    pub production_validation: StageOutcome,
}

impl CapabilityEvidence {
    /// Required hardware is absent: every stage is skipped with `reason`.
    pub fn absent(capability: HardwareCapability, reason: impl Into<String>) -> Self {
        let reason = reason.into();
        Self {
            capability,
            device: None,
            discovery: StageOutcome::skipped(reason.clone()),
            compute: StageOutcome::skipped(reason.clone()),
            production_validation: StageOutcome::skipped(reason),
        }
    }

    /// A capability is production-validated only when all three stages
    /// passed on a named Device.
    pub fn is_production_validated(&self) -> bool {
        self.device.is_some()
            && self.discovery.is_passed()
            && self.compute.is_passed()
            && self.production_validation.is_passed()
    }
}

/// Identity of the machine the matrix was produced on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HostIdentity {
    pub os: String,
    pub arch: String,
    pub cpu_model: Option<String>,
    pub logical_cpus: Option<usize>,
}

impl HostIdentity {
    pub fn discover() -> Self {
        Self {
            os: std::env::consts::OS.into(),
            arch: std::env::consts::ARCH.into(),
            cpu_model: fs::read_to_string("/proc/cpuinfo")
                .ok()
                .and_then(|cpuinfo| parse_cpu_model(&cpuinfo)),
            logical_cpus: std::thread::available_parallelism().ok().map(usize::from),
        }
    }

    /// A human-readable Device name for the host CPU.
    pub fn cpu_device_name(&self) -> String {
        format!(
            "host-cpu:{}",
            self.cpu_model.as_deref().unwrap_or(self.arch.as_str())
        )
    }
}

fn parse_cpu_model(cpuinfo: &str) -> Option<String> {
    cpuinfo.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        matches!(key.trim(), "model name" | "Model" | "Hardware")
            .then(|| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
}

/// The full matrix for one qualification run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HardwareCapabilityMatrix {
    pub host: HostIdentity,
    pub entries: Vec<CapabilityEvidence>,
}

impl HardwareCapabilityMatrix {
    pub fn new(host: HostIdentity, mut entries: Vec<CapabilityEvidence>) -> Self {
        entries.sort_by_key(|entry| entry.capability);
        Self { host, entries }
    }

    pub fn entries_for(&self, capability: HardwareCapability) -> Vec<&CapabilityEvidence> {
        self.entries
            .iter()
            .filter(|entry| entry.capability == capability)
            .collect()
    }

    /// Capabilities this matrix may claim as production-validated.
    pub fn production_validated(&self) -> Vec<HardwareCapability> {
        self.entries
            .iter()
            .filter(|entry| entry.is_production_validated())
            .map(|entry| entry.capability)
            .collect()
    }
}

/// Discovers the host CPU's SIMD instruction sets. Discovery uses the
/// running CPU's feature flags; ISA-specific compute and production stages
/// stay skipped until a Provider kernel actually selects that ISA.
pub fn probe_host_cpu_simd(host: &HostIdentity) -> Vec<CapabilityEvidence> {
    let device = host.cpu_device_name();
    let no_isa_kernel = |isa: &str| {
        StageOutcome::skipped(format!(
            "no Provider kernel selects {isa} yet; ISA-specific compute is \
             optimize-cpu-simd-and-locality (B08) scope"
        ))
    };
    let entry = |capability: HardwareCapability, isa: &str, detected: Option<bool>, arch: &str| {
        match detected {
            None => CapabilityEvidence::absent(capability, format!("not an {arch} host")),
            Some(false) => CapabilityEvidence {
                capability,
                device: Some(device.clone()),
                discovery: StageOutcome::failed(format!("{isa} not reported by this CPU")),
                compute: StageOutcome::skipped(format!("{isa} not available")),
                production_validation: StageOutcome::skipped(format!("{isa} not available")),
            },
            Some(true) => CapabilityEvidence {
                capability,
                device: Some(device.clone()),
                discovery: StageOutcome::passed(format!("{isa} reported by CPU feature flags")),
                compute: no_isa_kernel(isa),
                production_validation: no_isa_kernel(isa),
            },
        }
    };
    vec![
        entry(HardwareCapability::Avx2, "AVX2", detect_avx2(), "x86_64"),
        entry(
            HardwareCapability::Avx512F,
            "AVX-512F",
            detect_avx512f(),
            "x86_64",
        ),
        entry(HardwareCapability::Neon, "NEON", detect_neon(), "aarch64"),
    ]
}

#[cfg(target_arch = "x86_64")]
fn detect_avx2() -> Option<bool> {
    Some(std::arch::is_x86_feature_detected!("avx2"))
}

#[cfg(not(target_arch = "x86_64"))]
fn detect_avx2() -> Option<bool> {
    None
}

#[cfg(target_arch = "x86_64")]
fn detect_avx512f() -> Option<bool> {
    Some(std::arch::is_x86_feature_detected!("avx512f"))
}

#[cfg(not(target_arch = "x86_64"))]
fn detect_avx512f() -> Option<bool> {
    None
}

#[cfg(target_arch = "aarch64")]
fn detect_neon() -> Option<bool> {
    Some(std::arch::is_aarch64_feature_detected!("neon"))
}

#[cfg(not(target_arch = "aarch64"))]
fn detect_neon() -> Option<bool> {
    None
}

/// Executes a real host memory copy over `buffer_bytes` (`repetitions`
/// times) and records the best observed bandwidth. Bandwidth is evidence
/// for interpreting other results, not a capability to qualify, so its
/// production stage is always skipped.
pub fn measure_host_memory_bandwidth(
    host: &HostIdentity,
    buffer_bytes: usize,
    repetitions: usize,
) -> CapabilityEvidence {
    let device = Some(host.cpu_device_name());
    let informational = StageOutcome::skipped("bandwidth is informational evidence only");
    if buffer_bytes == 0 || repetitions == 0 {
        return CapabilityEvidence {
            capability: HardwareCapability::MemoryBandwidth,
            device,
            discovery: StageOutcome::passed("host memory"),
            compute: StageOutcome::skipped("no bandwidth test was requested"),
            production_validation: informational,
        };
    }
    let source = vec![1_u8; buffer_bytes];
    let mut destination = vec![0_u8; buffer_bytes];
    let best_seconds = (0..repetitions)
        .map(|_| {
            let started = Instant::now();
            destination.copy_from_slice(black_box(&source));
            black_box(&mut destination);
            started.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min);
    let compute = if destination[buffer_bytes - 1] != 1 {
        StageOutcome::failed("copied buffer does not match its source")
    } else if best_seconds > 0.0 {
        // Read plus write traffic per copy.
        let gigabytes_per_second = (2.0 * buffer_bytes as f64) / best_seconds / 1e9;
        StageOutcome::passed(format!(
            "best of {repetitions} copies of {buffer_bytes} bytes: \
             {gigabytes_per_second:.2} GB/s read+write"
        ))
    } else {
        StageOutcome::failed("copy completed below timer resolution")
    };
    CapabilityEvidence {
        capability: HardwareCapability::MemoryBandwidth,
        device,
        discovery: StageOutcome::passed("host memory"),
        compute,
        production_validation: informational,
    }
}

#[cfg(test)]
#[path = "hardware_tests.rs"]
mod tests;
