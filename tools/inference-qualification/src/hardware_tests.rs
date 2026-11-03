use super::*;

fn host() -> HostIdentity {
    HostIdentity::discover()
}

#[test]
fn absent_hardware_is_an_explicit_skip_and_claims_nothing() {
    let entry = CapabilityEvidence::absent(HardwareCapability::Hip, "no ROCm Device");
    assert_eq!(entry.device, None);
    assert_eq!(entry.discovery, StageOutcome::skipped("no ROCm Device"));
    assert_eq!(entry.compute, StageOutcome::skipped("no ROCm Device"));
    assert!(!entry.is_production_validated());

    let matrix = HardwareCapabilityMatrix::new(host(), vec![entry]);
    assert!(matrix.production_validated().is_empty());
}

#[test]
fn production_validation_requires_every_stage_on_a_named_device() {
    let mut entry = CapabilityEvidence {
        capability: HardwareCapability::F32Generation,
        device: Some("cpu".into()),
        discovery: StageOutcome::passed("found"),
        compute: StageOutcome::passed("ran"),
        production_validation: StageOutcome::passed("qualified"),
    };
    assert!(entry.is_production_validated());
    entry.production_validation = StageOutcome::skipped("gates pending");
    assert!(!entry.is_production_validated());
    entry.production_validation = StageOutcome::passed("qualified");
    entry.device = None;
    assert!(!entry.is_production_validated());
}

#[test]
fn cpu_simd_discovery_never_claims_compute_or_production() {
    let host = host();
    let entries = probe_host_cpu_simd(&host);
    assert_eq!(entries.len(), 3);
    for entry in &entries {
        assert!(!entry.compute.is_passed(), "{entry:?}");
        assert!(!entry.is_production_validated(), "{entry:?}");
    }
    let applicable = entries
        .iter()
        .filter(|entry| entry.device.is_some())
        .count();
    if cfg!(any(target_arch = "x86_64", target_arch = "aarch64")) {
        assert!(applicable >= 1, "the host's own ISA family is evaluated");
    }
}

#[test]
fn bandwidth_test_executes_a_real_copy() {
    let entry = measure_host_memory_bandwidth(&host(), 1 << 20, 3);
    assert_eq!(entry.capability, HardwareCapability::MemoryBandwidth);
    assert!(entry.compute.is_passed(), "{entry:?}");
    assert!(!entry.is_production_validated());

    let skipped = measure_host_memory_bandwidth(&host(), 0, 3);
    assert!(!skipped.compute.is_passed());
}

#[test]
fn matrix_sorts_and_filters_entries() {
    let matrix = HardwareCapabilityMatrix::new(
        host(),
        vec![
            CapabilityEvidence::absent(HardwareCapability::Wgpu, "x"),
            CapabilityEvidence::absent(HardwareCapability::Avx2, "y"),
        ],
    );
    assert_eq!(matrix.entries[0].capability, HardwareCapability::Avx2);
    assert_eq!(matrix.entries_for(HardwareCapability::Wgpu).len(), 1);
    assert!(matrix.entries_for(HardwareCapability::Neon).is_empty());
    assert_eq!(
        HardwareCapability::CudaGraphCapture.to_string(),
        "cuda_graph_capture"
    );
}

#[test]
fn parses_cpu_model_from_cpuinfo() {
    assert_eq!(
        parse_cpu_model("processor\t: 0\nmodel name\t: Example CPU @ 3GHz\n"),
        Some("Example CPU @ 3GHz".into())
    );
    assert_eq!(parse_cpu_model("processor\t: 0\n"), None);
    assert!(host().cpu_device_name().starts_with("host-cpu:"));
}
