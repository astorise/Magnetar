//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::capability::{CapabilityId, CapabilityVersion};
use crate::device::DeviceId;

use crate::compute::{
    COMPUTE_CAPABILITY_VERSION, ComputeOperationFamily, TensorResourceId, compute_capability,
};
use crate::memory::{MemoryPlacement, TensorResidency};
fn capability_binding(name: &str, version: CapabilityVersion) -> CapabilityBinding {
    CapabilityBinding::new(CapabilityId::new(name), version)
}

#[test]
fn affinity_constraints_preserve_compatible_facts_and_fallback_precedence() {
    let capability_a = capability_binding("magnetar:compute/run", CapabilityVersion::new(1, 1, 0));
    let capability_b = capability_binding("magnetar:tokenize/run", CapabilityVersion::new(1, 0, 0));
    let provider = ProviderBinding::new("provider-a");
    let device = DeviceBinding::new(DeviceId::new("gpu:0"));
    let context = ExecutionContextId::new(42);
    let group = AffinityGroupId::new(7);

    let model = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(provider.clone())
        .with_device(device.clone())
        .with_capability(capability_a.clone())
        .with_artifact(ArtifactBinding::new("model", "sha256:model"))
        .with_artifact(ArtifactBinding::new("bundle", "sha256:bundle"))
        .with_execution_context(context)
        .with_group(group);
    let tokenizer = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(provider)
        .with_device(device)
        .with_capability(capability_b.clone())
        .with_artifact(ArtifactBinding::new("tokenizer", "sha256:tokenizer"))
        .with_artifact(ArtifactBinding::new("bundle", "sha256:bundle"))
        .with_execution_context(context)
        .with_group(group);

    let constraints = AffinityConstraints::try_from_affinities([&model, &tokenizer]).unwrap();
    let aggregate = constraints.affinity();
    assert_eq!(aggregate.capability(capability_a.id()), Some(&capability_a));
    assert_eq!(aggregate.capability(capability_b.id()), Some(&capability_b));
    assert_eq!(
        aggregate.artifact("model").unwrap().fingerprint(),
        "sha256:model"
    );
    assert_eq!(
        aggregate.artifact("tokenizer").unwrap().fingerprint(),
        "sha256:tokenizer"
    );
    assert_eq!(
        aggregate.artifact("bundle").unwrap().fingerprint(),
        "sha256:bundle"
    );
    assert_eq!(aggregate.fallback(), FallbackClass::ProviderPinned);
}

#[test]
fn affinity_constraints_report_each_binding_conflict() {
    let base = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(ProviderBinding::new("provider-a"))
        .with_device(DeviceBinding::new(DeviceId::new("gpu:0")))
        .with_capability(capability_binding(
            "magnetar:compute/run",
            CapabilityVersion::new(1, 1, 0),
        ))
        .with_artifact(ArtifactBinding::new("bundle", "sha256:a"))
        .with_execution_context(ExecutionContextId::new(1))
        .with_group(AffinityGroupId::new(1));

    let provider_conflict = base
        .clone()
        .with_provider(ProviderBinding::new("provider-b"));
    assert!(matches!(
        base.validate_with(&provider_conflict),
        Err(AffinityError::ProviderMismatch { .. })
    ));

    let device_conflict = base
        .clone()
        .with_device(DeviceBinding::new(DeviceId::new("gpu:1")));
    assert!(matches!(
        base.validate_with(&device_conflict),
        Err(AffinityError::DeviceMismatch { .. })
    ));

    let capability_conflict = base.clone().with_capability(capability_binding(
        "magnetar:compute/run",
        CapabilityVersion::new(1, 2, 0),
    ));
    assert!(matches!(
        base.validate_with(&capability_conflict),
        Err(AffinityError::CapabilityMismatch { .. })
    ));

    let artifact_conflict = base
        .clone()
        .with_artifact(ArtifactBinding::new("bundle", "sha256:b"));
    assert!(matches!(
        base.validate_with(&artifact_conflict),
        Err(AffinityError::ArtifactMismatch { .. })
    ));

    let context_conflict = base
        .clone()
        .with_execution_context(ExecutionContextId::new(2));
    assert!(matches!(
        base.validate_with(&context_conflict),
        Err(AffinityError::ExecutionContextMismatch { .. })
    ));

    let group_conflict = base.clone().with_group(AffinityGroupId::new(2));
    assert!(matches!(
        base.validate_with(&group_conflict),
        Err(AffinityError::AffinityGroupMismatch { .. })
    ));
}

#[test]
fn affinity_resource_keeps_value_and_affinity_together() {
    let affinity = ResourceAffinity::new(FallbackClass::Restartable)
        .with_provider(ProviderBinding::new("provider-a"));
    let resource = AffinityResource::new("native-handle", affinity.clone());

    assert_eq!(resource.value(), &"native-handle");
    assert_eq!(resource.affinity(), &affinity);
    assert_eq!(resource.into_parts(), ("native-handle", affinity));
}

#[test]
fn health_reports_redact_diagnostics_and_track_freshness() {
    let provider = ProviderBinding::new("provider-a");
    let mut report = ProviderHealthReport::new(provider.clone(), HealthState::Saturated);
    report.timestamp = Some(HealthTimestamp::unix_millis(1_000));
    report.time_to_live = Some(HealthTimeToLive::millis(250));
    report.capacity.queue_depth = Some(8);
    report.diagnostics.push(
        HealthDiagnostic::new(HealthScope::Provider, HealthState::Saturated)
            .with_code("queue-pressure")
            .with_message("cuda stream /tmp/secret token=abc is saturated")
            .with_trace_id("trace-1"),
    );

    assert!(report.is_stale_at(HealthTimestamp::unix_millis(1_251)));
    assert_eq!(report.capacity.queue_depth, Some(8));
    let message = report.diagnostics[0].message.as_deref().unwrap();
    assert!(!message.contains("/tmp/secret"));
    assert!(!message.contains("token=abc"));
    assert_eq!(message, "[redacted backend diagnostic]");
    assert_eq!(report.diagnostics[0].trace_id.as_deref(), Some("trace-1"));

    let device_health = DeviceHealth::new(
        provider.clone(),
        DeviceBinding::new(DeviceId::new("gpu:0")),
        HealthState::Available,
    );
    let capability_health = CapabilityHealth::new(
        provider,
        CapabilityBinding::new(compute_capability().id, COMPUTE_CAPABILITY_VERSION),
        HealthState::Degraded,
    );
    assert!(matches!(
        HealthReport::Device(device_health),
        HealthReport::Device(_)
    ));
    assert!(matches!(
        HealthReport::Capability(capability_health),
        HealthReport::Capability(_)
    ));
}

#[test]
fn provider_status_snapshot_separates_health_readiness_pressure_and_admission() {
    let provider = ProviderBinding::new("provider-a");
    let mut snapshot = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        provider.clone(),
        HealthState::Available,
    ));
    snapshot.health = ProviderHealthState::Healthy;
    snapshot.readiness = ProviderReadinessState::NotReady;
    snapshot.pressure = ProviderPressureLevel::Low;
    snapshot.admission = provider_admission_from_dimensions(
        snapshot.lifecycle,
        snapshot.health,
        snapshot.readiness,
        snapshot.pressure,
    );
    snapshot.timestamp = Some(HealthTimestamp::unix_millis(10));
    snapshot.time_to_live = Some(HealthTimeToLive::millis(5));

    assert_eq!(snapshot.provider, provider);
    assert_eq!(snapshot.health, ProviderHealthState::Healthy);
    assert_eq!(snapshot.readiness, ProviderReadinessState::NotReady);
    assert_eq!(snapshot.pressure, ProviderPressureLevel::Low);
    assert_eq!(snapshot.admission, ProviderAdmissionDecision::Reject);
    assert!(!snapshot.accepts_new_work_by_default());
    assert!(snapshot.is_stale_at(HealthTimestamp::unix_millis(16)));
}

#[test]
fn operation_family_status_falls_back_to_capability_status_when_absent() {
    let provider = ProviderBinding::new("provider-a");
    let mut snapshot = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        provider.clone(),
        HealthState::Available,
    ));
    assert_eq!(
        snapshot.operation_family_or_capability_status(ComputeOperationFamily::LinearAlgebra),
        ProviderReadinessState::Ready
    );

    let unsupported =
        OperationFamilyStatus::unsupported(provider.clone(), ComputeOperationFamily::LinearAlgebra);
    snapshot = snapshot.with_operation_family_status(unsupported);
    assert_eq!(
        snapshot.operation_family_or_capability_status(ComputeOperationFamily::LinearAlgebra),
        ProviderReadinessState::NotReady
    );

    let mut saturated =
        OperationFamilyStatus::available(provider, ComputeOperationFamily::Elementwise);
    saturated.pressure = ProviderPressureLevel::Saturated;
    saturated.readiness = ProviderReadinessState::NotReady;
    snapshot = snapshot.with_operation_family_status(saturated);
    assert_eq!(
        snapshot
            .operation_family_status(ComputeOperationFamily::Elementwise)
            .unwrap()
            .pressure,
        ProviderPressureLevel::Saturated
    );
}

#[test]
fn provider_status_maps_interruption_to_health_readiness_and_admission() {
    let snapshot = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        ProviderBinding::new("provider-a"),
        HealthState::Interrupted,
    ));

    assert_eq!(snapshot.lifecycle, ProviderLifecycleState::Failed);
    assert_eq!(snapshot.health, ProviderHealthState::Failed);
    assert_eq!(snapshot.readiness, ProviderReadinessState::NotReady);
    assert_eq!(snapshot.admission, ProviderAdmissionDecision::Reject);
    assert_eq!(snapshot.health_reason, ProviderStatusReason::Interrupted);
    assert!(matches!(
        snapshot.interruption,
        Some(ProviderInterruptionReason::DriverLoss)
    ));
}

#[test]
fn tensor_residency_affinity_rejects_forged_device_binding() {
    let claimed = TensorResidency::new(
        TensorResourceId::new("forged"),
        MemoryPlacement::Device(DeviceBinding::new(DeviceId::new("gpu-0"))),
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_device(DeviceBinding::new(DeviceId::new("gpu-0"))),
    );
    let actual = TensorResidency::new(
        TensorResourceId::new("forged"),
        MemoryPlacement::Device(DeviceBinding::new(DeviceId::new("gpu-1"))),
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_device(DeviceBinding::new(DeviceId::new("gpu-1"))),
    );
    let error = AffinityConstraints::try_from_affinities([&claimed.affinity, &actual.affinity])
        .unwrap_err();
    assert!(matches!(error, AffinityError::DeviceMismatch { .. }));
}
