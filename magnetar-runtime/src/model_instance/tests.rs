//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::adapter::*;
use crate::affinity::ProviderBinding;
use crate::affinity::*;
use crate::capability::CapabilityVersion;
use crate::compute::*;
use crate::e2e_conformance::*;
use crate::inference_api::*;
use crate::kernel::{KernelId, KernelImplementationFamily, KernelOperatorVersionRange};
use crate::kernel_performance_model::KernelPerformanceFeedbackMode;
use crate::kv_cache::*;
use crate::memory::*;
use crate::model::*;
use crate::operator::{OperatorFamily, OperatorId};
use crate::prefix_cache::*;
use crate::provider::*;
use crate::reference_cpu::*;
use crate::runtime::*;
use crate::session::*;
use crate::tensor::*;
use crate::tests::*;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;
#[test]
fn model_instance_readiness_fails_when_kernel_preparation_failed() {
    let mut checks = ModelInstanceReadinessChecks::default();
    assert_eq!(checks.readiness(), ModelInstanceReadiness::Ready);
    assert!(checks.validate().is_ok());

    checks.kernel_preparation_ready = false;
    assert_eq!(checks.readiness(), ModelInstanceReadiness::Failed);
    assert!(matches!(
        checks.validate(),
        Err(ModelInstanceError::ModelInstanceKernelPreparationFailed)
    ));
}

fn conformance_kernel_id(name: &str) -> KernelId {
    KernelId::new(
        ProviderBinding::new("kernel-artifact-test-provider"),
        name,
        CapabilityVersion::new(1, 0, 0),
        OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra),
        KernelOperatorVersionRange::exact(1),
        KernelImplementationFamily::TestFixture,
    )
}

#[test]
fn reproducible_kernel_selection_forces_pinned_performance_feedback() {
    // Implements "Reproducible Mode Prevents Adaptation" (proposal, define-
    // kernel-performance-model-and-adaptive-feedback-contract): a pinned
    // Kernel selection always yields Pinned feedback mode regardless of the
    // Model Instance's requested mode, and dynamic selection leaves the
    // requested mode untouched.
    let policy = ModelInstancePolicy {
        performance_feedback: KernelPerformanceFeedbackMode::Adaptive,
        ..ModelInstancePolicy::default()
    };

    assert_eq!(
        effective_performance_feedback_mode(&policy, &KernelSelectionPolicy::Dynamic),
        KernelPerformanceFeedbackMode::Adaptive
    );

    let pinned = KernelSelectionPolicy::Pinned(PinnedKernelSelection::new(
        conformance_kernel_id("attn"),
        "digest-a",
    ));
    assert_eq!(
        effective_performance_feedback_mode(&policy, &pinned),
        KernelPerformanceFeedbackMode::Pinned
    );
}

/// Implements `seal-model-instance-readiness-authority` round 3, the
/// audit's own resume scenario: state that made an instance eligible for
/// `Ready` can change while it is suspended, so `resume_model_instance`
/// must re-derive readiness against *current* Runtime state, not assume
/// whatever was true before suspension still holds. Proven here by
/// invalidating the weight evidence during suspension (removing its
/// `TensorResidency`, a legitimate way real state can regress -- e.g. a
/// rollback or eviction elsewhere) rather than by mutating a `Provider`'s
/// status snapshot, which this test double has no interior mutability to
/// do after registration; both exercise the same property: resume must
/// look at current state, not trust the state from before suspension.
#[test]
fn inference_api_resume_model_instance_revalidates_and_rejects_stale_evidence() {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .build()
        .unwrap();
    let mut definition = model_instance_definition();
    definition.placement = ModelInstancePlacement::new(
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
    );
    let instance = runtime.model_instances_mut().create(definition).unwrap();
    reach_ready_with_real_weight(&mut runtime, &instance);
    assert_eq!(
        model_instance_status(&runtime, &instance)
            .unwrap()
            .lifecycle,
        ModelInstanceLifecycleState::Ready
    );

    suspend_model_instance(
        &mut runtime,
        &instance,
        ModelInstanceSuspensionReason::AdministrativePolicy,
    )
    .unwrap();

    // While suspended, the weight evidence that made this instance Ready
    // becomes invalid (its TensorResidency is removed) -- simulating real
    // state regressing during the suspension window, exactly the scenario
    // the audit's own resume test targets.
    let weight_resource = runtime
        .model_instance(&instance)
        .unwrap()
        .definition
        .resource_bindings
        .weights
        .get("weight")
        .unwrap()
        .clone();
    runtime
        .memory_mut()
        .remove_tensor_residency(&weight_resource);

    resume_model_instance(&mut runtime, &instance).unwrap_err();
    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_ne!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(!status.readiness.accepts_generation());
}

/// Implements `seal-model-instance-readiness-authority` round 3, audit
/// test 25 "Inventaire incomplet": a manifest declaring multiple mandatory
/// tensors, with only some of them bound, must not reach Ready even
/// though every bound entry is individually real (residency-backed,
/// Provider-written, and -- since `bind-model-loading-evidence-to-
/// validated-artifact` -- Runtime-evidenced). Round 2's fix only checked
/// that whatever *was* bound was real; it never checked the bound set was
/// *complete*. Materializes "weight.a" for real through the one authorized
/// transaction (which has no knowledge of `required_weight_names` --
/// that's `derive_effective_readiness_checks`'s own concern -- so it marks
/// the instance Ready on its own terms); a subsequent `warm_model_instance`
/// re-derivation must then find the mandatory inventory still incomplete
/// and demote it, the same defense-in-depth property the resume-
/// revalidation test above proves for stale residency.
#[test]
fn inference_api_warm_model_instance_rejects_incomplete_mandatory_weight_inventory() {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .build()
        .unwrap();
    let mut definition = model_instance_definition();
    definition.placement = ModelInstancePlacement::new(
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
    );
    definition.required_weight_names =
        BTreeSet::from(["weight.a".to_string(), "weight.b".to_string()]);
    let instance = runtime.model_instances_mut().create(definition).unwrap();

    // Really materialize only "weight.a" -- "weight.b" stays entirely
    // absent.
    let weights = BTreeMap::from([("weight.a".to_string(), HostTensor::new([1], [0.0]).unwrap())]);
    materialize_model_instance_weights(&mut runtime, &instance, "test", &weights).unwrap();

    let plan = ModelInstanceWarmupPlan {
        policy: ModelInstanceWarmupPolicy::ValidateMetadataOnly,
        steps: Vec::new(),
    };
    warm_model_instance(
        &mut runtime,
        &instance,
        &plan,
        &ModelInstanceReadinessChecks::default(),
    )
    .unwrap_err();

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_ne!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(!status.readiness.accepts_generation());
}

/// Implements `runtime-owned-model-instance-readiness-authority`. An
/// external audit of PR #36 found `warm_model_instance` trusted a caller's
/// `weights_materialized: true` claim outright, even though
/// `ModelInstanceReadinessChecks::default()` sets it `true` -- a caller
/// using the default checks against an instance whose weights were never
/// materialized could reach `Ready`. Covers the audit's test 18.1 (warmup
/// without materialization must not reach Ready) and 18.3 (a forged
/// `weights_materialized=true` claim against empty bindings is rejected).
#[test]
fn inference_api_warm_model_instance_rejects_forged_weights_materialized_claim() {
    let mut runtime = Runtime::builder().build().unwrap();
    let instance = runtime
        .model_instances_mut()
        .create(model_instance_definition())
        .unwrap();
    assert!(
        runtime
            .model_instance(&instance)
            .unwrap()
            .definition
            .resource_bindings
            .weights
            .is_empty(),
        "test precondition: no weights ever bound"
    );

    let plan = ModelInstanceWarmupPlan {
        policy: ModelInstanceWarmupPolicy::ValidateMetadataOnly,
        steps: Vec::new(),
    };
    // The caller asserts every fact is satisfied, exactly matching
    // `ModelInstanceReadinessChecks::default()` -- the audit's own
    // exploit scenario (section 11).
    let forged_checks = ModelInstanceReadinessChecks::default();
    assert!(forged_checks.weights_materialized);

    warm_model_instance(&mut runtime, &instance, &plan, &forged_checks).unwrap_err();

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_ne!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(!status.readiness.accepts_generation());
}

/// Implements `runtime-owned-model-instance-readiness-authority`, test
/// 18.2: `WarmupPolicy::Disabled` calls `ModelInstance::validate_readiness`
/// directly, without the lifecycle transition `warmup()`'s other policies
/// perform first -- so before this fix, a caller-forged `readiness =
/// Ready` could be published while `lifecycle` stayed `Loading`, an
/// internally inconsistent state. This test isolates that specific gap
/// from `weights_materialized` derivation (covered separately above) by
/// directly binding a weight resource, so the *only* thing preventing
/// `Ready` here is the lifecycle/readiness consistency check.
#[test]
fn inference_api_warm_model_instance_disabled_policy_cannot_forge_ready_readiness() {
    let mut runtime = Runtime::builder().build().unwrap();
    let instance = runtime
        .model_instances_mut()
        .create(model_instance_definition())
        .unwrap();
    runtime
        .model_instances_mut()
        .instance_mut(&instance)
        .unwrap()
        .definition
        .resource_bindings
        .weights
        .insert("weight".into(), TensorResourceId::new("test.weight"));
    assert_eq!(
        model_instance_status(&runtime, &instance)
            .unwrap()
            .lifecycle,
        ModelInstanceLifecycleState::Loading,
        "test precondition: still Loading, no transition has run"
    );

    let plan = ModelInstanceWarmupPlan {
        policy: ModelInstanceWarmupPolicy::Disabled,
        steps: Vec::new(),
    };
    let checks = ModelInstanceReadinessChecks::default();

    // Disabled policy's own contract (`ModelInstance::warmup`) means this
    // call does not have to fail -- `validate_readiness`'s own `Result`
    // reflects whether the checks *themselves* are internally coherent,
    // which they are. What matters is `self.readiness` afterward.
    let _ = warm_model_instance(&mut runtime, &instance, &plan, &checks);

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_eq!(
        status.lifecycle,
        ModelInstanceLifecycleState::Loading,
        "Disabled policy must not transition the lifecycle"
    );
    assert_ne!(
        status.readiness,
        ModelInstanceReadiness::Ready,
        "readiness must not report Ready while lifecycle is still Loading"
    );
}

/// Implements `runtime-owned-model-instance-readiness-authority`, test
/// 18.4: an internally inconsistent `lifecycle: Loading, readiness: Ready`
/// state -- however it might arise -- must never grant `acquire_usage` or
/// `generation_reference`. This is the structural safety net: it holds
/// regardless of which caller-forgeable path produced the inconsistency.
#[test]
fn model_instance_acquire_usage_rejects_ready_readiness_with_incompatible_lifecycle() {
    let mut manager = ModelInstanceManager::new();
    let id = manager.create(model_instance_definition()).unwrap();
    assert_eq!(
        manager.instance(&id).unwrap().lifecycle,
        ModelInstanceLifecycleState::Loading
    );

    // Force the inconsistent state directly (bypassing every public
    // entry point) to prove the check in `acquire_usage`/
    // `generation_reference` itself, independent of how the
    // inconsistency might be produced.
    manager.instance_mut(&id).unwrap().readiness = ModelInstanceReadiness::Ready;
    assert_eq!(
        manager.instance(&id).unwrap().lifecycle,
        ModelInstanceLifecycleState::Loading
    );
    assert_eq!(
        manager.instance(&id).unwrap().readiness,
        ModelInstanceReadiness::Ready
    );

    assert!(matches!(
        manager.instance_mut(&id).unwrap().acquire_usage(0),
        Err(ModelInstanceError::ModelInstanceLoading)
    ));
    assert!(matches!(
        manager.generation_reference(&id),
        Err(ModelInstanceError::ModelInstanceLoading)
    ));
}

/// Implements `runtime-owned-model-instance-readiness-authority`, test
/// 18.5: the happy path -- real weight materialization, a Provider the
/// Runtime can actually resolve, and no forged claims -- still reaches
/// Ready and accepts usage. Closing the forgery gap must not regress the
/// legitimate warmup path.
#[test]
fn inference_api_warm_model_instance_reaches_ready_when_weights_and_provider_are_real() {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .build()
        .unwrap();
    let mut definition = model_instance_definition();
    definition.placement = ModelInstancePlacement::new(
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
    );
    let instance = runtime.model_instances_mut().create(definition).unwrap();
    // Real, Provider-backed evidence, not just a residency record or a
    // bare map entry: `weights_materialized` derivation resolves each
    // residency's recorded Provider and reads the tensor back from it, so
    // a residency whose Provider never actually received a `write_tensor`
    // call does not count (Correctif: Runtime-owned ModelInstance
    // readiness authority, round 3 -- closing the "synthetic residency"
    // forgery a further audit of PR #36 demonstrated: a caller could
    // previously construct a valid-looking `TensorResidency` for a
    // resource no Provider ever wrote).
    reach_ready_with_real_weight(&mut runtime, &instance);

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_eq!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(status.readiness.accepts_generation());
    assert!(
        runtime
            .model_instances_mut()
            .acquire_usage(&instance, 0)
            .is_ok()
    );
}

/// `bind-model-loading-evidence-to-validated-artifact`: materialization
/// evidence is looked up by the instance's own id, so Model Instance B
/// cannot become materialized by having its weight bindings set to match
/// Model Instance A's already-legitimately-materialized bindings -- B's
/// lookup only ever finds evidence `commit` minted for B's own id (none),
/// regardless of what A's evidence says.
#[test]
fn inference_api_warm_model_instance_rejects_bindings_copied_from_another_instance() {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .build()
        .unwrap();
    let mut definition_a = model_instance_definition();
    definition_a.placement = ModelInstancePlacement::new(
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
    );
    let instance_a = runtime.model_instances_mut().create(definition_a).unwrap();
    reach_ready_with_real_weight(&mut runtime, &instance_a);

    let mut definition_b = model_instance_definition();
    definition_b.placement = ModelInstancePlacement::new(
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
    );
    let instance_b = runtime.model_instances_mut().create(definition_b).unwrap();

    // Copy A's already-legitimate bindings directly onto B -- the exact
    // "reuse another instance's materialization" scenario the audit's own
    // "Artifact mismatch" test asks for, here proven for same-artifact
    // reuse across two distinct instances (the stronger, more general
    // case: even identical bindings for the identical artifact do not
    // transfer, because evidence is instance-scoped, not artifact-scoped).
    let copied_weights = runtime
        .model_instance(&instance_a)
        .unwrap()
        .definition
        .resource_bindings
        .weights
        .clone();
    runtime
        .model_instances_mut()
        .instance_mut(&instance_b)
        .unwrap()
        .definition
        .resource_bindings
        .weights = copied_weights;

    let plan = ModelInstanceWarmupPlan {
        policy: ModelInstanceWarmupPolicy::ValidateMetadataOnly,
        steps: Vec::new(),
    };
    warm_model_instance(
        &mut runtime,
        &instance_b,
        &plan,
        &ModelInstanceReadinessChecks::default(),
    )
    .unwrap_err();

    let status_b = model_instance_status(&runtime, &instance_b).unwrap();
    assert_ne!(status_b.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(!status_b.readiness.accepts_generation());
    // A remains legitimately Ready throughout -- this is not a mutation of
    // A's own state, only B reading A's (still-valid) bindings.
    assert_eq!(
        model_instance_status(&runtime, &instance_a)
            .unwrap()
            .lifecycle,
        ModelInstanceLifecycleState::Ready
    );
}

/// `bind-model-loading-evidence-to-validated-artifact`: materialization
/// evidence records the `ModelArtifactId` that was current when it was
/// minted; if a Model Instance's declared artifact later differs from what
/// its evidence recorded, the evidence no longer matches and readiness
/// must reject it. `artifact` has no dedicated setter, but the field
/// remains `pub` on `ModelInstanceDefinition` -- unlike `resource_bindings`,
/// reassigning it can only ever make otherwise-valid evidence stop
/// matching (fail closed), never let a caller borrow a *different*
/// instance's evidence, since evidence lookup is keyed by instance id, not
/// artifact id (see the previous test). Proven directly here rather than
/// left as a hypothetical.
#[test]
fn inference_api_warm_model_instance_rejects_evidence_after_artifact_reassignment() {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .build()
        .unwrap();
    let mut definition = model_instance_definition();
    definition.placement = ModelInstancePlacement::new(
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
    );
    let instance = runtime.model_instances_mut().create(definition).unwrap();
    reach_ready_with_real_weight(&mut runtime, &instance);
    assert_eq!(
        model_instance_status(&runtime, &instance)
            .unwrap()
            .lifecycle,
        ModelInstanceLifecycleState::Ready
    );

    runtime
        .model_instances_mut()
        .instance_mut(&instance)
        .unwrap()
        .definition
        .artifact = ModelArtifactId::new(
        ModelArtifactKind::ModelWeights,
        ModelName::new("qwen").unwrap(),
        ModelRevision::new("2-different".to_string()).unwrap(),
        ModelDigest::sha256(b"different weights"),
    );

    let plan = ModelInstanceWarmupPlan {
        policy: ModelInstanceWarmupPolicy::ValidateMetadataOnly,
        steps: Vec::new(),
    };
    warm_model_instance(
        &mut runtime,
        &instance,
        &plan,
        &ModelInstanceReadinessChecks::default(),
    )
    .unwrap_err();

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_ne!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(!status.readiness.accepts_generation());
}

/// A further audit of PR #36 found that `WeightMaterializationTransaction::
/// commit` called `ModelInstanceManager::mark_ready` unconditionally right
/// after staging succeeded and evidence was minted -- so
/// `materialize_model_instance_weights(..., &BTreeMap::new())` against an
/// instance with a non-empty mandatory inventory could still reach
/// `Ready`, bypassing the exact same Runtime-derived readiness gate
/// `warm_model_instance`/`resume_model_instance` already correctly use.
/// Non-compliant with `model-loading`'s pre-existing "Model Loading Does
/// Not Bypass Instance Readiness" ("Successful materialization alone
/// SHALL not imply Model Instance readiness") and "Partial Loading
/// Policy" requirements. Proven here at the exact public entrypoint the
/// bug was in, not only through `warm_model_instance` (which was already
/// correct on its own and could not, by itself, prove this bypass was
/// closed).
#[test]
fn inference_api_materialize_model_instance_weights_does_not_ready_with_empty_map() {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .build()
        .unwrap();
    let mut definition = model_instance_definition();
    definition.placement = ModelInstancePlacement::new(
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
    );
    definition.required_weight_names =
        BTreeSet::from(["weight.a".to_string(), "weight.b".to_string()]);
    let instance = runtime.model_instances_mut().create(definition).unwrap();

    materialize_model_instance_weights(&mut runtime, &instance, "test", &BTreeMap::new()).unwrap();

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_ne!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(!status.readiness.accepts_generation());
}

/// Same root gap as the empty-map test above, for a strict subset of a
/// multi-tensor mandatory inventory -- audit scenario "subset partiel":
/// `commit` minted an evidence record that was *exactly correct* for what
/// it staged, but exact correctness for an incomplete set is still
/// incomplete, and `commit` never checked completeness before marking
/// Ready.
#[test]
fn inference_api_materialize_model_instance_weights_does_not_ready_with_partial_inventory() {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .build()
        .unwrap();
    let mut definition = model_instance_definition();
    definition.placement = ModelInstancePlacement::new(
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)),
    );
    definition.required_weight_names = BTreeSet::from([
        "weight.a".to_string(),
        "weight.b".to_string(),
        "weight.c".to_string(),
    ]);
    let instance = runtime.model_instances_mut().create(definition).unwrap();

    let weights = BTreeMap::from([
        ("weight.a".to_string(), HostTensor::new([1], [0.0]).unwrap()),
        ("weight.b".to_string(), HostTensor::new([1], [0.0]).unwrap()),
    ]);
    materialize_model_instance_weights(&mut runtime, &instance, "test", &weights).unwrap();

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_ne!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(!status.readiness.accepts_generation());

    // The partial materialization itself still succeeded for real: its
    // evidence exists and matches exactly what was staged, proving
    // `commit`'s gate rejects on inventory completeness specifically, not
    // by failing to publish anything at all (which would also incidentally
    // "fix" the symptom without fixing the actual defect).
    assert_eq!(
        runtime
            .model_instance(&instance)
            .unwrap()
            .definition
            .resource_bindings
            .weights
            .len(),
        2
    );

    // Completing the inventory in a second, independent call reaches
    // Ready -- proving incremental/progressive materialization (which
    // `commit`'s own evidence-recomputation is designed to support) still
    // works once the gate this fix adds is actually satisfied.
    let remaining =
        BTreeMap::from([("weight.c".to_string(), HostTensor::new([1], [0.0]).unwrap())]);
    materialize_model_instance_weights(&mut runtime, &instance, "test", &remaining).unwrap();
    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_eq!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(status.readiness.accepts_generation());
}

/// Implements `runtime-owned-model-instance-readiness-authority` round 2,
/// audit test 23.3: a `TensorResourceId` inserted directly into
/// `resource_bindings.weights`, with no matching `TensorResidency`, must
/// not let `warm_model_instance` derive `weights_materialized = true` --
/// the exact gap a follow-up audit of PR #36 found: the round-1 fix only
/// checked the map was non-empty, not that its entries were real.
#[test]
fn inference_api_warm_model_instance_rejects_weight_binding_without_residency() {
    let mut runtime = Runtime::builder().build().unwrap();
    let instance = runtime
        .model_instances_mut()
        .create(model_instance_definition())
        .unwrap();
    // Insert a plausible-looking `TensorResourceId` directly -- no
    // `MemoryManager::allocate`, no `record_tensor_residency`. This is
    // exactly what a caller with `&mut Runtime` could always do, and what
    // this Change's derivation must not trust.
    runtime
        .model_instances_mut()
        .instance_mut(&instance)
        .unwrap()
        .definition
        .resource_bindings
        .weights
        .insert("weight".into(), TensorResourceId::new("forged.weight"));
    assert!(
        runtime
            .memory()
            .tensor_residency(&TensorResourceId::new("forged.weight"))
            .is_none(),
        "test precondition: no residency was ever recorded for this resource"
    );

    let plan = ModelInstanceWarmupPlan {
        policy: ModelInstanceWarmupPolicy::ValidateMetadataOnly,
        steps: Vec::new(),
    };
    warm_model_instance(
        &mut runtime,
        &instance,
        &plan,
        &ModelInstanceReadinessChecks::default(),
    )
    .unwrap_err();

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_ne!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(!status.readiness.accepts_generation());
}

/// Implements `runtime-owned-model-instance-readiness-authority` round 2,
/// audit test 23.5: a Provider that is registered and offers a real
/// `execution_api()` but whose own status model reports it as not
/// accepting new work (here: `Saturated` pressure) must make
/// `provider_ready` derive `false`, even though the caller claims `true`
/// -- the round-1 fix only checked the Provider "exists and is executable
/// in principle" (`execution_api().is_some()`), not "is ready now".
#[test]
fn inference_api_warm_model_instance_rejects_provider_that_rejects_new_work() {
    let mut provider = TestProvider::new("saturated-provider");
    provider.execution_api = Some(Arc::new(TestProviderExecutionApi::new()));
    let mut snapshot = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        ProviderBinding::new("saturated-provider"),
        HealthState::Available,
    ));
    snapshot.pressure = ProviderPressureLevel::Saturated;
    snapshot.admission = provider_admission_from_dimensions(
        snapshot.lifecycle,
        snapshot.health,
        snapshot.readiness,
        snapshot.pressure,
    );
    provider.status_snapshot = Some(snapshot);
    assert!(
        !provider.status_snapshot().accepts_new_work_by_default(),
        "test precondition: this Provider's own status model rejects new work"
    );

    let mut runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let mut definition = model_instance_definition();
    definition.placement = ModelInstancePlacement::new(
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_provider(ProviderBinding::new("saturated-provider")),
    );
    let instance = runtime.model_instances_mut().create(definition).unwrap();
    let allocation = runtime
        .memory_mut()
        .allocate(MemoryAllocationRequest::new(
            MemoryAllocationClass::Tensor,
            1,
            MemoryPlacement::HostOrdinary,
            MemoryAllocationOwner::InferenceArtifact("test".into()),
        ))
        .unwrap();
    let weight_resource = TensorResourceId::new("test.weight");
    runtime
        .memory_mut()
        .record_tensor_residency(
            TensorResidency::new(
                weight_resource.clone(),
                MemoryPlacement::HostOrdinary,
                ResourceAffinity::new(FallbackClass::Transparent),
            )
            .with_allocation(allocation.id),
        )
        .unwrap();
    runtime
        .model_instances_mut()
        .instance_mut(&instance)
        .unwrap()
        .definition
        .resource_bindings
        .weights
        .insert("weight".into(), weight_resource);

    let plan = ModelInstanceWarmupPlan {
        policy: ModelInstanceWarmupPolicy::ValidateMetadataOnly,
        steps: Vec::new(),
    };
    // Real weight evidence is present; only `provider_ready` should be
    // what fails this attempt.
    warm_model_instance(
        &mut runtime,
        &instance,
        &plan,
        &ModelInstanceReadinessChecks::default(),
    )
    .unwrap_err();

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_ne!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(!status.readiness.accepts_generation());
}

/// A further audit of PR #36 found that `ModelInstanceDefinition`'s
/// `#[derive(Clone)]` copies every field -- including the already-sealed
/// `resource_bindings` -- regardless of the caller's own inability to name
/// those fields directly, and that `ModelInstanceManager::create` accepted
/// a caller-supplied definition as-is. So an external caller with only
/// `pub` access (`ModelInstance::definition()`, `Clone`,
/// `ModelInstanceManager::create`) could clone a `Ready` instance's
/// definition -- carrying its real weight bindings -- into a brand-new
/// instance, aliasing live Provider resources across two distinct
/// `ModelInstanceId`s. Fixed by having `create` unconditionally reset
/// `resource_bindings` regardless of what the supplied definition
/// contained.
#[test]
fn cloned_definition_does_not_inherit_weight_authority() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let id_a = runtime
        .model_instances_mut()
        .create(sealed_creation_definition())
        .unwrap();
    sealed_creation_reach_ready(&mut runtime, &id_a);
    assert_eq!(
        runtime.model_instance(&id_a).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );

    let cloned = runtime.model_instance(&id_a).unwrap().definition().clone();
    let id_b = runtime.model_instances_mut().create(cloned).unwrap();

    // B must not be Ready on arrival -- creation alone never implies
    // readiness, cloned definition or not.
    assert_ne!(
        runtime.model_instance(&id_b).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );

    // The real regression: an *empty* materialization attempt on B must
    // not be able to "adopt" A's already-committed bindings by minting
    // fresh evidence over whatever `resource_bindings.weights` happens to
    // contain. If `create` had not reset it, this would have reached
    // `Ready` for B using A's real Provider resource, with zero bytes
    // actually staged by this call.
    materialize_model_instance_weights(&mut runtime, &id_b, "test", &BTreeMap::new()).unwrap();
    assert_ne!(
        runtime.model_instance(&id_b).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );

    // A is unaffected throughout -- this is not a mutation of A's own
    // state, only B's (failed) attempt to inherit it.
    assert_eq!(
        runtime.model_instance(&id_a).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );
}

/// Same root cause as the test above, exercised through `reload` (which
/// also calls `ModelInstanceManager::create` internally with a
/// caller-supplied replacement definition) rather than a direct `create`
/// call.
#[test]
fn reload_replacement_does_not_inherit_original_weight_authority() {
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let id = runtime
        .model_instances_mut()
        .create(sealed_creation_definition())
        .unwrap();
    sealed_creation_reach_ready(&mut runtime, &id);
    let existing_definition = runtime.model_instance(&id).unwrap().definition().clone();

    let replacement_id = runtime
        .model_instances_mut()
        .reload(
            &id,
            ModelInstanceReloadRequest {
                replacement: existing_definition,
                migrate_sessions: false,
                allow_active_semantic_mutation: false,
            },
        )
        .unwrap();

    assert_ne!(
        runtime.model_instance(&replacement_id).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );
}

/// Proves `ModelInstanceManager::create_checked`'s own checks-validation
/// contract directly: `checks.validate()` must run and reject before
/// `create` is ever reached, and readiness checks are evaluated
/// independently afterward. `Runtime::create_model_instance` has no
/// parameter through which a caller can inject custom
/// `ModelInstanceCreationChecks`, so this cannot be re-expressed through
/// it.
#[test]
fn sealed_creation_and_readiness_checks_gate_ready_state() {
    let mut manager = ModelInstanceManager::new();
    let denied = ModelInstanceCreationChecks {
        artifact_trusted: false,
        ..ModelInstanceCreationChecks::default()
    };
    assert_eq!(
        manager.create_checked(sealed_creation_definition(), &denied),
        Err(ModelInstanceError::ModelInstancePolicyDenied)
    );

    let id = manager
        .create_checked(
            sealed_creation_definition(),
            &ModelInstanceCreationChecks::default(),
        )
        .unwrap();
    let checks = ModelInstanceReadinessChecks {
        provider_ready: false,
        ..ModelInstanceReadinessChecks::default()
    };
    assert_eq!(
        manager
            .instance_mut(&id)
            .unwrap()
            .validate_readiness(&checks),
        Err(ModelInstanceError::ModelInstanceProviderNotReady)
    );
}

/// Proves pre-create usage-dependency fields (`kv_cache_dependencies`,
/// `prefix_cache_dependencies`) set directly on a hand-built definition
/// are honored by `create`. `Runtime::create_model_instance` builds its
/// definition internally from `architecture`/`affinity` alone, with no
/// way for a caller to inject these before creation.
#[test]
fn sealed_creation_adapter_activation_records_mutation_and_invalidates_dependent_caches() {
    let mut manager = ModelInstanceManager::new();
    let mut def = sealed_creation_definition();
    def.usage
        .kv_cache_dependencies
        .insert(KvCacheId::new("cache-a").unwrap());
    def.usage
        .prefix_cache_dependencies
        .insert(PrefixCacheEntryId::new("prefix-a").unwrap());
    let id = manager.create(def).unwrap();

    let report = manager
        .activate_adapters(&id, AdapterSetId::empty(), "session:opaque", true)
        .unwrap();

    assert_eq!(report.kv_caches.len(), 1);
    assert_eq!(report.prefix_entries.len(), 1);
    assert_eq!(
        manager.instance(&id).unwrap().definition().mutation_version,
        1
    );
    assert!(
        manager
            .observations()
            .iter()
            .any(|event| event.kind == ModelInstanceObservationKind::CacheInvalidation)
    );
}

/// Proves pre-create session/usage/adapter dependency fields are honored
/// by `create` and correctly released on unload, and that `create`
/// unconditionally resets `resource_bindings` regardless of what the
/// caller-supplied `def` pre-populated -- so an out-of-band memory
/// allocation id set here cannot be pre-populated on `def` before
/// `create`, only injected afterward via the two narrow, post-creation
/// mutations `ModelInstance` still allows
/// (`set_provider_resource`/`track_memory_allocation`).
#[test]
fn sealed_creation_unload_releases_memory_provider_resources_adapters_and_cache_dependencies() {
    let mut def = sealed_creation_definition();
    def.associated_sessions
        .insert(InferenceSessionId::new("session-unload").unwrap());
    def.usage
        .kv_cache_dependencies
        .insert(KvCacheId::new("cache-unload").unwrap());
    def.usage
        .prefix_cache_dependencies
        .insert(PrefixCacheEntryId::new("prefix-unload").unwrap());
    def.usage.adapter_dependencies.insert(AdapterSetId::empty());
    let mut runtime = Runtime::initialize(RuntimeConfig::default());
    let id = runtime.model_instances_mut().create(def).unwrap();
    sealed_creation_reach_ready(&mut runtime, &id);
    {
        let instance = runtime.model_instances_mut().instance_mut(&id).unwrap();
        instance.set_provider_resource(Some(ProviderModelResource {
            provider: ProviderBinding::new("provider-a"),
            handle_kind: "opaque-model".into(),
            release_required: true,
        }));
        // A deliberately out-of-band id: `reach_ready` above issues its own
        // real allocation from the same `MemoryManager`, whose ids start
        // at 1, so a fixture id of `1` here would silently collide in the
        // `BTreeSet` and undercount `released_memory_allocations`.
        instance.track_memory_allocation(MemoryAllocationId::new(999));
    }

    let report = runtime
        .model_instances_mut()
        .unload(&id, ModelInstanceUnloadPolicy::DrainActiveUse)
        .unwrap();

    assert_eq!(report.invalidated.kv_caches.len(), 1);
    assert_eq!(report.invalidated.prefix_entries.len(), 1);
    assert_eq!(report.invalidated.adapters_released.len(), 1);
    // +1 relative to the fixture's single pre-set `memory_allocations`
    // entry: `reach_ready`'s own Memory Manager allocation for the bound
    // weight is released on unload too.
    assert_eq!(report.released_memory_allocations.len(), 2);
    assert_eq!(report.released_provider_resources.len(), 1);
    assert!(!report.dangling_session_references);
    assert_eq!(
        runtime.model_instance(&id).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Unloaded
    );
}

/// Proves pre-create `policy.browser_linear_memory_limit_bytes` is
/// enforced by `create` itself. `Runtime::create_model_instance` builds
/// `policy: ModelInstancePolicy::default()` internally with no way for a
/// caller to override it before creation.
#[test]
fn sealed_creation_browser_policy_rejects_native_or_oversized_instance_features() {
    let mut def = sealed_creation_definition();
    def.policy = ModelInstancePolicy {
        browser_linear_memory_limit_bytes: Some(1),
        ..ModelInstancePolicy::default()
    };
    let mut manager = ModelInstanceManager::new();
    assert_eq!(
        manager.create(def),
        Err(ModelInstanceError::ModelInstanceBrowserFeatureUnsupported)
    );
}
