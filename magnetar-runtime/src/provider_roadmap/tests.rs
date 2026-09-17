//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::affinity::{FallbackClass, ResourceAffinity};
use crate::conformance::ProviderConformanceProfile;
use crate::inference_api::validate_inference_scope;
use crate::reference_cpu::FallbackPolicyContext;
use std::collections::BTreeSet;

#[test]
fn provider_roadmap_features_are_all_optional_and_phase_tagged() {
    assert_eq!(PROVIDER_ROADMAP_FEATURES.len(), 31);
    for feature in PROVIDER_ROADMAP_FEATURES {
        assert!(feature.is_optional(), "{feature:?} must be optional");
    }
    assert_eq!(
        provider_roadmap_features_for_phase(ProviderRoadmapPhase::OptimizedCpu),
        vec![
            ProviderRoadmapFeature::Simd,
            ProviderRoadmapFeature::Blas,
            ProviderRoadmapFeature::ThreadPoolExecution,
            ProviderRoadmapFeature::CacheAwareKernels,
            ProviderRoadmapFeature::OptimizedCpuFusedKernels,
        ]
    );
    assert_eq!(
        provider_roadmap_features_for_phase(ProviderRoadmapPhase::Cuda).len(),
        8
    );
    assert_eq!(
        provider_roadmap_features_for_phase(ProviderRoadmapPhase::Metal).len(),
        5
    );
    assert_eq!(
        provider_roadmap_features_for_phase(ProviderRoadmapPhase::OpenVino).len(),
        4
    );
    assert_eq!(
        provider_roadmap_features_for_phase(ProviderRoadmapPhase::Qnn).len(),
        4
    );
    assert_eq!(
        provider_roadmap_features_for_phase(ProviderRoadmapPhase::WebGpu).len(),
        5
    );
    for feature in PROVIDER_ROADMAP_FEATURES {
        assert!(!feature.id().is_empty());
        assert!(
            provider_roadmap_features_for_phase(feature.phase()).contains(feature),
            "{feature:?} must appear under its own phase()"
        );
    }
}

#[test]
fn provider_roadmap_phase_readiness_requires_actually_passed_profiles() {
    let cuda = ProviderRoadmapPhase::Cuda;
    let required = cuda.required_conformance_gates();
    assert!(!phase_is_production_ready(cuda, &BTreeSet::new()));
    assert!(!phase_is_production_ready(
        cuda,
        &BTreeSet::from([ProviderConformanceProfile::ProviderCore])
    ));
    assert!(phase_is_production_ready(cuda, &required));
}

#[test]
fn provider_roadmap_reference_cpu_remains_correctness_baseline() {
    // Every post-baseline hardware family's primary fallback edge terminates
    // at Reference CPU (or an explicit browser-CPU-like path for WebGPU),
    // never at another hardware Provider -- Reference CPU stays the
    // correctness baseline the roadmap compares optimized output against.
    assert_eq!(
        ProviderRoadmapHardwareFamily::Cuda.primary_fallback_edge(),
        ProviderRoadmapFallbackEdge::CudaToReferenceCpu
    );
    assert_eq!(
        ProviderRoadmapHardwareFamily::Metal.primary_fallback_edge(),
        ProviderRoadmapFallbackEdge::MetalToReferenceCpu
    );
    assert_eq!(
        ProviderRoadmapHardwareFamily::OpenVino.primary_fallback_edge(),
        ProviderRoadmapFallbackEdge::OpenVinoToReferenceCpu
    );
    assert_eq!(
        ProviderRoadmapHardwareFamily::Qnn.primary_fallback_edge(),
        ProviderRoadmapFallbackEdge::QnnToReferenceCpu
    );
    assert_eq!(
        ProviderRoadmapHardwareFamily::WebGpu.primary_fallback_edge(),
        ProviderRoadmapFallbackEdge::WebGpuToBrowserCpuLike
    );
}

#[test]
fn provider_roadmap_denies_native_handle_exposure_for_every_hardware_family() {
    for family in [
        ProviderRoadmapHardwareFamily::Cuda,
        ProviderRoadmapHardwareFamily::Metal,
        ProviderRoadmapHardwareFamily::OpenVino,
        ProviderRoadmapHardwareFamily::Qnn,
    ] {
        assert!(
            !family.native_handle_kinds().is_empty(),
            "{family:?} should declare at least one native handle kind"
        );
        for handle_kind in family.native_handle_kinds() {
            let outcome = reject_native_handle_exposure(family, handle_kind);
            assert!(matches!(
                outcome,
                Err(ProviderRoadmapError::ProviderNativeHandleExposureDenied { .. })
            ));
        }
    }
    // WebGPU has no native-handle boundary of its own (browser sandboxing
    // constraints apply instead), but it still must not require native
    // Provider loading.
    assert!(
        ProviderRoadmapHardwareFamily::WebGpu
            .native_handle_kinds()
            .is_empty()
    );
    assert!(ProviderRoadmapHardwareFamily::WebGpu.requires_no_native_provider_loading());
    assert!(!ProviderRoadmapHardwareFamily::Cuda.requires_no_native_provider_loading());
}

#[test]
fn provider_roadmap_rejects_hidden_dequantization() {
    assert!(matches!(
        reject_hidden_dequantization(false),
        Err(ProviderRoadmapError::ProviderQuantizationUnsupported { .. })
    ));
    assert!(reject_hidden_dequantization(true).is_ok());
}

#[test]
fn provider_roadmap_advanced_attention_unsupported_path_fails_explicitly() {
    let outcome = reject_unsupported_advanced_attention(AdvancedAttentionVariant::FlashAttention);
    assert!(matches!(
        outcome,
        ProviderRoadmapError::ProviderAdvancedAttentionUnsupported { .. }
    ));
}

#[test]
fn provider_roadmap_fallback_observed_emits_considered_then_used_or_denied() {
    let affinity = ResourceAffinity::new(FallbackClass::Transparent);
    let denied_context = ProviderRoadmapFallbackContext::deny_by_default();
    let (observations, outcome) = evaluate_provider_roadmap_fallback_observed(
        ProviderRoadmapFallbackEdge::WebGpuToBrowserCpuLike,
        &affinity,
        &denied_context,
    );
    assert!(outcome.is_err());
    assert_eq!(observations.len(), 2);
    assert_eq!(
        observations[0].kind,
        ProviderRoadmapObservationKind::FallbackConsidered
    );
    assert_eq!(
        observations[1].kind,
        ProviderRoadmapObservationKind::FallbackDenied
    );

    let allowed_context = ProviderRoadmapFallbackContext {
        cpu: FallbackPolicyContext::new(true),
        memory_policy_allows_fallback: true,
        privacy_policy_allows_fallback: true,
        precision_policy_allows_fallback: true,
    };
    let (observations, outcome) = evaluate_provider_roadmap_fallback_observed(
        ProviderRoadmapFallbackEdge::WebGpuToBrowserCpuLike,
        &affinity,
        &allowed_context,
    );
    assert!(outcome.is_ok());
    assert_eq!(
        observations[1].kind,
        ProviderRoadmapObservationKind::FallbackUsed
    );
}

#[test]
fn provider_roadmap_runtime_api_remains_provider_independent() {
    for capability in PROVIDER_ROADMAP_FORBIDDEN_API_HANDLE_SCOPES {
        assert!(
            reject_provider_specific_handle_capability(capability).is_err(),
            "{capability} should have been rejected"
        );
    }
    // Ordinary inference scopes remain unaffected.
    assert!(reject_provider_specific_handle_capability("generation").is_ok());
    assert!(validate_inference_scope("generation").is_ok());
}

#[test]
fn provider_roadmap_cli_may_pass_policy_preference_without_authority() {
    let preference = ProviderRoadmapPolicyPreference {
        preferred_provider: Some("cuda".into()),
        allow_optimized_provider_fallback: true,
    };
    let echoed = cli_may_pass_policy_preference(&preference);
    assert_eq!(echoed, preference);
    assert!(reject_cli_raw_provider_handle_selection("cuda-device-pointer").is_err());
}

#[test]
fn provider_roadmap_conformance_profiles_are_declared_without_implying_readiness() {
    let ids = provider_roadmap_conformance_profile_ids();
    assert_eq!(ids.len(), 9);
    assert!(ids.values().all(|required| !required));
    assert!(ids.contains_key(ProviderConformanceProfile::Quantized.id()));
    assert!(ids.contains_key(ProviderConformanceProfile::AdvancedAttention.id()));
    assert!(ids.contains_key(ProviderConformanceProfile::FusedKernel.id()));
    assert!(ids.contains_key(ProviderConformanceProfile::Browser.id()));
    assert!(ids.contains_key(ProviderConformanceProfile::WebGpu.id()));
}

#[test]
fn provider_roadmap_error_display_is_non_empty_for_every_variant() {
    let variants = vec![
        ProviderRoadmapError::ProviderRoadmapUnsupported {
            reason: "example".into(),
        },
        ProviderRoadmapError::OptimizedCpuProviderUnavailable {
            reason: "example".into(),
        },
        ProviderRoadmapError::CudaProviderUnavailable {
            reason: "example".into(),
        },
        ProviderRoadmapError::MetalProviderUnavailable {
            reason: "example".into(),
        },
        ProviderRoadmapError::OpenVinoProviderUnavailable {
            reason: "example".into(),
        },
        ProviderRoadmapError::QnnProviderUnavailable {
            reason: "example".into(),
        },
        ProviderRoadmapError::WebGpuProviderUnavailable {
            reason: "example".into(),
        },
        ProviderRoadmapError::ProviderFeatureUnsupported {
            feature: "example".into(),
        },
        ProviderRoadmapError::ProviderLayoutUnsupported {
            layout: "example".into(),
        },
        ProviderRoadmapError::ProviderDTypeUnsupported {
            dtype: "example".into(),
        },
        ProviderRoadmapError::ProviderMemoryClassUnsupported {
            memory_class: "example".into(),
        },
        ProviderRoadmapError::ProviderAdvancedAttentionUnsupported {
            variant: "example".into(),
        },
        ProviderRoadmapError::ProviderQuantizationUnsupported {
            reason: "example".into(),
        },
        ProviderRoadmapError::ProviderFusionInvalid {
            reason: "example".into(),
        },
        ProviderRoadmapError::ProviderConformanceFailed {
            report: "example".into(),
        },
        ProviderRoadmapError::ProviderBenchmarkFailed {
            reason: "example".into(),
        },
        ProviderRoadmapError::ProviderFallbackDenied {
            reason: "example".into(),
        },
        ProviderRoadmapError::ProviderNativeHandleExposureDenied {
            handle_kind: "example".into(),
        },
        ProviderRoadmapError::InternalProviderRoadmapError {
            reason: "example".into(),
        },
    ];
    for variant in variants {
        let rendered = variant.to_string();
        assert!(!rendered.is_empty(), "{variant:?} rendered empty");
        assert!(!variant.id().is_empty());
    }
}

#[test]
fn provider_roadmap_observation_kind_round_trips_through_debug() {
    let kinds = [
        ProviderRoadmapObservationKind::RoadmapFeatureDiscovered,
        ProviderRoadmapObservationKind::CapabilityAdvertised,
        ProviderRoadmapObservationKind::CapabilityRejected,
        ProviderRoadmapObservationKind::OptimizedProviderSelected,
        ProviderRoadmapObservationKind::AdvancedAttentionSelected,
        ProviderRoadmapObservationKind::QuantizedKernelSelected,
        ProviderRoadmapObservationKind::FusedKernelSelected,
        ProviderRoadmapObservationKind::FallbackConsidered,
        ProviderRoadmapObservationKind::FallbackUsed,
        ProviderRoadmapObservationKind::FallbackDenied,
        ProviderRoadmapObservationKind::ConformancePassed,
        ProviderRoadmapObservationKind::ConformanceFailed,
        ProviderRoadmapObservationKind::BenchmarkExecuted,
        ProviderRoadmapObservationKind::BenchmarkSkipped,
    ];
    assert_eq!(kinds.len(), 14);
    for kind in kinds {
        let observation = ProviderRoadmapObservation::new(kind);
        assert_eq!(observation.kind, kind);
    }
}

#[test]
fn provider_roadmap_device_metadata_templates_carry_family_memory_classes() {
    for family in [
        ProviderRoadmapHardwareFamily::Cuda,
        ProviderRoadmapHardwareFamily::Metal,
        ProviderRoadmapHardwareFamily::OpenVino,
        ProviderRoadmapHardwareFamily::Qnn,
        ProviderRoadmapHardwareFamily::WebGpu,
    ] {
        let device = family.device_metadata_template();
        assert_eq!(device.memory_class_support, family.memory_classes());
        assert!(!device.vendor.is_empty());
        assert!(!device.architecture.is_empty());
    }
}
