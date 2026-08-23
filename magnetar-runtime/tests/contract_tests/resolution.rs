use super::support::ContractProvider;
use magnetar_runtime::{
    Capability, CapabilityDescriptor, CapabilityId, CapabilityVersion, ProviderHealth,
    ResolutionDecisionReason, Runtime, WitInterface,
};
use std::sync::Arc;

#[test]
fn resolution_prefers_available_provider_over_unavailable_candidate() {
    let capability = Capability::new(
        CapabilityId::new("magnetar:contract/run"),
        CapabilityVersion::new(1, 0, 0),
        CapabilityDescriptor::new("contract test")
            .with_contract(WitInterface::new("magnetar:contract/run", "1.0.0")),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(
            ContractProvider::new("available").with_capability(capability.clone()),
        ))
        .register_provider(Arc::new(
            ContractProvider::new("unavailable")
                .with_capability(capability.clone())
                .with_health(ProviderHealth::Unavailable),
        ))
        .build()
        .unwrap();

    let resolved = runtime
        .resolve_with_affinity(
            &capability,
            &[],
            magnetar_runtime::FallbackClass::Transparent,
        )
        .unwrap();

    assert_eq!(resolved.provider().metadata().name, "available");
    assert_eq!(
        resolved.decision().reason,
        ResolutionDecisionReason::SelectedDeterministically
    );
    assert_eq!(resolved.decision().rejected_candidates.len(), 1);
}
