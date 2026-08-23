use magnetar_runtime::{AffinityError, FallbackClass, ProviderBinding, ResourceAffinity};

#[test]
fn resource_affinity_reports_provider_conflict() {
    let left =
        ResourceAffinity::new(FallbackClass::Transparent).with_provider(ProviderBinding::new("a"));
    let right =
        ResourceAffinity::new(FallbackClass::Transparent).with_provider(ProviderBinding::new("b"));

    assert!(matches!(
        left.validate_with(&right),
        Err(AffinityError::ProviderMismatch { .. })
    ));
}
