use super::support::ContractProvider;
use magnetar_runtime::{ProviderError, ProviderLoader};
use magnetar_runtime::{ProviderLoadingMode, ProviderLoadingPolicy};
use std::sync::Arc;

#[test]
fn provider_registry_rejects_duplicate_provider_ids() {
    let mut loader = ProviderLoader::new();
    loader
        .register_provider(Arc::new(ContractProvider::new("duplicate")))
        .unwrap();

    assert!(matches!(
        loader.register_provider(Arc::new(ContractProvider::new("duplicate"))),
        Err(ProviderError::ProviderAlreadyRegistered(name)) if name == "duplicate"
    ));
}

#[test]
fn provider_loading_policy_is_explicit_for_dynamic_and_development_modes() {
    let root = std::path::PathBuf::from("target/providers");
    let provider = root.join("provider.dll");
    let outside = std::path::PathBuf::from("target/other/provider.dll");
    let dynamic = ProviderLoadingPolicy::dynamic_library([root.clone()]);
    let development = ProviderLoadingPolicy::development([root.clone()]);

    assert_eq!(dynamic.mode, ProviderLoadingMode::DynamicLibrary);
    assert!(!dynamic.development_mode);
    assert!(dynamic.allows(&provider));
    assert!(!dynamic.allows(&outside));
    assert_eq!(development.mode, ProviderLoadingMode::DevelopmentProvider);
    assert!(development.development_mode);
    assert!(development.allows(&provider));
    assert!(!ProviderLoadingPolicy::default().allows(&provider));
}
