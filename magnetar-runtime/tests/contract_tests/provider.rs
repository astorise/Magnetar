use super::support::ContractProvider;
use magnetar_runtime::{ProviderError, ProviderLoader};
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
