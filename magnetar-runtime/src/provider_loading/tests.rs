//! Unit tests for the parent module.

use super::*;
use crate::capability::CapabilityVersion;
use crate::tests::*;
use magnetar_provider_api::provider_api::*;
use std::sync::Arc;
use std::sync::atomic::Ordering;

#[test]
fn reject_incompatible() {
    let mut p = TestProvider::new("old");
    p.metadata.api_version += 1;
    assert!(matches!(
        ProviderLoader::new().register_provider(Arc::new(p)),
        Err(ProviderError::IncompatibleApiVersion { .. })
    ));
}

#[test]
fn reject_duplicate() {
    let mut m = ProviderLoader::new();
    m.register_provider(Arc::new(TestProvider::new("same")))
        .unwrap();
    assert!(matches!(
        m.register_provider(Arc::new(TestProvider::new("same"))),
        Err(ProviderError::ProviderAlreadyRegistered(_))
    ));
}

#[test]
fn dynamic_provider_loading_denies_paths_by_default() {
    let mut loader = ProviderLoader::new();
    let path = std::path::PathBuf::from("target/test-provider.dll");

    let result = unsafe { loader.load_dynamic(&path) };

    assert!(matches!(
        result,
        Err(ProviderError::ProviderPathDenied { path: denied }) if denied == path
    ));
}

#[test]
fn load_valid_provider() {
    let p = Arc::new(TestProvider::new("valid"));
    let mut m = ProviderLoader::new();
    m.register_provider(p.clone()).unwrap();
    assert!(m.provider("valid").is_some());
    assert!(p.initialized.load(Ordering::SeqCst));
}

#[test]
fn provider_shutdown_releases_registered_provider() {
    let p = TestProvider::new("provider");
    let p = Arc::new(p);
    let mut m = ProviderLoader::new();
    m.register_provider(p.clone()).unwrap();
    assert!(m.provider("provider").is_some());
    m.shutdown().unwrap();
    assert!(p.shut_down.load(Ordering::SeqCst));
}

#[test]
fn registers_capabilities_and_resolves_fallbacks_by_name() {
    let capability = capability("magnetar:runtime/execute", CapabilityVersion::new(1, 0, 0));
    let mut primary = TestProvider::new("a-primary");
    primary.metadata.capabilities.insert(capability.clone());
    let mut fallback = TestProvider::new("z-fallback");
    fallback.metadata.capabilities.insert(capability.clone());
    let mut loader = ProviderLoader::new();
    loader.register_provider(Arc::new(fallback)).unwrap();
    loader.register_provider(Arc::new(primary)).unwrap();

    assert!(loader.registry().has_capability(&capability));
    let names = loader
        .resolve_providers(&capability)
        .into_iter()
        .map(|provider| provider.metadata().name)
        .collect::<Vec<_>>();
    assert_eq!(names, ["a-primary", "z-fallback"]);
}

#[test]
fn dynamic_provider_loading_rejects_legacy_trait_object_factory_contract() {
    let mut loader = ProviderLoader::new();
    let root = std::path::PathBuf::from("target");
    let path = root.join("test-provider.dll");
    let policy = ProviderLoadingPolicy::development([root]);

    let result = unsafe { loader.load_dynamic_with_policy(&path, &policy) };

    assert!(matches!(
        result,
        Err(ProviderError::UnsupportedDynamicAbi {
            expected_symbol: PROVIDER_ABI_FACTORY_SYMBOL_V1,
            expected_descriptor,
            ..
        }) if expected_descriptor.abi_version == ProviderAbiVersion::CURRENT
    ));
    assert!(
        !include_str!("../provider.rs").contains("Box<dyn Provider>"),
        "stable dynamic loading must not use Rust trait-object factories"
    );
}
