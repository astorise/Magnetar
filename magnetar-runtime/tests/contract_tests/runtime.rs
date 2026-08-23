use super::support::ContractProvider;
use magnetar_runtime::{
    Capability, CapabilityDescriptor, CapabilityId, CapabilityVersion, Runtime, WitInterface,
};
use std::sync::Arc;

#[test]
fn runtime_resolves_component_imports_without_component_provider_pin() {
    let interface = WitInterface::new("magnetar:contract/run", "1.0.0");
    let capability = Capability::new(
        CapabilityId::new("magnetar:contract/run"),
        CapabilityVersion::new(1, 0, 0),
        CapabilityDescriptor::new("contract test").with_contract(interface.clone()),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(
            ContractProvider::new("contract-provider").with_capability(capability),
        ))
        .build()
        .unwrap();

    let providers = runtime.resolve_component_import(&interface).unwrap();

    assert_eq!(providers.len(), 1);
    assert_eq!(providers[0].metadata().name, "contract-provider");
}
