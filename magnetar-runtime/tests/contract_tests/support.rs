use magnetar_runtime::{
    Capability, Device, Provider, ProviderError, ProviderHealth, ProviderMetadata,
    ProviderRegistry, ProviderStatusSnapshot,
};
use std::sync::Arc;

pub struct ContractProvider {
    metadata: ProviderMetadata,
    health: ProviderHealth,
}

impl ContractProvider {
    pub fn new(name: &str) -> Self {
        Self {
            metadata: ProviderMetadata::new(name, "1", "contract-tests", "test provider"),
            health: ProviderHealth::Available,
        }
    }

    pub fn with_capability(mut self, capability: Capability) -> Self {
        self.metadata.capabilities.insert(capability);
        self
    }

    pub fn with_health(mut self, health: ProviderHealth) -> Self {
        self.health = health;
        self
    }
}

impl Provider for ContractProvider {
    fn metadata(&self) -> ProviderMetadata {
        self.metadata.clone()
    }

    fn register(&self, _registry: &mut ProviderRegistry) -> Result<(), ProviderError> {
        Ok(())
    }

    fn health(&self) -> ProviderHealth {
        self.health
    }

    fn status_snapshot(&self) -> ProviderStatusSnapshot {
        ProviderStatusSnapshot::from_health_report(self.health_report())
    }

    fn devices(&self) -> Vec<Arc<dyn Device>> {
        Vec::new()
    }
}
