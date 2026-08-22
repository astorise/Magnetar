use crate::*;
use std::{collections::BTreeSet, fmt};
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DeviceId(String);
impl DeviceId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DeviceType {
    Cpu,
    Gpu,
    Npu,
    Tpu,
    Other,
}

/// Immutable metadata describing a hardware execution target.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceMetadata {
    pub id: DeviceId,
    pub name: String,
    pub device_type: DeviceType,
    pub vendor: String,
    pub architecture: String,
    pub memory_capacity: u64,
    pub compute_units: u32,
    pub execution_capabilities: BTreeSet<CapabilityId>,
    /// Stable name of the Provider that discovered this device.
    pub provider: String,
}
impl DeviceMetadata {
    pub fn new(
        id: DeviceId,
        name: impl Into<String>,
        device_type: DeviceType,
        provider: impl Into<String>,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            device_type,
            vendor: String::new(),
            architecture: String::new(),
            memory_capacity: 0,
            compute_units: 0,
            execution_capabilities: BTreeSet::new(),
            provider: provider.into(),
        }
    }
}

/// A reusable concrete device implementation backed by [`DeviceMetadata`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceDescriptor {
    pub metadata: DeviceMetadata,
}
impl DeviceDescriptor {
    pub fn new(metadata: DeviceMetadata) -> Self {
        Self { metadata }
    }
}
pub trait Device: Send + Sync {
    fn metadata(&self) -> &DeviceMetadata;
    fn id(&self) -> &DeviceId {
        &self.metadata().id
    }
    fn device_type(&self) -> DeviceType {
        self.metadata().device_type
    }
    fn availability(&self) -> DeviceAvailability {
        DeviceAvailability::Available
    }
    fn health_report(&self) -> DeviceHealth {
        DeviceHealth::new(
            ProviderBinding::new(&self.metadata().provider),
            DeviceBinding::new(self.id().clone()),
            self.availability(),
        )
    }
}
impl Device for DeviceDescriptor {
    fn metadata(&self) -> &DeviceMetadata {
        &self.metadata
    }
}
