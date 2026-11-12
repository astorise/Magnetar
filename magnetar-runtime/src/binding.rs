//! Identifiers that bind a resource to where it may be used: execution
//! context, affinity group, Provider, Device, Capability and artifact.
//!
//! Core layer (`quality/layers.toml`). Re-exported from [`crate::affinity`],
//! where these types were first defined, so existing paths are unchanged.

use crate::capability::*;
use crate::device::*;
use crate::resolution::*;
use std::{collections::BTreeMap, error::Error, fmt, path::Path};

/// Process-local identity of a Runtime execution context.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExecutionContextId(u64);

impl ExecutionContextId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ExecutionContextId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Process-local identity for resources resolved as one affinity cohort.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AffinityGroupId(u64);

impl AffinityGroupId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for AffinityGroupId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Stable name of the Provider that owns a live resource in one Runtime.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProviderBinding(String);

impl ProviderBinding {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ProviderBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Globally unique Device selected for a device-resident resource.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DeviceBinding(DeviceId);

impl DeviceBinding {
    pub fn new(id: DeviceId) -> Self {
        Self(id)
    }
    pub fn id(&self) -> &DeviceId {
        &self.0
    }
}

impl fmt::Display for DeviceBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Exact Capability implementation that created or constrains a live resource.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CapabilityBinding {
    id: CapabilityId,
    version: CapabilityVersion,
}

impl CapabilityBinding {
    pub fn new(id: CapabilityId, version: CapabilityVersion) -> Self {
        Self { id, version }
    }
    pub fn id(&self) -> &CapabilityId {
        &self.id
    }
    pub const fn version(&self) -> CapabilityVersion {
        self.version
    }
}

impl fmt::Display for CapabilityBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.id, self.version)
    }
}

/// Canonical content fingerprint attached under a semantic artifact role.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ArtifactBinding {
    role: String,
    fingerprint: String,
}

impl ArtifactBinding {
    pub fn new(role: impl Into<String>, fingerprint: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            fingerprint: fingerprint.into(),
        }
    }
    pub fn role(&self) -> &str {
        &self.role
    }
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
}

/// Recovery classification for state associated with an affinity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum FallbackClass {
    Transparent,
    Restartable,
    ProviderPinned,
}

/// Immutable ownership and compatibility facts carried by one opaque resource.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceAffinity {
    provider: Option<ProviderBinding>,
    device: Option<DeviceBinding>,
    capabilities: BTreeMap<CapabilityId, CapabilityBinding>,
    artifacts: BTreeMap<String, ArtifactBinding>,
    execution_context: Option<ExecutionContextId>,
    group: Option<AffinityGroupId>,
    fallback: FallbackClass,
}

impl ResourceAffinity {
    pub fn new(fallback: FallbackClass) -> Self {
        Self {
            provider: None,
            device: None,
            capabilities: BTreeMap::new(),
            artifacts: BTreeMap::new(),
            execution_context: None,
            group: None,
            fallback,
        }
    }
    pub fn with_provider(mut self, binding: ProviderBinding) -> Self {
        self.provider = Some(binding);
        self
    }
    pub fn with_device(mut self, binding: DeviceBinding) -> Self {
        self.device = Some(binding);
        self
    }
    pub fn with_capability(mut self, binding: CapabilityBinding) -> Self {
        self.capabilities.insert(binding.id.clone(), binding);
        self
    }
    pub fn with_artifact(mut self, binding: ArtifactBinding) -> Self {
        self.artifacts.insert(binding.role.clone(), binding);
        self
    }
    pub fn with_execution_context(mut self, id: ExecutionContextId) -> Self {
        self.execution_context = Some(id);
        self
    }
    pub fn with_group(mut self, id: AffinityGroupId) -> Self {
        self.group = Some(id);
        self
    }
    pub fn with_fallback(mut self, fallback: FallbackClass) -> Self {
        self.fallback = fallback;
        self
    }
    pub fn provider(&self) -> Option<&ProviderBinding> {
        self.provider.as_ref()
    }
    pub fn device(&self) -> Option<&DeviceBinding> {
        self.device.as_ref()
    }
    pub fn capability(&self, id: &CapabilityId) -> Option<&CapabilityBinding> {
        self.capabilities.get(id)
    }
    pub fn capabilities(&self) -> impl Iterator<Item = &CapabilityBinding> {
        self.capabilities.values()
    }
    pub fn artifact(&self, role: &str) -> Option<&ArtifactBinding> {
        self.artifacts.get(role)
    }
    pub fn artifacts(&self) -> impl Iterator<Item = &ArtifactBinding> {
        self.artifacts.values()
    }
    pub const fn execution_context(&self) -> Option<ExecutionContextId> {
        self.execution_context
    }
    pub const fn group(&self) -> Option<AffinityGroupId> {
        self.group
    }
    pub const fn fallback(&self) -> FallbackClass {
        self.fallback
    }
    pub fn validate_with(&self, other: &Self) -> Result<(), AffinityError> {
        AffinityConstraints::try_from_affinities([self, other]).map(|_| ())
    }
}

/// Structured validation and constrained-resolution failures for affinities.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AffinityError {
    ProviderMismatch {
        expected: ProviderBinding,
        found: ProviderBinding,
    },
    DeviceMismatch {
        expected: DeviceBinding,
        found: DeviceBinding,
    },
    CapabilityMismatch {
        id: CapabilityId,
        expected: CapabilityVersion,
        found: CapabilityVersion,
    },
    ArtifactMismatch {
        role: String,
        expected: String,
        found: String,
    },
    ExecutionContextMismatch {
        expected: ExecutionContextId,
        found: ExecutionContextId,
    },
    AffinityGroupMismatch {
        expected: AffinityGroupId,
        found: AffinityGroupId,
    },
    BoundProviderUnavailable(ProviderBinding),
    BoundDeviceUnavailable(DeviceBinding),
    DeviceProviderMismatch {
        device: DeviceBinding,
        provider: ProviderBinding,
        owner: ProviderBinding,
    },
    ProviderDoesNotImplementCapability {
        provider: ProviderBinding,
        capability: CapabilityBinding,
    },
    NoCompatibleProvider(CapabilityBinding),
    PolicyRejectedProvider {
        capability: CapabilityBinding,
        policy: ResolutionPolicyId,
    },
    RuntimeNotInitialized,
}

impl fmt::Display for AffinityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProviderMismatch { expected, found } => write!(
                f,
                "resource provider mismatch: expected '{expected}', found '{found}'"
            ),
            Self::DeviceMismatch { expected, found } => write!(
                f,
                "resource device mismatch: expected '{expected}', found '{found}'"
            ),
            Self::CapabilityMismatch {
                id,
                expected,
                found,
            } => write!(
                f,
                "resource capability mismatch for '{id}': expected {expected}, found {found}"
            ),
            Self::ArtifactMismatch {
                role,
                expected,
                found,
            } => write!(
                f,
                "resource artifact mismatch for role '{role}': expected '{expected}', found '{found}'"
            ),
            Self::ExecutionContextMismatch { expected, found } => write!(
                f,
                "resource execution-context mismatch: expected {expected}, found {found}"
            ),
            Self::AffinityGroupMismatch { expected, found } => write!(
                f,
                "resource affinity-group mismatch: expected {expected}, found {found}"
            ),
            Self::BoundProviderUnavailable(provider) => {
                write!(f, "bound provider '{provider}' is unavailable")
            }
            Self::BoundDeviceUnavailable(device) => {
                write!(f, "bound device '{device}' is unavailable")
            }
            Self::DeviceProviderMismatch {
                device,
                provider,
                owner,
            } => write!(
                f,
                "bound device '{device}' belongs to provider '{owner}', not '{provider}'"
            ),
            Self::ProviderDoesNotImplementCapability {
                provider,
                capability,
            } => write!(
                f,
                "bound provider '{provider}' does not implement compatible capability '{capability}'"
            ),
            Self::NoCompatibleProvider(capability) => {
                write!(
                    f,
                    "no Provider implements compatible capability '{capability}'"
                )
            }
            Self::PolicyRejectedProvider { capability, policy } => write!(
                f,
                "resolution policy '{policy}' rejected every Provider for capability '{capability}'"
            ),
            Self::RuntimeNotInitialized => {
                write!(f, "runtime is not initialized for affinity resolution")
            }
        }
    }
}

impl Error for AffinityError {}

/// A conflict-checked aggregation of all affinities consumed by one call.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AffinityConstraints {
    affinity: ResourceAffinity,
}

impl AffinityConstraints {
    pub fn new(fallback: FallbackClass) -> Self {
        Self {
            affinity: ResourceAffinity::new(fallback),
        }
    }
    pub fn try_from_affinities<'a>(
        affinities: impl IntoIterator<Item = &'a ResourceAffinity>,
    ) -> Result<Self, AffinityError> {
        let mut constraints = Self::new(FallbackClass::Transparent);
        for affinity in affinities {
            constraints.merge(affinity)?;
        }
        Ok(constraints)
    }
    pub fn affinity(&self) -> &ResourceAffinity {
        &self.affinity
    }
    pub fn into_affinity(self) -> ResourceAffinity {
        self.affinity
    }
    pub fn require_fallback(&mut self, fallback: FallbackClass) {
        self.affinity.fallback = self.affinity.fallback.max(fallback);
    }
    pub(crate) fn merge(&mut self, incoming: &ResourceAffinity) -> Result<(), AffinityError> {
        if let Some(found) = &incoming.provider {
            match &self.affinity.provider {
                Some(expected) if expected != found => {
                    return Err(AffinityError::ProviderMismatch {
                        expected: expected.clone(),
                        found: found.clone(),
                    });
                }
                None => self.affinity.provider = Some(found.clone()),
                _ => {}
            }
        }
        if let Some(found) = &incoming.device {
            match &self.affinity.device {
                Some(expected) if expected != found => {
                    return Err(AffinityError::DeviceMismatch {
                        expected: expected.clone(),
                        found: found.clone(),
                    });
                }
                None => self.affinity.device = Some(found.clone()),
                _ => {}
            }
        }
        if let Some(found) = incoming.execution_context {
            match self.affinity.execution_context {
                Some(expected) if expected != found => {
                    return Err(AffinityError::ExecutionContextMismatch { expected, found });
                }
                None => self.affinity.execution_context = Some(found),
                _ => {}
            }
        }
        if let Some(found) = incoming.group {
            match self.affinity.group {
                Some(expected) if expected != found => {
                    return Err(AffinityError::AffinityGroupMismatch { expected, found });
                }
                None => self.affinity.group = Some(found),
                _ => {}
            }
        }
        for (id, found) in &incoming.capabilities {
            match self.affinity.capabilities.get(id) {
                Some(expected) if expected != found => {
                    return Err(AffinityError::CapabilityMismatch {
                        id: id.clone(),
                        expected: expected.version,
                        found: found.version,
                    });
                }
                None => {
                    self.affinity.capabilities.insert(id.clone(), found.clone());
                }
                _ => {}
            }
        }
        for (role, found) in &incoming.artifacts {
            match self.affinity.artifacts.get(role) {
                Some(expected) if expected != found => {
                    return Err(AffinityError::ArtifactMismatch {
                        role: role.clone(),
                        expected: expected.fingerprint.clone(),
                        found: found.fingerprint.clone(),
                    });
                }
                None => {
                    self.affinity.artifacts.insert(role.clone(), found.clone());
                }
                _ => {}
            }
        }
        self.affinity.fallback = self.affinity.fallback.max(incoming.fallback);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComponentArtifactReference<'a> {
    LocalPath(&'a Path),
}
