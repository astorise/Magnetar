use crate::component::WitInterface;
use crate::provider::ProviderError;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CapabilityId(String);
impl CapabilityId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for CapabilityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A semantic capability version.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CapabilityVersion {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
}
impl CapabilityVersion {
    pub const fn new(major: u64, minor: u64, patch: u64) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
    /// Whether this available version satisfies `required`.
    pub fn is_compatible_with(&self, required: Self) -> bool {
        if self.major != required.major {
            return false;
        }
        if self.major == 0 {
            return self == &required;
        }
        self >= &required
    }
}
impl fmt::Display for CapabilityVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Declarative contracts and dependencies of a capability.
#[derive(Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct CapabilityDescriptor {
    pub description: String,
    pub contracts: BTreeSet<WitInterface>,
    pub dependencies: BTreeMap<CapabilityId, CapabilityVersion>,
}
impl CapabilityDescriptor {
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            ..Self::default()
        }
    }
    pub fn with_contract(mut self, contract: WitInterface) -> Self {
        self.contracts.insert(contract);
        self
    }
    pub fn with_dependency(mut self, id: CapabilityId, version: CapabilityVersion) -> Self {
        self.dependencies.insert(id, version);
        self
    }
}

/// A versioned, independently registered runtime capability contract.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Capability {
    pub id: CapabilityId,
    pub version: CapabilityVersion,
    pub descriptor: CapabilityDescriptor,
}
impl Capability {
    pub fn new(
        id: CapabilityId,
        version: CapabilityVersion,
        descriptor: CapabilityDescriptor,
    ) -> Self {
        Self {
            id,
            version,
            descriptor,
        }
    }
    pub fn from_wit(interface: WitInterface) -> Result<Self, ProviderError> {
        let version = parse_capability_version(&interface.version)?;
        Ok(Self::new(
            CapabilityId::new(&interface.name),
            version,
            CapabilityDescriptor::default().with_contract(interface),
        ))
    }
}

pub(crate) fn parse_capability_version(value: &str) -> Result<CapabilityVersion, ProviderError> {
    let mut segments = value.split('.');
    let parse = |segment: Option<&str>| {
        segment
            .ok_or_else(|| ProviderError::InvalidCapabilityVersion(value.into()))?
            .parse::<u64>()
            .map_err(|_| ProviderError::InvalidCapabilityVersion(value.into()))
    };
    let version = CapabilityVersion::new(
        parse(segments.next())?,
        parse(segments.next())?,
        parse(segments.next())?,
    );
    if segments.next().is_some() {
        return Err(ProviderError::InvalidCapabilityVersion(value.into()));
    }
    Ok(version)
}
