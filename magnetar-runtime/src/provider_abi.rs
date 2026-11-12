//! Provider ABI version and memory rule: the part of the Provider ABI that
//! prepared Kernels are checked against.
//!
//! Core layer (`quality/layers.toml`). Re-exported from [`crate::provider`],
//! where these items were first defined, so existing paths are unchanged.

pub const PROVIDER_ABI_MAJOR_VERSION: u16 = 1;

pub const PROVIDER_ABI_MINOR_VERSION: u16 = 0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProviderAbiVersion {
    pub major: u16,
    pub minor: u16,
}

impl ProviderAbiVersion {
    pub const CURRENT: Self = Self {
        major: PROVIDER_ABI_MAJOR_VERSION,
        minor: PROVIDER_ABI_MINOR_VERSION,
    };

    pub const fn new(major: u16, minor: u16) -> Self {
        Self { major, minor }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderAbiMemoryRule {
    pub owner: ProviderAbiMemoryOwner,
    pub retention: ProviderAbiRetentionPolicy,
    pub release_required: bool,
}

impl ProviderAbiMemoryRule {
    pub const fn runtime_borrowed() -> Self {
        Self {
            owner: ProviderAbiMemoryOwner::Runtime,
            retention: ProviderAbiRetentionPolicy::BorrowedForCall,
            release_required: false,
        }
    }

    pub const fn provider_released() -> Self {
        Self {
            owner: ProviderAbiMemoryOwner::Provider,
            retention: ProviderAbiRetentionPolicy::RetainedUntilRelease,
            release_required: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderAbiMemoryOwner {
    Runtime,
    Provider,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderAbiRetentionPolicy {
    BorrowedForCall,
    RetainedUntilRelease,
    OwnershipTransferred,
}
