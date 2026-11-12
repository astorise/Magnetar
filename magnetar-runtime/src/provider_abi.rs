//! Provider ABI version and memory rule: the part of the Provider ABI that
//! prepared Kernels are checked against.
//!
//! Core layer (`quality/layers.toml`). Re-exported from [`crate::provider`],
//! where these items were first defined, so existing paths are unchanged.

use crate::provider_status::ProviderError;
use std::collections::BTreeSet;

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

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProviderAbiFeature {
    ProviderMetadata,
    CapabilityAdvertisement,
    DeviceListing,
    StatusReporting,
    Execution,
    Cancellation,
    BlockingExecution,
    AsyncExecution,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderAbiThreadingModel {
    SingleThreaded,
    RuntimeSynchronized,
    InternallyThreadSafe,
    Reentrant,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderAbiExecutionBehavior {
    Blocking,
    AsyncCapable,
    LongRunning,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderAbiOwnershipRules {
    pub strings: ProviderAbiMemoryRule,
    pub lists: ProviderAbiMemoryRule,
    pub descriptors: ProviderAbiMemoryRule,
    pub error_messages: ProviderAbiMemoryRule,
    pub opaque_handles: ProviderAbiMemoryRule,
    pub runtime_buffers: ProviderAbiMemoryRule,
    pub provider_buffers: ProviderAbiMemoryRule,
}

impl Default for ProviderAbiOwnershipRules {
    fn default() -> Self {
        Self {
            strings: ProviderAbiMemoryRule::provider_released(),
            lists: ProviderAbiMemoryRule::provider_released(),
            descriptors: ProviderAbiMemoryRule::provider_released(),
            error_messages: ProviderAbiMemoryRule::provider_released(),
            opaque_handles: ProviderAbiMemoryRule::provider_released(),
            runtime_buffers: ProviderAbiMemoryRule::runtime_borrowed(),
            provider_buffers: ProviderAbiMemoryRule::provider_released(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProviderAbiHandle(u64);

impl ProviderAbiHandle {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderAbiHandleKind {
    ProviderInstance,
    ProviderResource,
    Operation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderAbiHandleDescriptor {
    pub kind: ProviderAbiHandleKind,
    pub handle: ProviderAbiHandle,
    pub destroy_required: bool,
}

impl ProviderAbiHandleDescriptor {
    pub const fn new(kind: ProviderAbiHandleKind, handle: ProviderAbiHandle) -> Self {
        Self {
            kind,
            handle,
            destroy_required: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderAbiLoadingLifecycle {
    Discovered,
    LibraryLoaded,
    DescriptorValidated,
    Initialized,
    Registered,
    Ready,
    Draining,
    Stopped,
    Failed,
    Destroyed,
}

impl ProviderAbiLoadingLifecycle {
    pub const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Discovered, Self::LibraryLoaded)
                | (Self::LibraryLoaded, Self::DescriptorValidated)
                | (Self::LibraryLoaded, Self::Failed)
                | (Self::DescriptorValidated, Self::Initialized)
                | (Self::DescriptorValidated, Self::Failed)
                | (Self::Initialized, Self::Registered)
                | (Self::Initialized, Self::Failed)
                | (Self::Registered, Self::Ready)
                | (Self::Registered, Self::Failed)
                | (Self::Ready, Self::Draining)
                | (Self::Ready, Self::Failed)
                | (Self::Draining, Self::Stopped)
                | (Self::Draining, Self::Failed)
                | (Self::Stopped, Self::Destroyed)
                | (Self::Failed, Self::Destroyed)
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderAbiUnloadPolicy {
    NeverUnload,
    UnloadAfterNoResources,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderAbiErrorCode {
    InvalidAbiDescriptor,
    UnsupportedAbiVersion,
    InvalidMetadata,
    InvalidAdvertisement,
    InvalidDeviceMetadata,
    InitializationFailure,
    ProviderNotReady,
    ProviderDraining,
    ProviderSaturated,
    ExecutionRejected,
    ExecutionFailed,
    CancellationUnsupported,
    CancellationFailed,
    ResourceInvalid,
    InternalProviderError,
    PanicOrUnwindViolation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderAbiFunctionTable {
    pub metadata: bool,
    pub advertisements: bool,
    pub devices: bool,
    pub status: bool,
    pub execution: bool,
    pub release: bool,
    pub destroy: bool,
}

impl ProviderAbiFunctionTable {
    pub const REQUIRED: Self = Self {
        metadata: true,
        advertisements: true,
        devices: true,
        status: true,
        execution: true,
        release: true,
        destroy: true,
    };
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderAbiDescriptor {
    pub descriptor_size: usize,
    pub abi_version: ProviderAbiVersion,
    pub features: BTreeSet<ProviderAbiFeature>,
    pub functions: ProviderAbiFunctionTable,
    pub ownership: ProviderAbiOwnershipRules,
    pub threading: ProviderAbiThreadingModel,
    pub execution_behavior: ProviderAbiExecutionBehavior,
    pub unload_policy: ProviderAbiUnloadPolicy,
}

impl ProviderAbiDescriptor {
    pub fn current() -> Self {
        let features = [
            ProviderAbiFeature::ProviderMetadata,
            ProviderAbiFeature::CapabilityAdvertisement,
            ProviderAbiFeature::DeviceListing,
            ProviderAbiFeature::StatusReporting,
            ProviderAbiFeature::Execution,
            ProviderAbiFeature::Cancellation,
            ProviderAbiFeature::BlockingExecution,
        ]
        .into_iter()
        .collect();
        Self {
            descriptor_size: std::mem::size_of::<Self>(),
            abi_version: ProviderAbiVersion::CURRENT,
            features,
            functions: ProviderAbiFunctionTable::REQUIRED,
            ownership: ProviderAbiOwnershipRules::default(),
            threading: ProviderAbiThreadingModel::RuntimeSynchronized,
            execution_behavior: ProviderAbiExecutionBehavior::Blocking,
            unload_policy: ProviderAbiUnloadPolicy::NeverUnload,
        }
    }

    pub fn validate(&self) -> Result<(), ProviderError> {
        if self.descriptor_size < std::mem::size_of::<Self>() {
            return Err(ProviderError::InvalidAbiDescriptor(
                "descriptor size is smaller than the current layout".into(),
            ));
        }
        if self.abi_version.major != PROVIDER_ABI_MAJOR_VERSION {
            return Err(ProviderError::UnsupportedAbiVersion {
                expected: ProviderAbiVersion::CURRENT,
                found: self.abi_version,
            });
        }
        if self.abi_version.minor > PROVIDER_ABI_MINOR_VERSION {
            return Err(ProviderError::UnsupportedAbiVersion {
                expected: ProviderAbiVersion::CURRENT,
                found: self.abi_version,
            });
        }
        if !self.functions.metadata
            || !self.functions.advertisements
            || !self.functions.devices
            || !self.functions.status
            || !self.functions.execution
            || !self.functions.release
            || !self.functions.destroy
        {
            return Err(ProviderError::InvalidAbiDescriptor(
                "required ABI function is missing".into(),
            ));
        }
        if !self.ownership.strings.release_required
            || !self.ownership.lists.release_required
            || !self.ownership.descriptors.release_required
            || !self.ownership.error_messages.release_required
            || !self.ownership.opaque_handles.release_required
        {
            return Err(ProviderError::InvalidAbiDescriptor(
                "provider-owned ABI memory requires provider release functions".into(),
            ));
        }
        Ok(())
    }
}
