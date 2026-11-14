//! What a Provider advertises for the Compute Capability.
//!
//! The Compute Capability's identifier and version, the operation schemas,
//! data-movement descriptors, per-Device support tables and placement
//! intent a Provider reports, and the validation of a data movement against
//! a placement. Defined below `compute` (graph validation, effective
//! advertisement, execution) so that a Provider can implement the
//! Capability without depending on it.

use crate::binding::*;
use crate::capability::*;
use crate::compute_types::*;
use crate::device::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

pub const COMPUTE_CAPABILITY_ID: &str = "magnetar:compute/run";

/// Current stable version of the executable Compute capability WIT contract.
pub const COMPUTE_CAPABILITY_VERSION: CapabilityVersion = CapabilityVersion::new(2, 0, 0);

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DTypeSupport {
    pub portable: BTreeSet<ComputeDType>,
    pub provider_specific: BTreeSet<String>,
}

impl DTypeSupport {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_portable(mut self, dtypes: impl IntoIterator<Item = ComputeDType>) -> Self {
        self.portable.extend(dtypes);
        self
    }
    pub fn with_provider_specific(
        mut self,
        dtypes: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.provider_specific
            .extend(dtypes.into_iter().map(Into::into));
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayoutSupport {
    pub input: BTreeSet<ComputeLayout>,
    pub output: BTreeSet<ComputeLayout>,
    pub consumes_views: bool,
    pub requires_materialization: bool,
}

impl LayoutSupport {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_input(mut self, layouts: impl IntoIterator<Item = ComputeLayout>) -> Self {
        self.input.extend(layouts);
        self
    }
    pub fn with_output(mut self, layouts: impl IntoIterator<Item = ComputeLayout>) -> Self {
        self.output.extend(layouts);
        self
    }
    pub const fn with_view_consumption(mut self) -> Self {
        self.consumes_views = true;
        self
    }
    pub const fn with_materialization_required(mut self) -> Self {
        self.requires_materialization = true;
        self
    }
}

impl Default for LayoutSupport {
    fn default() -> Self {
        Self {
            input: BTreeSet::new(),
            output: BTreeSet::new(),
            consumes_views: true,
            requires_materialization: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShapeLimitSupport {
    pub descriptor_limits: TensorDescriptorLimits,
    pub max_broadcast_rank: Option<u64>,
    pub max_batch_dimensions: Option<u64>,
}

impl ShapeLimitSupport {
    pub fn new(limits: TensorDescriptorLimits) -> Self {
        Self {
            descriptor_limits: limits,
            max_broadcast_rank: None,
            max_batch_dimensions: None,
        }
    }
    pub const fn with_broadcast_rank(mut self, rank: u64) -> Self {
        self.max_broadcast_rank = Some(rank);
        self
    }
    pub const fn with_batch_dimensions(mut self, dimensions: u64) -> Self {
        self.max_batch_dimensions = Some(dimensions);
        self
    }
}

impl Default for ShapeLimitSupport {
    fn default() -> Self {
        Self::new(TensorDescriptorLimits::default())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PrecisionSupport {
    pub modes: BTreeSet<ComputePrecision>,
    pub accumulation_dtypes: BTreeSet<ComputeDType>,
    pub approximate_math: bool,
    pub deterministic_execution: bool,
    pub deterministic_random_generation: bool,
}

impl PrecisionSupport {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_modes(mut self, modes: impl IntoIterator<Item = ComputePrecision>) -> Self {
        self.modes.extend(modes);
        self
    }
    pub fn with_accumulation_dtypes(
        mut self,
        dtypes: impl IntoIterator<Item = ComputeDType>,
    ) -> Self {
        self.accumulation_dtypes.extend(dtypes);
        self
    }
    pub const fn with_approximate_math(mut self) -> Self {
        self.approximate_math = true;
        self
    }
    pub const fn with_deterministic_execution(mut self) -> Self {
        self.deterministic_execution = true;
        self
    }
    pub const fn with_deterministic_random_generation(mut self) -> Self {
        self.deterministic_random_generation = true;
        self
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ComputeOperationSupport {
    pub dtypes: BTreeSet<ComputeDType>,
    pub provider_specific_dtypes: BTreeSet<String>,
    pub layouts: BTreeSet<ComputeLayout>,
    pub precision_modes: BTreeSet<ComputePrecision>,
    pub descriptor_limits: TensorDescriptorLimits,
}

impl ComputeOperationSupport {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_dtypes(mut self, dtypes: impl IntoIterator<Item = ComputeDType>) -> Self {
        self.dtypes.extend(dtypes);
        self
    }
    pub fn with_provider_specific_dtypes(
        mut self,
        dtypes: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.provider_specific_dtypes
            .extend(dtypes.into_iter().map(Into::into));
        self
    }
    pub fn with_layouts(mut self, layouts: impl IntoIterator<Item = ComputeLayout>) -> Self {
        self.layouts.extend(layouts);
        self
    }
    pub fn with_descriptor_limits(mut self, limits: TensorDescriptorLimits) -> Self {
        self.descriptor_limits = limits;
        self
    }
    pub fn with_precision_modes(
        mut self,
        precision_modes: impl IntoIterator<Item = ComputePrecision>,
    ) -> Self {
        self.precision_modes.extend(precision_modes);
        self
    }
    #[doc(hidden)]
    pub fn supports(
        &self,
        operation: &ComputeOperationDescriptor,
    ) -> Result<(), ComputeValidationError> {
        if let Some(dtype) = operation.dtype
            && !self.dtypes.is_empty()
            && !self.dtypes.contains(&dtype)
        {
            return Err(ComputeValidationError::UnsupportedDType {
                family: operation.family,
                dtype,
            });
        }
        if let Some(layout) = operation.layout
            && !self.layouts.is_empty()
            && !self.layouts.contains(&layout)
        {
            return Err(ComputeValidationError::UnsupportedLayout {
                family: operation.family,
                layout,
            });
        }
        for tensor in &operation.tensors {
            tensor.validate(&self.descriptor_limits)?;
            self.supports_dtype(&tensor.dtype, operation.family)?;
            self.supports_layout(tensor.layout.kind(), operation.family)?;
            if let Some(view) = &tensor.view {
                self.supports_layout(ComputeLayout::Strided, operation.family)?;
                validate_strides(&tensor.shape, &view.strides_elements, view.offset_elements)?;
            }
        }
        if let Some(precision) = operation.precision
            && !self.precision_modes.is_empty()
            && !self.precision_modes.contains(&precision)
        {
            return Err(ComputeValidationError::UnsupportedPrecision {
                family: operation.family,
                precision,
            });
        }
        Ok(())
    }
    pub(crate) fn supports_dtype(
        &self,
        dtype: &DTypeDescriptor,
        family: ComputeOperationFamily,
    ) -> Result<(), ComputeValidationError> {
        match dtype {
            DTypeDescriptor::Portable(dtype) => {
                if !self.dtypes.is_empty() && !self.dtypes.contains(dtype) {
                    return Err(ComputeValidationError::UnsupportedDType {
                        family,
                        dtype: *dtype,
                    });
                }
            }
            DTypeDescriptor::ProviderSpecific { id, .. } => {
                if !self.provider_specific_dtypes.contains(id) {
                    return Err(ComputeValidationError::UnsupportedProviderDType {
                        family,
                        dtype: id.clone(),
                    });
                }
            }
        }
        Ok(())
    }
    pub(crate) fn supports_layout(
        &self,
        layout: ComputeLayout,
        family: ComputeOperationFamily,
    ) -> Result<(), ComputeValidationError> {
        if !self.layouts.is_empty() && !self.layouts.contains(&layout) {
            return Err(ComputeValidationError::UnsupportedLayout { family, layout });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeCapabilitySupport {
    pub capability_id: CapabilityId,
    pub versions: BTreeSet<CapabilityVersion>,
    pub operation_catalog_revision: String,
    pub operation_schema_revision: String,
    pub experimental_extensions: BTreeSet<String>,
}

impl Default for ComputeCapabilitySupport {
    fn default() -> Self {
        Self {
            capability_id: CapabilityId::new(COMPUTE_CAPABILITY_ID),
            versions: BTreeSet::new(),
            operation_catalog_revision: String::new(),
            operation_schema_revision: String::new(),
            experimental_extensions: BTreeSet::new(),
        }
    }
}

impl ComputeCapabilitySupport {
    pub fn with_versions(mut self, versions: impl IntoIterator<Item = CapabilityVersion>) -> Self {
        self.versions.extend(versions);
        self
    }
    pub fn with_operation_catalog_revision(mut self, revision: impl Into<String>) -> Self {
        self.operation_catalog_revision = revision.into();
        self
    }
    pub fn with_operation_schema_revision(mut self, revision: impl Into<String>) -> Self {
        self.operation_schema_revision = revision.into();
        self
    }
    pub fn with_experimental_extensions(
        mut self,
        extensions: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.experimental_extensions
            .extend(extensions.into_iter().map(Into::into));
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationFamilySupport {
    pub family: ComputeOperationFamily,
    pub dtypes: DTypeSupport,
    pub layouts: LayoutSupport,
    pub shapes: ShapeLimitSupport,
    pub precision: PrecisionSupport,
    pub portable: bool,
}

impl OperationFamilySupport {
    pub fn new(family: ComputeOperationFamily) -> Self {
        Self {
            family,
            dtypes: DTypeSupport::default(),
            layouts: LayoutSupport::default(),
            shapes: ShapeLimitSupport::default(),
            precision: PrecisionSupport::default(),
            portable: true,
        }
    }
    pub fn from_operation_support(
        family: ComputeOperationFamily,
        support: ComputeOperationSupport,
    ) -> Self {
        Self {
            family,
            dtypes: DTypeSupport {
                portable: support.dtypes,
                provider_specific: support.provider_specific_dtypes,
            },
            layouts: LayoutSupport {
                input: support.layouts.clone(),
                output: support.layouts,
                ..LayoutSupport::default()
            },
            shapes: ShapeLimitSupport::new(support.descriptor_limits),
            precision: PrecisionSupport {
                modes: support.precision_modes,
                ..PrecisionSupport::default()
            },
            portable: true,
        }
    }
    #[doc(hidden)]
    pub fn operation_support(&self) -> ComputeOperationSupport {
        ComputeOperationSupport {
            dtypes: self.dtypes.portable.clone(),
            provider_specific_dtypes: self.dtypes.provider_specific.clone(),
            layouts: self
                .layouts
                .input
                .union(&self.layouts.output)
                .copied()
                .collect(),
            precision_modes: self.precision.modes.clone(),
            descriptor_limits: self.shapes.descriptor_limits.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationSchemaSupport {
    pub operation: ComputeOperationId,
    pub family: ComputeOperationFamily,
    pub dtypes: DTypeSupport,
    pub layouts: LayoutSupport,
    pub shapes: ShapeLimitSupport,
    pub precision: PrecisionSupport,
    pub portable: bool,
}

impl OperationSchemaSupport {
    pub fn new(operation: ComputeOperationId, family: ComputeOperationFamily) -> Self {
        Self {
            operation,
            family,
            dtypes: DTypeSupport::default(),
            layouts: LayoutSupport::default(),
            shapes: ShapeLimitSupport::default(),
            precision: PrecisionSupport::default(),
            portable: true,
        }
    }
    pub fn from_operation_support(
        operation: ComputeOperationId,
        family: ComputeOperationFamily,
        support: ComputeOperationSupport,
    ) -> Self {
        Self {
            operation,
            family,
            dtypes: DTypeSupport {
                portable: support.dtypes,
                provider_specific: support.provider_specific_dtypes,
            },
            layouts: LayoutSupport {
                input: support.layouts.clone(),
                output: support.layouts,
                ..LayoutSupport::default()
            },
            shapes: ShapeLimitSupport::new(support.descriptor_limits),
            precision: PrecisionSupport {
                modes: support.precision_modes,
                ..PrecisionSupport::default()
            },
            portable: true,
        }
    }
    #[doc(hidden)]
    pub fn operation_support(&self) -> ComputeOperationSupport {
        OperationFamilySupport {
            family: self.family,
            dtypes: self.dtypes.clone(),
            layouts: self.layouts.clone(),
            shapes: self.shapes.clone(),
            precision: self.precision.clone(),
            portable: self.portable,
        }
        .operation_support()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DataMovementSupport {
    pub kind: ComputeDataMovementKind,
    pub dtypes: DTypeSupport,
    pub layouts: LayoutSupport,
    pub host_encodings: BTreeSet<HostBufferEncoding>,
    pub shapes: ShapeLimitSupport,
    pub allow_host_staging: bool,
}

impl DataMovementSupport {
    pub fn new(kind: ComputeDataMovementKind) -> Self {
        Self {
            kind,
            dtypes: DTypeSupport::default(),
            layouts: LayoutSupport::default(),
            host_encodings: BTreeSet::new(),
            shapes: ShapeLimitSupport::default(),
            allow_host_staging: false,
        }
    }
    pub fn from_compute_support(
        kind: ComputeDataMovementKind,
        support: ComputeDataMovementSupport,
    ) -> Self {
        Self {
            kind,
            dtypes: DTypeSupport {
                portable: support.dtypes,
                provider_specific: support.provider_specific_dtypes,
            },
            layouts: LayoutSupport {
                input: support.layouts.clone(),
                output: support.layouts,
                ..LayoutSupport::default()
            },
            host_encodings: support.host_encodings,
            shapes: ShapeLimitSupport::new(support.descriptor_limits),
            allow_host_staging: support.allow_host_staging,
        }
    }
    #[doc(hidden)]
    pub fn movement_support(&self) -> ComputeDataMovementSupport {
        ComputeDataMovementSupport {
            dtypes: self.dtypes.portable.clone(),
            provider_specific_dtypes: self.dtypes.provider_specific.clone(),
            layouts: self
                .layouts
                .input
                .union(&self.layouts.output)
                .copied()
                .collect(),
            host_encodings: self.host_encodings.clone(),
            descriptor_limits: self.shapes.descriptor_limits.clone(),
            allow_host_staging: self.allow_host_staging,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum HostBufferEncoding {
    RawBytes,
    NativeEndian,
    LittleEndian,
    BigEndian,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostBufferDescriptor {
    pub byte_len: u64,
    pub encoding: HostBufferEncoding,
}

impl HostBufferDescriptor {
    pub const fn new(byte_len: u64, encoding: HostBufferEncoding) -> Self {
        Self { byte_len, encoding }
    }
    pub fn validate_for(&self, tensor: &TensorDescriptor) -> Result<(), ComputeValidationError> {
        let expected = tensor.byte_size()?;
        if self.byte_len != expected {
            return Err(ComputeValidationError::InvalidHostBuffer {
                reason: format!(
                    "host buffer byte length {} does not match tensor byte size {expected}",
                    self.byte_len
                ),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComputeDataMovementSource {
    Host(HostBufferDescriptor),
    Tensor(TensorResourceDescriptor),
}

impl ComputeDataMovementSource {
    #[doc(hidden)]
    pub fn tensor(&self) -> Option<&TensorResourceDescriptor> {
        match self {
            Self::Host(_) => None,
            Self::Tensor(tensor) => Some(tensor),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputePlacementIntent {
    PreserveSourceAffinity,
    RuntimeSelected,
    HostAccessible,
}

impl ComputePlacementIntent {
    pub const fn id(self) -> &'static str {
        match self {
            Self::PreserveSourceAffinity => "preserve-source-affinity",
            Self::RuntimeSelected => "runtime-selected",
            Self::HostAccessible => "host-accessible",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeDataMovementDescriptor {
    pub kind: ComputeDataMovementKind,
    pub source: ComputeDataMovementSource,
    pub output: TensorDescriptor,
    pub placement: ComputePlacementIntent,
    pub host_staging: HostStagingPolicy,
}

impl ComputeDataMovementDescriptor {
    pub fn upload(host: HostBufferDescriptor, output: TensorDescriptor) -> Self {
        Self::new(
            ComputeDataMovementKind::Upload,
            ComputeDataMovementSource::Host(host),
            output,
        )
    }
    pub fn download(source: TensorResourceDescriptor, output: TensorDescriptor) -> Self {
        Self::new(
            ComputeDataMovementKind::Download,
            ComputeDataMovementSource::Tensor(source),
            output,
        )
    }
    pub fn copy(source: TensorResourceDescriptor, output: TensorDescriptor) -> Self {
        Self::new(
            ComputeDataMovementKind::Copy,
            ComputeDataMovementSource::Tensor(source),
            output,
        )
    }
    pub fn materialize(source: TensorResourceDescriptor, output: TensorDescriptor) -> Self {
        Self::new(
            ComputeDataMovementKind::Materialize,
            ComputeDataMovementSource::Tensor(source),
            output,
        )
    }
    pub fn transfer(source: TensorResourceDescriptor, output: TensorDescriptor) -> Self {
        Self::new(
            ComputeDataMovementKind::Transfer,
            ComputeDataMovementSource::Tensor(source),
            output,
        )
    }
    pub fn dtype_conversion(source: TensorResourceDescriptor, output: TensorDescriptor) -> Self {
        Self::new(
            ComputeDataMovementKind::DTypeConversion,
            ComputeDataMovementSource::Tensor(source),
            output,
        )
    }
    pub fn placement_conversion(
        source: TensorResourceDescriptor,
        output: TensorDescriptor,
    ) -> Self {
        Self::new(
            ComputeDataMovementKind::PlacementConversion,
            ComputeDataMovementSource::Tensor(source),
            output,
        )
    }
    pub(crate) fn new(
        kind: ComputeDataMovementKind,
        source: ComputeDataMovementSource,
        output: TensorDescriptor,
    ) -> Self {
        let placement = match kind {
            ComputeDataMovementKind::Upload
            | ComputeDataMovementKind::Transfer
            | ComputeDataMovementKind::PlacementConversion => {
                ComputePlacementIntent::RuntimeSelected
            }
            ComputeDataMovementKind::Download => ComputePlacementIntent::HostAccessible,
            ComputeDataMovementKind::Copy
            | ComputeDataMovementKind::Materialize
            | ComputeDataMovementKind::DTypeConversion => {
                ComputePlacementIntent::PreserveSourceAffinity
            }
        };
        Self {
            kind,
            source,
            output,
            placement,
            host_staging: HostStagingPolicy::Forbid,
        }
    }
    pub const fn with_placement(mut self, placement: ComputePlacementIntent) -> Self {
        self.placement = placement;
        self
    }
    pub const fn with_host_staging_policy(mut self, host_staging: HostStagingPolicy) -> Self {
        self.host_staging = host_staging;
        self
    }
    pub const fn permit_host_staging(mut self) -> Self {
        self.host_staging = HostStagingPolicy::Permit;
        self
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ComputeDataMovementSupport {
    pub dtypes: BTreeSet<ComputeDType>,
    pub provider_specific_dtypes: BTreeSet<String>,
    pub layouts: BTreeSet<ComputeLayout>,
    pub host_encodings: BTreeSet<HostBufferEncoding>,
    pub descriptor_limits: TensorDescriptorLimits,
    pub allow_host_staging: bool,
}

impl ComputeDataMovementSupport {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_dtypes(mut self, dtypes: impl IntoIterator<Item = ComputeDType>) -> Self {
        self.dtypes.extend(dtypes);
        self
    }
    pub fn with_provider_specific_dtypes(
        mut self,
        dtypes: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.provider_specific_dtypes
            .extend(dtypes.into_iter().map(Into::into));
        self
    }
    pub fn with_layouts(mut self, layouts: impl IntoIterator<Item = ComputeLayout>) -> Self {
        self.layouts.extend(layouts);
        self
    }
    pub fn with_host_encodings(
        mut self,
        encodings: impl IntoIterator<Item = HostBufferEncoding>,
    ) -> Self {
        self.host_encodings.extend(encodings);
        self
    }
    pub fn with_descriptor_limits(mut self, limits: TensorDescriptorLimits) -> Self {
        self.descriptor_limits = limits;
        self
    }
    pub const fn with_host_staging(mut self) -> Self {
        self.allow_host_staging = true;
        self
    }
    #[doc(hidden)]
    pub fn supports(
        &self,
        _provider: &ProviderBinding,
        movement: &ComputeDataMovementDescriptor,
    ) -> Result<(), ComputeValidationError> {
        validate_data_movement_placement(movement.kind, movement.placement)?;
        movement.output.validate(&self.descriptor_limits)?;
        self.supports_dtype(&movement.output.dtype, movement.kind)?;
        self.supports_layout(movement.output.layout.kind(), movement.kind)?;
        if let Some(source) = movement.source.tensor() {
            source.descriptor.validate(&self.descriptor_limits)?;
            self.supports_dtype(&source.descriptor.dtype, movement.kind)?;
            self.supports_layout(source.descriptor.layout.kind(), movement.kind)?;
        }
        match &movement.source {
            ComputeDataMovementSource::Host(host) => {
                if movement.kind != ComputeDataMovementKind::Upload {
                    return Err(ComputeValidationError::InvalidTransfer {
                        reason: "host buffers are valid only as upload sources".into(),
                    });
                }
                if !self.host_encodings.is_empty() && !self.host_encodings.contains(&host.encoding)
                {
                    return Err(ComputeValidationError::InvalidHostBuffer {
                        reason: format!("host encoding {:?} is not supported", host.encoding),
                    });
                }
                host.validate_for(&movement.output)?;
            }
            ComputeDataMovementSource::Tensor(source) => {
                if movement.kind == ComputeDataMovementKind::Upload {
                    return Err(ComputeValidationError::InvalidTransfer {
                        reason: "upload requires a host buffer source".into(),
                    });
                }
                if movement.kind == ComputeDataMovementKind::Download {
                    source.descriptor.byte_size()?;
                }
                if movement.kind == ComputeDataMovementKind::Materialize
                    && source.descriptor.view.is_none()
                {
                    return Err(ComputeValidationError::MaterializationRequired {
                        reason: "materialize requires a tensor view source".into(),
                    });
                }
                if movement.kind == ComputeDataMovementKind::DTypeConversion
                    && source.descriptor.dtype == movement.output.dtype
                {
                    return Err(ComputeValidationError::UnsupportedConversion {
                        reason: "dtype conversion requires a different output dtype".into(),
                    });
                }
            }
        }
        Ok(())
    }
    pub(crate) fn supports_dtype(
        &self,
        dtype: &DTypeDescriptor,
        kind: ComputeDataMovementKind,
    ) -> Result<(), ComputeValidationError> {
        match dtype {
            DTypeDescriptor::Portable(dtype) => {
                if !self.dtypes.is_empty() && !self.dtypes.contains(dtype) {
                    return Err(ComputeValidationError::UnsupportedConversion {
                        reason: format!(
                            "data movement '{}' does not support dtype {dtype:?}",
                            kind.id()
                        ),
                    });
                }
            }
            DTypeDescriptor::ProviderSpecific { id, .. } => {
                if !self.provider_specific_dtypes.contains(id) {
                    return Err(ComputeValidationError::UnsupportedConversion {
                        reason: format!(
                            "data movement '{}' does not support provider-specific dtype '{id}'",
                            kind.id()
                        ),
                    });
                }
            }
        }
        Ok(())
    }
    pub(crate) fn supports_layout(
        &self,
        layout: ComputeLayout,
        kind: ComputeDataMovementKind,
    ) -> Result<(), ComputeValidationError> {
        if !self.layouts.is_empty() && !self.layouts.contains(&layout) {
            return Err(ComputeValidationError::UnsupportedConversion {
                reason: format!(
                    "data movement '{}' does not support layout {layout:?}",
                    kind.id()
                ),
            });
        }
        Ok(())
    }
}

pub(crate) fn validate_data_movement_placement(
    kind: ComputeDataMovementKind,
    placement: ComputePlacementIntent,
) -> Result<(), ComputeValidationError> {
    let valid = match kind {
        ComputeDataMovementKind::Upload => {
            matches!(placement, ComputePlacementIntent::RuntimeSelected)
        }
        ComputeDataMovementKind::Download => {
            matches!(placement, ComputePlacementIntent::HostAccessible)
        }
        ComputeDataMovementKind::Copy
        | ComputeDataMovementKind::Materialize
        | ComputeDataMovementKind::DTypeConversion => matches!(
            placement,
            ComputePlacementIntent::PreserveSourceAffinity | ComputePlacementIntent::HostAccessible
        ),
        ComputeDataMovementKind::Transfer | ComputeDataMovementKind::PlacementConversion => {
            matches!(
                placement,
                ComputePlacementIntent::RuntimeSelected | ComputePlacementIntent::HostAccessible
            )
        }
    };
    if valid {
        Ok(())
    } else {
        Err(ComputeValidationError::InvalidTransfer {
            reason: format!(
                "placement intent '{}' is not valid for data movement '{}'",
                placement.id(),
                kind.id()
            ),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceComputeSupport {
    pub device: DeviceId,
    pub memory_bytes: Option<u64>,
    pub operation_families: BTreeMap<ComputeOperationFamily, OperationFamilySupport>,
    pub operation_schemas: BTreeMap<ComputeOperationId, OperationSchemaSupport>,
    pub data_movement: BTreeMap<ComputeDataMovementKind, DataMovementSupport>,
}

impl DeviceComputeSupport {
    pub fn new(device: DeviceId) -> Self {
        Self {
            device,
            memory_bytes: None,
            operation_families: BTreeMap::new(),
            operation_schemas: BTreeMap::new(),
            data_movement: BTreeMap::new(),
        }
    }
    pub const fn with_memory_bytes(mut self, memory_bytes: u64) -> Self {
        self.memory_bytes = Some(memory_bytes);
        self
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProviderComputeAdvertisement {
    pub capability: ComputeCapabilitySupport,
    pub operation_families: BTreeMap<ComputeOperationFamily, OperationFamilySupport>,
    pub operation_schemas: BTreeMap<ComputeOperationId, OperationSchemaSupport>,
    pub unsupported_operation_schemas: BTreeSet<ComputeOperationId>,
    pub provider_extension_schemas: BTreeSet<ComputeOperationId>,
    pub data_movement: BTreeMap<ComputeDataMovementKind, DataMovementSupport>,
    pub devices: BTreeMap<DeviceId, DeviceComputeSupport>,
    pub diagnostics: BTreeMap<String, String>,
}

impl ProviderComputeAdvertisement {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn is_empty(&self) -> bool {
        self.capability.versions.is_empty()
            && self.operation_families.is_empty()
            && self.operation_schemas.is_empty()
            && self.unsupported_operation_schemas.is_empty()
            && self.provider_extension_schemas.is_empty()
            && self.data_movement.is_empty()
            && self.devices.is_empty()
            && self.diagnostics.is_empty()
    }
    pub fn supports_capability_version(&self, required: CapabilityVersion) -> bool {
        self.capability.versions.is_empty()
            || self
                .capability
                .versions
                .iter()
                .any(|version| version.is_compatible_with(required))
    }
    pub fn with_capability(mut self, capability: ComputeCapabilitySupport) -> Self {
        self.capability = capability;
        self
    }
    pub fn with_operation_family(mut self, support: OperationFamilySupport) -> Self {
        self.operation_families.insert(support.family, support);
        self
    }
    pub fn with_operation_schema(mut self, support: OperationSchemaSupport) -> Self {
        self.operation_schemas
            .insert(support.operation.clone(), support);
        self
    }
    pub fn with_unsupported_operation_schema(mut self, operation: ComputeOperationId) -> Self {
        self.unsupported_operation_schemas.insert(operation);
        self
    }
    pub fn with_provider_extension_schema(mut self, operation: ComputeOperationId) -> Self {
        self.provider_extension_schemas.insert(operation);
        self
    }
    pub fn with_data_movement(mut self, support: DataMovementSupport) -> Self {
        self.data_movement.insert(support.kind, support);
        self
    }
    pub fn with_device(mut self, support: DeviceComputeSupport) -> Self {
        self.devices.insert(support.device.clone(), support);
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeAttributeKind {
    Boolean,
    Integer,
    Float,
    String,
    DType,
    Shape,
    Axes,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ComputeOperationAttribute {
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
    DType(ComputeDType),
    Shape(ShapeDescriptor),
    Axes(Vec<u64>),
}

impl ComputeOperationAttribute {
    pub const fn kind(&self) -> ComputeAttributeKind {
        match self {
            Self::Boolean(_) => ComputeAttributeKind::Boolean,
            Self::Integer(_) => ComputeAttributeKind::Integer,
            Self::Float(_) => ComputeAttributeKind::Float,
            Self::String(_) => ComputeAttributeKind::String,
            Self::DType(_) => ComputeAttributeKind::DType,
            Self::Shape(_) => ComputeAttributeKind::Shape,
            Self::Axes(_) => ComputeAttributeKind::Axes,
        }
    }
}

impl Eq for ComputeOperationAttribute {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeOperationAttributeRule {
    pub kind: ComputeAttributeKind,
    pub required: bool,
}

impl ComputeOperationAttributeRule {
    pub const fn required(kind: ComputeAttributeKind) -> Self {
        Self {
            kind,
            required: true,
        }
    }
    pub const fn optional(kind: ComputeAttributeKind) -> Self {
        Self {
            kind,
            required: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeOperationInputRule {
    pub min_inputs: usize,
    pub max_inputs: Option<usize>,
    pub require_same_dtype: bool,
    pub allow_broadcast: bool,
    pub boolean_inputs: BTreeSet<usize>,
    pub integer_index_inputs: BTreeSet<usize>,
}

impl ComputeOperationInputRule {
    pub fn exactly(count: usize) -> Self {
        Self {
            min_inputs: count,
            max_inputs: Some(count),
            require_same_dtype: false,
            allow_broadcast: false,
            boolean_inputs: BTreeSet::new(),
            integer_index_inputs: BTreeSet::new(),
        }
    }
    pub fn at_least(count: usize) -> Self {
        Self {
            min_inputs: count,
            max_inputs: None,
            require_same_dtype: false,
            allow_broadcast: false,
            boolean_inputs: BTreeSet::new(),
            integer_index_inputs: BTreeSet::new(),
        }
    }
    pub fn with_same_dtype(mut self) -> Self {
        self.require_same_dtype = true;
        self
    }
    pub fn with_broadcast(mut self) -> Self {
        self.allow_broadcast = true;
        self
    }
    pub fn with_boolean_input(mut self, index: usize) -> Self {
        self.boolean_inputs.insert(index);
        self
    }
    pub fn with_integer_index_input(mut self, index: usize) -> Self {
        self.integer_index_inputs.insert(index);
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComputeOutputDTypeRule {
    SameAsInput(usize),
    Boolean,
    ExplicitAttribute(String),
    IntegerIndex,
    ProviderDefined,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComputeOutputShapeRule {
    SameAsInput(usize),
    ExplicitAttribute(String),
    BroadcastInputs,
    Reduction {
        axes_attribute: String,
        keep_dimensions_attribute: String,
    },
    MatrixMultiplication,
    BatchedMatrixMultiplication,
    Concatenation {
        axis_attribute: String,
    },
    ProviderDefined,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeOperationOutputRule {
    pub output_count: usize,
    pub dtype: ComputeOutputDTypeRule,
    pub shape: ComputeOutputShapeRule,
}

impl ComputeOperationOutputRule {
    pub fn new(
        output_count: usize,
        dtype: ComputeOutputDTypeRule,
        shape: ComputeOutputShapeRule,
    ) -> Self {
        Self {
            output_count,
            dtype,
            shape,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeOperationSchema {
    pub id: ComputeOperationId,
    pub family: ComputeOperationFamily,
    pub attributes: BTreeMap<String, ComputeOperationAttributeRule>,
    pub input_rule: ComputeOperationInputRule,
    pub output_rule: ComputeOperationOutputRule,
    pub provider_specific_semantics: bool,
}

impl ComputeOperationSchema {
    pub fn new(
        id: impl Into<String>,
        family: ComputeOperationFamily,
        input_rule: ComputeOperationInputRule,
        output_rule: ComputeOperationOutputRule,
    ) -> Self {
        Self {
            id: ComputeOperationId::new(id),
            family,
            attributes: BTreeMap::new(),
            input_rule,
            output_rule,
            provider_specific_semantics: false,
        }
    }
    pub fn with_attribute(
        mut self,
        name: impl Into<String>,
        rule: ComputeOperationAttributeRule,
    ) -> Self {
        self.attributes.insert(name.into(), rule);
        self
    }
    pub const fn with_provider_specific_semantics(mut self) -> Self {
        self.provider_specific_semantics = true;
        self
    }
}

/// Portable operation-specific schema descriptor inside `magnetar:compute/run`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeOperationDescriptor {
    pub schema_id: Option<ComputeOperationId>,
    pub family: ComputeOperationFamily,
    pub dtype: Option<ComputeDType>,
    pub layout: Option<ComputeLayout>,
    pub precision: Option<ComputePrecision>,
    pub attributes: BTreeMap<String, ComputeOperationAttribute>,
    pub tensors: Vec<TensorDescriptor>,
}

impl ComputeOperationDescriptor {
    pub fn new(family: ComputeOperationFamily) -> Self {
        Self {
            schema_id: None,
            family,
            dtype: None,
            layout: None,
            precision: None,
            attributes: BTreeMap::new(),
            tensors: Vec::new(),
        }
    }
    pub fn from_schema(schema: &ComputeOperationSchema) -> Self {
        Self {
            schema_id: Some(schema.id.clone()),
            family: schema.family,
            dtype: None,
            layout: None,
            precision: None,
            attributes: BTreeMap::new(),
            tensors: Vec::new(),
        }
    }
    pub fn with_dtype(mut self, dtype: ComputeDType) -> Self {
        self.dtype = Some(dtype);
        self
    }
    pub fn with_layout(mut self, layout: ComputeLayout) -> Self {
        self.layout = Some(layout);
        self
    }
    pub fn with_precision(mut self, precision: ComputePrecision) -> Self {
        self.precision = Some(precision);
        self
    }
    pub fn with_attribute(
        mut self,
        name: impl Into<String>,
        value: ComputeOperationAttribute,
    ) -> Self {
        self.attributes.insert(name.into(), value);
        self
    }
    pub fn with_tensor(mut self, tensor: TensorDescriptor) -> Self {
        self.tensors.push(tensor);
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeErrorPhase {
    Validation,
    Resolution,
    AffinityValidation,
    Planning,
    DataMovement,
    Materialization,
    MemoryPlanning,
    Submission,
    Execution,
    Cancellation,
    Completion,
    Interruption,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeErrorSeverity {
    Recoverable,
    Terminal,
    Fatal,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RecoveryHint {
    NotRetryable,
    RetryBeforeState,
    RestartableWithReplay,
    ExplicitTransferRequired,
    ExplicitMaterializationRequired,
    ProviderPinned,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeErrorCode {
    InvalidTensorDescriptor,
    InvalidShape,
    InvalidDType,
    InvalidLayout,
    InvalidOperationAttribute,
    InvalidOperationArity,
    InvalidOutputDescriptor,
    SizeOverflow,
    InvalidGraph,
    CyclicGraph,
    MissingInput,
    MissingOutput,
    NoCompatibleProvider,
    NoCompatibleDevice,
    PolicyRejectedProvider,
    ProviderUnavailable,
    DeviceUnavailable,
    CapabilityVersionMismatch,
    UnsupportedOperation,
    UnsupportedOperationFamily,
    UnsupportedDType,
    UnsupportedLayout,
    UnsupportedDataMovement,
    IncompatibleResourceAffinity,
    ProviderPinnedResource,
    DeviceBoundResource,
    ArtifactFingerprintMismatch,
    AffinityGroupMismatch,
    ExecutionFailed,
    ExecutionInterrupted,
    ExecutionCancelled,
    OperationTimeout,
    PlanningFailed,
    InvalidExecutionPlan,
    DataMovementRequired,
    UnsupportedTransfer,
    MemoryPlanningFailed,
    OutOfMemory,
    ResourceExhausted,
    ProviderMemoryLimitExceeded,
    DeviceMemoryLimitExceeded,
    InvalidHostBuffer,
    InvalidTransfer,
    UnsupportedConversion,
    MaterializationRequired,
    InvalidState,
    Internal,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ComputeDiagnostic {
    pub provider: Option<ProviderBinding>,
    pub device: Option<DeviceBinding>,
    pub capability: Option<CapabilityBinding>,
    pub operation_family: Option<ComputeOperationFamily>,
    pub rejected_candidates: Vec<ProviderBinding>,
    pub backend_message: Option<String>,
    pub debug_trace_id: Option<String>,
}

impl ComputeDiagnostic {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_provider(mut self, provider: ProviderBinding) -> Self {
        self.provider = Some(provider);
        self
    }
    pub fn with_device(mut self, device: DeviceBinding) -> Self {
        self.device = Some(device);
        self
    }
    pub fn with_capability(mut self, capability: CapabilityBinding) -> Self {
        self.capability = Some(capability);
        self
    }
    pub fn with_operation_family(mut self, family: ComputeOperationFamily) -> Self {
        self.operation_family = Some(family);
        self
    }
    pub fn with_rejected_candidate(mut self, provider: ProviderBinding) -> Self {
        self.rejected_candidates.push(provider);
        self
    }
    pub fn with_backend_message(mut self, message: impl AsRef<str>) -> Self {
        self.backend_message = Some(redact_backend_diagnostic(message.as_ref()));
        self
    }
    pub fn with_debug_trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.debug_trace_id = Some(trace_id.into());
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeError {
    pub code: ComputeErrorCode,
    pub phase: ComputeErrorPhase,
    pub severity: ComputeErrorSeverity,
    pub message: String,
    pub diagnostics: Vec<ComputeDiagnostic>,
    pub recovery_hints: Vec<RecoveryHint>,
}

impl ComputeError {
    pub fn new(
        code: ComputeErrorCode,
        phase: ComputeErrorPhase,
        severity: ComputeErrorSeverity,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code,
            phase,
            severity,
            message: message.into(),
            diagnostics: Vec::new(),
            recovery_hints: Vec::new(),
        }
    }
    pub fn with_diagnostic(mut self, diagnostic: ComputeDiagnostic) -> Self {
        self.diagnostics.push(diagnostic);
        self
    }
    pub fn with_recovery_hint(mut self, hint: RecoveryHint) -> Self {
        if !self.recovery_hints.contains(&hint) {
            self.recovery_hints.push(hint);
        }
        self
    }
    pub fn validation(code: ComputeErrorCode, message: impl Into<String>) -> Self {
        Self::new(
            code,
            ComputeErrorPhase::Validation,
            ComputeErrorSeverity::Terminal,
            message,
        )
        .with_recovery_hint(RecoveryHint::NotRetryable)
    }
}

impl fmt::Display for ComputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "compute error {:?} during {:?}: {}",
            self.code, self.phase, self.message
        )
    }
}

impl Error for ComputeError {}

impl From<MemoryPlanningError> for ComputeError {
    fn from(error: MemoryPlanningError) -> Self {
        let message = error.to_string();
        match error {
            MemoryPlanningError::MemoryPlanningFailed { report, .. } => ComputeError::new(
                ComputeErrorCode::MemoryPlanningFailed,
                ComputeErrorPhase::MemoryPlanning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(memory_pressure_diagnostic(&report))
            .with_recovery_hint(RecoveryHint::NotRetryable),
            MemoryPlanningError::OutOfMemory { report, .. } => ComputeError::new(
                ComputeErrorCode::OutOfMemory,
                ComputeErrorPhase::MemoryPlanning,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(memory_pressure_diagnostic(&report)),
            MemoryPlanningError::ResourceExhausted { report, .. } => ComputeError::new(
                ComputeErrorCode::ResourceExhausted,
                ComputeErrorPhase::MemoryPlanning,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(memory_pressure_diagnostic(&report)),
            MemoryPlanningError::SizeOverflow { report, .. } => ComputeError::new(
                ComputeErrorCode::SizeOverflow,
                ComputeErrorPhase::MemoryPlanning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(memory_pressure_diagnostic(&report)),
            MemoryPlanningError::IncompatibleResourceAffinity(error) => ComputeError::from(error),
            MemoryPlanningError::UnsupportedLayout { report, .. } => ComputeError::new(
                ComputeErrorCode::UnsupportedLayout,
                ComputeErrorPhase::MemoryPlanning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(memory_pressure_diagnostic(&report)),
            MemoryPlanningError::MaterializationRequired { report, .. } => ComputeError::new(
                ComputeErrorCode::MaterializationRequired,
                ComputeErrorPhase::MemoryPlanning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(memory_pressure_diagnostic(&report))
            .with_recovery_hint(RecoveryHint::ExplicitMaterializationRequired),
            MemoryPlanningError::TransferRequired { report, .. } => ComputeError::new(
                ComputeErrorCode::InvalidTransfer,
                ComputeErrorPhase::MemoryPlanning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(memory_pressure_diagnostic(&report))
            .with_recovery_hint(RecoveryHint::ExplicitTransferRequired),
            MemoryPlanningError::ProviderMemoryLimitExceeded { report, .. } => ComputeError::new(
                ComputeErrorCode::ProviderMemoryLimitExceeded,
                ComputeErrorPhase::MemoryPlanning,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(memory_pressure_diagnostic(&report)),
            MemoryPlanningError::DeviceMemoryLimitExceeded { report, .. } => ComputeError::new(
                ComputeErrorCode::DeviceMemoryLimitExceeded,
                ComputeErrorPhase::MemoryPlanning,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(memory_pressure_diagnostic(&report)),
        }
    }
}

pub(crate) fn memory_pressure_diagnostic(report: &MemoryPressureReport) -> ComputeDiagnostic {
    let mut diagnostic = ComputeDiagnostic::new();
    if let Some(provider) = &report.selected_provider {
        diagnostic = diagnostic.with_provider(provider.clone());
    }
    if let Some(device) = &report.selected_device {
        diagnostic = diagnostic.with_device(device.clone());
    }
    diagnostic.with_backend_message(format!(
        "memory pressure: required={} peak={} materialization={} transfer={}",
        report.estimated_required_bytes,
        report.estimated_peak_bytes,
        report.materialization_cost_bytes,
        report.transfer_buffer_cost_bytes
    ))
}

impl From<AffinityError> for ComputeError {
    fn from(error: AffinityError) -> Self {
        let message = error.to_string();
        match error {
            AffinityError::ProviderMismatch { expected, found } => ComputeError::new(
                ComputeErrorCode::ProviderPinnedResource,
                ComputeErrorPhase::AffinityValidation,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(
                ComputeDiagnostic::new()
                    .with_provider(found)
                    .with_rejected_candidate(expected),
            )
            .with_recovery_hint(RecoveryHint::ProviderPinned),
            AffinityError::DeviceMismatch { expected: _, found } => ComputeError::new(
                ComputeErrorCode::DeviceBoundResource,
                ComputeErrorPhase::AffinityValidation,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_device(found))
            .with_recovery_hint(RecoveryHint::ExplicitTransferRequired),
            AffinityError::CapabilityMismatch {
                id,
                expected,
                found: _,
            } => ComputeError::new(
                ComputeErrorCode::CapabilityVersionMismatch,
                ComputeErrorPhase::AffinityValidation,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(
                ComputeDiagnostic::new().with_capability(CapabilityBinding::new(id, expected)),
            ),
            AffinityError::ArtifactMismatch { .. } => ComputeError::new(
                ComputeErrorCode::ArtifactFingerprintMismatch,
                ComputeErrorPhase::AffinityValidation,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::ProviderPinned),
            AffinityError::ExecutionContextMismatch { .. } => ComputeError::new(
                ComputeErrorCode::ProviderPinnedResource,
                ComputeErrorPhase::Interruption,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::ProviderPinned),
            AffinityError::AffinityGroupMismatch { .. } => ComputeError::new(
                ComputeErrorCode::AffinityGroupMismatch,
                ComputeErrorPhase::AffinityValidation,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::ExplicitTransferRequired),
            AffinityError::BoundProviderUnavailable(provider) => ComputeError::new(
                ComputeErrorCode::ProviderUnavailable,
                ComputeErrorPhase::Interruption,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_provider(provider))
            .with_recovery_hint(RecoveryHint::ProviderPinned),
            AffinityError::BoundDeviceUnavailable(device) => ComputeError::new(
                ComputeErrorCode::DeviceUnavailable,
                ComputeErrorPhase::Interruption,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_device(device))
            .with_recovery_hint(RecoveryHint::ProviderPinned),
            AffinityError::DeviceProviderMismatch {
                device,
                provider,
                owner,
            } => ComputeError::new(
                ComputeErrorCode::DeviceBoundResource,
                ComputeErrorPhase::AffinityValidation,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(
                ComputeDiagnostic::new()
                    .with_device(device)
                    .with_provider(owner)
                    .with_rejected_candidate(provider),
            ),
            AffinityError::ProviderDoesNotImplementCapability {
                provider,
                capability,
            } => ComputeError::new(
                ComputeErrorCode::UnsupportedOperation,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(
                ComputeDiagnostic::new()
                    .with_provider(provider)
                    .with_capability(capability),
            ),
            AffinityError::NoCompatibleProvider(capability) => ComputeError::new(
                ComputeErrorCode::NoCompatibleProvider,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_capability(capability)),
            AffinityError::PolicyRejectedProvider {
                capability,
                policy: _,
            } => ComputeError::new(
                ComputeErrorCode::PolicyRejectedProvider,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_capability(capability)),
            AffinityError::RuntimeNotInitialized => ComputeError::new(
                ComputeErrorCode::ProviderUnavailable,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_recovery_hint(RecoveryHint::RetryBeforeState),
        }
    }
}

impl From<ComputeValidationError> for ComputeError {
    fn from(error: ComputeValidationError) -> Self {
        let message = error.to_string();
        match error {
            ComputeValidationError::UnknownOperationFamily(family) => {
                ComputeError::validation(ComputeErrorCode::UnsupportedOperationFamily, message)
                    .with_diagnostic(
                        ComputeDiagnostic::new()
                            .with_backend_message(format!("unknown operation family: {family}")),
                    )
            }
            ComputeValidationError::UnknownOperationSchema(operation) => {
                ComputeError::validation(ComputeErrorCode::UnsupportedOperation, message)
                    .with_diagnostic(
                        ComputeDiagnostic::new()
                            .with_backend_message(format!("unknown operation schema: {operation}")),
                    )
            }
            ComputeValidationError::InvalidGraph { .. } => {
                ComputeError::validation(ComputeErrorCode::InvalidGraph, message)
            }
            ComputeValidationError::MissingInput { .. } => {
                ComputeError::validation(ComputeErrorCode::MissingInput, message)
            }
            ComputeValidationError::MissingOutput { .. } => {
                ComputeError::validation(ComputeErrorCode::MissingOutput, message)
            }
            ComputeValidationError::CyclicGraph { .. } => {
                ComputeError::validation(ComputeErrorCode::CyclicGraph, message)
            }
            ComputeValidationError::InvalidState { .. } => ComputeError::new(
                ComputeErrorCode::InvalidState,
                ComputeErrorPhase::Submission,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputeValidationError::InvalidOperationAttribute { .. } => {
                ComputeError::validation(ComputeErrorCode::InvalidOperationAttribute, message)
            }
            ComputeValidationError::InvalidOperationArity { .. } => {
                ComputeError::validation(ComputeErrorCode::InvalidOperationArity, message)
            }
            ComputeValidationError::InvalidOutputDescriptor { .. } => {
                ComputeError::validation(ComputeErrorCode::InvalidOutputDescriptor, message)
            }
            ComputeValidationError::InvalidShape { .. } => {
                ComputeError::validation(ComputeErrorCode::InvalidShape, message)
            }
            ComputeValidationError::InvalidLayout { .. } => {
                ComputeError::validation(ComputeErrorCode::InvalidLayout, message)
            }
            ComputeValidationError::SizeOverflow { .. } => {
                ComputeError::validation(ComputeErrorCode::SizeOverflow, message)
            }
            ComputeValidationError::UnsupportedOperationFamily { provider, family } => {
                ComputeError::validation(ComputeErrorCode::UnsupportedOperationFamily, message)
                    .with_diagnostic(
                        ComputeDiagnostic::new()
                            .with_provider(provider)
                            .with_operation_family(family),
                    )
            }
            ComputeValidationError::UnsupportedOperationSchema {
                provider,
                operation,
            } => ComputeError::validation(ComputeErrorCode::UnsupportedOperation, message)
                .with_diagnostic(
                    ComputeDiagnostic::new()
                        .with_provider(provider)
                        .with_backend_message(format!("unsupported operation schema: {operation}")),
                ),
            ComputeValidationError::UnsupportedAdvertisement { provider, reason } => {
                ComputeError::validation(ComputeErrorCode::NoCompatibleProvider, message)
                    .with_diagnostic(
                        ComputeDiagnostic::new()
                            .with_provider(provider)
                            .with_backend_message(reason),
                    )
            }
            ComputeValidationError::UnsupportedDType { family, .. }
            | ComputeValidationError::UnsupportedProviderDType { family, .. } => {
                ComputeError::validation(ComputeErrorCode::UnsupportedDType, message)
                    .with_diagnostic(ComputeDiagnostic::new().with_operation_family(family))
            }
            ComputeValidationError::UnsupportedLayout { family, layout: _ } => {
                ComputeError::validation(ComputeErrorCode::UnsupportedLayout, message)
                    .with_diagnostic(ComputeDiagnostic::new().with_operation_family(family))
            }
            ComputeValidationError::UnsupportedPrecision { family, .. } => {
                ComputeError::validation(ComputeErrorCode::UnsupportedOperation, message)
                    .with_diagnostic(ComputeDiagnostic::new().with_operation_family(family))
            }
            ComputeValidationError::UnsupportedDataMovement { provider, .. } => {
                ComputeError::validation(ComputeErrorCode::UnsupportedDataMovement, message)
                    .with_diagnostic(ComputeDiagnostic::new().with_provider(provider))
                    .with_recovery_hint(RecoveryHint::ExplicitTransferRequired)
            }
            ComputeValidationError::InvalidHostBuffer { .. } => {
                ComputeError::validation(ComputeErrorCode::InvalidHostBuffer, message)
            }
            ComputeValidationError::InvalidTransfer { .. } => {
                ComputeError::validation(ComputeErrorCode::InvalidTransfer, message)
                    .with_recovery_hint(RecoveryHint::ExplicitTransferRequired)
            }
            ComputeValidationError::UnsupportedConversion { .. } => {
                ComputeError::validation(ComputeErrorCode::UnsupportedConversion, message)
            }
            ComputeValidationError::MaterializationRequired { .. } => {
                ComputeError::validation(ComputeErrorCode::MaterializationRequired, message)
                    .with_recovery_hint(RecoveryHint::ExplicitMaterializationRequired)
            }
            ComputeValidationError::MemoryPlanning(error) => ComputeError::from(error),
            ComputeValidationError::ProviderUnavailable(provider) => ComputeError::new(
                ComputeErrorCode::ProviderUnavailable,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_provider(provider))
            .with_recovery_hint(RecoveryHint::RetryBeforeState),
            ComputeValidationError::IncompatibleResourceAffinity(error) => {
                ComputeError::from(error)
            }
        }
    }
}
