//! Portable compute data types shared by every Runtime layer.
//!
//! Core layer (`quality/layers.toml`): nothing here depends on another
//! Runtime module. The types are re-exported from [`crate::compute`], where
//! they were first defined, so `magnetar_runtime::compute::…` paths and the
//! crate-root re-exports are unchanged.

use crate::binding::*;
use std::{collections::BTreeMap, error::Error, fmt};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeDType {
    Boolean,
    UInt8,
    SInt8,
    UInt16,
    SInt16,
    UInt32,
    SInt32,
    UInt64,
    SInt64,
    Float16,
    BrainFloat16,
    Float32,
    Float64,
}

impl ComputeDType {
    pub const fn size_bytes(self) -> u64 {
        match self {
            Self::Boolean | Self::UInt8 | Self::SInt8 => 1,
            Self::UInt16 | Self::SInt16 | Self::Float16 | Self::BrainFloat16 => 2,
            Self::UInt32 | Self::SInt32 | Self::Float32 => 4,
            Self::UInt64 | Self::SInt64 | Self::Float64 => 8,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DTypeDescriptor {
    Portable(ComputeDType),
    ProviderSpecific { id: String, size_bytes: u64 },
}

impl DTypeDescriptor {
    pub const fn portable(dtype: ComputeDType) -> Self {
        Self::Portable(dtype)
    }
    pub fn size_bytes(&self) -> u64 {
        match self {
            Self::Portable(dtype) => dtype.size_bytes(),
            Self::ProviderSpecific { size_bytes, .. } => *size_bytes,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TensorResourceId(String);

impl TensorResourceId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TensorResourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputePrecision {
    Exact,
    Default,
    Reduced,
    Mixed,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum HostStagingPolicy {
    Forbid,
    Permit,
}

impl HostStagingPolicy {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Forbid => "forbid",
            Self::Permit => "permit",
        }
    }
}

impl ComputeDType {
    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Self::UInt8
                | Self::SInt8
                | Self::UInt16
                | Self::SInt16
                | Self::UInt32
                | Self::SInt32
                | Self::UInt64
                | Self::SInt64
        )
    }
}

/// Public (not `pub(crate)`): unlike [`crate::model::ModelTrustDecision`]'s
/// fields, this carries no authority -- it's a stateless text redaction the
/// `magnetar-roadmap-contracts` crate's own release/security diagnostic
/// contracts (#62) need to reproduce identically outside this crate.
pub fn redact_backend_diagnostic(message: &str) -> String {
    let contains_native_handle = message.contains("0x") || message.contains("handle=");
    let contains_path = message.contains('\\') || message.contains('/');
    if contains_native_handle || contains_path {
        "[redacted backend diagnostic]".into()
    } else {
        message.into()
    }
}

/// Semantic operation families covered by the portable Compute capability.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeOperationFamily {
    DescriptorAndView,
    ConstructionAndAllocation,
    DataMovementAndConversion,
    Elementwise,
    ComparisonAndSelection,
    Reduction,
    LinearAlgebra,
    ConvolutionAndSpatialTransform,
    IndexingAndUpdate,
    RandomGeneration,
    SynchronizationAndCompletion,
}

impl ComputeOperationFamily {
    pub const ALL: [Self; 11] = [
        Self::DescriptorAndView,
        Self::ConstructionAndAllocation,
        Self::DataMovementAndConversion,
        Self::Elementwise,
        Self::ComparisonAndSelection,
        Self::Reduction,
        Self::LinearAlgebra,
        Self::ConvolutionAndSpatialTransform,
        Self::IndexingAndUpdate,
        Self::RandomGeneration,
        Self::SynchronizationAndCompletion,
    ];
    pub const fn id(self) -> &'static str {
        match self {
            Self::DescriptorAndView => "descriptor-and-view",
            Self::ConstructionAndAllocation => "construction-and-allocation",
            Self::DataMovementAndConversion => "data-movement-and-conversion",
            Self::Elementwise => "elementwise",
            Self::ComparisonAndSelection => "comparison-and-selection",
            Self::Reduction => "reduction",
            Self::LinearAlgebra => "linear-algebra",
            Self::ConvolutionAndSpatialTransform => "convolution-and-spatial-transform",
            Self::IndexingAndUpdate => "indexing-and-update",
            Self::RandomGeneration => "random-generation",
            Self::SynchronizationAndCompletion => "synchronization-and-completion",
        }
    }
    pub const fn metadata(self) -> ComputeOperationFamilyMetadata {
        match self {
            Self::DescriptorAndView => ComputeOperationFamilyMetadata {
                family: self,
                name: "Descriptor and view",
                scope: "tensor metadata and view transformations",
                examples: &[
                    "shape",
                    "dtype",
                    "reshape",
                    "flatten",
                    "squeeze",
                    "unsqueeze",
                    "transpose",
                    "permute",
                    "narrow",
                    "slice",
                    "broadcast",
                ],
            },
            Self::ConstructionAndAllocation => ComputeOperationFamilyMetadata {
                family: self,
                name: "Construction and allocation",
                scope: "portable tensor construction and allocation requests",
                examples: &["scalar", "zeros", "ones", "range", "allocate"],
            },
            Self::DataMovementAndConversion => ComputeOperationFamilyMetadata {
                family: self,
                name: "Data movement and conversion",
                scope: "explicit transfer, copy, materialization and dtype conversion",
                examples: &[
                    "upload",
                    "download",
                    "copy",
                    "materialize",
                    "convert",
                    "transfer",
                ],
            },
            Self::Elementwise => ComputeOperationFamilyMetadata {
                family: self,
                name: "Elementwise",
                scope: "portable unary, binary, activation and affine tensor operations",
                examples: &["add", "sub", "mul", "div", "exp", "log", "relu", "pow"],
            },
            Self::ComparisonAndSelection => ComputeOperationFamilyMetadata {
                family: self,
                name: "Comparison and selection",
                scope: "comparisons and conditional selection",
                examples: &["eq", "lt", "gt", "where"],
            },
            Self::Reduction => ComputeOperationFamilyMetadata {
                family: self,
                name: "Reduction",
                scope: "axis-based reductions with future schema-defined edge behavior",
                examples: &["sum", "mean", "min", "max", "argmin", "argmax"],
            },
            Self::LinearAlgebra => ComputeOperationFamilyMetadata {
                family: self,
                name: "Linear algebra",
                scope: "matrix and batched matrix operations",
                examples: &["matmul", "batched-matmul", "broadcast-matmul"],
            },
            Self::ConvolutionAndSpatialTransform => ComputeOperationFamilyMetadata {
                family: self,
                name: "Convolution and spatial transform",
                scope: "convolutions, pooling and spatial resampling",
                examples: &[
                    "conv",
                    "conv-transpose",
                    "pool",
                    "upsample-nearest",
                    "upsample-bilinear",
                ],
            },
            Self::IndexingAndUpdate => ComputeOperationFamilyMetadata {
                family: self,
                name: "Indexing and update",
                scope: "indexing, scatter/gather and explicit update-like result semantics",
                examples: &[
                    "gather",
                    "index-select",
                    "index-add",
                    "scatter",
                    "scatter-add",
                    "concat",
                ],
            },
            Self::RandomGeneration => ComputeOperationFamilyMetadata {
                family: self,
                name: "Random generation",
                scope: "provider-owned random tensor generation with optional seeds",
                examples: &["uniform", "normal", "seeded-generation"],
            },
            Self::SynchronizationAndCompletion => ComputeOperationFamilyMetadata {
                family: self,
                name: "Synchronization and completion",
                scope: "coarse operation status, await, cancellation and output retrieval",
                examples: &["status", "await", "cancel", "take-outputs"],
            },
        }
    }
    pub fn from_id(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|family| family.id() == value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ComputeOperationFamilyMetadata {
    pub family: ComputeOperationFamily,
    pub name: &'static str,
    pub scope: &'static str,
    pub examples: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeLayout {
    Dense,
    Strided,
    Blocked,
    Paged,
    PackedQuantized,
    AttentionSpecific,
    BrowserCompatible,
    ProviderOpaque,
}

/// How a Paged layout's logical length may grow after creation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PagedAppendBehavior {
    Forbidden,
    FixedCapacity,
    GrowOnDemand,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LayoutDescriptor {
    Contiguous,
    Strided {
        strides_elements: Vec<i64>,
        offset_elements: u64,
    },
    /// Reserved for tiled or block-structured storage; MAY be a future or
    /// Provider-specific placeholder in the first implementation scope.
    Blocked {
        block_dimensions: Vec<u64>,
    },
    /// Page/block-based storage, especially for future KV cache and
    /// attention paths. Raw page pointers SHALL not be exposed: the
    /// logical-to-physical map holds page *indices*, never addresses.
    Paged {
        page_size_elements: u64,
        block_size_elements: u64,
        capacity_pages: Option<u64>,
        current_length_elements: Option<u64>,
        logical_to_physical: Option<BTreeMap<u64, u64>>,
        append_behavior: Option<PagedAppendBehavior>,
    },
    /// Quantized packed storage. MAY be future or placeholder initially.
    PackedQuantized {
        method: String,
        bits_per_value: u32,
        group_size: Option<u64>,
        scale_dtype: Option<Box<DTypeDescriptor>>,
        zero_point_dtype: Option<Box<DTypeDescriptor>>,
        packing_order: Option<String>,
        dequantization_requirements: Option<String>,
    },
    AttentionSpecific {
        layout_id: String,
    },
    BrowserCompatible {
        layout_id: String,
    },
    ProviderOpaque {
        layout_id: String,
    },
}

impl LayoutDescriptor {
    pub const fn kind(&self) -> ComputeLayout {
        match self {
            Self::Contiguous => ComputeLayout::Dense,
            Self::Strided { .. } => ComputeLayout::Strided,
            Self::Blocked { .. } => ComputeLayout::Blocked,
            Self::Paged { .. } => ComputeLayout::Paged,
            Self::PackedQuantized { .. } => ComputeLayout::PackedQuantized,
            Self::AttentionSpecific { .. } => ComputeLayout::AttentionSpecific,
            Self::BrowserCompatible { .. } => ComputeLayout::BrowserCompatible,
            Self::ProviderOpaque { .. } => ComputeLayout::ProviderOpaque,
        }
    }
}

/// Per-dimension declaration for shapes that describe more than one
/// concrete extent: a `Fixed` dimension must match the concrete value in
/// `ShapeDescriptor::dimensions`; `Symbolic` names a dimension shared across
/// tensors (e.g. a sequence-length symbol); `Dynamic` marks a dimension
/// whose concrete value may change between invocations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SymbolicDimension {
    Fixed(u64),
    Symbolic(String),
    Dynamic,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShapeDescriptor {
    pub dimensions: Vec<u64>,
    pub symbolic: Option<Vec<SymbolicDimension>>,
}

impl ShapeDescriptor {
    pub fn new(dimensions: impl Into<Vec<u64>>) -> Self {
        Self {
            dimensions: dimensions.into(),
            symbolic: None,
        }
    }
    pub fn with_symbolic_dimensions(mut self, symbolic: impl Into<Vec<SymbolicDimension>>) -> Self {
        self.symbolic = Some(symbolic.into());
        self
    }
    pub fn rank(&self) -> u64 {
        self.dimensions.len() as u64
    }
    pub fn element_count(&self) -> Result<u64, ComputeValidationError> {
        self.dimensions.iter().try_fold(1_u64, |acc, dimension| {
            acc.checked_mul(*dimension)
                .ok_or(ComputeValidationError::SizeOverflow {
                    reason: "tensor element count overflows u64".into(),
                })
        })
    }
    /// Row-major (C-order) strides implied by this shape, in elements. This
    /// is the explicit dimension order Contiguous layout uses.
    pub fn row_major_strides(&self) -> Vec<i64> {
        let mut strides = vec![1_i64; self.dimensions.len()];
        for index in (0..self.dimensions.len().saturating_sub(1)).rev() {
            strides[index] = strides[index + 1] * self.dimensions[index + 1] as i64;
        }
        strides
    }
    /// Symbolic dimension count must match rank, and any `Fixed` entry must
    /// agree with the concrete dimension it annotates.
    pub fn validate_symbolic(&self) -> Result<(), ComputeValidationError> {
        let Some(symbolic) = &self.symbolic else {
            return Ok(());
        };
        if symbolic.len() != self.dimensions.len() {
            return Err(ComputeValidationError::InvalidShape {
                reason: format!(
                    "symbolic dimension count {} does not match rank {}",
                    symbolic.len(),
                    self.dimensions.len()
                ),
            });
        }
        for (symbol, concrete) in symbolic.iter().zip(&self.dimensions) {
            if let SymbolicDimension::Fixed(value) = symbol
                && value != concrete
            {
                return Err(ComputeValidationError::InvalidShape {
                    reason: format!(
                        "fixed symbolic dimension {value} does not match concrete dimension {concrete}"
                    ),
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorViewSource {
    Descriptor,
    Resource(TensorResourceId),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ViewDescriptor {
    pub source: TensorViewSource,
    pub offset_elements: u64,
    pub strides_elements: Vec<i64>,
}

impl ViewDescriptor {
    pub fn from_resource(
        source: TensorResourceId,
        offset_elements: u64,
        strides_elements: impl Into<Vec<i64>>,
    ) -> Self {
        Self {
            source: TensorViewSource::Resource(source),
            offset_elements,
            strides_elements: strides_elements.into(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorDescriptor {
    pub shape: ShapeDescriptor,
    pub dtype: DTypeDescriptor,
    pub layout: LayoutDescriptor,
    pub view: Option<ViewDescriptor>,
    pub storage_dtype: Option<DTypeDescriptor>,
    pub compute_dtype: Option<DTypeDescriptor>,
    pub memory_class_intent: Option<crate::TensorMemoryClass>,
    pub mutability_intent: Option<crate::TensorMutabilityKind>,
    pub aliasing_intent: Option<crate::TensorAliasingKind>,
    pub affinity_constraints: Option<Box<ResourceAffinity>>,
    pub semantic_role: Option<crate::TensorRole>,
    pub dimension_roles: Option<Vec<crate::DimensionRole>>,
}

impl TensorDescriptor {
    pub fn new(shape: ShapeDescriptor, dtype: DTypeDescriptor, layout: LayoutDescriptor) -> Self {
        Self {
            shape,
            dtype,
            layout,
            view: None,
            storage_dtype: None,
            compute_dtype: None,
            memory_class_intent: None,
            mutability_intent: None,
            aliasing_intent: None,
            affinity_constraints: None,
            semantic_role: None,
            dimension_roles: None,
        }
    }
    pub fn with_view(mut self, view: ViewDescriptor) -> Self {
        self.view = Some(view);
        self
    }
    pub fn with_storage_dtype(mut self, dtype: DTypeDescriptor) -> Self {
        self.storage_dtype = Some(dtype);
        self
    }
    pub fn with_compute_dtype(mut self, dtype: DTypeDescriptor) -> Self {
        self.compute_dtype = Some(dtype);
        self
    }
    pub fn with_memory_class_intent(mut self, class: crate::TensorMemoryClass) -> Self {
        self.memory_class_intent = Some(class);
        self
    }
    pub fn with_mutability_intent(mut self, mutability: crate::TensorMutabilityKind) -> Self {
        self.mutability_intent = Some(mutability);
        self
    }
    pub fn with_aliasing_intent(mut self, aliasing: crate::TensorAliasingKind) -> Self {
        self.aliasing_intent = Some(aliasing);
        self
    }
    pub fn with_affinity_constraints(mut self, affinity: ResourceAffinity) -> Self {
        self.affinity_constraints = Some(Box::new(affinity));
        self
    }
    pub fn with_semantic_role(mut self, role: crate::TensorRole) -> Self {
        self.semantic_role = Some(role);
        self
    }
    pub fn with_dimension_roles(mut self, roles: impl Into<Vec<crate::DimensionRole>>) -> Self {
        self.dimension_roles = Some(roles.into());
        self
    }
    pub fn materialized(shape: ShapeDescriptor, dtype: DTypeDescriptor) -> Self {
        Self::new(shape, dtype, LayoutDescriptor::Contiguous)
    }
    pub fn byte_size(&self) -> Result<u64, ComputeValidationError> {
        self.shape
            .element_count()?
            .checked_mul(self.dtype.size_bytes())
            .ok_or(ComputeValidationError::SizeOverflow {
                reason: "tensor byte size overflows u64".into(),
            })
    }
    /// Conservative byte-size estimate honoring packed/quantized layout
    /// metadata. Falls back to `byte_size()` for layouts without explicit
    /// packing, since bits-per-value there is the same as the dtype width.
    pub fn estimated_byte_size(&self) -> Result<u64, ComputeValidationError> {
        match &self.layout {
            LayoutDescriptor::PackedQuantized { bits_per_value, .. } => {
                let elements = self.shape.element_count()?;
                let total_bits = elements.checked_mul(u64::from(*bits_per_value)).ok_or(
                    ComputeValidationError::SizeOverflow {
                        reason: "packed quantized tensor bit size overflows u64".into(),
                    },
                )?;
                Ok(total_bits.div_ceil(8))
            }
            _ => self.byte_size(),
        }
    }
    pub fn validate(&self, limits: &TensorDescriptorLimits) -> Result<(), ComputeValidationError> {
        limits.validate_shape(&self.shape)?;
        self.shape.validate_symbolic()?;
        let byte_size = self.byte_size()?;
        if byte_size > limits.max_bytes {
            return Err(ComputeValidationError::SizeOverflow {
                reason: format!(
                    "tensor byte size {byte_size} exceeds provider limit {}",
                    limits.max_bytes
                ),
            });
        }
        validate_layout_bounds(&self.shape, &self.layout)?;
        if let Some(view) = &self.view {
            validate_strides(&self.shape, &view.strides_elements, view.offset_elements)?;
        }
        if let Some(roles) = &self.dimension_roles
            && roles.len() as u64 != self.shape.rank()
        {
            return Err(ComputeValidationError::InvalidShape {
                reason: format!(
                    "dimension role count {} does not match tensor rank {}",
                    roles.len(),
                    self.shape.rank()
                ),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorResourceDescriptor {
    pub id: TensorResourceId,
    pub descriptor: TensorDescriptor,
    pub affinity: ResourceAffinity,
}

impl TensorResourceDescriptor {
    pub fn new(
        id: TensorResourceId,
        descriptor: TensorDescriptor,
        affinity: ResourceAffinity,
    ) -> Self {
        Self {
            id,
            descriptor,
            affinity,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorDescriptorLimits {
    pub max_rank: u64,
    pub max_dimension: u64,
    pub max_elements: u64,
    pub max_bytes: u64,
    pub allow_zero_sized: bool,
}

impl Default for TensorDescriptorLimits {
    fn default() -> Self {
        Self {
            max_rank: 64,
            max_dimension: u64::MAX,
            max_elements: u64::MAX,
            max_bytes: u64::MAX,
            allow_zero_sized: false,
        }
    }
}

impl TensorDescriptorLimits {
    pub fn validate_shape(&self, shape: &ShapeDescriptor) -> Result<(), ComputeValidationError> {
        if shape.rank() > self.max_rank {
            return Err(ComputeValidationError::InvalidShape {
                reason: format!(
                    "tensor rank {} exceeds provider limit {}",
                    shape.rank(),
                    self.max_rank
                ),
            });
        }
        for dimension in &shape.dimensions {
            if *dimension > self.max_dimension {
                return Err(ComputeValidationError::InvalidShape {
                    reason: format!(
                        "tensor dimension {dimension} exceeds provider limit {}",
                        self.max_dimension
                    ),
                });
            }
            if *dimension == 0 && !self.allow_zero_sized {
                return Err(ComputeValidationError::InvalidShape {
                    reason: "zero-sized tensor dimensions are not supported".into(),
                });
            }
        }
        let element_count = shape.element_count()?;
        if element_count > self.max_elements {
            return Err(ComputeValidationError::SizeOverflow {
                reason: format!(
                    "tensor element count {element_count} exceeds provider limit {}",
                    self.max_elements
                ),
            });
        }
        Ok(())
    }
}

pub(crate) fn validate_layout_bounds(
    shape: &ShapeDescriptor,
    layout: &LayoutDescriptor,
) -> Result<(), ComputeValidationError> {
    match layout {
        LayoutDescriptor::Contiguous
        | LayoutDescriptor::ProviderOpaque { .. }
        | LayoutDescriptor::Blocked { .. }
        | LayoutDescriptor::Paged { .. }
        | LayoutDescriptor::PackedQuantized { .. }
        | LayoutDescriptor::AttentionSpecific { .. }
        | LayoutDescriptor::BrowserCompatible { .. } => Ok(()),
        LayoutDescriptor::Strided {
            strides_elements,
            offset_elements,
        } => validate_strides(shape, strides_elements, *offset_elements),
    }
}

#[doc(hidden)]
pub fn validate_strides(
    shape: &ShapeDescriptor,
    strides_elements: &[i64],
    offset_elements: u64,
) -> Result<(), ComputeValidationError> {
    if strides_elements.len() as u64 != shape.rank() {
        return Err(ComputeValidationError::InvalidLayout {
            reason: format!(
                "stride rank {} does not match tensor rank {}",
                strides_elements.len(),
                shape.rank()
            ),
        });
    }
    let element_count = shape.element_count()?;
    if element_count == 0 {
        return Ok(());
    }
    if offset_elements >= element_count {
        return Err(ComputeValidationError::InvalidLayout {
            reason: "view offset is outside tensor bounds".into(),
        });
    }
    let max_relative_offset = shape.dimensions.iter().zip(strides_elements).try_fold(
        0_u64,
        |acc, (dimension, stride)| {
            let stride = stride.unsigned_abs();
            let extent = dimension.saturating_sub(1);
            let span = extent
                .checked_mul(stride)
                .ok_or(ComputeValidationError::SizeOverflow {
                    reason: "strided layout span overflows u64".into(),
                })?;
            acc.checked_add(span)
                .ok_or(ComputeValidationError::SizeOverflow {
                    reason: "strided layout span overflows u64".into(),
                })
        },
    )?;
    let max_offset = offset_elements.checked_add(max_relative_offset).ok_or(
        ComputeValidationError::SizeOverflow {
            reason: "strided layout offset overflows u64".into(),
        },
    )?;
    if max_offset >= element_count {
        return Err(ComputeValidationError::InvalidLayout {
            reason: "strided layout addresses elements outside tensor bounds".into(),
        });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeDataMovementKind {
    Upload,
    Download,
    Copy,
    Materialize,
    Transfer,
    DTypeConversion,
    PlacementConversion,
}

impl ComputeDataMovementKind {
    pub const ALL: [Self; 7] = [
        Self::Upload,
        Self::Download,
        Self::Copy,
        Self::Materialize,
        Self::Transfer,
        Self::DTypeConversion,
        Self::PlacementConversion,
    ];
    pub const fn id(self) -> &'static str {
        match self {
            Self::Upload => "upload",
            Self::Download => "download",
            Self::Copy => "copy",
            Self::Materialize => "materialize",
            Self::Transfer => "transfer",
            Self::DTypeConversion => "dtype-conversion",
            Self::PlacementConversion => "placement-conversion",
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ComputeOperationId(String);

impl ComputeOperationId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ComputeOperationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ComputeGraphId(String);

impl ComputeGraphId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ComputeGraphId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ComputeNodeId(String);

impl ComputeNodeId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ComputeNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ComputeInputId(String);

impl ComputeInputId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ComputeInputId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ComputeOutputId(String);

impl ComputeOutputId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ComputeOutputId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComputeValidationError {
    UnknownOperationFamily(String),
    UnknownOperationSchema(ComputeOperationId),
    InvalidGraph {
        reason: String,
    },
    MissingInput {
        input: ComputeInputId,
    },
    MissingOutput {
        node: ComputeNodeId,
        output: ComputeOutputId,
    },
    CyclicGraph {
        node: ComputeNodeId,
        depends_on: ComputeNodeId,
    },
    InvalidState {
        reason: String,
    },
    InvalidOperationAttribute {
        operation: ComputeOperationId,
        attribute: String,
        reason: String,
    },
    InvalidOperationArity {
        operation: ComputeOperationId,
        expected: String,
        found: usize,
    },
    InvalidOutputDescriptor {
        operation: ComputeOperationId,
        reason: String,
    },
    InvalidShape {
        reason: String,
    },
    InvalidLayout {
        reason: String,
    },
    SizeOverflow {
        reason: String,
    },
    UnsupportedOperationFamily {
        provider: ProviderBinding,
        family: ComputeOperationFamily,
    },
    UnsupportedOperationSchema {
        provider: ProviderBinding,
        operation: ComputeOperationId,
    },
    UnsupportedAdvertisement {
        provider: ProviderBinding,
        reason: String,
    },
    UnsupportedDType {
        family: ComputeOperationFamily,
        dtype: ComputeDType,
    },
    UnsupportedProviderDType {
        family: ComputeOperationFamily,
        dtype: String,
    },
    UnsupportedLayout {
        family: ComputeOperationFamily,
        layout: ComputeLayout,
    },
    UnsupportedPrecision {
        family: ComputeOperationFamily,
        precision: ComputePrecision,
    },
    UnsupportedDataMovement {
        provider: ProviderBinding,
        kind: ComputeDataMovementKind,
    },
    InvalidHostBuffer {
        reason: String,
    },
    InvalidTransfer {
        reason: String,
    },
    UnsupportedConversion {
        reason: String,
    },
    MaterializationRequired {
        reason: String,
    },
    MemoryPlanning(MemoryPlanningError),
    ProviderUnavailable(ProviderBinding),
    IncompatibleResourceAffinity(AffinityError),
}

impl fmt::Display for ComputeValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownOperationFamily(family) => {
                write!(f, "unknown compute operation family '{family}'")
            }
            Self::UnknownOperationSchema(operation) => {
                write!(f, "unknown compute operation schema '{operation}'")
            }
            Self::InvalidGraph { reason } => write!(f, "invalid compute graph: {reason}"),
            Self::MissingInput { input } => {
                write!(f, "compute graph references missing input '{input}'")
            }
            Self::MissingOutput { node, output } => write!(
                f,
                "compute graph references missing output '{output}' on node '{node}'"
            ),
            Self::CyclicGraph { node, depends_on } => write!(
                f,
                "compute graph node '{node}' depends on future or cyclic node '{depends_on}'"
            ),
            Self::InvalidState { reason } => {
                write!(f, "invalid compute submission state: {reason}")
            }
            Self::InvalidOperationAttribute {
                operation,
                attribute,
                reason,
            } => write!(
                f,
                "invalid attribute '{attribute}' for operation schema '{operation}': {reason}"
            ),
            Self::InvalidOperationArity {
                operation,
                expected,
                found,
            } => write!(
                f,
                "invalid arity for operation schema '{operation}': expected {expected}, found {found}"
            ),
            Self::InvalidOutputDescriptor { operation, reason } => write!(
                f,
                "invalid output descriptor for operation schema '{operation}': {reason}"
            ),
            Self::InvalidShape { reason } => write!(f, "invalid tensor shape: {reason}"),
            Self::InvalidLayout { reason } => write!(f, "invalid tensor layout: {reason}"),
            Self::SizeOverflow { reason } => write!(f, "invalid tensor size: {reason}"),
            Self::UnsupportedOperationFamily { provider, family } => write!(
                f,
                "provider '{provider}' does not support compute operation family '{}'",
                family.id()
            ),
            Self::UnsupportedOperationSchema {
                provider,
                operation,
            } => write!(
                f,
                "provider '{provider}' does not support compute operation schema '{operation}'"
            ),
            Self::UnsupportedAdvertisement { provider, reason } => {
                write!(
                    f,
                    "provider '{provider}' compute advertisement is unsupported: {reason}"
                )
            }
            Self::UnsupportedDType { family, dtype } => write!(
                f,
                "compute operation family '{}' does not support dtype {dtype:?}",
                family.id()
            ),
            Self::UnsupportedProviderDType { family, dtype } => write!(
                f,
                "compute operation family '{}' does not support provider-specific dtype '{dtype}'",
                family.id()
            ),
            Self::UnsupportedLayout { family, layout } => write!(
                f,
                "compute operation family '{}' does not support layout {layout:?}",
                family.id()
            ),
            Self::UnsupportedPrecision { family, precision } => write!(
                f,
                "compute operation family '{}' does not support precision {precision:?}",
                family.id()
            ),
            Self::UnsupportedDataMovement { provider, kind } => write!(
                f,
                "provider '{provider}' does not support compute data movement '{}'",
                kind.id()
            ),
            Self::InvalidHostBuffer { reason } => {
                write!(f, "invalid host buffer: {reason}")
            }
            Self::InvalidTransfer { reason } => {
                write!(f, "invalid compute data transfer: {reason}")
            }
            Self::UnsupportedConversion { reason } => {
                write!(f, "unsupported compute data conversion: {reason}")
            }
            Self::MaterializationRequired { reason } => {
                write!(f, "materialization required: {reason}")
            }
            Self::MemoryPlanning(error) => {
                write!(f, "{error}")
            }
            Self::ProviderUnavailable(provider) => {
                write!(f, "provider '{provider}' is unavailable")
            }
            Self::IncompatibleResourceAffinity(error) => {
                write!(f, "incompatible tensor resource affinity: {error}")
            }
        }
    }
}

impl Error for ComputeValidationError {}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MemoryPressureReport {
    pub estimated_required_bytes: u64,
    pub estimated_peak_bytes: u64,
    pub selected_provider: Option<ProviderBinding>,
    pub selected_device: Option<DeviceBinding>,
    pub rejected_device_limit: Option<u64>,
    pub materialization_cost_bytes: u64,
    pub transfer_buffer_cost_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryPlanningError {
    MemoryPlanningFailed {
        reason: String,
        report: MemoryPressureReport,
    },
    OutOfMemory {
        required: u64,
        limit: u64,
        report: MemoryPressureReport,
    },
    ResourceExhausted {
        reason: String,
        report: MemoryPressureReport,
    },
    SizeOverflow {
        reason: String,
        report: MemoryPressureReport,
    },
    IncompatibleResourceAffinity(AffinityError),
    UnsupportedLayout {
        layout: ComputeLayout,
        report: MemoryPressureReport,
    },
    MaterializationRequired {
        reason: String,
        report: MemoryPressureReport,
    },
    TransferRequired {
        reason: String,
        report: MemoryPressureReport,
    },
    ProviderMemoryLimitExceeded {
        provider: ProviderBinding,
        required: u64,
        limit: u64,
        report: MemoryPressureReport,
    },
    DeviceMemoryLimitExceeded {
        device: DeviceBinding,
        required: u64,
        limit: u64,
        report: MemoryPressureReport,
    },
}

impl fmt::Display for MemoryPlanningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MemoryPlanningFailed { reason, .. } => {
                write!(f, "memory planning failed: {reason}")
            }
            Self::OutOfMemory {
                required, limit, ..
            } => write!(
                f,
                "memory planning requires {required} bytes but only {limit} bytes are available"
            ),
            Self::ResourceExhausted { reason, .. } => write!(f, "resource exhausted: {reason}"),
            Self::SizeOverflow { reason, .. } => write!(f, "memory size overflow: {reason}"),
            Self::IncompatibleResourceAffinity(error) => {
                write!(f, "incompatible memory resource affinity: {error}")
            }
            Self::UnsupportedLayout { layout, .. } => {
                write!(f, "memory planning does not support layout {layout:?}")
            }
            Self::MaterializationRequired { reason, .. } => {
                write!(f, "memory planning requires materialization: {reason}")
            }
            Self::TransferRequired { reason, .. } => {
                write!(f, "memory planning requires explicit transfer: {reason}")
            }
            Self::ProviderMemoryLimitExceeded {
                provider,
                required,
                limit,
                ..
            } => write!(
                f,
                "provider '{provider}' memory limit exceeded: required {required}, limit {limit}"
            ),
            Self::DeviceMemoryLimitExceeded {
                device,
                required,
                limit,
                ..
            } => write!(
                f,
                "device '{device}' memory limit exceeded: required {required}, limit {limit}"
            ),
        }
    }
}

impl Error for MemoryPlanningError {}

/// Explicit Tensor Resource mutability, finer-grained than the coarse
/// mutable/immutable classification used on Execution Graph edges.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TensorMutabilityKind {
    #[default]
    Immutable,
    Mutable,
    SingleWriter,
    MultiReader,
    RuntimeInternal,
    ProviderOwned,
}

impl TensorMutabilityKind {
    pub const fn allows_mutation(self) -> bool {
        matches!(
            self,
            Self::Mutable | Self::SingleWriter | Self::RuntimeInternal | Self::ProviderOwned
        )
    }
}

/// Explicit Tensor Resource aliasing relationship.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TensorAliasingKind {
    #[default]
    NoAlias,
    ReadOnlyAlias,
    MutableAlias,
    InputOutputAlias,
    ViewAlias,
    InternalTemporaryAlias,
}

impl TensorAliasingKind {
    pub const fn requires_in_place_support(self) -> bool {
        matches!(self, Self::MutableAlias | Self::InputOutputAlias)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelMemoryClass {
    Host,
    PinnedHost,
    Device,
    Unified,
    Shared,
    ProviderOwned,
    BrowserLinearMemory,
    FutureWebGpuBuffer,
}

/// Portable classification of where a Tensor Resource's storage lives.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TensorMemoryClass {
    Host,
    PinnedHost,
    Device,
    Unified,
    Shared,
    ProviderOwned,
    BrowserLinearMemory,
    FutureWebgpuBuffer,
}

impl TensorMemoryClass {
    pub const fn is_host_accessible(self) -> bool {
        matches!(
            self,
            Self::Host | Self::PinnedHost | Self::Unified | Self::Shared
        )
    }
}

/// Semantic role of one Shape dimension, for descriptors that choose to
/// annotate it. Purely advisory metadata; it does not change validation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DimensionRole {
    Batch,
    Sequence,
    Hidden,
    Head,
    Other,
}

/// Structured Tensor failure categories.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorError {
    DescriptorInvalid { reason: String },
    ResourceNotFound { id: TensorResourceId },
    ResourceNotReady { reason: String },
    ResourceInvalid { reason: String },
    ResourceReleased,
    ShapeInvalid { reason: String },
    ShapeMismatch { reason: String },
    RankUnsupported { rank: u64 },
    DTypeUnsupported { reason: String },
    DTypeConversionRequired { reason: String },
    DTypeConversionUnsupported { reason: String },
    LayoutUnsupported { reason: String },
    LayoutConversionRequired { reason: String },
    LayoutConversionUnsupported { reason: String },
    MemoryClassUnsupported { class: TensorMemoryClass },
    ResidencyUnavailable { reason: String },
    ResourceAffinityConflict { reason: String },
    AliasingViolation { reason: String },
    MutabilityViolation { reason: String },
    ViewInvalid { reason: String },
    ViewBaseUnavailable { base: TensorResourceId },
    SizeUnknown { reason: String },
    MaterializationFailed { reason: String },
    TransferFailed { reason: String },
    BrowserFeatureUnsupported { feature: String },
    Internal { reason: String },
}

impl TensorError {
    pub fn descriptor_invalid(reason: impl AsRef<str>) -> Self {
        Self::DescriptorInvalid {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn resource_not_ready(reason: impl AsRef<str>) -> Self {
        Self::ResourceNotReady {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn resource_invalid(reason: impl AsRef<str>) -> Self {
        Self::ResourceInvalid {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn shape_invalid(reason: impl AsRef<str>) -> Self {
        Self::ShapeInvalid {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn shape_mismatch(reason: impl AsRef<str>) -> Self {
        Self::ShapeMismatch {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn dtype_unsupported(reason: impl AsRef<str>) -> Self {
        Self::DTypeUnsupported {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn layout_unsupported(reason: impl AsRef<str>) -> Self {
        Self::LayoutUnsupported {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn residency_unavailable(reason: impl AsRef<str>) -> Self {
        Self::ResidencyUnavailable {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn resource_affinity_conflict(reason: impl AsRef<str>) -> Self {
        Self::ResourceAffinityConflict {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn aliasing_violation(reason: impl AsRef<str>) -> Self {
        Self::AliasingViolation {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn mutability_violation(reason: impl AsRef<str>) -> Self {
        Self::MutabilityViolation {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn view_invalid(reason: impl AsRef<str>) -> Self {
        Self::ViewInvalid {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn size_unknown(reason: impl AsRef<str>) -> Self {
        Self::SizeUnknown {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn materialization_failed(reason: impl AsRef<str>) -> Self {
        Self::MaterializationFailed {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn transfer_failed(reason: impl AsRef<str>) -> Self {
        Self::TransferFailed {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
    pub fn internal(reason: impl AsRef<str>) -> Self {
        Self::Internal {
            reason: redact_backend_diagnostic(reason.as_ref()),
        }
    }
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DescriptorInvalid { reason } => write!(f, "tensor descriptor invalid: {reason}"),
            Self::ResourceNotFound { id } => write!(f, "tensor resource not found: {id}"),
            Self::ResourceNotReady { reason } => write!(f, "tensor resource not ready: {reason}"),
            Self::ResourceInvalid { reason } => write!(f, "tensor resource invalid: {reason}"),
            Self::ResourceReleased => write!(f, "tensor resource was released"),
            Self::ShapeInvalid { reason } => write!(f, "tensor shape invalid: {reason}"),
            Self::ShapeMismatch { reason } => write!(f, "tensor shape mismatch: {reason}"),
            Self::RankUnsupported { rank } => write!(f, "tensor rank {rank} is unsupported"),
            Self::DTypeUnsupported { reason } => write!(f, "tensor dtype unsupported: {reason}"),
            Self::DTypeConversionRequired { reason } => {
                write!(f, "tensor dtype conversion required: {reason}")
            }
            Self::DTypeConversionUnsupported { reason } => {
                write!(f, "tensor dtype conversion unsupported: {reason}")
            }
            Self::LayoutUnsupported { reason } => write!(f, "tensor layout unsupported: {reason}"),
            Self::LayoutConversionRequired { reason } => {
                write!(f, "tensor layout conversion required: {reason}")
            }
            Self::LayoutConversionUnsupported { reason } => {
                write!(f, "tensor layout conversion unsupported: {reason}")
            }
            Self::MemoryClassUnsupported { class } => {
                write!(f, "tensor memory class unsupported: {class:?}")
            }
            Self::ResidencyUnavailable { reason } => {
                write!(f, "tensor residency unavailable: {reason}")
            }
            Self::ResourceAffinityConflict { reason } => {
                write!(f, "tensor resource affinity conflict: {reason}")
            }
            Self::AliasingViolation { reason } => write!(f, "tensor aliasing violation: {reason}"),
            Self::MutabilityViolation { reason } => {
                write!(f, "tensor mutability violation: {reason}")
            }
            Self::ViewInvalid { reason } => write!(f, "tensor view invalid: {reason}"),
            Self::ViewBaseUnavailable { base } => {
                write!(f, "tensor view base unavailable: {base}")
            }
            Self::SizeUnknown { reason } => write!(f, "tensor size unknown: {reason}"),
            Self::MaterializationFailed { reason } => {
                write!(f, "tensor materialization failed: {reason}")
            }
            Self::TransferFailed { reason } => write!(f, "tensor transfer failed: {reason}"),
            Self::BrowserFeatureUnsupported { feature } => {
                write!(f, "tensor browser feature unsupported: {feature}")
            }
            Self::Internal { reason } => write!(f, "internal tensor error: {reason}"),
        }
    }
}

impl Error for TensorError {}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CompletionTokenId {
    value: u64,
    generation: u64,
}

impl CompletionTokenId {
    pub const fn new(value: u64, generation: u64) -> Self {
        Self { value, generation }
    }

    pub const fn value(self) -> u64 {
        self.value
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }
}

impl fmt::Display for CompletionTokenId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "completion-token:{}:{}", self.value, self.generation)
    }
}
