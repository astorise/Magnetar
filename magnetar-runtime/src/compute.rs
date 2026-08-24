use crate::planning::memory_pressure_diagnostic;
use crate::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};
pub const COMPUTE_CAPABILITY_ID: &str = "magnetar:compute/run";
/// WIT package that defines the Compute capability contract.
pub const COMPUTE_WIT_PACKAGE: &str = "magnetar:compute";
/// WIT interface implemented by Compute providers.
pub const COMPUTE_WIT_INTERFACE: &str = COMPUTE_CAPABILITY_ID;
/// Current stable version of the executable Compute capability WIT contract.
pub const COMPUTE_CAPABILITY_VERSION: CapabilityVersion = CapabilityVersion::new(2, 0, 0);

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

fn validate_layout_bounds(
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

fn validate_strides(
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
pub enum ComputePrecision {
    Exact,
    Default,
    Reduced,
    Mixed,
}

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
    pub(crate) fn supports(
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
    fn supports_dtype(
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
    fn supports_layout(
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
    pub(crate) fn operation_support(&self) -> ComputeOperationSupport {
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
    pub(crate) fn operation_support(&self) -> ComputeOperationSupport {
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
    pub(crate) fn movement_support(&self) -> ComputeDataMovementSupport {
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
    pub(crate) fn tensor(&self) -> Option<&TensorResourceDescriptor> {
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
    fn new(
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
    pub(crate) fn supports(
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
    fn supports_dtype(
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
    fn supports_layout(
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

fn validate_data_movement_placement(
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

pub(crate) fn effective_compute_advertisement(
    metadata: &ProviderMetadata,
) -> ProviderComputeAdvertisement {
    let mut advertisement = metadata.compute_advertisement.clone();
    for capability in metadata
        .capabilities
        .iter()
        .filter(|capability| capability.id.as_str() == COMPUTE_CAPABILITY_ID)
    {
        advertisement.capability.versions.insert(capability.version);
    }
    for (family, support) in &metadata.compute_operation_support {
        advertisement
            .operation_families
            .entry(*family)
            .or_insert_with(|| {
                OperationFamilySupport::from_operation_support(*family, support.clone())
            });
    }
    for (operation, support) in &metadata.compute_operation_schema_support {
        let family = initial_compute_operation_schemas()
            .get(operation)
            .map(|schema| schema.family)
            .unwrap_or(ComputeOperationFamily::DescriptorAndView);
        advertisement
            .operation_schemas
            .entry(operation.clone())
            .or_insert_with(|| {
                OperationSchemaSupport::from_operation_support(
                    operation.clone(),
                    family,
                    support.clone(),
                )
            });
    }
    for (kind, support) in &metadata.compute_data_movement_support {
        advertisement
            .data_movement
            .entry(*kind)
            .or_insert_with(|| DataMovementSupport::from_compute_support(*kind, support.clone()));
    }
    advertisement
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeOperationValidationResult {
    pub schema: ComputeOperationId,
    pub family: ComputeOperationFamily,
    pub input_count: usize,
    pub output_count: usize,
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

pub fn initial_compute_operation_schemas() -> BTreeMap<ComputeOperationId, ComputeOperationSchema> {
    let mut schemas = BTreeMap::new();
    let same_output = || {
        ComputeOperationOutputRule::new(
            1,
            ComputeOutputDTypeRule::SameAsInput(0),
            ComputeOutputShapeRule::SameAsInput(0),
        )
    };
    for id in [
        "tensor.transpose",
        "tensor.permute",
        "tensor.slice",
        "tensor.squeeze",
        "tensor.unsqueeze",
    ] {
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                id,
                ComputeOperationFamily::DescriptorAndView,
                ComputeOperationInputRule::exactly(1),
                same_output(),
            ),
        );
    }
    insert_schema(
        &mut schemas,
        ComputeOperationSchema::new(
            "tensor.reshape",
            ComputeOperationFamily::DescriptorAndView,
            ComputeOperationInputRule::exactly(1),
            ComputeOperationOutputRule::new(
                1,
                ComputeOutputDTypeRule::SameAsInput(0),
                ComputeOutputShapeRule::ExplicitAttribute("shape".into()),
            ),
        )
        .with_attribute(
            "shape",
            ComputeOperationAttributeRule::required(ComputeAttributeKind::Shape),
        ),
    );
    insert_schema(
        &mut schemas,
        ComputeOperationSchema::new(
            "tensor.broadcast",
            ComputeOperationFamily::DescriptorAndView,
            ComputeOperationInputRule::exactly(1).with_broadcast(),
            ComputeOperationOutputRule::new(
                1,
                ComputeOutputDTypeRule::SameAsInput(0),
                ComputeOutputShapeRule::ExplicitAttribute("shape".into()),
            ),
        )
        .with_attribute(
            "shape",
            ComputeOperationAttributeRule::required(ComputeAttributeKind::Shape),
        ),
    );
    for op in [
        "abs", "neg", "exp", "log", "sqrt", "recip", "sin", "cos", "tanh", "relu", "silu", "gelu",
        "erf", "floor", "ceil", "round",
    ] {
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                format!("elementwise.unary.{op}"),
                ComputeOperationFamily::Elementwise,
                ComputeOperationInputRule::exactly(1),
                same_output(),
            ),
        );
    }
    for op in ["add", "sub", "mul", "div", "maximum", "minimum"] {
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                format!("elementwise.binary.{op}"),
                ComputeOperationFamily::Elementwise,
                ComputeOperationInputRule::exactly(2)
                    .with_same_dtype()
                    .with_broadcast(),
                ComputeOperationOutputRule::new(
                    1,
                    ComputeOutputDTypeRule::SameAsInput(0),
                    ComputeOutputShapeRule::BroadcastInputs,
                ),
            ),
        );
    }
    for op in ["eq", "ne", "lt", "le", "gt", "ge"] {
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                format!("comparison.{op}"),
                ComputeOperationFamily::ComparisonAndSelection,
                ComputeOperationInputRule::exactly(2)
                    .with_same_dtype()
                    .with_broadcast(),
                ComputeOperationOutputRule::new(
                    1,
                    ComputeOutputDTypeRule::Boolean,
                    ComputeOutputShapeRule::BroadcastInputs,
                ),
            ),
        );
    }
    insert_schema(
        &mut schemas,
        ComputeOperationSchema::new(
            "selection.where",
            ComputeOperationFamily::ComparisonAndSelection,
            ComputeOperationInputRule::exactly(3)
                .with_boolean_input(0)
                .with_broadcast(),
            ComputeOperationOutputRule::new(
                1,
                ComputeOutputDTypeRule::SameAsInput(1),
                ComputeOutputShapeRule::BroadcastInputs,
            ),
        ),
    );
    for op in ["sum", "mean", "min", "max", "argmin", "argmax"] {
        let dtype = if matches!(op, "argmin" | "argmax") {
            ComputeOutputDTypeRule::IntegerIndex
        } else {
            ComputeOutputDTypeRule::SameAsInput(0)
        };
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                format!("reduction.{op}"),
                ComputeOperationFamily::Reduction,
                ComputeOperationInputRule::exactly(1),
                ComputeOperationOutputRule::new(
                    1,
                    dtype,
                    ComputeOutputShapeRule::Reduction {
                        axes_attribute: "axes".into(),
                        keep_dimensions_attribute: "keep-dimensions".into(),
                    },
                ),
            )
            .with_attribute(
                "axes",
                ComputeOperationAttributeRule::required(ComputeAttributeKind::Axes),
            )
            .with_attribute(
                "keep-dimensions",
                ComputeOperationAttributeRule::optional(ComputeAttributeKind::Boolean),
            ),
        );
    }
    for (id, shape_rule) in [
        (
            "linalg.matmul",
            ComputeOutputShapeRule::MatrixMultiplication,
        ),
        (
            "linalg.batched-matmul",
            ComputeOutputShapeRule::BatchedMatrixMultiplication,
        ),
    ] {
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                id,
                ComputeOperationFamily::LinearAlgebra,
                ComputeOperationInputRule::exactly(2).with_same_dtype(),
                ComputeOperationOutputRule::new(
                    1,
                    ComputeOutputDTypeRule::SameAsInput(0),
                    shape_rule,
                ),
            )
            .with_attribute(
                "transpose-a",
                ComputeOperationAttributeRule::optional(ComputeAttributeKind::Boolean),
            )
            .with_attribute(
                "transpose-b",
                ComputeOperationAttributeRule::optional(ComputeAttributeKind::Boolean),
            )
            .with_attribute(
                "accumulation-dtype",
                ComputeOperationAttributeRule::optional(ComputeAttributeKind::DType),
            )
            .with_attribute(
                "precision",
                ComputeOperationAttributeRule::optional(ComputeAttributeKind::String),
            ),
        );
    }
    for id in ["tensor.gather", "tensor.index-select"] {
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                id,
                ComputeOperationFamily::IndexingAndUpdate,
                ComputeOperationInputRule::exactly(2).with_integer_index_input(1),
                ComputeOperationOutputRule::new(
                    1,
                    ComputeOutputDTypeRule::SameAsInput(0),
                    ComputeOutputShapeRule::ProviderDefined,
                ),
            )
            .with_attribute(
                "axis",
                ComputeOperationAttributeRule::required(ComputeAttributeKind::Integer),
            ),
        );
    }
    for id in ["tensor.scatter", "tensor.scatter-add"] {
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                id,
                ComputeOperationFamily::IndexingAndUpdate,
                ComputeOperationInputRule::exactly(3)
                    .with_same_dtype()
                    .with_integer_index_input(1),
                ComputeOperationOutputRule::new(
                    1,
                    ComputeOutputDTypeRule::SameAsInput(0),
                    ComputeOutputShapeRule::SameAsInput(0),
                ),
            )
            .with_attribute(
                "axis",
                ComputeOperationAttributeRule::required(ComputeAttributeKind::Integer),
            )
            .with_provider_specific_semantics(),
        );
    }
    insert_schema(
        &mut schemas,
        ComputeOperationSchema::new(
            "tensor.concat",
            ComputeOperationFamily::IndexingAndUpdate,
            ComputeOperationInputRule::at_least(1).with_same_dtype(),
            ComputeOperationOutputRule::new(
                1,
                ComputeOutputDTypeRule::SameAsInput(0),
                ComputeOutputShapeRule::Concatenation {
                    axis_attribute: "axis".into(),
                },
            ),
        )
        .with_attribute(
            "axis",
            ComputeOperationAttributeRule::required(ComputeAttributeKind::Integer),
        ),
    );
    for id in ["random.uniform", "random.normal"] {
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                id,
                ComputeOperationFamily::RandomGeneration,
                ComputeOperationInputRule::exactly(0),
                ComputeOperationOutputRule::new(
                    1,
                    ComputeOutputDTypeRule::ExplicitAttribute("dtype".into()),
                    ComputeOutputShapeRule::ExplicitAttribute("shape".into()),
                ),
            )
            .with_attribute(
                "shape",
                ComputeOperationAttributeRule::required(ComputeAttributeKind::Shape),
            )
            .with_attribute(
                "dtype",
                ComputeOperationAttributeRule::required(ComputeAttributeKind::DType),
            )
            .with_attribute(
                "seed",
                ComputeOperationAttributeRule::optional(ComputeAttributeKind::Integer),
            ),
        );
    }
    schemas
}

fn insert_schema(
    schemas: &mut BTreeMap<ComputeOperationId, ComputeOperationSchema>,
    schema: ComputeOperationSchema,
) {
    schemas.insert(schema.id.clone(), schema);
}

pub(crate) fn validate_compute_operation_schema(
    operation: &ComputeOperationDescriptor,
) -> Result<Option<ComputeOperationValidationResult>, ComputeValidationError> {
    let Some(schema_id) = &operation.schema_id else {
        return Ok(None);
    };
    let schemas = initial_compute_operation_schemas();
    let schema = schemas
        .get(schema_id)
        .ok_or_else(|| ComputeValidationError::UnknownOperationSchema(schema_id.clone()))?;
    if schema.family != operation.family {
        return Err(ComputeValidationError::UnknownOperationFamily(
            operation.family.id().into(),
        ));
    }
    validate_operation_attributes(schema, operation)?;

    if operation.tensors.len() < schema.output_rule.output_count {
        return Err(ComputeValidationError::InvalidOutputDescriptor {
            operation: schema.id.clone(),
            reason: format!(
                "operation declares {} tensor descriptors but schema requires {} output(s)",
                operation.tensors.len(),
                schema.output_rule.output_count
            ),
        });
    }
    let input_count = operation.tensors.len() - schema.output_rule.output_count;
    validate_operation_arity(schema, input_count)?;
    let (inputs, outputs) = operation.tensors.split_at(input_count);
    validate_operation_input_rule(schema, inputs)?;
    validate_operation_output_rule(schema, operation, inputs, outputs)?;
    Ok(Some(ComputeOperationValidationResult {
        schema: schema.id.clone(),
        family: schema.family,
        input_count,
        output_count: outputs.len(),
    }))
}

fn validate_operation_attributes(
    schema: &ComputeOperationSchema,
    operation: &ComputeOperationDescriptor,
) -> Result<(), ComputeValidationError> {
    for name in operation.attributes.keys() {
        if !schema.attributes.contains_key(name) {
            return Err(ComputeValidationError::InvalidOperationAttribute {
                operation: schema.id.clone(),
                attribute: name.clone(),
                reason: "attribute is not defined by the operation schema".into(),
            });
        }
    }
    for (name, rule) in &schema.attributes {
        match operation.attributes.get(name) {
            Some(value) if value.kind() == rule.kind => {}
            Some(value) => {
                return Err(ComputeValidationError::InvalidOperationAttribute {
                    operation: schema.id.clone(),
                    attribute: name.clone(),
                    reason: format!("expected {:?}, found {:?}", rule.kind, value.kind()),
                });
            }
            None if rule.required => {
                return Err(ComputeValidationError::InvalidOperationAttribute {
                    operation: schema.id.clone(),
                    attribute: name.clone(),
                    reason: "required attribute is missing".into(),
                });
            }
            None => {}
        }
    }
    Ok(())
}

fn validate_operation_arity(
    schema: &ComputeOperationSchema,
    input_count: usize,
) -> Result<(), ComputeValidationError> {
    let too_few = input_count < schema.input_rule.min_inputs;
    let too_many = schema
        .input_rule
        .max_inputs
        .is_some_and(|max| input_count > max);
    if too_few || too_many {
        let expected = match schema.input_rule.max_inputs {
            Some(max) if max == schema.input_rule.min_inputs => max.to_string(),
            Some(max) => format!("{}..={max}", schema.input_rule.min_inputs),
            None => format!("at least {}", schema.input_rule.min_inputs),
        };
        return Err(ComputeValidationError::InvalidOperationArity {
            operation: schema.id.clone(),
            expected,
            found: input_count,
        });
    }
    Ok(())
}

fn validate_operation_input_rule(
    schema: &ComputeOperationSchema,
    inputs: &[TensorDescriptor],
) -> Result<(), ComputeValidationError> {
    for index in &schema.input_rule.boolean_inputs {
        match inputs.get(*index).map(|tensor| &tensor.dtype) {
            Some(DTypeDescriptor::Portable(ComputeDType::Boolean)) => {}
            _ => {
                return Err(ComputeValidationError::InvalidOperationAttribute {
                    operation: schema.id.clone(),
                    attribute: format!("input[{index}]"),
                    reason: "input must have boolean dtype".into(),
                });
            }
        }
    }
    for index in &schema.input_rule.integer_index_inputs {
        match inputs.get(*index).map(|tensor| &tensor.dtype) {
            Some(DTypeDescriptor::Portable(dtype)) if dtype.is_integer() => {}
            _ => {
                return Err(ComputeValidationError::InvalidOperationAttribute {
                    operation: schema.id.clone(),
                    attribute: format!("input[{index}]"),
                    reason: "index input must have an integer dtype".into(),
                });
            }
        }
    }
    if schema.input_rule.require_same_dtype {
        let Some(first) = inputs.first().map(|tensor| &tensor.dtype) else {
            return Ok(());
        };
        if inputs.iter().any(|tensor| &tensor.dtype != first) {
            return Err(ComputeValidationError::UnsupportedDType {
                family: schema.family,
                dtype: portable_dtype(first).unwrap_or(ComputeDType::UInt8),
            });
        }
    }
    if schema.input_rule.allow_broadcast && inputs.len() > 1 {
        broadcast_shape(inputs.iter().map(|tensor| &tensor.shape)).map_err(|reason| {
            ComputeValidationError::InvalidShape {
                reason: format!("operation schema '{}': {reason}", schema.id),
            }
        })?;
    }
    Ok(())
}

fn validate_operation_output_rule(
    schema: &ComputeOperationSchema,
    operation: &ComputeOperationDescriptor,
    inputs: &[TensorDescriptor],
    outputs: &[TensorDescriptor],
) -> Result<(), ComputeValidationError> {
    if outputs.len() != schema.output_rule.output_count {
        return Err(ComputeValidationError::InvalidOutputDescriptor {
            operation: schema.id.clone(),
            reason: format!(
                "expected {} output(s), found {}",
                schema.output_rule.output_count,
                outputs.len()
            ),
        });
    }
    for output in outputs {
        validate_output_dtype(schema, operation, inputs, output)?;
    }
    validate_output_shape(schema, operation, inputs, outputs)
}

fn validate_output_dtype(
    schema: &ComputeOperationSchema,
    operation: &ComputeOperationDescriptor,
    inputs: &[TensorDescriptor],
    output: &TensorDescriptor,
) -> Result<(), ComputeValidationError> {
    match &schema.output_rule.dtype {
        ComputeOutputDTypeRule::SameAsInput(index) => {
            let Some(input) = inputs.get(*index) else {
                return Ok(());
            };
            if output.dtype != input.dtype {
                return Err(ComputeValidationError::InvalidOutputDescriptor {
                    operation: schema.id.clone(),
                    reason: "output dtype must match input dtype".into(),
                });
            }
        }
        ComputeOutputDTypeRule::Boolean => {
            if output.dtype != DTypeDescriptor::Portable(ComputeDType::Boolean) {
                return Err(ComputeValidationError::InvalidOutputDescriptor {
                    operation: schema.id.clone(),
                    reason: "output dtype must be boolean".into(),
                });
            }
        }
        ComputeOutputDTypeRule::ExplicitAttribute(name) => {
            if let Some(ComputeOperationAttribute::DType(dtype)) = operation.attributes.get(name)
                && output.dtype != DTypeDescriptor::Portable(*dtype)
            {
                return Err(ComputeValidationError::InvalidOutputDescriptor {
                    operation: schema.id.clone(),
                    reason: format!("output dtype must match '{name}' attribute"),
                });
            }
        }
        ComputeOutputDTypeRule::IntegerIndex => {
            if !matches!(&output.dtype, DTypeDescriptor::Portable(dtype) if dtype.is_integer()) {
                return Err(ComputeValidationError::InvalidOutputDescriptor {
                    operation: schema.id.clone(),
                    reason: "output dtype must be an integer index dtype".into(),
                });
            }
        }
        ComputeOutputDTypeRule::ProviderDefined => {}
    }
    Ok(())
}

fn validate_output_shape(
    schema: &ComputeOperationSchema,
    operation: &ComputeOperationDescriptor,
    inputs: &[TensorDescriptor],
    outputs: &[TensorDescriptor],
) -> Result<(), ComputeValidationError> {
    let Some(output) = outputs.first() else {
        return Ok(());
    };
    let expected = match &schema.output_rule.shape {
        ComputeOutputShapeRule::SameAsInput(index) => inputs.get(*index).map(|t| t.shape.clone()),
        ComputeOutputShapeRule::ExplicitAttribute(name) => match operation.attributes.get(name) {
            Some(ComputeOperationAttribute::Shape(shape)) => Some(shape.clone()),
            _ => None,
        },
        ComputeOutputShapeRule::BroadcastInputs => {
            Some(broadcast_shape(inputs.iter().map(|t| &t.shape))?)
        }
        ComputeOutputShapeRule::Reduction {
            axes_attribute,
            keep_dimensions_attribute,
        } => Some(reduction_shape(
            &schema.id,
            inputs.first(),
            operation.attributes.get(axes_attribute),
            operation.attributes.get(keep_dimensions_attribute),
        )?),
        ComputeOutputShapeRule::MatrixMultiplication => Some(matmul_shape(&schema.id, inputs)?),
        ComputeOutputShapeRule::BatchedMatrixMultiplication => {
            Some(batched_matmul_shape(&schema.id, inputs)?)
        }
        ComputeOutputShapeRule::Concatenation { axis_attribute } => Some(concat_shape(
            &schema.id,
            inputs,
            operation.attributes.get(axis_attribute),
        )?),
        ComputeOutputShapeRule::ProviderDefined => None,
    };
    if let Some(expected) = expected
        && output.shape != expected
    {
        return Err(ComputeValidationError::InvalidOutputDescriptor {
            operation: schema.id.clone(),
            reason: format!(
                "output shape {:?} does not match expected {:?}",
                output.shape.dimensions, expected.dimensions
            ),
        });
    }
    Ok(())
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

fn portable_dtype(dtype: &DTypeDescriptor) -> Option<ComputeDType> {
    match dtype {
        DTypeDescriptor::Portable(dtype) => Some(*dtype),
        DTypeDescriptor::ProviderSpecific { .. } => None,
    }
}

fn broadcast_shape<'a>(
    shapes: impl IntoIterator<Item = &'a ShapeDescriptor>,
) -> Result<ShapeDescriptor, ComputeValidationError> {
    let shapes = shapes.into_iter().collect::<Vec<_>>();
    let max_rank = shapes
        .iter()
        .map(|shape| shape.dimensions.len())
        .max()
        .unwrap_or(0);
    let mut result = vec![1_u64; max_rank];
    for shape in shapes {
        for (offset, dimension) in shape.dimensions.iter().rev().enumerate() {
            let index = max_rank - 1 - offset;
            let current = result[index];
            if current == 1 {
                result[index] = *dimension;
            } else if *dimension == 1 || current == *dimension {
                continue;
            } else {
                return Err(ComputeValidationError::InvalidShape {
                    reason: format!(
                        "dimensions {current} and {dimension} are not broadcast-compatible"
                    ),
                });
            }
        }
    }
    Ok(ShapeDescriptor::new(result))
}

fn reduction_shape(
    operation: &ComputeOperationId,
    input: Option<&TensorDescriptor>,
    axes: Option<&ComputeOperationAttribute>,
    keep_dimensions: Option<&ComputeOperationAttribute>,
) -> Result<ShapeDescriptor, ComputeValidationError> {
    let input = input.ok_or_else(|| ComputeValidationError::InvalidOperationArity {
        operation: operation.clone(),
        expected: "1".into(),
        found: 0,
    })?;
    let axes = match axes {
        Some(ComputeOperationAttribute::Axes(axes)) => axes,
        _ => {
            return Err(ComputeValidationError::InvalidOperationAttribute {
                operation: operation.clone(),
                attribute: "axes".into(),
                reason: "axes attribute is required".into(),
            });
        }
    };
    let keep = matches!(
        keep_dimensions,
        Some(ComputeOperationAttribute::Boolean(true))
    );
    let rank = input.shape.dimensions.len() as u64;
    for axis in axes {
        if *axis >= rank {
            return Err(ComputeValidationError::InvalidShape {
                reason: format!("reduction axis {axis} is outside rank {rank}"),
            });
        }
    }
    let axis_set = axes.iter().copied().collect::<BTreeSet<_>>();
    let dimensions = input
        .shape
        .dimensions
        .iter()
        .enumerate()
        .filter_map(|(index, dimension)| {
            if axis_set.contains(&(index as u64)) {
                keep.then_some(1)
            } else {
                Some(*dimension)
            }
        })
        .collect::<Vec<_>>();
    Ok(ShapeDescriptor::new(dimensions))
}

fn matmul_shape(
    operation: &ComputeOperationId,
    inputs: &[TensorDescriptor],
) -> Result<ShapeDescriptor, ComputeValidationError> {
    let [lhs, rhs] = inputs else {
        return Err(ComputeValidationError::InvalidOperationArity {
            operation: operation.clone(),
            expected: "2".into(),
            found: inputs.len(),
        });
    };
    if lhs.shape.dimensions.len() != 2 || rhs.shape.dimensions.len() != 2 {
        return Err(ComputeValidationError::InvalidShape {
            reason: "matrix multiplication requires rank-2 inputs".into(),
        });
    }
    if lhs.shape.dimensions[1] != rhs.shape.dimensions[0] {
        return Err(ComputeValidationError::InvalidShape {
            reason: "matrix multiplication inner dimensions are incompatible".into(),
        });
    }
    Ok(ShapeDescriptor::new([
        lhs.shape.dimensions[0],
        rhs.shape.dimensions[1],
    ]))
}

fn batched_matmul_shape(
    operation: &ComputeOperationId,
    inputs: &[TensorDescriptor],
) -> Result<ShapeDescriptor, ComputeValidationError> {
    let [lhs, rhs] = inputs else {
        return Err(ComputeValidationError::InvalidOperationArity {
            operation: operation.clone(),
            expected: "2".into(),
            found: inputs.len(),
        });
    };
    if lhs.shape.dimensions.len() < 3 || rhs.shape.dimensions.len() < 3 {
        return Err(ComputeValidationError::InvalidShape {
            reason: "batched matrix multiplication requires rank >= 3 inputs".into(),
        });
    }
    let lhs_rank = lhs.shape.dimensions.len();
    let rhs_rank = rhs.shape.dimensions.len();
    if lhs.shape.dimensions[lhs_rank - 1] != rhs.shape.dimensions[rhs_rank - 2] {
        return Err(ComputeValidationError::InvalidShape {
            reason: "batched matrix multiplication inner dimensions are incompatible".into(),
        });
    }
    let mut batch = broadcast_shape(
        [
            ShapeDescriptor::new(lhs.shape.dimensions[..lhs_rank - 2].to_vec()),
            ShapeDescriptor::new(rhs.shape.dimensions[..rhs_rank - 2].to_vec()),
        ]
        .iter(),
    )?
    .dimensions;
    batch.push(lhs.shape.dimensions[lhs_rank - 2]);
    batch.push(rhs.shape.dimensions[rhs_rank - 1]);
    Ok(ShapeDescriptor::new(batch))
}

fn concat_shape(
    operation: &ComputeOperationId,
    inputs: &[TensorDescriptor],
    axis: Option<&ComputeOperationAttribute>,
) -> Result<ShapeDescriptor, ComputeValidationError> {
    let Some(first) = inputs.first() else {
        return Err(ComputeValidationError::InvalidOperationArity {
            operation: operation.clone(),
            expected: "at least 1".into(),
            found: 0,
        });
    };
    let axis = match axis {
        Some(ComputeOperationAttribute::Integer(axis)) if *axis >= 0 => *axis as usize,
        _ => {
            return Err(ComputeValidationError::InvalidOperationAttribute {
                operation: operation.clone(),
                attribute: "axis".into(),
                reason: "axis must be a non-negative integer".into(),
            });
        }
    };
    if axis >= first.shape.dimensions.len() {
        return Err(ComputeValidationError::InvalidShape {
            reason: "concatenation axis is outside input rank".into(),
        });
    }
    let mut dimensions = first.shape.dimensions.clone();
    for input in &inputs[1..] {
        if input.shape.dimensions.len() != dimensions.len() {
            return Err(ComputeValidationError::InvalidShape {
                reason: "all concatenation inputs must have the same rank".into(),
            });
        }
        for (index, dimension) in input.shape.dimensions.iter().enumerate() {
            if index == axis {
                dimensions[index] = dimensions[index].checked_add(*dimension).ok_or(
                    ComputeValidationError::SizeOverflow {
                        reason: "concatenated dimension overflows u64".into(),
                    },
                )?;
            } else if dimensions[index] != *dimension {
                return Err(ComputeValidationError::InvalidShape {
                    reason: "non-concatenated dimensions must match".into(),
                });
            }
        }
    }
    Ok(ShapeDescriptor::new(dimensions))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeOperationRequest {
    pub family_id: String,
    pub dtype: Option<ComputeDType>,
    pub layout: Option<ComputeLayout>,
    pub precision: Option<ComputePrecision>,
    pub tensors: Vec<TensorDescriptor>,
}
impl ComputeOperationRequest {
    pub fn new(family_id: impl Into<String>) -> Self {
        Self {
            family_id: family_id.into(),
            dtype: None,
            layout: None,
            precision: None,
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
    pub fn with_tensor(mut self, tensor: TensorDescriptor) -> Self {
        self.tensors.push(tensor);
        self
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
pub enum ComputeInputValue {
    TensorResource(TensorResourceDescriptor),
    TensorDescriptor(TensorDescriptor),
    Constant(TensorDescriptor),
}
impl ComputeInputValue {
    pub(crate) fn descriptor(&self) -> &TensorDescriptor {
        match self {
            Self::TensorResource(resource) => &resource.descriptor,
            Self::TensorDescriptor(descriptor) | Self::Constant(descriptor) => descriptor,
        }
    }
    pub(crate) fn affinity(&self) -> Option<&ResourceAffinity> {
        match self {
            Self::TensorResource(resource) => Some(&resource.affinity),
            Self::TensorDescriptor(_) | Self::Constant(_) => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeInput {
    pub id: ComputeInputId,
    pub value: ComputeInputValue,
}
impl ComputeInput {
    pub fn new(id: ComputeInputId, value: ComputeInputValue) -> Self {
        Self { id, value }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeValueRef {
    Input(ComputeInputId),
    NodeOutput {
        node: ComputeNodeId,
        output: ComputeOutputId,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeNodeOutput {
    pub id: ComputeOutputId,
    pub descriptor: TensorDescriptor,
}
impl ComputeNodeOutput {
    pub fn new(id: ComputeOutputId, descriptor: TensorDescriptor) -> Self {
        Self { id, descriptor }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeNode {
    pub id: ComputeNodeId,
    pub operation: ComputeOperationDescriptor,
    pub inputs: Vec<ComputeValueRef>,
    pub outputs: Vec<ComputeNodeOutput>,
}
impl ComputeNode {
    pub fn new(id: ComputeNodeId, operation: ComputeOperationDescriptor) -> Self {
        Self {
            id,
            operation,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }
    pub fn with_input(mut self, input: ComputeValueRef) -> Self {
        self.inputs.push(input);
        self
    }
    pub fn with_output(mut self, output: ComputeNodeOutput) -> Self {
        self.outputs.push(output);
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeOutput {
    pub id: ComputeOutputId,
    pub source: ComputeValueRef,
}
impl ComputeOutput {
    pub fn new(id: ComputeOutputId, source: ComputeValueRef) -> Self {
        Self { id, source }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeGraph {
    pub id: ComputeGraphId,
    pub inputs: Vec<ComputeInput>,
    pub nodes: Vec<ComputeNode>,
    pub outputs: Vec<ComputeOutput>,
}
impl ComputeGraph {
    pub fn new(id: ComputeGraphId) -> Self {
        Self {
            id,
            inputs: Vec::new(),
            nodes: Vec::new(),
            outputs: Vec::new(),
        }
    }
    pub fn with_input(mut self, input: ComputeInput) -> Self {
        self.inputs.push(input);
        self
    }
    pub fn with_node(mut self, node: ComputeNode) -> Self {
        self.nodes.push(node);
        self
    }
    pub fn with_output(mut self, output: ComputeOutput) -> Self {
        self.outputs.push(output);
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComputeSubmissionState {
    Pending,
    Running,
    Completed,
    Cancelled,
    Failed,
}
impl ComputeSubmissionState {
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled | Self::Failed)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeExecutionResult {
    pub state: ComputeSubmissionState,
    pub outputs: Vec<TensorResourceDescriptor>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeSubmission {
    pub graph: ComputeGraphId,
    pub provider: ProviderBinding,
    pub affinity: ResourceAffinity,
    state: ComputeSubmissionState,
    result: Option<ComputeExecutionResult>,
}
impl ComputeSubmission {
    pub fn new(
        graph: ComputeGraphId,
        provider: ProviderBinding,
        affinity: ResourceAffinity,
    ) -> Self {
        Self {
            graph,
            provider,
            affinity,
            state: ComputeSubmissionState::Pending,
            result: None,
        }
    }
    pub fn state(&self) -> ComputeSubmissionState {
        self.state
    }
    pub fn start(&mut self) -> Result<(), ComputeValidationError> {
        if self.state != ComputeSubmissionState::Pending {
            return Err(ComputeValidationError::InvalidState {
                reason: format!("cannot start submission from {:?} state", self.state),
            });
        }
        self.state = ComputeSubmissionState::Running;
        Ok(())
    }
    pub fn complete(
        &mut self,
        outputs: Vec<TensorResourceDescriptor>,
    ) -> Result<(), ComputeValidationError> {
        if self.state.is_terminal() {
            return Err(ComputeValidationError::InvalidState {
                reason: "submission is already terminal".into(),
            });
        }
        self.state = ComputeSubmissionState::Completed;
        self.result = Some(ComputeExecutionResult {
            state: self.state,
            outputs,
        });
        Ok(())
    }
    pub fn cancel(&mut self) -> Result<(), ComputeValidationError> {
        if self.state.is_terminal() {
            return Err(ComputeValidationError::InvalidState {
                reason: "submission is already terminal".into(),
            });
        }
        self.state = ComputeSubmissionState::Cancelled;
        self.result = Some(ComputeExecutionResult {
            state: self.state,
            outputs: Vec::new(),
        });
        Ok(())
    }
    pub fn fail(&mut self) -> Result<(), ComputeValidationError> {
        if self.state.is_terminal() {
            return Err(ComputeValidationError::InvalidState {
                reason: "submission is already terminal".into(),
            });
        }
        self.state = ComputeSubmissionState::Failed;
        self.result = Some(ComputeExecutionResult {
            state: self.state,
            outputs: Vec::new(),
        });
        Ok(())
    }
    pub fn result(&self) -> Option<&ComputeExecutionResult> {
        self.result.as_ref()
    }
}
pub struct ComputeGraphValidationReport {
    pub provider: ProviderBinding,
    pub graph: ComputeGraphId,
    pub node_count: usize,
    pub input_count: usize,
    pub output_count: usize,
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

pub(crate) fn redact_backend_diagnostic(message: &str) -> String {
    let contains_native_handle = message.contains("0x") || message.contains("handle=");
    let contains_path = message.contains('\\') || message.contains('/');
    if contains_native_handle || contains_path {
        "[redacted backend diagnostic]".into()
    } else {
        message.into()
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

pub(crate) fn ensure_non_empty_id(kind: &str, value: &str) -> Result<(), ComputeValidationError> {
    if value.trim().is_empty() {
        return Err(ComputeValidationError::InvalidGraph {
            reason: format!("{kind} identifier must not be empty"),
        });
    }
    Ok(())
}

pub(crate) fn insert_unique<T: Ord + Clone + fmt::Display>(
    ids: &mut BTreeSet<T>,
    kind: &str,
    id: &T,
) -> Result<(), ComputeValidationError> {
    if !ids.insert(id.clone()) {
        return Err(ComputeValidationError::InvalidGraph {
            reason: format!("duplicate {kind} identifier '{id}'"),
        });
    }
    Ok(())
}

pub(crate) fn resolve_compute_value_descriptor<'a>(
    current_node: Option<&ComputeNodeId>,
    value: &ComputeValueRef,
    input_descriptors: &'a BTreeMap<ComputeInputId, TensorDescriptor>,
    output_descriptors: &'a BTreeMap<(ComputeNodeId, ComputeOutputId), TensorDescriptor>,
    completed_nodes: &BTreeSet<ComputeNodeId>,
) -> Result<&'a TensorDescriptor, ComputeValidationError> {
    match value {
        ComputeValueRef::Input(input) => {
            input_descriptors
                .get(input)
                .ok_or_else(|| ComputeValidationError::MissingInput {
                    input: input.clone(),
                })
        }
        ComputeValueRef::NodeOutput { node, output } => {
            if !completed_nodes.contains(node) {
                return Err(ComputeValidationError::CyclicGraph {
                    node: current_node.cloned().unwrap_or_else(|| node.clone()),
                    depends_on: node.clone(),
                });
            }
            output_descriptors
                .get(&(node.clone(), output.clone()))
                .ok_or_else(|| ComputeValidationError::MissingOutput {
                    node: node.clone(),
                    output: output.clone(),
                })
        }
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

impl From<ComputePlanningError> for ComputeError {
    fn from(error: ComputePlanningError) -> Self {
        let message = error.to_string();
        match error {
            ComputePlanningError::PlanningFailed { .. } => ComputeError::new(
                ComputeErrorCode::PlanningFailed,
                ComputeErrorPhase::Planning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::NoCompatibleProvider { capability } => ComputeError::new(
                ComputeErrorCode::NoCompatibleProvider,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_capability(capability))
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::NoCompatibleDevice { provider } => ComputeError::new(
                ComputeErrorCode::NoCompatibleDevice,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_provider(provider))
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::PolicyRejectedProvider { capability, .. } => ComputeError::new(
                ComputeErrorCode::PolicyRejectedProvider,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_capability(capability))
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::UnsupportedOperation(operation) => ComputeError::new(
                ComputeErrorCode::UnsupportedOperation,
                ComputeErrorPhase::Planning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(
                ComputeDiagnostic::new()
                    .with_backend_message(format!("unsupported operation schema: {operation}")),
            )
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::UnsupportedDType(_) => ComputeError::new(
                ComputeErrorCode::UnsupportedDType,
                ComputeErrorPhase::Planning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::UnsupportedLayout(_) => ComputeError::new(
                ComputeErrorCode::UnsupportedLayout,
                ComputeErrorPhase::Planning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::UnsupportedPrecisionPolicy(_) => ComputeError::new(
                ComputeErrorCode::UnsupportedOperation,
                ComputeErrorPhase::Planning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::IncompatibleResourceAffinity(error) => ComputeError::from(error),
            ComputePlanningError::UnresolvedAffinityGroup(_) => ComputeError::new(
                ComputeErrorCode::AffinityGroupMismatch,
                ComputeErrorPhase::Planning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::MemoryPlanFailed(error) => ComputeError::from(error),
            ComputePlanningError::DataMovementRequired { .. } => ComputeError::new(
                ComputeErrorCode::DataMovementRequired,
                ComputeErrorPhase::DataMovement,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::ExplicitTransferRequired),
            ComputePlanningError::UnsupportedTransfer { .. } => ComputeError::new(
                ComputeErrorCode::UnsupportedTransfer,
                ComputeErrorPhase::DataMovement,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::ExplicitTransferRequired),
            ComputePlanningError::MaterializationRequired { .. } => ComputeError::new(
                ComputeErrorCode::MaterializationRequired,
                ComputeErrorPhase::Materialization,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::ExplicitMaterializationRequired),
            ComputePlanningError::ProviderUnavailable(provider) => ComputeError::new(
                ComputeErrorCode::ProviderUnavailable,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_provider(provider))
            .with_recovery_hint(RecoveryHint::RetryBeforeState),
            ComputePlanningError::DeviceUnavailable(device) => ComputeError::new(
                ComputeErrorCode::DeviceUnavailable,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_device(device))
            .with_recovery_hint(RecoveryHint::RetryBeforeState),
            ComputePlanningError::InvalidExecutionPlan { .. } => ComputeError::new(
                ComputeErrorCode::InvalidExecutionPlan,
                ComputeErrorPhase::Planning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::NotRetryable),
        }
    }
}

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

/// Returns the canonical hardware-independent Compute capability declaration.
///
/// Providers add this value to their metadata to advertise support for the
/// `magnetar:compute/run@2.0.0` WIT contract.
pub fn compute_capability() -> Capability {
    Capability::new(
        CapabilityId::new(COMPUTE_CAPABILITY_ID),
        COMPUTE_CAPABILITY_VERSION,
        CapabilityDescriptor::new("coarse provider-owned graph execution").with_contract(
            WitInterface::new(
                COMPUTE_WIT_INTERFACE,
                COMPUTE_CAPABILITY_VERSION.to_string(),
            ),
        ),
    )
}
