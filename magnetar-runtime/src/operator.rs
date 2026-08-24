//! Portable Operator contract.
//!
//! Operators describe inference semantics. They are intentionally separate from
//! Provider kernels: Providers may advertise and execute compatible operators,
//! but operator identity, attributes, tensor contracts, and validation stay
//! Runtime-owned and platform-neutral.

use crate::{
    CapabilityVersion, ComputeDType, DTypeDescriptor, LayoutDescriptor, ResourceAffinity,
    TensorDescriptor,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

pub const OPERATOR_CATALOG_VERSION: CapabilityVersion = CapabilityVersion::new(1, 0, 0);
pub const OPERATOR_NAMESPACE: &str = "magnetar:operator";

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum OperatorFamily {
    Tensor,
    LinearAlgebra,
    Normalization,
    PositionEncoding,
    Attention,
    Activation,
    Quantization,
    Layout,
    SamplingSupport,
    Control,
}

impl OperatorFamily {
    pub const ALL: [Self; 10] = [
        Self::Tensor,
        Self::LinearAlgebra,
        Self::Normalization,
        Self::PositionEncoding,
        Self::Attention,
        Self::Activation,
        Self::Quantization,
        Self::Layout,
        Self::SamplingSupport,
        Self::Control,
    ];

    pub const fn id(self) -> &'static str {
        match self {
            Self::Tensor => "tensor",
            Self::LinearAlgebra => "linear-algebra",
            Self::Normalization => "normalization",
            Self::PositionEncoding => "position-encoding",
            Self::Attention => "attention",
            Self::Activation => "activation",
            Self::Quantization => "quantization",
            Self::Layout => "layout",
            Self::SamplingSupport => "sampling-support",
            Self::Control => "control",
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct OperatorId {
    namespace: String,
    name: String,
    version: u32,
    family: OperatorFamily,
}

impl OperatorId {
    pub fn new(
        namespace: impl Into<String>,
        name: impl Into<String>,
        version: u32,
        family: OperatorFamily,
    ) -> Self {
        Self {
            namespace: namespace.into(),
            name: name.into(),
            version,
            family,
        }
    }

    pub fn magnetar(name: impl Into<String>, version: u32, family: OperatorFamily) -> Self {
        Self::new(OPERATOR_NAMESPACE, name, version, family)
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn version(&self) -> u32 {
        self.version
    }

    pub const fn family(&self) -> OperatorFamily {
        self.family
    }
}

impl fmt::Display for OperatorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}@{}", self.namespace, self.name, self.version)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TensorRole {
    Input,
    Output,
    Storage,
    Compute,
    Accumulation,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TensorLayoutKind {
    Contiguous,
    Strided,
    Blocked,
    Paged,
    ProviderOpaque,
    QuantizedPacked,
    AttentionSpecific,
    BrowserCompatible,
}

impl TensorLayoutKind {
    pub const fn component_visible(self) -> bool {
        !matches!(self, Self::ProviderOpaque)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperatorDTypeContract {
    pub role: TensorRole,
    pub supported: BTreeSet<ComputeDType>,
}

impl OperatorDTypeContract {
    pub fn new(role: TensorRole, supported: impl IntoIterator<Item = ComputeDType>) -> Self {
        Self {
            role,
            supported: supported.into_iter().collect(),
        }
    }

    pub fn accepts(&self, dtype: ComputeDType) -> bool {
        self.supported.is_empty() || self.supported.contains(&dtype)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperatorLayoutContract {
    pub supported: BTreeSet<TensorLayoutKind>,
    pub explicit_conversion_required: bool,
}

impl OperatorLayoutContract {
    pub fn new(supported: impl IntoIterator<Item = TensorLayoutKind>) -> Self {
        Self {
            supported: supported.into_iter().collect(),
            explicit_conversion_required: true,
        }
    }

    pub fn accepts(&self, layout: TensorLayoutKind) -> bool {
        self.supported.is_empty() || self.supported.contains(&layout)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShapeRule {
    Any,
    SameRank,
    SameShape,
    Matmul,
    Rank(u64),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OperatorMemoryBehavior {
    pub reads_input: bool,
    pub writes_output: bool,
    pub mutates_input: bool,
    pub aliases_output: bool,
    pub requires_workspace: bool,
    pub supports_in_place: bool,
    pub requires_host_visible: bool,
    pub requires_device_resident: bool,
    pub requires_pinned_memory: bool,
    pub supports_streaming_output: bool,
    pub supports_paged_kv_cache: bool,
}

impl OperatorMemoryBehavior {
    pub const fn pure() -> Self {
        Self {
            reads_input: true,
            writes_output: true,
            mutates_input: false,
            aliases_output: false,
            requires_workspace: false,
            supports_in_place: false,
            requires_host_visible: false,
            requires_device_resident: false,
            requires_pinned_memory: false,
            supports_streaming_output: false,
            supports_paged_kv_cache: false,
        }
    }

    pub const fn with_workspace(mut self) -> Self {
        self.requires_workspace = true;
        self
    }

    pub const fn with_paged_kv_cache(mut self) -> Self {
        self.supports_paged_kv_cache = true;
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperatorDeterminism {
    pub deterministic: bool,
    pub influenced_by_dtype: bool,
    pub influenced_by_provider: bool,
    pub influenced_by_device: bool,
    pub influenced_by_kernel: bool,
    pub influenced_by_reduction: bool,
    pub influenced_by_layout: bool,
}

impl Default for OperatorDeterminism {
    fn default() -> Self {
        Self {
            deterministic: true,
            influenced_by_dtype: true,
            influenced_by_provider: true,
            influenced_by_device: true,
            influenced_by_kernel: true,
            influenced_by_reduction: false,
            influenced_by_layout: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum OperatorAttributeKind {
    Boolean,
    Integer,
    Float,
    String,
    DType,
    Layout,
}

#[derive(Clone, Debug, PartialEq)]
pub enum OperatorAttributeValue {
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
    DType(ComputeDType),
    Layout(TensorLayoutKind),
}

impl OperatorAttributeValue {
    pub const fn kind(&self) -> OperatorAttributeKind {
        match self {
            Self::Boolean(_) => OperatorAttributeKind::Boolean,
            Self::Integer(_) => OperatorAttributeKind::Integer,
            Self::Float(_) => OperatorAttributeKind::Float,
            Self::String(_) => OperatorAttributeKind::String,
            Self::DType(_) => OperatorAttributeKind::DType,
            Self::Layout(_) => OperatorAttributeKind::Layout,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperatorAttributeRule {
    pub kind: OperatorAttributeKind,
    pub required: bool,
}

impl OperatorAttributeRule {
    pub const fn required(kind: OperatorAttributeKind) -> Self {
        Self {
            kind,
            required: true,
        }
    }

    pub const fn optional(kind: OperatorAttributeKind) -> Self {
        Self {
            kind,
            required: false,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OperatorAttributeSchema {
    pub rules: BTreeMap<String, OperatorAttributeRule>,
}

impl OperatorAttributeSchema {
    pub fn with_rule(mut self, name: impl Into<String>, rule: OperatorAttributeRule) -> Self {
        self.rules.insert(name.into(), rule);
        self
    }

    pub fn validate(
        &self,
        attributes: &BTreeMap<String, OperatorAttributeValue>,
    ) -> Result<(), OperatorError> {
        for forbidden in ["provider", "provider_id", "device", "device_id", "kernel"] {
            if attributes.contains_key(forbidden) {
                return Err(OperatorError::OperatorAttributeInvalid {
                    attribute: forbidden.into(),
                    reason: "operator attributes must not select Provider, Device, or kernel"
                        .into(),
                });
            }
        }
        for (name, rule) in &self.rules {
            match attributes.get(name) {
                Some(value) if value.kind() == rule.kind => {}
                Some(_) => {
                    return Err(OperatorError::OperatorAttributeInvalid {
                        attribute: name.clone(),
                        reason: "attribute kind mismatch".into(),
                    });
                }
                None if rule.required => {
                    return Err(OperatorError::OperatorAttributeInvalid {
                        attribute: name.clone(),
                        reason: "required attribute missing".into(),
                    });
                }
                None => {}
            }
        }
        for name in attributes.keys() {
            if !self.rules.contains_key(name) {
                return Err(OperatorError::OperatorAttributeInvalid {
                    attribute: name.clone(),
                    reason: "attribute is not defined by schema".into(),
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperatorSpec {
    pub id: OperatorId,
    pub inputs: usize,
    pub outputs: usize,
    pub attributes: OperatorAttributeSchema,
    pub shape_rule: ShapeRule,
    pub dtype_contracts: Vec<OperatorDTypeContract>,
    pub layout_contract: OperatorLayoutContract,
    pub memory: OperatorMemoryBehavior,
    pub determinism: OperatorDeterminism,
}

impl OperatorSpec {
    pub fn new(id: OperatorId, inputs: usize, outputs: usize) -> Self {
        Self {
            id,
            inputs,
            outputs,
            attributes: OperatorAttributeSchema::default(),
            shape_rule: ShapeRule::Any,
            dtype_contracts: Vec::new(),
            layout_contract: OperatorLayoutContract::new([TensorLayoutKind::Contiguous]),
            memory: OperatorMemoryBehavior::pure(),
            determinism: OperatorDeterminism::default(),
        }
    }

    pub fn with_attributes(mut self, attributes: OperatorAttributeSchema) -> Self {
        self.attributes = attributes;
        self
    }

    pub fn with_shape_rule(mut self, shape_rule: ShapeRule) -> Self {
        self.shape_rule = shape_rule;
        self
    }

    pub fn with_dtype_contract(mut self, contract: OperatorDTypeContract) -> Self {
        self.dtype_contracts.push(contract);
        self
    }

    pub fn with_layouts(mut self, layouts: impl IntoIterator<Item = TensorLayoutKind>) -> Self {
        self.layout_contract = OperatorLayoutContract::new(layouts);
        self
    }

    pub const fn with_memory(mut self, memory: OperatorMemoryBehavior) -> Self {
        self.memory = memory;
        self
    }

    pub fn validate_invocation(
        &self,
        inputs: &[TensorDescriptor],
        outputs: &[TensorDescriptor],
        attributes: &BTreeMap<String, OperatorAttributeValue>,
    ) -> Result<(), OperatorError> {
        self.attributes.validate(attributes)?;
        if inputs.len() != self.inputs {
            return Err(OperatorError::InputArityInvalid {
                expected: self.inputs,
                actual: inputs.len(),
            });
        }
        if outputs.len() != self.outputs {
            return Err(OperatorError::OutputArityInvalid {
                expected: self.outputs,
                actual: outputs.len(),
            });
        }
        validate_shape_rule(&self.shape_rule, inputs, outputs)?;
        for tensor in inputs.iter().chain(outputs) {
            validate_dtype_contracts(&self.dtype_contracts, tensor)?;
            let layout = layout_kind(&tensor.layout);
            if !layout.component_visible() || !self.layout_contract.accepts(layout) {
                return Err(OperatorError::LayoutUnsupported { layout });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperatorCatalog {
    pub version: CapabilityVersion,
    pub operators: BTreeMap<OperatorId, OperatorSpec>,
}

impl OperatorCatalog {
    pub fn new(version: CapabilityVersion) -> Self {
        Self {
            version,
            operators: BTreeMap::new(),
        }
    }

    pub fn with_operator(mut self, spec: OperatorSpec) -> Self {
        self.operators.insert(spec.id.clone(), spec);
        self
    }

    pub fn get(&self, id: &OperatorId) -> Result<&OperatorSpec, OperatorError> {
        self.operators.get(id).ok_or_else(|| {
            if self
                .operators
                .keys()
                .any(|known| known.namespace == id.namespace && known.name == id.name)
            {
                OperatorError::OperatorVersionUnsupported {
                    operator: id.clone(),
                }
            } else {
                OperatorError::OperatorNotFound {
                    operator: id.clone(),
                }
            }
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum OperatorObservationKind {
    OperatorPlanned,
    DataMovementInserted,
    DTypeConversionInserted,
    LayoutConversionInserted,
    WorkspaceRequested,
    OperatorExecutionStarted,
    OperatorExecutionCompleted,
    OperatorExecutionFailed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperatorObservation {
    pub kind: OperatorObservationKind,
    pub operator: OperatorId,
    pub redacted_metadata: BTreeMap<String, String>,
}

impl OperatorObservation {
    pub fn new(kind: OperatorObservationKind, operator: OperatorId) -> Self {
        Self {
            kind,
            operator,
            redacted_metadata: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OperatorError {
    OperatorNotFound {
        operator: OperatorId,
    },
    OperatorVersionUnsupported {
        operator: OperatorId,
    },
    OperatorAttributeInvalid {
        attribute: String,
        reason: String,
    },
    InputArityInvalid {
        expected: usize,
        actual: usize,
    },
    OutputArityInvalid {
        expected: usize,
        actual: usize,
    },
    ShapeMismatch {
        reason: String,
    },
    ShapeUnsupported {
        reason: String,
    },
    DTypeUnsupported {
        dtype: ComputeDType,
    },
    DTypeConversionRequired {
        from: ComputeDType,
        to: ComputeDType,
    },
    DTypeConversionUnsupported {
        from: ComputeDType,
        to: ComputeDType,
    },
    LayoutUnsupported {
        layout: TensorLayoutKind,
    },
    LayoutConversionRequired {
        from: TensorLayoutKind,
        to: TensorLayoutKind,
    },
    LayoutConversionUnsupported {
        from: TensorLayoutKind,
        to: TensorLayoutKind,
    },
    MemoryBehaviorUnsupported {
        reason: String,
    },
    WorkspaceUnavailable {
        bytes: u64,
    },
    ResourceAffinityConflict {
        reason: String,
    },
    ProviderCapabilityUnavailable {
        capability: String,
    },
    KernelUnavailable {
        operator: OperatorId,
    },
    GraphValidationFailed {
        reason: String,
    },
    GraphPlanningFailed {
        reason: String,
    },
    GraphExecutionFailed {
        reason: String,
    },
    BrowserFeatureUnsupported {
        feature: String,
    },
    InternalOperator {
        reason: String,
    },
}

impl fmt::Display for OperatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OperatorNotFound { operator } => write!(f, "operator not found: {operator}"),
            Self::OperatorVersionUnsupported { operator } => {
                write!(f, "operator version unsupported: {operator}")
            }
            Self::OperatorAttributeInvalid { attribute, reason } => {
                write!(f, "operator attribute invalid: {attribute}: {reason}")
            }
            Self::InputArityInvalid { expected, actual } => {
                write!(f, "input arity invalid: expected {expected}, got {actual}")
            }
            Self::OutputArityInvalid { expected, actual } => {
                write!(f, "output arity invalid: expected {expected}, got {actual}")
            }
            Self::ShapeMismatch { reason } => write!(f, "shape mismatch: {reason}"),
            Self::ShapeUnsupported { reason } => write!(f, "shape unsupported: {reason}"),
            Self::DTypeUnsupported { dtype } => write!(f, "dtype unsupported: {dtype:?}"),
            Self::DTypeConversionRequired { from, to } => {
                write!(f, "dtype conversion required: {from:?} to {to:?}")
            }
            Self::DTypeConversionUnsupported { from, to } => {
                write!(f, "dtype conversion unsupported: {from:?} to {to:?}")
            }
            Self::LayoutUnsupported { layout } => write!(f, "layout unsupported: {layout:?}"),
            Self::LayoutConversionRequired { from, to } => {
                write!(f, "layout conversion required: {from:?} to {to:?}")
            }
            Self::LayoutConversionUnsupported { from, to } => {
                write!(f, "layout conversion unsupported: {from:?} to {to:?}")
            }
            Self::MemoryBehaviorUnsupported { reason } => {
                write!(f, "memory behavior unsupported: {reason}")
            }
            Self::WorkspaceUnavailable { bytes } => write!(f, "workspace unavailable: {bytes}"),
            Self::ResourceAffinityConflict { reason } => {
                write!(f, "resource affinity conflict: {reason}")
            }
            Self::ProviderCapabilityUnavailable { capability } => {
                write!(f, "provider capability unavailable: {capability}")
            }
            Self::KernelUnavailable { operator } => write!(f, "kernel unavailable: {operator}"),
            Self::GraphValidationFailed { reason } => {
                write!(f, "graph validation failed: {reason}")
            }
            Self::GraphPlanningFailed { reason } => write!(f, "graph planning failed: {reason}"),
            Self::GraphExecutionFailed { reason } => write!(f, "graph execution failed: {reason}"),
            Self::BrowserFeatureUnsupported { feature } => {
                write!(f, "browser feature unsupported: {feature}")
            }
            Self::InternalOperator { reason } => write!(f, "internal operator error: {reason}"),
        }
    }
}

impl Error for OperatorError {}

pub fn layout_kind(layout: &LayoutDescriptor) -> TensorLayoutKind {
    match layout {
        LayoutDescriptor::Contiguous => TensorLayoutKind::Contiguous,
        LayoutDescriptor::Strided { .. } => TensorLayoutKind::Strided,
        LayoutDescriptor::ProviderOpaque { .. } => TensorLayoutKind::ProviderOpaque,
    }
}

pub fn validate_affinity_compatibility(
    a: &ResourceAffinity,
    b: &ResourceAffinity,
) -> Result<(), OperatorError> {
    a.validate_with(b)
        .map_err(|error| OperatorError::ResourceAffinityConflict {
            reason: error.to_string(),
        })
}

pub fn initial_operator_catalog() -> OperatorCatalog {
    let floats = [
        ComputeDType::Float16,
        ComputeDType::BrainFloat16,
        ComputeDType::Float32,
        ComputeDType::Float64,
    ];
    let numeric = [
        ComputeDType::UInt8,
        ComputeDType::SInt8,
        ComputeDType::UInt16,
        ComputeDType::SInt16,
        ComputeDType::UInt32,
        ComputeDType::SInt32,
        ComputeDType::UInt64,
        ComputeDType::SInt64,
        ComputeDType::Float16,
        ComputeDType::BrainFloat16,
        ComputeDType::Float32,
        ComputeDType::Float64,
    ];
    let mut catalog = OperatorCatalog::new(OPERATOR_CATALOG_VERSION);
    for (name, family, inputs, outputs, shape) in [
        (
            "matmul",
            OperatorFamily::LinearAlgebra,
            2,
            1,
            ShapeRule::Matmul,
        ),
        (
            "batched-matmul",
            OperatorFamily::LinearAlgebra,
            2,
            1,
            ShapeRule::Any,
        ),
        ("embedding", OperatorFamily::Tensor, 2, 1, ShapeRule::Any),
        (
            "rmsnorm",
            OperatorFamily::Normalization,
            2,
            1,
            ShapeRule::SameShape,
        ),
        (
            "layernorm",
            OperatorFamily::Normalization,
            2,
            1,
            ShapeRule::SameShape,
        ),
        (
            "rope",
            OperatorFamily::PositionEncoding,
            1,
            1,
            ShapeRule::SameShape,
        ),
        ("attention", OperatorFamily::Attention, 3, 1, ShapeRule::Any),
        (
            "paged-attention",
            OperatorFamily::Attention,
            3,
            1,
            ShapeRule::Any,
        ),
        (
            "softmax",
            OperatorFamily::Activation,
            1,
            1,
            ShapeRule::SameShape,
        ),
        (
            "activation",
            OperatorFamily::Activation,
            1,
            1,
            ShapeRule::SameShape,
        ),
        (
            "gelu",
            OperatorFamily::Activation,
            1,
            1,
            ShapeRule::SameShape,
        ),
        (
            "silu",
            OperatorFamily::Activation,
            1,
            1,
            ShapeRule::SameShape,
        ),
        ("add", OperatorFamily::Tensor, 2, 1, ShapeRule::SameShape),
        ("mul", OperatorFamily::Tensor, 2, 1, ShapeRule::SameShape),
        (
            "residual-add",
            OperatorFamily::Tensor,
            2,
            1,
            ShapeRule::SameShape,
        ),
        (
            "dtype-conversion",
            OperatorFamily::Tensor,
            1,
            1,
            ShapeRule::SameShape,
        ),
        (
            "layout-conversion",
            OperatorFamily::Layout,
            1,
            1,
            ShapeRule::SameShape,
        ),
        (
            "quantize",
            OperatorFamily::Quantization,
            1,
            1,
            ShapeRule::SameShape,
        ),
        (
            "dequantize",
            OperatorFamily::Quantization,
            1,
            1,
            ShapeRule::SameShape,
        ),
        (
            "sampling-helper",
            OperatorFamily::SamplingSupport,
            1,
            1,
            ShapeRule::Any,
        ),
    ] {
        let mut spec = OperatorSpec::new(OperatorId::magnetar(name, 1, family), inputs, outputs)
            .with_shape_rule(shape)
            .with_dtype_contract(OperatorDTypeContract::new(TensorRole::Input, numeric))
            .with_dtype_contract(OperatorDTypeContract::new(TensorRole::Output, numeric))
            .with_layouts([
                TensorLayoutKind::Contiguous,
                TensorLayoutKind::Strided,
                TensorLayoutKind::Blocked,
                TensorLayoutKind::Paged,
                TensorLayoutKind::QuantizedPacked,
                TensorLayoutKind::AttentionSpecific,
                TensorLayoutKind::BrowserCompatible,
            ]);
        spec.attributes = default_attribute_schema(name);
        if matches!(family, OperatorFamily::Attention) {
            spec.memory = OperatorMemoryBehavior::pure()
                .with_workspace()
                .with_paged_kv_cache();
        }
        if matches!(
            family,
            OperatorFamily::Normalization | OperatorFamily::Activation
        ) {
            spec.dtype_contracts = vec![
                OperatorDTypeContract::new(TensorRole::Input, floats),
                OperatorDTypeContract::new(TensorRole::Output, floats),
                OperatorDTypeContract::new(TensorRole::Accumulation, floats),
            ];
        }
        catalog = catalog.with_operator(spec);
    }
    catalog
}

fn default_attribute_schema(name: &str) -> OperatorAttributeSchema {
    match name {
        "matmul" | "batched-matmul" => OperatorAttributeSchema::default()
            .with_rule(
                "transpose_a",
                OperatorAttributeRule::optional(OperatorAttributeKind::Boolean),
            )
            .with_rule(
                "transpose_b",
                OperatorAttributeRule::optional(OperatorAttributeKind::Boolean),
            )
            .with_rule(
                "accumulation_dtype",
                OperatorAttributeRule::optional(OperatorAttributeKind::DType),
            ),
        "attention" | "paged-attention" => OperatorAttributeSchema::default()
            .with_rule(
                "causal",
                OperatorAttributeRule::optional(OperatorAttributeKind::Boolean),
            )
            .with_rule(
                "window_size",
                OperatorAttributeRule::optional(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "head_count",
                OperatorAttributeRule::required(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "kv_head_count",
                OperatorAttributeRule::optional(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "head_dimension",
                OperatorAttributeRule::required(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "attention_mask_kind",
                OperatorAttributeRule::optional(OperatorAttributeKind::String),
            ),
        "rope" => OperatorAttributeSchema::default()
            .with_rule(
                "base",
                OperatorAttributeRule::required(OperatorAttributeKind::Float),
            )
            .with_rule(
                "scale",
                OperatorAttributeRule::optional(OperatorAttributeKind::Float),
            )
            .with_rule(
                "dimension",
                OperatorAttributeRule::required(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "position_mode",
                OperatorAttributeRule::optional(OperatorAttributeKind::String),
            ),
        "rmsnorm" | "layernorm" => OperatorAttributeSchema::default()
            .with_rule(
                "epsilon",
                OperatorAttributeRule::required(OperatorAttributeKind::Float),
            )
            .with_rule(
                "normalized_dimension",
                OperatorAttributeRule::optional(OperatorAttributeKind::Integer),
            )
            .with_rule(
                "accumulation_dtype",
                OperatorAttributeRule::optional(OperatorAttributeKind::DType),
            ),
        "activation" => OperatorAttributeSchema::default().with_rule(
            "kind",
            OperatorAttributeRule::required(OperatorAttributeKind::String),
        ),
        "quantize" | "dequantize" => OperatorAttributeSchema::default()
            .with_rule(
                "storage_dtype",
                OperatorAttributeRule::required(OperatorAttributeKind::DType),
            )
            .with_rule(
                "scale",
                OperatorAttributeRule::optional(OperatorAttributeKind::Float),
            ),
        "dtype-conversion" => OperatorAttributeSchema::default().with_rule(
            "dtype",
            OperatorAttributeRule::required(OperatorAttributeKind::DType),
        ),
        "layout-conversion" => OperatorAttributeSchema::default().with_rule(
            "layout",
            OperatorAttributeRule::required(OperatorAttributeKind::Layout),
        ),
        _ => OperatorAttributeSchema::default(),
    }
}

fn validate_shape_rule(
    rule: &ShapeRule,
    inputs: &[TensorDescriptor],
    outputs: &[TensorDescriptor],
) -> Result<(), OperatorError> {
    match rule {
        ShapeRule::Any => Ok(()),
        ShapeRule::Rank(rank) => inputs
            .iter()
            .chain(outputs)
            .all(|tensor| tensor.shape.rank() == *rank)
            .then_some(())
            .ok_or_else(|| OperatorError::ShapeUnsupported {
                reason: format!("expected rank {rank}"),
            }),
        ShapeRule::SameRank => {
            let Some(first) = inputs.first().or_else(|| outputs.first()) else {
                return Ok(());
            };
            inputs
                .iter()
                .chain(outputs)
                .all(|tensor| tensor.shape.rank() == first.shape.rank())
                .then_some(())
                .ok_or_else(|| OperatorError::ShapeMismatch {
                    reason: "tensor ranks differ".into(),
                })
        }
        ShapeRule::SameShape => {
            let Some(first) = inputs.first().or_else(|| outputs.first()) else {
                return Ok(());
            };
            inputs
                .iter()
                .chain(outputs)
                .all(|tensor| tensor.shape.dimensions == first.shape.dimensions)
                .then_some(())
                .ok_or_else(|| OperatorError::ShapeMismatch {
                    reason: "tensor shapes differ".into(),
                })
        }
        ShapeRule::Matmul => {
            if inputs.len() < 2 {
                return Err(OperatorError::InputArityInvalid {
                    expected: 2,
                    actual: inputs.len(),
                });
            }
            let a = &inputs[0].shape.dimensions;
            let b = &inputs[1].shape.dimensions;
            if a.len() < 2 || b.len() < 2 || a[a.len() - 1] != b[b.len() - 2] {
                return Err(OperatorError::ShapeMismatch {
                    reason: "matmul inner dimensions differ".into(),
                });
            }
            Ok(())
        }
    }
}

fn validate_dtype_contracts(
    contracts: &[OperatorDTypeContract],
    tensor: &TensorDescriptor,
) -> Result<(), OperatorError> {
    let DTypeDescriptor::Portable(dtype) = &tensor.dtype else {
        return Err(OperatorError::DTypeUnsupported {
            dtype: ComputeDType::UInt8,
        });
    };
    if contracts.iter().any(|contract| contract.accepts(*dtype)) {
        Ok(())
    } else {
        Err(OperatorError::DTypeUnsupported { dtype: *dtype })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ShapeDescriptor, TensorDescriptor};

    #[test]
    fn operator_catalog_contains_required_families_and_initial_operators() {
        let families = OperatorFamily::ALL
            .into_iter()
            .map(OperatorFamily::id)
            .collect::<BTreeSet<_>>();
        assert!(families.contains("attention"));
        let catalog = initial_operator_catalog();
        for name in ["matmul", "attention", "rmsnorm", "rope", "sampling-helper"] {
            assert!(
                catalog
                    .operators
                    .keys()
                    .any(|operator| operator.name() == name)
            );
        }
    }

    #[test]
    fn operator_attributes_reject_provider_device_and_unknown_selectors() {
        let catalog = initial_operator_catalog();
        let matmul = catalog
            .get(&OperatorId::magnetar(
                "matmul",
                1,
                OperatorFamily::LinearAlgebra,
            ))
            .unwrap();
        let mut attributes = BTreeMap::new();
        attributes.insert(
            "provider".into(),
            OperatorAttributeValue::String("cuda".into()),
        );
        assert!(matches!(
            matmul.attributes.validate(&attributes),
            Err(OperatorError::OperatorAttributeInvalid { .. })
        ));
    }

    #[test]
    fn operator_validation_rejects_shape_dtype_layout_errors() {
        let catalog = initial_operator_catalog();
        let matmul = catalog
            .get(&OperatorId::magnetar(
                "matmul",
                1,
                OperatorFamily::LinearAlgebra,
            ))
            .unwrap();
        let a = TensorDescriptor::materialized(
            ShapeDescriptor::new([2, 3]),
            DTypeDescriptor::portable(ComputeDType::Float32),
        );
        let b = TensorDescriptor::materialized(
            ShapeDescriptor::new([4, 2]),
            DTypeDescriptor::portable(ComputeDType::Float32),
        );
        let out = TensorDescriptor::materialized(
            ShapeDescriptor::new([2, 2]),
            DTypeDescriptor::portable(ComputeDType::Float32),
        );
        assert!(matches!(
            matmul.validate_invocation(&[a, b], &[out], &BTreeMap::new()),
            Err(OperatorError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn opaque_layout_is_not_component_visible() {
        assert!(!TensorLayoutKind::ProviderOpaque.component_visible());
    }
}
