//! First operator implementation scope.
//!
//! This module describes the deliberately small, platform-neutral operator
//! surface used by the first correctness-oriented executable path. It is
//! Runtime-owned metadata: Providers and Kernels may cover entries in the
//! scope, but they do not define the scope.

use crate::{
    ComputeDType, ExecutionGraph, KernelAdvertisement, KernelSelectionRequest,
    OperatorAttributeValue, OperatorFamily, OperatorId, OperatorRequirement, TensorLayoutKind,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

pub const FIRST_OPERATOR_SCOPE_VERSION: &str = "first-operator-scope-v1";

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum OperatorScopeTier {
    RequiredNow,
    RequiredForFirstDecoderModel,
    Placeholder,
    ExplicitlyUnsupported,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum FutureOptimizedOperator {
    FlashAttention,
    FusedRmsNorm,
    FusedMlp,
    FusedRopeAttention,
    TensorCoreMatmul,
    SimdRmsNorm,
    PagedAttention,
    QuantizedMatmul,
}

impl FutureOptimizedOperator {
    pub const ALL: [Self; 8] = [
        Self::FlashAttention,
        Self::FusedRmsNorm,
        Self::FusedMlp,
        Self::FusedRopeAttention,
        Self::TensorCoreMatmul,
        Self::SimdRmsNorm,
        Self::PagedAttention,
        Self::QuantizedMatmul,
    ];

    pub const fn id(self) -> &'static str {
        match self {
            Self::FlashAttention => "flash-attention",
            Self::FusedRmsNorm => "fused-rmsnorm",
            Self::FusedMlp => "fused-mlp",
            Self::FusedRopeAttention => "fused-rope-attention",
            Self::TensorCoreMatmul => "tensorcore-matmul",
            Self::SimdRmsNorm => "simd-rmsnorm",
            Self::PagedAttention => "paged-attention",
            Self::QuantizedMatmul => "quantized-matmul",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum FirstScopeDTypeTier {
    Required,
    Placeholder,
    ExplicitlyUnsupported,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum FirstScopeLayoutTier {
    Required,
    Placeholder,
    Future,
    ProviderInternalOnly,
    ExplicitlyUnsupported,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum FirstScopeObservationKind {
    OperatorAccepted,
    OperatorRejected,
    PlaceholderOperatorEncountered,
    UnsupportedOperatorEncountered,
    RequiredKernelMissing,
    DTypeUnsupported,
    LayoutUnsupported,
    ShapeUnsupported,
    ConformancePassed,
    ConformanceFailed,
}

impl FirstScopeObservationKind {
    pub const fn id(self) -> &'static str {
        match self {
            Self::OperatorAccepted => "first-scope-operator-accepted",
            Self::OperatorRejected => "first-scope-operator-rejected",
            Self::PlaceholderOperatorEncountered => "placeholder-operator-encountered",
            Self::UnsupportedOperatorEncountered => "unsupported-operator-encountered",
            Self::RequiredKernelMissing => "required-kernel-missing",
            Self::DTypeUnsupported => "dtype-unsupported-in-first-scope",
            Self::LayoutUnsupported => "layout-unsupported-in-first-scope",
            Self::ShapeUnsupported => "shape-unsupported-in-first-scope",
            Self::ConformancePassed => "first-scope-conformance-passed",
            Self::ConformanceFailed => "first-scope-conformance-failed",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FirstScopeObservation {
    pub kind: FirstScopeObservationKind,
    pub operator: Option<OperatorId>,
    pub redacted_metadata: Vec<(&'static str, String)>,
}

impl FirstScopeObservation {
    pub fn new(kind: FirstScopeObservationKind) -> Self {
        Self {
            kind,
            operator: None,
            redacted_metadata: Vec::new(),
        }
    }

    pub fn with_operator(mut self, operator: OperatorId) -> Self {
        self.operator = Some(operator);
        self
    }

    pub fn with_redacted_metadata(mut self, key: &'static str, value: impl Into<String>) -> Self {
        self.redacted_metadata.push((key, value.into()));
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OperatorScopeEntry {
    pub name: &'static str,
    pub family: OperatorFamily,
    pub tier: OperatorScopeTier,
    pub required_for_first_decoder_model: bool,
    pub requires_reference_cpu_kernel: bool,
    pub requires_conformance_fixture: bool,
}

impl OperatorScopeEntry {
    pub fn id(self) -> OperatorId {
        OperatorId::magnetar(self.name, 1, self.family)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum FirstScopeErrorCode {
    OperatorOutOfFirstScope,
    OperatorPlaceholderOnly,
    OperatorExplicitlyUnsupported,
    DTypeUnsupported,
    LayoutUnsupported,
    ShapeUnsupported,
    AttributeUnsupported,
    KernelMissing,
    ConformanceMissing,
    ConformanceFailed,
    GraphPlanningFailed,
    Internal,
}

impl FirstScopeErrorCode {
    pub const fn id(self) -> &'static str {
        match self {
            Self::OperatorOutOfFirstScope => "operator-out-of-first-scope",
            Self::OperatorPlaceholderOnly => "operator-placeholder-only",
            Self::OperatorExplicitlyUnsupported => "operator-explicitly-unsupported",
            Self::DTypeUnsupported => "first-scope-dtype-unsupported",
            Self::LayoutUnsupported => "first-scope-layout-unsupported",
            Self::ShapeUnsupported => "first-scope-shape-unsupported",
            Self::AttributeUnsupported => "first-scope-attribute-unsupported",
            Self::KernelMissing => "first-scope-kernel-missing",
            Self::ConformanceMissing => "first-scope-conformance-missing",
            Self::ConformanceFailed => "first-scope-conformance-failed",
            Self::GraphPlanningFailed => "first-scope-graph-planning-failed",
            Self::Internal => "internal-first-operator-scope",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FirstScopeError {
    pub code: FirstScopeErrorCode,
    pub operator: Option<OperatorId>,
    pub reason: String,
}

impl FirstScopeError {
    pub fn new(code: FirstScopeErrorCode, reason: impl Into<String>) -> Self {
        Self {
            code,
            operator: None,
            reason: reason.into(),
        }
    }

    pub fn with_operator(mut self, operator: OperatorId) -> Self {
        self.operator = Some(operator);
        self
    }

    pub const fn id(&self) -> &'static str {
        self.code.id()
    }
}

impl fmt::Display for FirstScopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.operator {
            Some(operator) => write!(f, "{}: {operator}: {}", self.id(), self.reason),
            None => write!(f, "{}: {}", self.id(), self.reason),
        }
    }
}

impl Error for FirstScopeError {}

pub fn first_operator_scope() -> &'static [OperatorScopeEntry] {
    &FIRST_OPERATOR_SCOPE
}

pub fn future_optimized_operators() -> &'static [FutureOptimizedOperator] {
    &FutureOptimizedOperator::ALL
}

pub fn operator_scope_entry(operator: &OperatorId) -> Option<OperatorScopeEntry> {
    first_operator_scope()
        .iter()
        .copied()
        .find(|entry| entry.name == operator.name())
}

pub fn validate_required_now_operator(
    operator: &OperatorId,
) -> Result<OperatorScopeEntry, FirstScopeError> {
    let entry = operator_scope_entry(operator).ok_or_else(|| {
        FirstScopeError::new(
            FirstScopeErrorCode::OperatorOutOfFirstScope,
            "operator is not part of the first implementation scope",
        )
        .with_operator(operator.clone())
    })?;
    match entry.tier {
        OperatorScopeTier::RequiredNow => Ok(entry),
        OperatorScopeTier::RequiredForFirstDecoderModel => Ok(entry),
        OperatorScopeTier::Placeholder => Err(FirstScopeError::new(
            FirstScopeErrorCode::OperatorPlaceholderOnly,
            "operator identity is reserved but not required for execution",
        )
        .with_operator(operator.clone())),
        OperatorScopeTier::ExplicitlyUnsupported => Err(FirstScopeError::new(
            FirstScopeErrorCode::OperatorExplicitlyUnsupported,
            "operator is explicitly unsupported in the first implementation scope",
        )
        .with_operator(operator.clone())),
    }
}

pub fn validate_first_scope_graph(graph: &ExecutionGraph) -> Result<(), FirstScopeError> {
    for node in graph.nodes.values() {
        validate_required_now_operator(&node.operator)?;
        validate_first_scope_attributes(&node.operator, &node.attributes)?;
    }
    Ok(())
}

pub fn validate_model_component_first_scope_requirements(
    requirements: &[OperatorRequirement],
) -> Result<(), FirstScopeError> {
    for requirement in requirements {
        if validate_required_now_operator(&requirement.operator).is_ok() {
            continue;
        }
        if requirement
            .alternatives
            .iter()
            .any(|alternative| validate_required_now_operator(alternative).is_ok())
        {
            continue;
        }
        return Err(validate_required_now_operator(&requirement.operator)
            .expect_err("invalid requirement should produce a first-scope error"));
    }
    Ok(())
}

pub fn validate_first_scope_kernel_selection_request(
    request: &KernelSelectionRequest,
) -> Result<(), FirstScopeError> {
    validate_required_now_operator(&request.operator)?;
    for dtype in &request.dtype_requirements {
        validate_first_scope_dtype(*dtype)?;
    }
    for layout in &request.layout_requirements {
        validate_first_scope_layout(*layout)?;
    }
    Ok(())
}

pub fn first_scope_dtype_tier(dtype: ComputeDType) -> FirstScopeDTypeTier {
    match dtype {
        ComputeDType::Float32
        | ComputeDType::SInt32
        | ComputeDType::UInt32
        | ComputeDType::Boolean => FirstScopeDTypeTier::Required,
        ComputeDType::Float16
        | ComputeDType::BrainFloat16
        | ComputeDType::SInt8
        | ComputeDType::UInt8 => FirstScopeDTypeTier::Placeholder,
        ComputeDType::UInt16
        | ComputeDType::SInt16
        | ComputeDType::UInt64
        | ComputeDType::SInt64
        | ComputeDType::Float64 => FirstScopeDTypeTier::ExplicitlyUnsupported,
    }
}

pub fn validate_first_scope_dtype(dtype: ComputeDType) -> Result<(), FirstScopeError> {
    match first_scope_dtype_tier(dtype) {
        FirstScopeDTypeTier::Required => Ok(()),
        FirstScopeDTypeTier::Placeholder => Err(FirstScopeError::new(
            FirstScopeErrorCode::DTypeUnsupported,
            format!("{dtype:?} compute is a placeholder and requires explicit conversion"),
        )),
        FirstScopeDTypeTier::ExplicitlyUnsupported => Err(FirstScopeError::new(
            FirstScopeErrorCode::DTypeUnsupported,
            format!("{dtype:?} compute is outside the first scope"),
        )),
    }
}

pub fn first_scope_layout_tier(layout: TensorLayoutKind) -> FirstScopeLayoutTier {
    match layout {
        TensorLayoutKind::Contiguous => FirstScopeLayoutTier::Required,
        TensorLayoutKind::Strided
        | TensorLayoutKind::Paged
        | TensorLayoutKind::AttentionSpecific => FirstScopeLayoutTier::Placeholder,
        TensorLayoutKind::Blocked | TensorLayoutKind::QuantizedPacked => {
            FirstScopeLayoutTier::Future
        }
        TensorLayoutKind::ProviderOpaque => FirstScopeLayoutTier::ProviderInternalOnly,
        TensorLayoutKind::BrowserCompatible => FirstScopeLayoutTier::ExplicitlyUnsupported,
    }
}

pub fn validate_first_scope_layout(layout: TensorLayoutKind) -> Result<(), FirstScopeError> {
    match first_scope_layout_tier(layout) {
        FirstScopeLayoutTier::Required => Ok(()),
        FirstScopeLayoutTier::Placeholder
        | FirstScopeLayoutTier::Future
        | FirstScopeLayoutTier::ProviderInternalOnly
        | FirstScopeLayoutTier::ExplicitlyUnsupported => Err(FirstScopeError::new(
            FirstScopeErrorCode::LayoutUnsupported,
            format!("{layout:?} layout requires explicit conversion or rejection"),
        )),
    }
}

pub fn validate_reference_cpu_required_kernel_coverage(
    advertisements: &[KernelAdvertisement],
) -> Result<(), FirstScopeError> {
    let advertised = advertisements
        .iter()
        .map(|advertisement| advertisement.implemented_operator.name().to_owned())
        .collect::<BTreeSet<_>>();
    for entry in first_operator_scope()
        .iter()
        .filter(|entry| entry.requires_reference_cpu_kernel)
    {
        if !advertised.contains(entry.name) {
            return Err(FirstScopeError::new(
                FirstScopeErrorCode::KernelMissing,
                format!(
                    "Reference CPU does not advertise required-now kernel '{}'",
                    entry.name
                ),
            )
            .with_operator(entry.id()));
        }
    }
    Ok(())
}

pub fn validate_no_placeholder_kernel_advertisements(
    advertisements: &[KernelAdvertisement],
) -> Result<(), FirstScopeError> {
    for advertisement in advertisements {
        if let Some(entry) = operator_scope_entry(&advertisement.implemented_operator)
            && entry.tier == OperatorScopeTier::Placeholder
        {
            return Err(FirstScopeError::new(
                FirstScopeErrorCode::OperatorPlaceholderOnly,
                "placeholder operator must not be advertised as implemented",
            )
            .with_operator(advertisement.implemented_operator.clone()));
        }
    }
    Ok(())
}

pub fn first_scope_required_fixture_names() -> BTreeSet<&'static str> {
    first_operator_scope()
        .iter()
        .filter(|entry| entry.requires_conformance_fixture)
        .map(|entry| entry.name)
        .collect()
}

fn validate_first_scope_attributes(
    operator: &OperatorId,
    attributes: &BTreeMap<String, OperatorAttributeValue>,
) -> Result<(), FirstScopeError> {
    match operator.name() {
        "attention" => {
            if let Some(OperatorAttributeValue::String(mask_kind)) =
                attributes.get("attention_mask_kind")
                && !matches!(mask_kind.as_str(), "causal" | "bidirectional")
            {
                return Err(FirstScopeError::new(
                    FirstScopeErrorCode::AttributeUnsupported,
                    format!("attention mask kind '{mask_kind}' is outside first scope"),
                )
                .with_operator(operator.clone()));
            }
            Ok(())
        }
        "rope" => {
            if let Some(OperatorAttributeValue::String(position_mode)) =
                attributes.get("position_mode")
                && position_mode != "sequential"
            {
                return Err(FirstScopeError::new(
                    FirstScopeErrorCode::AttributeUnsupported,
                    format!("RoPE position mode '{position_mode}' is outside first scope"),
                )
                .with_operator(operator.clone()));
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

const FIRST_OPERATOR_SCOPE: [OperatorScopeEntry; 31] = [
    required_now("embedding", OperatorFamily::Tensor, true),
    required_now("matmul", OperatorFamily::LinearAlgebra, true),
    required_now("rmsnorm", OperatorFamily::Normalization, true),
    required_now("rope", OperatorFamily::PositionEncoding, true),
    required_now("attention", OperatorFamily::Attention, true),
    required_now("softmax", OperatorFamily::Activation, true),
    required_now("silu", OperatorFamily::Activation, true),
    required_now("add", OperatorFamily::Tensor, true),
    required_now("mul", OperatorFamily::Tensor, true),
    required_now("residual-add", OperatorFamily::Tensor, true),
    required_now("dtype-conversion", OperatorFamily::Tensor, false),
    required_now("layout-conversion", OperatorFamily::Layout, false),
    placeholder("batched-matmul", OperatorFamily::LinearAlgebra),
    placeholder("layernorm", OperatorFamily::Normalization),
    first_decoder_optional("gelu", OperatorFamily::Activation),
    placeholder("dequantize", OperatorFamily::Quantization),
    placeholder("quantize", OperatorFamily::Quantization),
    placeholder("requantize", OperatorFamily::Quantization),
    placeholder("quantized-matmul", OperatorFamily::Quantization),
    placeholder("paged-attention", OperatorFamily::Attention),
    placeholder("sampling-helper", OperatorFamily::SamplingSupport),
    placeholder("logits-processor-helper", OperatorFamily::SamplingSupport),
    placeholder("layout-pack", OperatorFamily::Layout),
    placeholder("layout-unpack", OperatorFamily::Layout),
    unsupported("flash-attention", OperatorFamily::Attention),
    unsupported("grouped-quantization", OperatorFamily::Quantization),
    unsupported("moe-dispatch", OperatorFamily::Control),
    unsupported(
        "speculative-decoding-helper",
        OperatorFamily::SamplingSupport,
    ),
    unsupported("beam-search-helper", OperatorFamily::SamplingSupport),
    unsupported("training-operator", OperatorFamily::Control),
    unsupported("gradient-operator", OperatorFamily::Control),
];

const fn required_now(
    name: &'static str,
    family: OperatorFamily,
    required_for_first_decoder_model: bool,
) -> OperatorScopeEntry {
    OperatorScopeEntry {
        name,
        family,
        tier: OperatorScopeTier::RequiredNow,
        required_for_first_decoder_model,
        requires_reference_cpu_kernel: true,
        requires_conformance_fixture: true,
    }
}

const fn first_decoder_optional(name: &'static str, family: OperatorFamily) -> OperatorScopeEntry {
    OperatorScopeEntry {
        name,
        family,
        tier: OperatorScopeTier::RequiredForFirstDecoderModel,
        required_for_first_decoder_model: false,
        requires_reference_cpu_kernel: false,
        requires_conformance_fixture: false,
    }
}

const fn placeholder(name: &'static str, family: OperatorFamily) -> OperatorScopeEntry {
    OperatorScopeEntry {
        name,
        family,
        tier: OperatorScopeTier::Placeholder,
        required_for_first_decoder_model: false,
        requires_reference_cpu_kernel: false,
        requires_conformance_fixture: false,
    }
}

const fn unsupported(name: &'static str, family: OperatorFamily) -> OperatorScopeEntry {
    OperatorScopeEntry {
        name,
        family,
        tier: OperatorScopeTier::ExplicitlyUnsupported,
        required_for_first_decoder_model: false,
        requires_reference_cpu_kernel: false,
        requires_conformance_fixture: false,
    }
}
