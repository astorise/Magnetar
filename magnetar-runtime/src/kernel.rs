//! Provider-owned Kernel contract.
//!
//! Kernels are concrete implementations of portable Operators. The contract is
//! intentionally metadata-first: Runtime validates Operator semantics, tensor
//! constraints, memory ownership, Resource Affinity, and policy before any
//! Provider-owned execution path is reached.

use crate::{
    CapabilityVersion, ComputeDType, ComputePrecision, DTypeDescriptor, DeviceBinding, DeviceType,
    MemoryAllocationId, OperatorAttributeValue, OperatorId, OperatorMemoryBehavior, OperatorSpec,
    ProviderBinding, ResourceAffinity, TensorAliasingKind, TensorDescriptor, TensorLayoutKind,
    TensorResourceDescriptor, layout_kind, validate_affinity_compatibility,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelInvocationId(String);

impl KernelInvocationId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for KernelInvocationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelOperatorVersionRange {
    pub min: u32,
    pub max: u32,
}

impl KernelOperatorVersionRange {
    pub const fn exact(version: u32) -> Self {
        Self {
            min: version,
            max: version,
        }
    }

    pub const fn contains(&self, version: u32) -> bool {
        self.min <= version && version <= self.max
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelImplementationFamily {
    CpuScalar,
    CpuVector,
    Cuda,
    Metal,
    OpenVino,
    Qnn,
    Wasm,
    WebGpu,
    ProviderFused,
    TestFixture,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelId {
    pub provider: ProviderBinding,
    pub name: String,
    pub version: CapabilityVersion,
    pub operator: OperatorId,
    pub operator_versions: KernelOperatorVersionRange,
    pub features: BTreeSet<String>,
    pub family: KernelImplementationFamily,
    pub build_fingerprint: Option<String>,
    pub conformance_profile: Option<String>,
}

impl KernelId {
    pub fn new(
        provider: ProviderBinding,
        name: impl Into<String>,
        version: CapabilityVersion,
        operator: OperatorId,
        operator_versions: KernelOperatorVersionRange,
        family: KernelImplementationFamily,
    ) -> Self {
        Self {
            provider,
            name: name.into(),
            version,
            operator,
            operator_versions,
            features: BTreeSet::new(),
            family,
            build_fingerprint: None,
            conformance_profile: None,
        }
    }

    pub fn with_features(mut self, features: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.features.extend(features.into_iter().map(Into::into));
        self
    }

    pub fn with_build_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.build_fingerprint = Some(fingerprint.into());
        self
    }

    pub fn with_conformance_profile(mut self, profile: impl Into<String>) -> Self {
        self.conformance_profile = Some(profile.into());
        self
    }

    pub fn stable_key(&self) -> String {
        format!(
            "{}:{}@{}:{}",
            self.provider, self.name, self.version, self.operator
        )
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

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelExecutionMode {
    Synchronous,
    Asynchronous,
    Streamed,
    Batched,
    GraphCaptured,
    ProviderFused,
    BrowserCompatible,
    TestFixture,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelCancellationSupport {
    NotSupported,
    BeforeDispatchOnly,
    Cooperative,
    Interruptible,
    TimeoutOnly,
    ProviderSpecific,
}

impl KernelCancellationSupport {
    pub const fn can_cancel_during_execution(self) -> bool {
        matches!(self, Self::Cooperative | Self::Interruptible)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelShapeConstraints {
    pub rank: Option<u64>,
    pub static_dimensions: BTreeMap<usize, u64>,
    pub supports_dynamic_dimensions: bool,
    pub alignment: Option<u64>,
    pub max_batch_size: Option<u64>,
    pub max_sequence_length: Option<u64>,
    pub max_head_count: Option<u64>,
    pub max_head_dimension: Option<u64>,
    pub matrix_tile: Option<(u64, u64)>,
    pub block_size: Option<u64>,
    pub page_size: Option<u64>,
    pub max_total_elements: Option<u64>,
    pub max_total_tokens: Option<u64>,
}

impl Default for KernelShapeConstraints {
    fn default() -> Self {
        Self {
            rank: None,
            static_dimensions: BTreeMap::new(),
            supports_dynamic_dimensions: true,
            alignment: None,
            max_batch_size: None,
            max_sequence_length: None,
            max_head_count: None,
            max_head_dimension: None,
            matrix_tile: None,
            block_size: None,
            page_size: None,
            max_total_elements: None,
            max_total_tokens: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelWorkspaceRequirements {
    pub required: bool,
    pub size_bytes_upper_bound: Option<u64>,
    pub memory_class: KernelMemoryClass,
    pub alignment_bytes: u64,
    pub lifetime: KernelWorkspaceLifetime,
    pub reuse: KernelWorkspaceReuse,
}

impl KernelWorkspaceRequirements {
    pub const fn none() -> Self {
        Self {
            required: false,
            size_bytes_upper_bound: Some(0),
            memory_class: KernelMemoryClass::Host,
            alignment_bytes: 1,
            lifetime: KernelWorkspaceLifetime::Operation,
            reuse: KernelWorkspaceReuse::NoReuse,
        }
    }

    pub const fn required(
        size_bytes_upper_bound: u64,
        memory_class: KernelMemoryClass,
        alignment_bytes: u64,
    ) -> Self {
        Self {
            required: true,
            size_bytes_upper_bound: Some(size_bytes_upper_bound),
            memory_class,
            alignment_bytes,
            lifetime: KernelWorkspaceLifetime::Operation,
            reuse: KernelWorkspaceReuse::PerOperation,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelWorkspaceLifetime {
    Operation,
    Batch,
    ExecutionContext,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelWorkspaceReuse {
    NoReuse,
    PerOperation,
    PerBatch,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelAliasing {
    pub allows_input_output_alias: bool,
    pub supports_in_place: bool,
    pub output_aliases_input: bool,
    pub mutates_input: bool,
    pub reads_input: bool,
    pub writes_output: bool,
}

impl Default for KernelAliasing {
    fn default() -> Self {
        Self {
            allows_input_output_alias: false,
            supports_in_place: false,
            output_aliases_input: false,
            mutates_input: false,
            reads_input: true,
            writes_output: true,
        }
    }
}

impl KernelAliasing {
    pub const fn compatible_with_operator(self, memory: OperatorMemoryBehavior) -> bool {
        (!self.mutates_input || memory.mutates_input)
            && (!self.output_aliases_input || memory.aliases_output)
            && (!self.supports_in_place || memory.supports_in_place)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelDeterminism {
    pub deterministic: bool,
    pub depends_on_dtype: bool,
    pub depends_on_device: bool,
    pub depends_on_execution_mode: bool,
    pub depends_on_parallel_reduction: bool,
    pub depends_on_accumulation_order: bool,
    pub depends_on_atomic_operations: bool,
    pub depends_on_kernel_version: bool,
    pub depends_on_provider_version: bool,
    pub depends_on_hardware_features: bool,
}

impl Default for KernelDeterminism {
    fn default() -> Self {
        Self {
            deterministic: true,
            depends_on_dtype: true,
            depends_on_device: true,
            depends_on_execution_mode: true,
            depends_on_parallel_reduction: false,
            depends_on_accumulation_order: false,
            depends_on_atomic_operations: false,
            depends_on_kernel_version: true,
            depends_on_provider_version: true,
            depends_on_hardware_features: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelPrecisionMetadata {
    pub accumulation_dtype: Option<ComputeDType>,
    pub rounding_mode: Option<String>,
    pub approximate_math: bool,
    pub fused_operation_semantics: bool,
    pub tolerance_profile: Option<String>,
    pub quantization_error_profile: Option<String>,
    pub deterministic_tolerance_profile: Option<String>,
}

impl Default for KernelPrecisionMetadata {
    fn default() -> Self {
        Self {
            accumulation_dtype: None,
            rounding_mode: None,
            approximate_math: false,
            fused_operation_semantics: false,
            tolerance_profile: Some("operator-default".into()),
            quantization_error_profile: None,
            deterministic_tolerance_profile: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelFusionMetadata {
    pub operator_group: Vec<OperatorId>,
    pub preserves_graph_semantics: bool,
}

/// Quantization numeric method a post-baseline Kernel declares, from the
/// `define-post-baseline-provider-roadmap` change's "Quantized Execution"
/// section. This does not implement any quantized numerics -- it is metadata
/// a Kernel advertisement carries so Runtime can validate that quantized
/// execution is explicit rather than a hidden substitution.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelQuantizationMethod {
    Int8,
    Int4,
    Fp8,
    NormalFloat4,
    GroupwiseAffine,
    Custom,
}

/// Whether a quantized Kernel dequantizes explicitly (as its own graph step)
/// or fuses dequantization into the Kernel itself. Either is acceptable, but
/// one of the two SHALL be declared -- "no hidden quantization or
/// dequantization SHALL occur".
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelDequantizationBehavior {
    ExplicitBeforeOperator,
    FusedIntoKernel,
}

/// Post-baseline quantization declaration for one Kernel: method, storage
/// dtype, compute dtype, accumulation dtype, scale/zero-point metadata, group
/// size, packing layout, dequantization behavior, supported Operators, and a
/// conformance tolerance profile. See `specs/kernel-registry/spec.md` and
/// `specs/provider-roadmap/spec.md` in
/// `openspec/changes/define-post-baseline-provider-roadmap` for the
/// requirements this metadata satisfies.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelQuantizationMetadata {
    pub method: KernelQuantizationMethod,
    pub storage_dtype: ComputeDType,
    pub compute_dtype: ComputeDType,
    pub accumulation_dtype: ComputeDType,
    pub scale_dtype: ComputeDType,
    pub zero_point_dtype: Option<ComputeDType>,
    pub group_size: Option<u64>,
    pub packing_layout: TensorLayoutKind,
    pub dequantization: KernelDequantizationBehavior,
    pub supported_operators: BTreeSet<OperatorId>,
    pub conformance_tolerance_profile: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelAdapterMetadata {
    pub methods: BTreeSet<String>,
    pub max_rank: Option<u32>,
    pub dtypes: BTreeSet<ComputeDType>,
    pub merge_strategy: Option<String>,
    pub target_modules: BTreeSet<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelKvCacheMetadata {
    pub layouts: BTreeSet<String>,
    pub paged_cache: bool,
    pub append: bool,
    pub read: bool,
    pub dtypes: BTreeSet<ComputeDType>,
    pub memory_classes: BTreeSet<KernelMemoryClass>,
    pub affinity: Option<ResourceAffinity>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelPrefixCacheMetadata {
    pub supports_adjusted_sequence_length: bool,
    pub supports_adjusted_context_length: bool,
    pub supports_reused_prefix_boundary: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelBatchMetadata {
    pub max_batch_size: Option<u64>,
    pub max_active_sequences: Option<u64>,
    pub max_total_tokens: Option<u64>,
    pub supports_ragged_batches: bool,
    pub supports_paged_kv_cache: bool,
    pub per_operation_output_mapping: bool,
    pub batch_slot_compatible: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelAdvertisement {
    pub id: KernelId,
    pub implemented_operator: OperatorId,
    pub supported_dtypes: BTreeMap<crate::TensorRole, BTreeSet<ComputeDType>>,
    pub supported_layouts: BTreeSet<TensorLayoutKind>,
    pub shape: KernelShapeConstraints,
    pub memory_classes: BTreeSet<KernelMemoryClass>,
    pub devices: BTreeSet<DeviceBinding>,
    pub device_classes: Vec<DeviceType>,
    pub workspace: KernelWorkspaceRequirements,
    pub execution_modes: BTreeSet<KernelExecutionMode>,
    pub cancellation: KernelCancellationSupport,
    pub determinism: KernelDeterminism,
    pub precision: KernelPrecisionMetadata,
    pub aliasing: KernelAliasing,
    pub performance_hints: BTreeMap<String, String>,
    pub fallback_hints: BTreeSet<KernelFallbackClass>,
    pub required_provider_features: BTreeSet<String>,
    pub required_device_features: BTreeSet<String>,
    pub fusion: Option<KernelFusionMetadata>,
    pub adapter: Option<KernelAdapterMetadata>,
    pub kv_cache: Option<KernelKvCacheMetadata>,
    pub prefix_cache: Option<KernelPrefixCacheMetadata>,
    pub batching: Option<KernelBatchMetadata>,
    pub browser_compatible: bool,
}

impl KernelAdvertisement {
    pub fn new(id: KernelId) -> Self {
        Self {
            implemented_operator: id.operator.clone(),
            id,
            supported_dtypes: BTreeMap::new(),
            supported_layouts: BTreeSet::new(),
            shape: KernelShapeConstraints::default(),
            memory_classes: BTreeSet::new(),
            devices: BTreeSet::new(),
            device_classes: Vec::new(),
            workspace: KernelWorkspaceRequirements::none(),
            execution_modes: [KernelExecutionMode::Synchronous].into_iter().collect(),
            cancellation: KernelCancellationSupport::NotSupported,
            determinism: KernelDeterminism::default(),
            precision: KernelPrecisionMetadata::default(),
            aliasing: KernelAliasing::default(),
            performance_hints: BTreeMap::new(),
            fallback_hints: BTreeSet::new(),
            required_provider_features: BTreeSet::new(),
            required_device_features: BTreeSet::new(),
            fusion: None,
            adapter: None,
            kv_cache: None,
            prefix_cache: None,
            batching: None,
            browser_compatible: false,
        }
    }

    pub fn with_dtypes(
        mut self,
        role: crate::TensorRole,
        dtypes: impl IntoIterator<Item = ComputeDType>,
    ) -> Self {
        self.supported_dtypes
            .entry(role)
            .or_default()
            .extend(dtypes);
        self
    }

    pub fn with_layouts(mut self, layouts: impl IntoIterator<Item = TensorLayoutKind>) -> Self {
        self.supported_layouts.extend(layouts);
        self
    }

    pub fn with_memory_classes(
        mut self,
        classes: impl IntoIterator<Item = KernelMemoryClass>,
    ) -> Self {
        self.memory_classes.extend(classes);
        self
    }

    pub fn with_devices(mut self, devices: impl IntoIterator<Item = DeviceBinding>) -> Self {
        self.devices.extend(devices);
        self
    }

    pub fn validate_invocation(
        &self,
        operator: &OperatorSpec,
        invocation: &KernelInvocation,
    ) -> Result<(), KernelError> {
        self.validate_operator(operator, &invocation.attributes)?;
        if invocation.kernel != self.id {
            return Err(KernelError::KernelNotFound {
                kernel: invocation.kernel.stable_key(),
            });
        }
        if !self.execution_modes.contains(&invocation.execution_mode) {
            return Err(KernelError::KernelExecutionFailed {
                reason: "execution mode unsupported".into(),
            });
        }
        if invocation.deterministic_required && !self.determinism.deterministic {
            return Err(KernelError::KernelDeterminismUnsupported);
        }
        if invocation.precision == ComputePrecision::Exact && self.precision.approximate_math {
            return Err(KernelError::KernelPrecisionUnsupported);
        }
        if self.workspace.required && invocation.workspace.is_none() {
            return Err(KernelError::KernelWorkspaceUnavailable);
        }
        if let Some(device) = invocation.device.as_ref()
            && !self.devices.is_empty()
            && !self.devices.contains(device)
        {
            return Err(KernelError::KernelDeviceUnsupported {
                device: device.to_string(),
            });
        }
        for input in &invocation.inputs {
            self.validate_resource(input, crate::TensorRole::Input)?;
        }
        for output in &invocation.outputs {
            self.validate_resource(output, crate::TensorRole::Output)?;
        }
        for output in &invocation.outputs {
            validate_affinity_compatibility(&invocation.affinity, &output.resource.affinity)
                .map_err(|error| KernelError::KernelResourceAffinityConflict {
                    reason: error.to_string(),
                })?;
        }
        let inputs = invocation
            .inputs
            .iter()
            .map(|resource| resource.resource.descriptor.clone())
            .collect::<Vec<_>>();
        let outputs = invocation
            .outputs
            .iter()
            .map(|resource| resource.resource.descriptor.clone())
            .collect::<Vec<_>>();
        operator
            .validate_invocation(&inputs, &outputs, &invocation.attributes)
            .map_err(KernelError::from_operator_error)?;
        if !self.aliasing.compatible_with_operator(operator.memory) {
            return Err(KernelError::KernelAliasingUnsupported);
        }
        Ok(())
    }

    fn validate_operator(
        &self,
        operator: &OperatorSpec,
        attributes: &BTreeMap<String, OperatorAttributeValue>,
    ) -> Result<(), KernelError> {
        if self.implemented_operator.namespace() != operator.id.namespace()
            || self.implemented_operator.name() != operator.id.name()
        {
            return Err(KernelError::KernelOperatorMismatch {
                expected: operator.id.to_string(),
                found: self.implemented_operator.to_string(),
            });
        }
        if !self.id.operator_versions.contains(operator.id.version()) {
            return Err(KernelError::KernelVersionUnsupported {
                kernel: self.id.stable_key(),
            });
        }
        operator
            .attributes
            .validate(attributes)
            .map_err(KernelError::from_operator_error)
    }

    fn validate_resource(
        &self,
        resource: &KernelResource,
        role: crate::TensorRole,
    ) -> Result<(), KernelError> {
        let descriptor = &resource.resource.descriptor;
        self.validate_shape(descriptor)?;
        self.validate_dtype(descriptor, role)?;
        let layout = layout_kind(&descriptor.layout);
        if !layout.component_visible() {
            return Err(KernelError::KernelLayoutUnsupported {
                layout: format!("{layout:?}"),
            });
        }
        if !self.supported_layouts.is_empty() && !self.supported_layouts.contains(&layout) {
            return Err(KernelError::KernelLayoutUnsupported {
                layout: format!("{layout:?}"),
            });
        }
        if !self.memory_classes.is_empty() && !self.memory_classes.contains(&resource.memory_class)
        {
            return Err(KernelError::KernelMemoryClassUnsupported {
                memory_class: format!("{:?}", resource.memory_class),
            });
        }
        Ok(())
    }

    fn validate_shape(&self, descriptor: &TensorDescriptor) -> Result<(), KernelError> {
        let rank = descriptor.shape.rank();
        if let Some(expected) = self.shape.rank
            && rank != expected
        {
            return Err(KernelError::KernelShapeUnsupported {
                reason: format!("expected rank {expected}, got {rank}"),
            });
        }
        for (index, expected) in &self.shape.static_dimensions {
            if descriptor.shape.dimensions.get(*index) != Some(expected) {
                return Err(KernelError::KernelShapeUnsupported {
                    reason: format!("dimension {index} must be {expected}"),
                });
            }
        }
        if let Some(alignment) = self.shape.alignment {
            for dimension in &descriptor.shape.dimensions {
                if dimension % alignment != 0 {
                    return Err(KernelError::KernelShapeUnsupported {
                        reason: format!("dimension {dimension} is not aligned to {alignment}"),
                    });
                }
            }
        }
        let elements = descriptor.shape.element_count().map_err(|error| {
            KernelError::KernelShapeUnsupported {
                reason: error.to_string(),
            }
        })?;
        if let Some(max) = self.shape.max_total_elements
            && elements > max
        {
            return Err(KernelError::KernelShapeUnsupported {
                reason: format!("element count {elements} exceeds {max}"),
            });
        }
        Ok(())
    }

    fn validate_dtype(
        &self,
        descriptor: &TensorDescriptor,
        role: crate::TensorRole,
    ) -> Result<(), KernelError> {
        let DTypeDescriptor::Portable(dtype) = descriptor.dtype else {
            return Err(KernelError::KernelDTypeUnsupported {
                dtype: "provider-specific".into(),
            });
        };
        if let Some(supported) = self.supported_dtypes.get(&role)
            && !supported.is_empty()
            && !supported.contains(&dtype)
        {
            return Err(KernelError::KernelDTypeUnsupported {
                dtype: format!("{dtype:?}"),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelResource {
    pub resource: TensorResourceDescriptor,
    pub memory_class: KernelMemoryClass,
}

impl KernelResource {
    pub const fn new(resource: TensorResourceDescriptor, memory_class: KernelMemoryClass) -> Self {
        Self {
            resource,
            memory_class,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelInvocation {
    pub id: KernelInvocationId,
    pub operator: OperatorId,
    pub kernel: KernelId,
    pub inputs: Vec<KernelResource>,
    pub outputs: Vec<KernelResource>,
    pub workspace: Option<MemoryAllocationId>,
    pub execution_mode: KernelExecutionMode,
    pub provider: ProviderBinding,
    pub device: Option<DeviceBinding>,
    pub affinity: ResourceAffinity,
    pub cancellation: Option<String>,
    pub deadline_millis: Option<u64>,
    pub observability_correlation: Option<String>,
    pub policy: BTreeMap<String, String>,
    pub attributes: BTreeMap<String, OperatorAttributeValue>,
    pub deterministic_required: bool,
    pub precision: ComputePrecision,
}

impl KernelInvocation {
    pub fn new(
        id: KernelInvocationId,
        operator: OperatorId,
        kernel: KernelId,
        provider: ProviderBinding,
        affinity: ResourceAffinity,
    ) -> Self {
        Self {
            id,
            operator,
            kernel,
            inputs: Vec::new(),
            outputs: Vec::new(),
            workspace: None,
            execution_mode: KernelExecutionMode::Synchronous,
            provider,
            device: None,
            affinity,
            cancellation: None,
            deadline_millis: None,
            observability_correlation: None,
            policy: BTreeMap::new(),
            attributes: BTreeMap::new(),
            deterministic_required: false,
            precision: ComputePrecision::Default,
        }
    }

    pub fn with_input(mut self, input: KernelResource) -> Self {
        self.inputs.push(input);
        self
    }

    pub fn with_output(mut self, output: KernelResource) -> Self {
        self.outputs.push(output);
        self
    }

    pub const fn with_workspace(mut self, workspace: MemoryAllocationId) -> Self {
        self.workspace = Some(workspace);
        self
    }

    pub fn with_attributes(mut self, attributes: BTreeMap<String, OperatorAttributeValue>) -> Self {
        self.attributes = attributes;
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelResult {
    pub invocation: KernelInvocationId,
    pub status: KernelResultStatus,
    pub output_readiness: BTreeMap<String, bool>,
    pub updated_resources: Vec<TensorResourceDescriptor>,
    /// Aliasing relationship Kernel execution left each named tensor in,
    /// where it differs from what was requested (e.g. an in-place op
    /// resolving `InputOutputAlias` on its output).
    pub updated_aliasing: BTreeMap<String, TensorAliasingKind>,
    pub workspace_release: Option<MemoryAllocationId>,
    pub timing_micros: Option<u64>,
    pub determinism: Option<KernelDeterminism>,
    pub precision: Option<KernelPrecisionMetadata>,
    pub provider_diagnostics: BTreeMap<String, String>,
    pub device_diagnostics: BTreeMap<String, String>,
    pub error: Option<KernelError>,
}

impl KernelResult {
    pub fn success(invocation: KernelInvocationId) -> Self {
        Self {
            invocation,
            status: KernelResultStatus::Succeeded,
            output_readiness: BTreeMap::new(),
            updated_resources: Vec::new(),
            updated_aliasing: BTreeMap::new(),
            workspace_release: None,
            timing_micros: None,
            determinism: None,
            precision: None,
            provider_diagnostics: BTreeMap::new(),
            device_diagnostics: BTreeMap::new(),
            error: None,
        }
    }

    pub fn failure(invocation: KernelInvocationId, error: KernelError) -> Self {
        Self {
            invocation,
            status: KernelResultStatus::Failed,
            output_readiness: BTreeMap::new(),
            updated_resources: Vec::new(),
            updated_aliasing: BTreeMap::new(),
            workspace_release: None,
            timing_micros: None,
            determinism: None,
            precision: None,
            provider_diagnostics: BTreeMap::new(),
            device_diagnostics: BTreeMap::new(),
            error: Some(error),
        }
    }

    pub fn with_aliasing_update(
        mut self,
        tensor_name: impl Into<String>,
        aliasing: TensorAliasingKind,
    ) -> Self {
        self.updated_aliasing.insert(tensor_name.into(), aliasing);
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelResultStatus {
    Succeeded,
    Failed,
    Cancelled,
    TimedOut,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelFallbackClass {
    AlternateKernel,
    AlternateProvider,
    AlternateDevice,
    ExplicitDTypeConversion,
    ExplicitLayoutConversion,
    HostExecution,
    Rejection,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelConformanceProfile {
    pub id: String,
    pub operator_semantics: bool,
    pub shape: bool,
    pub dtype: bool,
    pub layout: bool,
    pub memory: bool,
    pub aliasing: bool,
    pub workspace: bool,
    pub resource_affinity: bool,
    pub cancellation: bool,
    pub determinism: bool,
    pub precision: bool,
    pub error_mapping: bool,
    pub observability: bool,
}

impl KernelConformanceProfile {
    pub fn standard(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            operator_semantics: true,
            shape: true,
            dtype: true,
            layout: true,
            memory: true,
            aliasing: true,
            workspace: true,
            resource_affinity: true,
            cancellation: true,
            determinism: true,
            precision: true,
            error_mapping: true,
            observability: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelObservationKind {
    ProviderRegistered,
    DeviceDetected,
    KernelAdvertised,
    KernelAdvertisementReceived,
    KernelAdvertisementAccepted,
    KernelAdvertisementRejected,
    KernelRegistryUpdated,
    KernelRegistryInvalidated,
    KernelCandidateLookup,
    KernelCandidateRejected,
    KernelCandidateRanked,
    KernelSelected,
    KernelDispatchPlanCreated,
    KernelDispatchPlanRevalidated,
    KernelDispatchSubmitted,
    KernelDispatchRunning,
    KernelInvocationCreated,
    KernelDispatchStarted,
    KernelDispatchCompleted,
    KernelDispatchFailed,
    KernelMemoryFeasibilityFailed,
    KernelWorkspaceRequested,
    KernelCancellationRequested,
    KernelCancelled,
    KernelTimeout,
    KernelFallbackConsidered,
    KernelFallbackSelected,
    KernelFallbackFailed,
    KernelFallbackUsed,
    KernelConformanceResult,
    KernelConformanceGatingApplied,
    KernelResourceAffinityConflict,
    KernelDeterminismLimitation,
    KernelPrecisionDiagnostic,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelObservation {
    pub kind: KernelObservationKind,
    pub kernel: Option<String>,
    pub invocation: Option<KernelInvocationId>,
    pub redacted_metadata: BTreeMap<String, String>,
}

impl KernelObservation {
    pub fn new(kind: KernelObservationKind) -> Self {
        Self {
            kind,
            kernel: None,
            invocation: None,
            redacted_metadata: BTreeMap::new(),
        }
    }

    pub fn with_kernel(mut self, kernel: &KernelId) -> Self {
        self.kernel = Some(kernel.stable_key());
        self
    }

    pub fn with_invocation(mut self, invocation: KernelInvocationId) -> Self {
        self.invocation = Some(invocation);
        self
    }

    pub fn with_redacted_metadata(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.redacted_metadata.insert(key.into(), value.into());
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelErrorCode {
    KernelNotFound,
    KernelVersionUnsupported,
    KernelOperatorMismatch,
    KernelAttributeUnsupported,
    KernelShapeUnsupported,
    KernelDTypeUnsupported,
    KernelLayoutUnsupported,
    KernelMemoryClassUnsupported,
    KernelWorkspaceUnavailable,
    KernelAliasingUnsupported,
    KernelResourceAffinityConflict,
    KernelDeviceUnsupported,
    KernelProviderUnavailable,
    KernelProviderNotReady,
    KernelProviderSaturated,
    KernelExecutionFailed,
    KernelCancellationUnsupported,
    KernelCancelled,
    KernelTimeout,
    KernelDeterminismUnsupported,
    KernelPrecisionUnsupported,
    KernelConformanceFailed,
    KernelBrowserFeatureUnsupported,
    InternalKernel,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelError {
    KernelNotFound { kernel: String },
    KernelVersionUnsupported { kernel: String },
    KernelOperatorMismatch { expected: String, found: String },
    KernelAttributeUnsupported { attribute: String },
    KernelShapeUnsupported { reason: String },
    KernelDTypeUnsupported { dtype: String },
    KernelLayoutUnsupported { layout: String },
    KernelMemoryClassUnsupported { memory_class: String },
    KernelWorkspaceUnavailable,
    KernelAliasingUnsupported,
    KernelResourceAffinityConflict { reason: String },
    KernelDeviceUnsupported { device: String },
    KernelProviderUnavailable { provider: String },
    KernelProviderNotReady { provider: String },
    KernelProviderSaturated { provider: String },
    KernelExecutionFailed { reason: String },
    KernelCancellationUnsupported,
    KernelCancelled,
    KernelTimeout,
    KernelDeterminismUnsupported,
    KernelPrecisionUnsupported,
    KernelConformanceFailed { report: String },
    KernelBrowserFeatureUnsupported { feature: String },
    InternalKernel { reason: String },
}

impl KernelError {
    pub const fn id(&self) -> &'static str {
        self.code().id()
    }

    pub const fn code(&self) -> KernelErrorCode {
        match self {
            Self::KernelNotFound { .. } => KernelErrorCode::KernelNotFound,
            Self::KernelVersionUnsupported { .. } => KernelErrorCode::KernelVersionUnsupported,
            Self::KernelOperatorMismatch { .. } => KernelErrorCode::KernelOperatorMismatch,
            Self::KernelAttributeUnsupported { .. } => KernelErrorCode::KernelAttributeUnsupported,
            Self::KernelShapeUnsupported { .. } => KernelErrorCode::KernelShapeUnsupported,
            Self::KernelDTypeUnsupported { .. } => KernelErrorCode::KernelDTypeUnsupported,
            Self::KernelLayoutUnsupported { .. } => KernelErrorCode::KernelLayoutUnsupported,
            Self::KernelMemoryClassUnsupported { .. } => {
                KernelErrorCode::KernelMemoryClassUnsupported
            }
            Self::KernelWorkspaceUnavailable => KernelErrorCode::KernelWorkspaceUnavailable,
            Self::KernelAliasingUnsupported => KernelErrorCode::KernelAliasingUnsupported,
            Self::KernelResourceAffinityConflict { .. } => {
                KernelErrorCode::KernelResourceAffinityConflict
            }
            Self::KernelDeviceUnsupported { .. } => KernelErrorCode::KernelDeviceUnsupported,
            Self::KernelProviderUnavailable { .. } => KernelErrorCode::KernelProviderUnavailable,
            Self::KernelProviderNotReady { .. } => KernelErrorCode::KernelProviderNotReady,
            Self::KernelProviderSaturated { .. } => KernelErrorCode::KernelProviderSaturated,
            Self::KernelExecutionFailed { .. } => KernelErrorCode::KernelExecutionFailed,
            Self::KernelCancellationUnsupported => KernelErrorCode::KernelCancellationUnsupported,
            Self::KernelCancelled => KernelErrorCode::KernelCancelled,
            Self::KernelTimeout => KernelErrorCode::KernelTimeout,
            Self::KernelDeterminismUnsupported => KernelErrorCode::KernelDeterminismUnsupported,
            Self::KernelPrecisionUnsupported => KernelErrorCode::KernelPrecisionUnsupported,
            Self::KernelConformanceFailed { .. } => KernelErrorCode::KernelConformanceFailed,
            Self::KernelBrowserFeatureUnsupported { .. } => {
                KernelErrorCode::KernelBrowserFeatureUnsupported
            }
            Self::InternalKernel { .. } => KernelErrorCode::InternalKernel,
        }
    }

    fn from_operator_error(error: crate::OperatorError) -> Self {
        use crate::OperatorError;
        match error {
            OperatorError::OperatorNotFound { operator }
            | OperatorError::OperatorVersionUnsupported { operator } => {
                Self::KernelOperatorMismatch {
                    expected: operator.to_string(),
                    found: "unavailable".into(),
                }
            }
            OperatorError::OperatorAttributeInvalid { attribute, .. } => {
                Self::KernelAttributeUnsupported { attribute }
            }
            OperatorError::InputArityInvalid { .. } | OperatorError::OutputArityInvalid { .. } => {
                Self::KernelOperatorMismatch {
                    expected: "operator arity".into(),
                    found: "kernel invocation arity".into(),
                }
            }
            OperatorError::ShapeMismatch { reason }
            | OperatorError::ShapeUnsupported { reason } => Self::KernelShapeUnsupported { reason },
            OperatorError::DTypeUnsupported { dtype }
            | OperatorError::DTypeConversionRequired { from: dtype, .. }
            | OperatorError::DTypeConversionUnsupported { from: dtype, .. } => {
                Self::KernelDTypeUnsupported {
                    dtype: format!("{dtype:?}"),
                }
            }
            OperatorError::LayoutUnsupported { layout }
            | OperatorError::LayoutConversionRequired { from: layout, .. }
            | OperatorError::LayoutConversionUnsupported { from: layout, .. } => {
                Self::KernelLayoutUnsupported {
                    layout: format!("{layout:?}"),
                }
            }
            OperatorError::MemoryBehaviorUnsupported { reason } => {
                Self::KernelExecutionFailed { reason }
            }
            OperatorError::WorkspaceUnavailable { .. } => Self::KernelWorkspaceUnavailable,
            OperatorError::ResourceAffinityConflict { reason } => {
                Self::KernelResourceAffinityConflict { reason }
            }
            OperatorError::ProviderCapabilityUnavailable { capability } => {
                Self::KernelProviderUnavailable {
                    provider: capability,
                }
            }
            OperatorError::KernelUnavailable { operator } => Self::KernelNotFound {
                kernel: operator.to_string(),
            },
            OperatorError::BrowserFeatureUnsupported { feature } => {
                Self::KernelBrowserFeatureUnsupported { feature }
            }
            other => Self::InternalKernel {
                reason: other.to_string(),
            },
        }
    }
}

impl KernelErrorCode {
    pub const fn id(self) -> &'static str {
        match self {
            Self::KernelNotFound => "kernel-not-found",
            Self::KernelVersionUnsupported => "kernel-version-unsupported",
            Self::KernelOperatorMismatch => "kernel-operator-mismatch",
            Self::KernelAttributeUnsupported => "kernel-attribute-unsupported",
            Self::KernelShapeUnsupported => "kernel-shape-unsupported",
            Self::KernelDTypeUnsupported => "kernel-dtype-unsupported",
            Self::KernelLayoutUnsupported => "kernel-layout-unsupported",
            Self::KernelMemoryClassUnsupported => "kernel-memory-class-unsupported",
            Self::KernelWorkspaceUnavailable => "kernel-workspace-unavailable",
            Self::KernelAliasingUnsupported => "kernel-aliasing-unsupported",
            Self::KernelResourceAffinityConflict => "kernel-resource-affinity-conflict",
            Self::KernelDeviceUnsupported => "kernel-device-unsupported",
            Self::KernelProviderUnavailable => "kernel-provider-unavailable",
            Self::KernelProviderNotReady => "kernel-provider-not-ready",
            Self::KernelProviderSaturated => "kernel-provider-saturated",
            Self::KernelExecutionFailed => "kernel-execution-failed",
            Self::KernelCancellationUnsupported => "kernel-cancellation-unsupported",
            Self::KernelCancelled => "kernel-cancelled",
            Self::KernelTimeout => "kernel-timeout",
            Self::KernelDeterminismUnsupported => "kernel-determinism-unsupported",
            Self::KernelPrecisionUnsupported => "kernel-precision-unsupported",
            Self::KernelConformanceFailed => "kernel-conformance-failed",
            Self::KernelBrowserFeatureUnsupported => "kernel-browser-feature-unsupported",
            Self::InternalKernel => "internal-kernel",
        }
    }
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KernelNotFound { kernel } => write!(f, "kernel not found: {kernel}"),
            Self::KernelVersionUnsupported { kernel } => {
                write!(f, "kernel version unsupported: {kernel}")
            }
            Self::KernelOperatorMismatch { expected, found } => {
                write!(
                    f,
                    "kernel Operator mismatch: expected {expected}, found {found}"
                )
            }
            Self::KernelAttributeUnsupported { attribute } => {
                write!(f, "kernel attribute unsupported: {attribute}")
            }
            Self::KernelShapeUnsupported { reason } => {
                write!(f, "kernel shape unsupported: {reason}")
            }
            Self::KernelDTypeUnsupported { dtype } => {
                write!(f, "kernel dtype unsupported: {dtype}")
            }
            Self::KernelLayoutUnsupported { layout } => {
                write!(f, "kernel layout unsupported: {layout}")
            }
            Self::KernelMemoryClassUnsupported { memory_class } => {
                write!(f, "kernel memory class unsupported: {memory_class}")
            }
            Self::KernelWorkspaceUnavailable => write!(f, "kernel workspace unavailable"),
            Self::KernelAliasingUnsupported => write!(f, "kernel aliasing unsupported"),
            Self::KernelResourceAffinityConflict { reason } => {
                write!(f, "kernel Resource Affinity conflict: {reason}")
            }
            Self::KernelDeviceUnsupported { device } => {
                write!(f, "kernel Device unsupported: {device}")
            }
            Self::KernelProviderUnavailable { provider } => {
                write!(f, "kernel Provider unavailable: {provider}")
            }
            Self::KernelProviderNotReady { provider } => {
                write!(f, "kernel Provider not ready: {provider}")
            }
            Self::KernelProviderSaturated { provider } => {
                write!(f, "kernel Provider saturated: {provider}")
            }
            Self::KernelExecutionFailed { reason } => {
                write!(f, "kernel execution failed: {reason}")
            }
            Self::KernelCancellationUnsupported => write!(f, "kernel cancellation unsupported"),
            Self::KernelCancelled => write!(f, "kernel cancelled"),
            Self::KernelTimeout => write!(f, "kernel timeout"),
            Self::KernelDeterminismUnsupported => write!(f, "kernel determinism unsupported"),
            Self::KernelPrecisionUnsupported => write!(f, "kernel precision unsupported"),
            Self::KernelConformanceFailed { report } => {
                write!(f, "kernel conformance failed: {report}")
            }
            Self::KernelBrowserFeatureUnsupported { feature } => {
                write!(f, "kernel browser feature unsupported: {feature}")
            }
            Self::InternalKernel { reason } => write!(f, "internal kernel error: {reason}"),
        }
    }
}

impl Error for KernelError {}
