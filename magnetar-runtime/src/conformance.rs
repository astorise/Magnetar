use crate::affinity::*;
use crate::component::*;
use crate::compute::redact_backend_diagnostic;
use crate::compute::*;
use crate::device::DeviceType;
use crate::provider::*;
use crate::reference_cpu::{REFERENCE_CPU_DEVICE_ID, REFERENCE_CPU_PROVIDER_NAME};
use crate::runtime::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

pub const PROVIDER_CONFORMANCE_SUITE_VERSION: &str = "0.1.0";
pub const FIRST_NATIVE_MODEL_EXECUTION_PROFILE_VERSION: &str = "0.1.0";

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
pub enum FirstNativeProfileCapability {
    LocalRuntime,
    PlatformComponentEngine,
    WasmtimeComponentEngine,
    ModelWasmComponent,
    ModelArtifact,
    Tokenizer,
    OperatorCatalog,
    ExecutionGraph,
    PreparedExecutionPlan,
    KernelRegistry,
    ReferenceCpuProvider,
    LogicalCpuDevice,
    F32Execution,
    TensorResource,
    RuntimeMemoryManager,
    KvCache,
    IncrementalDecode,
    GreedySampling,
    StreamingOutput,
    ObservabilityRedaction,
}

impl FirstNativeProfileCapability {
    pub const fn id(self) -> &'static str {
        match self {
            Self::LocalRuntime => "local-runtime",
            Self::PlatformComponentEngine => "platform-component-engine",
            Self::WasmtimeComponentEngine => "wasmtime-component-engine",
            Self::ModelWasmComponent => "model-wasm-component",
            Self::ModelArtifact => "model-artifact",
            Self::Tokenizer => "tokenizer",
            Self::OperatorCatalog => "operator-catalog",
            Self::ExecutionGraph => "execution-graph",
            Self::PreparedExecutionPlan => "prepared-execution-plan",
            Self::KernelRegistry => "kernel-registry",
            Self::ReferenceCpuProvider => "reference-cpu-provider",
            Self::LogicalCpuDevice => "logical-cpu-device",
            Self::F32Execution => "f32-execution",
            Self::TensorResource => "tensor-resource",
            Self::RuntimeMemoryManager => "runtime-memory-manager",
            Self::KvCache => "kv-cache",
            Self::IncrementalDecode => "incremental-decode",
            Self::GreedySampling => "greedy-sampling",
            Self::StreamingOutput => "streaming-output",
            Self::ObservabilityRedaction => "observability-redaction",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
pub enum FirstNativeDeferredCapability {
    MultiDevicePlacement,
    TensorParallel,
    Collectives,
    GeneratedKernels,
    ProviderRuntimeCompilation,
    KernelArtifactIngestion,
    HotSwap,
    RuntimeAutotuning,
    AdaptivePerformanceFeedback,
    PerformanceModelReplacement,
    AcceleratedProviders,
    ReducedPrecision,
    Quantization,
    CrossProviderZeroCopy,
    AdvancedMemoryPools,
    PagedKvCache,
    PrefixCacheOptimization,
    ProductionContinuousBatching,
    AdvancedAsyncExecutionStreams,
}

impl FirstNativeDeferredCapability {
    pub const fn id(self) -> &'static str {
        match self {
            Self::MultiDevicePlacement => "multi-device-placement",
            Self::TensorParallel => "tensor-parallel",
            Self::Collectives => "collectives",
            Self::GeneratedKernels => "generated-kernels",
            Self::ProviderRuntimeCompilation => "provider-runtime-compilation",
            Self::KernelArtifactIngestion => "kernel-artifact-ingestion",
            Self::HotSwap => "hot-swap",
            Self::RuntimeAutotuning => "runtime-autotuning",
            Self::AdaptivePerformanceFeedback => "adaptive-performance-feedback",
            Self::PerformanceModelReplacement => "performance-model-replacement",
            Self::AcceleratedProviders => "accelerated-providers",
            Self::ReducedPrecision => "reduced-precision",
            Self::Quantization => "quantization",
            Self::CrossProviderZeroCopy => "cross-provider-zero-copy",
            Self::AdvancedMemoryPools => "advanced-memory-pools",
            Self::PagedKvCache => "paged-kv-cache",
            Self::PrefixCacheOptimization => "prefix-cache-optimization",
            Self::ProductionContinuousBatching => "production-continuous-batching",
            Self::AdvancedAsyncExecutionStreams => "advanced-async-execution-streams",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct FirstNativeModelExecutionProfile {
    pub version: String,
    pub mandatory_capabilities: BTreeSet<FirstNativeProfileCapability>,
    pub deferred_capabilities: BTreeSet<FirstNativeDeferredCapability>,
}

impl Default for FirstNativeModelExecutionProfile {
    fn default() -> Self {
        first_native_model_execution_profile()
    }
}

impl FirstNativeModelExecutionProfile {
    pub fn validate(&self) -> Result<(), FirstNativeModelExecutionProfileError> {
        if self.version.trim().is_empty() {
            return Err(FirstNativeModelExecutionProfileError::MissingVersion);
        }
        let missing_mandatory = first_native_mandatory_capabilities()
            .difference(&self.mandatory_capabilities)
            .copied()
            .collect::<Vec<_>>();
        if !missing_mandatory.is_empty() {
            return Err(
                FirstNativeModelExecutionProfileError::MissingMandatoryCapability(
                    missing_mandatory[0],
                ),
            );
        }
        let missing_deferred = first_native_deferred_capabilities()
            .difference(&self.deferred_capabilities)
            .copied()
            .collect::<Vec<_>>();
        if !missing_deferred.is_empty() {
            return Err(
                FirstNativeModelExecutionProfileError::MissingDeferredCapability(
                    missing_deferred[0],
                ),
            );
        }
        Ok(())
    }

    pub fn mandatory_ids(&self) -> BTreeSet<&'static str> {
        self.mandatory_capabilities
            .iter()
            .map(|capability| capability.id())
            .collect()
    }

    pub fn deferred_ids(&self) -> BTreeSet<&'static str> {
        self.deferred_capabilities
            .iter()
            .map(|capability| capability.id())
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FirstNativeModelExecutionProfileError {
    MissingVersion,
    MissingMandatoryCapability(FirstNativeProfileCapability),
    MissingDeferredCapability(FirstNativeDeferredCapability),
    RuntimeNotInitialized,
    ReferenceCpuProviderMissing,
    LogicalCpuDeviceMissing,
    LogicalCpuDeviceNotCpu,
    ComponentEngineProfileMismatch,
    ComponentEngineFeatureMissing(ComponentEngineFeature),
}

impl std::fmt::Display for FirstNativeModelExecutionProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingVersion => write!(f, "first native profile version is missing"),
            Self::MissingMandatoryCapability(capability) => {
                write!(f, "mandatory capability '{}' is missing", capability.id())
            }
            Self::MissingDeferredCapability(capability) => {
                write!(f, "deferred capability '{}' is missing", capability.id())
            }
            Self::RuntimeNotInitialized => write!(f, "runtime is not initialized"),
            Self::ReferenceCpuProviderMissing => {
                write!(f, "Reference CPU Provider is missing from runtime")
            }
            Self::LogicalCpuDeviceMissing => {
                write!(f, "Reference CPU logical Device is missing from runtime")
            }
            Self::LogicalCpuDeviceNotCpu => {
                write!(f, "Reference CPU logical Device is not a CPU Device")
            }
            Self::ComponentEngineProfileMismatch => {
                write!(f, "Component Engine is not using the native profile")
            }
            Self::ComponentEngineFeatureMissing(feature) => {
                write!(
                    f,
                    "Component Engine feature '{}' is missing",
                    feature.as_str()
                )
            }
        }
    }
}

impl std::error::Error for FirstNativeModelExecutionProfileError {}

pub fn first_native_mandatory_capabilities() -> BTreeSet<FirstNativeProfileCapability> {
    BTreeSet::from([
        FirstNativeProfileCapability::LocalRuntime,
        FirstNativeProfileCapability::PlatformComponentEngine,
        FirstNativeProfileCapability::WasmtimeComponentEngine,
        FirstNativeProfileCapability::ModelWasmComponent,
        FirstNativeProfileCapability::ModelArtifact,
        FirstNativeProfileCapability::Tokenizer,
        FirstNativeProfileCapability::OperatorCatalog,
        FirstNativeProfileCapability::ExecutionGraph,
        FirstNativeProfileCapability::PreparedExecutionPlan,
        FirstNativeProfileCapability::KernelRegistry,
        FirstNativeProfileCapability::ReferenceCpuProvider,
        FirstNativeProfileCapability::LogicalCpuDevice,
        FirstNativeProfileCapability::F32Execution,
        FirstNativeProfileCapability::TensorResource,
        FirstNativeProfileCapability::RuntimeMemoryManager,
        FirstNativeProfileCapability::KvCache,
        FirstNativeProfileCapability::IncrementalDecode,
        FirstNativeProfileCapability::GreedySampling,
        FirstNativeProfileCapability::StreamingOutput,
        FirstNativeProfileCapability::ObservabilityRedaction,
    ])
}

pub fn first_native_deferred_capabilities() -> BTreeSet<FirstNativeDeferredCapability> {
    BTreeSet::from([
        FirstNativeDeferredCapability::MultiDevicePlacement,
        FirstNativeDeferredCapability::TensorParallel,
        FirstNativeDeferredCapability::Collectives,
        FirstNativeDeferredCapability::GeneratedKernels,
        FirstNativeDeferredCapability::ProviderRuntimeCompilation,
        FirstNativeDeferredCapability::KernelArtifactIngestion,
        FirstNativeDeferredCapability::HotSwap,
        FirstNativeDeferredCapability::RuntimeAutotuning,
        FirstNativeDeferredCapability::AdaptivePerformanceFeedback,
        FirstNativeDeferredCapability::PerformanceModelReplacement,
        FirstNativeDeferredCapability::AcceleratedProviders,
        FirstNativeDeferredCapability::ReducedPrecision,
        FirstNativeDeferredCapability::Quantization,
        FirstNativeDeferredCapability::CrossProviderZeroCopy,
        FirstNativeDeferredCapability::AdvancedMemoryPools,
        FirstNativeDeferredCapability::PagedKvCache,
        FirstNativeDeferredCapability::PrefixCacheOptimization,
        FirstNativeDeferredCapability::ProductionContinuousBatching,
        FirstNativeDeferredCapability::AdvancedAsyncExecutionStreams,
    ])
}

pub fn first_native_model_execution_profile() -> FirstNativeModelExecutionProfile {
    FirstNativeModelExecutionProfile {
        version: FIRST_NATIVE_MODEL_EXECUTION_PROFILE_VERSION.into(),
        mandatory_capabilities: first_native_mandatory_capabilities(),
        deferred_capabilities: first_native_deferred_capabilities(),
    }
}

pub fn validate_first_native_single_host_topology(
    runtime: &Runtime,
) -> Result<(), FirstNativeModelExecutionProfileError> {
    first_native_model_execution_profile().validate()?;
    if !runtime.is_initialized() {
        return Err(FirstNativeModelExecutionProfileError::RuntimeNotInitialized);
    }
    if runtime
        .providers()
        .provider(REFERENCE_CPU_PROVIDER_NAME)
        .is_none()
    {
        return Err(FirstNativeModelExecutionProfileError::ReferenceCpuProviderMissing);
    }
    let device = runtime
        .device(&crate::DeviceId::new(REFERENCE_CPU_DEVICE_ID))
        .ok_or(FirstNativeModelExecutionProfileError::LogicalCpuDeviceMissing)?;
    if device.metadata().device_type != DeviceType::Cpu {
        return Err(FirstNativeModelExecutionProfileError::LogicalCpuDeviceNotCpu);
    }
    Ok(())
}

pub fn validate_first_native_component_engine_capabilities(
    capabilities: &ComponentEngineCapabilities,
) -> Result<(), FirstNativeModelExecutionProfileError> {
    first_native_model_execution_profile().validate()?;
    if capabilities.profile != ComponentEngineProfile::Native {
        return Err(FirstNativeModelExecutionProfileError::ComponentEngineProfileMismatch);
    }
    for feature in [
        ComponentEngineFeature::ComponentModel,
        ComponentEngineFeature::ResourceLimits,
        ComponentEngineFeature::NativeProviderEndpoints,
        ComponentEngineFeature::ControlledWasi,
    ] {
        if !capabilities.supports(feature) {
            return Err(
                FirstNativeModelExecutionProfileError::ComponentEngineFeatureMissing(feature),
            );
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
pub enum ProviderConformanceProfile {
    ProviderCore,
    ProviderCompute,
    ProviderDataMovement,
    ProviderCancellation,
    ProviderObservability,
    ProviderDynamicAbi,
    Cuda,
    Metal,
    OpenVino,
    Qnn,
    /// Post-baseline WebGPU hardware profile (`define-post-baseline-provider-roadmap`).
    WebGpu,
    /// Post-baseline quantized execution profile.
    Quantized,
    /// Post-baseline advanced attention (flash/paged/sliding-window/...) profile.
    AdvancedAttention,
    /// Post-baseline fused-kernel profile.
    FusedKernel,
    /// Post-baseline browser-execution profile (WebGPU/WASM constraints).
    Browser,
}

impl ProviderConformanceProfile {
    pub const fn id(self) -> &'static str {
        match self {
            Self::ProviderCore => "provider-core",
            Self::ProviderCompute => "provider-compute",
            Self::ProviderDataMovement => "provider-data-movement",
            Self::ProviderCancellation => "provider-cancellation",
            Self::ProviderObservability => "provider-observability",
            Self::ProviderDynamicAbi => "provider-dynamic-abi",
            Self::Cuda => "provider-hardware-cuda",
            Self::Metal => "provider-hardware-metal",
            Self::OpenVino => "provider-hardware-openvino",
            Self::Qnn => "provider-hardware-qnn",
            Self::WebGpu => "provider-hardware-webgpu",
            Self::Quantized => "provider-quantized",
            Self::AdvancedAttention => "provider-advanced-attention",
            Self::FusedKernel => "provider-fused-kernel",
            Self::Browser => "provider-browser",
        }
    }

    pub const fn required_by_default(self) -> bool {
        matches!(
            self,
            Self::ProviderCore
                | Self::ProviderCompute
                | Self::ProviderDataMovement
                | Self::ProviderCancellation
                | Self::ProviderObservability
                | Self::ProviderDynamicAbi
        )
    }
}

#[derive(Clone)]
pub enum ProviderConformanceTarget {
    BuiltIn {
        provider: Arc<dyn Provider>,
    },
    Mock {
        provider: Arc<dyn Provider>,
    },
    DynamicLibrary {
        path: PathBuf,
        policy: ProviderLoadingPolicy,
    },
    Development {
        path: PathBuf,
        policy: ProviderLoadingPolicy,
    },
}

impl ProviderConformanceTarget {
    pub fn built_in(provider: Arc<dyn Provider>) -> Self {
        Self::BuiltIn { provider }
    }

    pub fn mock(provider: Arc<dyn Provider>) -> Self {
        Self::Mock { provider }
    }

    pub fn dynamic_library(path: impl Into<PathBuf>, policy: ProviderLoadingPolicy) -> Self {
        Self::DynamicLibrary {
            path: path.into(),
            policy,
        }
    }

    pub fn development(path: impl Into<PathBuf>, policy: ProviderLoadingPolicy) -> Self {
        Self::Development {
            path: path.into(),
            policy,
        }
    }

    pub const fn kind(&self) -> ProviderConformanceTargetKind {
        match self {
            Self::BuiltIn { .. } => ProviderConformanceTargetKind::BuiltIn,
            Self::Mock { .. } => ProviderConformanceTargetKind::Mock,
            Self::DynamicLibrary { .. } => ProviderConformanceTargetKind::DynamicLibrary,
            Self::Development { .. } => ProviderConformanceTargetKind::Development,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
pub enum ProviderConformanceTargetKind {
    BuiltIn,
    DynamicLibrary,
    Mock,
    Development,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
pub enum ProviderConformanceTestStatus {
    Passed,
    Failed,
    Skipped,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct ProviderConformanceTestResult {
    pub profile: ProviderConformanceProfile,
    pub requirement: String,
    pub status: ProviderConformanceTestStatus,
    pub diagnostic: Option<String>,
}

impl ProviderConformanceTestResult {
    pub fn passed(profile: ProviderConformanceProfile, requirement: impl Into<String>) -> Self {
        Self {
            profile,
            requirement: requirement.into(),
            status: ProviderConformanceTestStatus::Passed,
            diagnostic: None,
        }
    }

    pub fn failed(
        profile: ProviderConformanceProfile,
        requirement: impl Into<String>,
        diagnostic: impl Into<String>,
    ) -> Self {
        Self {
            profile,
            requirement: requirement.into(),
            status: ProviderConformanceTestStatus::Failed,
            diagnostic: Some(redact_backend_diagnostic(&diagnostic.into())),
        }
    }

    pub fn skipped(
        profile: ProviderConformanceProfile,
        requirement: impl Into<String>,
        diagnostic: impl Into<String>,
    ) -> Self {
        Self {
            profile,
            requirement: requirement.into(),
            status: ProviderConformanceTestStatus::Skipped,
            diagnostic: Some(redact_backend_diagnostic(&diagnostic.into())),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct ProviderConformanceReport {
    pub provider_identity: String,
    pub provider_version: String,
    pub runtime_version: String,
    pub suite_version: String,
    pub target_kind: ProviderConformanceTargetKind,
    pub selected_profiles: BTreeSet<ProviderConformanceProfile>,
    pub passed_tests: Vec<ProviderConformanceTestResult>,
    pub failed_tests: Vec<ProviderConformanceTestResult>,
    pub skipped_tests: Vec<ProviderConformanceTestResult>,
    pub unsupported_optional_features: BTreeSet<String>,
    pub diagnostics: Vec<String>,
    pub timestamp_unix_seconds: u64,
}

impl ProviderConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.failed_tests.is_empty()
    }

    pub fn record(&mut self, result: ProviderConformanceTestResult) {
        match result.status {
            ProviderConformanceTestStatus::Passed => self.passed_tests.push(result),
            ProviderConformanceTestStatus::Failed => self.failed_tests.push(result),
            ProviderConformanceTestStatus::Skipped => self.skipped_tests.push(result),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderConformanceConfig {
    pub profiles: BTreeSet<ProviderConformanceProfile>,
    pub runtime_config: RuntimeConfig,
}

impl Default for ProviderConformanceConfig {
    fn default() -> Self {
        Self {
            profiles: BTreeSet::from([ProviderConformanceProfile::ProviderCore]),
            runtime_config: RuntimeConfig::default(),
        }
    }
}

impl ProviderConformanceConfig {
    pub fn with_profiles(
        mut self,
        profiles: impl IntoIterator<Item = ProviderConformanceProfile>,
    ) -> Self {
        self.profiles = profiles.into_iter().collect();
        self
    }
}

#[derive(Clone, Debug, Default)]
pub struct ProviderConformanceSuite {
    config: ProviderConformanceConfig,
}

impl ProviderConformanceSuite {
    pub fn new(config: ProviderConformanceConfig) -> Self {
        Self { config }
    }

    pub fn run(&self, target: ProviderConformanceTarget) -> ProviderConformanceReport {
        match target {
            ProviderConformanceTarget::BuiltIn { provider } => {
                self.run_provider(provider, ProviderConformanceTargetKind::BuiltIn)
            }
            ProviderConformanceTarget::Mock { provider } => {
                self.run_provider(provider, ProviderConformanceTargetKind::Mock)
            }
            ProviderConformanceTarget::DynamicLibrary { path, policy } => {
                self.run_dynamic(path, policy, ProviderConformanceTargetKind::DynamicLibrary)
            }
            ProviderConformanceTarget::Development { path, policy } => {
                self.run_dynamic(path, policy, ProviderConformanceTargetKind::Development)
            }
        }
    }

    fn run_provider(
        &self,
        provider: Arc<dyn Provider>,
        target_kind: ProviderConformanceTargetKind,
    ) -> ProviderConformanceReport {
        let metadata = provider.metadata();
        let mut report = empty_report(&metadata, target_kind, self.config.profiles.clone());
        let runtime = Runtime::builder()
            .config(self.config.runtime_config.clone())
            .register_provider(provider)
            .build();

        let runtime = match runtime {
            Ok(runtime) => {
                report.record(ProviderConformanceTestResult::passed(
                    ProviderConformanceProfile::ProviderCore,
                    "runtime registration",
                ));
                runtime
            }
            Err(error) => {
                report.record(ProviderConformanceTestResult::failed(
                    ProviderConformanceProfile::ProviderCore,
                    "runtime registration",
                    error.to_string(),
                ));
                return report;
            }
        };

        let Some(provider) = runtime.providers().provider(&metadata.name) else {
            report.record(ProviderConformanceTestResult::failed(
                ProviderConformanceProfile::ProviderCore,
                "registered provider lookup",
                "provider was not available after registration",
            ));
            return report;
        };

        for profile in self.effective_profiles(&metadata) {
            match profile {
                ProviderConformanceProfile::ProviderCore => {
                    validate_core(provider, &runtime, &mut report)
                }
                ProviderConformanceProfile::ProviderCompute => {
                    validate_compute(provider, &runtime, &mut report)
                }
                ProviderConformanceProfile::ProviderDataMovement => {
                    validate_data_movement(provider, &runtime, &mut report)
                }
                ProviderConformanceProfile::ProviderCancellation => {
                    validate_cancellation(provider, &mut report)
                }
                ProviderConformanceProfile::ProviderObservability => {
                    validate_observability(provider, &mut report)
                }
                ProviderConformanceProfile::ProviderDynamicAbi => {
                    report.record(ProviderConformanceTestResult::skipped(
                        profile,
                        "dynamic ABI",
                        "provider target is not a dynamic library",
                    ))
                }
                ProviderConformanceProfile::Cuda
                | ProviderConformanceProfile::Metal
                | ProviderConformanceProfile::OpenVino
                | ProviderConformanceProfile::Qnn
                | ProviderConformanceProfile::WebGpu => {
                    report
                        .unsupported_optional_features
                        .insert(profile.id().into());
                    report.record(ProviderConformanceTestResult::skipped(
                        profile,
                        "optional hardware profile",
                        "hardware-specific profiles are opt-in and not part of default CI",
                    ));
                }
                ProviderConformanceProfile::Quantized
                | ProviderConformanceProfile::AdvancedAttention
                | ProviderConformanceProfile::FusedKernel
                | ProviderConformanceProfile::Browser => {
                    // Post-baseline roadmap profiles (`define-post-baseline-provider-roadmap`):
                    // declaring them here does not imply a Provider has
                    // implemented or passed them -- see
                    // `crate::provider_roadmap::run_provider_roadmap_conformance`
                    // for the roadmap-level checks these profiles gate.
                    report
                        .unsupported_optional_features
                        .insert(profile.id().into());
                    report.record(ProviderConformanceTestResult::skipped(
                        profile,
                        "optional post-baseline roadmap profile",
                        "post-baseline roadmap profiles are opt-in and not part of default CI",
                    ));
                }
            }
        }

        report
    }

    fn run_dynamic(
        &self,
        path: PathBuf,
        policy: ProviderLoadingPolicy,
        target_kind: ProviderConformanceTargetKind,
    ) -> ProviderConformanceReport {
        let metadata = ProviderMetadata::new(
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("dynamic-provider"),
            "unknown",
            "unknown",
            "dynamic provider conformance target",
        );
        let mut report = empty_report(&metadata, target_kind, self.config.profiles.clone());
        if !policy.development_mode
            && matches!(target_kind, ProviderConformanceTargetKind::Development)
        {
            report.record(ProviderConformanceTestResult::failed(
                ProviderConformanceProfile::ProviderDynamicAbi,
                "development mode explicit",
                "development target requires an explicit development loading policy",
            ));
        }
        if !policy.allows(&path) {
            report.record(ProviderConformanceTestResult::failed(
                ProviderConformanceProfile::ProviderDynamicAbi,
                "allowed path loading",
                format!("provider path '{}' is denied by policy", path.display()),
            ));
            return report;
        }
        report.record(ProviderConformanceTestResult::passed(
            ProviderConformanceProfile::ProviderDynamicAbi,
            "allowed path loading",
        ));
        match ProviderAbiDescriptor::current().validate() {
            Ok(()) => report.record(ProviderConformanceTestResult::passed(
                ProviderConformanceProfile::ProviderDynamicAbi,
                "ABI descriptor structure",
            )),
            Err(error) => report.record(ProviderConformanceTestResult::failed(
                ProviderConformanceProfile::ProviderDynamicAbi,
                "ABI descriptor structure",
                error.to_string(),
            )),
        }
        report.record(ProviderConformanceTestResult::skipped(
            ProviderConformanceProfile::ProviderDynamicAbi,
            "factory symbol exists",
            "dynamic library loading is policy-defined but not implemented in this runtime revision",
        ));
        report
    }

    fn effective_profiles(
        &self,
        metadata: &ProviderMetadata,
    ) -> BTreeSet<ProviderConformanceProfile> {
        let mut profiles = self.config.profiles.clone();
        profiles.insert(ProviderConformanceProfile::ProviderCore);
        let advertises_compute = metadata.capabilities.iter().any(|capability| {
            capability.id.as_str() == COMPUTE_CAPABILITY_ID
                && capability
                    .version
                    .is_compatible_with(COMPUTE_CAPABILITY_VERSION)
        }) || !metadata.compute_advertisement.is_empty();
        if advertises_compute {
            profiles.insert(ProviderConformanceProfile::ProviderCompute);
        }
        if !metadata.compute_advertisement.data_movement.is_empty() {
            profiles.insert(ProviderConformanceProfile::ProviderDataMovement);
        }
        profiles
    }
}

fn empty_report(
    metadata: &ProviderMetadata,
    target_kind: ProviderConformanceTargetKind,
    selected_profiles: BTreeSet<ProviderConformanceProfile>,
) -> ProviderConformanceReport {
    ProviderConformanceReport {
        provider_identity: metadata.name.clone(),
        provider_version: metadata.version.clone(),
        runtime_version: MAGNETAR_RUNTIME_VERSION.into(),
        suite_version: PROVIDER_CONFORMANCE_SUITE_VERSION.into(),
        target_kind,
        selected_profiles,
        passed_tests: Vec::new(),
        failed_tests: Vec::new(),
        skipped_tests: Vec::new(),
        unsupported_optional_features: BTreeSet::new(),
        diagnostics: Vec::new(),
        timestamp_unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or_default(),
    }
}

fn validate_core(
    provider: &dyn Provider,
    runtime: &Runtime,
    report: &mut ProviderConformanceReport,
) {
    let metadata = provider.metadata();
    check(
        report,
        ProviderConformanceProfile::ProviderCore,
        "ProviderId syntax",
        is_stable_identifier(&metadata.name),
        "provider name must be a non-empty stable identifier",
    );
    check(
        report,
        ProviderConformanceProfile::ProviderCore,
        "Provider name presence",
        !metadata.name.trim().is_empty(),
        "provider name is empty",
    );
    check(
        report,
        ProviderConformanceProfile::ProviderCore,
        "Provider version syntax",
        !metadata.version.trim().is_empty(),
        "provider version is empty",
    );
    check(
        report,
        ProviderConformanceProfile::ProviderCore,
        "Runtime compatibility metadata",
        metadata.api_version == PROVIDER_API_VERSION,
        format!(
            "provider API {} does not match runtime API {}",
            metadata.api_version, PROVIDER_API_VERSION
        ),
    );
    check(
        report,
        ProviderConformanceProfile::ProviderCore,
        "vendor metadata",
        !metadata.vendor.trim().is_empty(),
        "provider vendor is empty",
    );
    check(
        report,
        ProviderConformanceProfile::ProviderCore,
        "metadata redaction",
        !contains_native_handle_text(&[
            metadata.name.as_str(),
            metadata.version.as_str(),
            metadata.vendor.as_str(),
            metadata.description.as_str(),
        ]),
        "public provider metadata appears to expose a native handle",
    );

    let mut device_ids = BTreeSet::new();
    for device in runtime.devices() {
        let device_metadata = device.metadata();
        check(
            report,
            ProviderConformanceProfile::ProviderCore,
            "DeviceId syntax",
            is_stable_identifier(device_metadata.id.as_str()),
            format!("invalid DeviceId '{}'", device_metadata.id),
        );
        check(
            report,
            ProviderConformanceProfile::ProviderCore,
            "Device provider ownership",
            device_metadata.provider == metadata.name,
            format!(
                "device '{}' is owned by '{}' instead of '{}'",
                device_metadata.id, device_metadata.provider, metadata.name
            ),
        );
        check(
            report,
            ProviderConformanceProfile::ProviderCore,
            "duplicate DeviceId per Provider",
            device_ids.insert(device_metadata.id.clone()),
            format!("duplicate DeviceId '{}'", device_metadata.id),
        );
        check(
            report,
            ProviderConformanceProfile::ProviderCore,
            "Device metadata redaction",
            !contains_native_handle_text(&[
                device_metadata.id.as_str(),
                device_metadata.name.as_str(),
                device_metadata.vendor.as_str(),
                device_metadata.architecture.as_str(),
            ]),
            "public Device metadata appears to expose a native handle",
        );
    }

    let status = provider.status_snapshot();
    check(
        report,
        ProviderConformanceProfile::ProviderCore,
        "lifecycle/readiness/pressure admission",
        status.accepts_new_work_by_default()
            == matches!(status.admission, ProviderAdmissionDecision::Admit),
        "admission does not match lifecycle, health, readiness, and pressure dimensions",
    );
    check(
        report,
        ProviderConformanceProfile::ProviderCore,
        "healthy distinct from ready",
        !(matches!(status.health, ProviderHealthState::Healthy)
            && matches!(status.readiness, ProviderReadinessState::NotReady)
            && matches!(status.admission, ProviderAdmissionDecision::Admit)),
        "not-ready provider admits new work",
    );
    check(
        report,
        ProviderConformanceProfile::ProviderCore,
        "saturated distinct from failed",
        !(matches!(status.pressure, ProviderPressureLevel::Saturated)
            && matches!(status.health, ProviderHealthState::Failed)),
        "provider reports saturation as generic failure",
    );
}

fn validate_compute(
    provider: &dyn Provider,
    runtime: &Runtime,
    report: &mut ProviderConformanceReport,
) {
    let metadata = provider.metadata();
    check(
        report,
        ProviderConformanceProfile::ProviderCompute,
        "Compute Capability version",
        metadata
            .compute_advertisement
            .supports_capability_version(COMPUTE_CAPABILITY_VERSION),
        "provider does not advertise a compatible Compute capability version",
    );
    for family in metadata.compute_advertisement.operation_families.keys() {
        let operation = ComputeOperationDescriptor::new(*family)
            .with_dtype(ComputeDType::Float32)
            .with_layout(ComputeLayout::Dense)
            .with_tensor(TensorDescriptor::materialized(
                ShapeDescriptor::new([1]),
                DTypeDescriptor::portable(ComputeDType::Float32),
            ));
        match runtime.validate_compute_operations(&metadata.name, &[operation]) {
            Ok(()) => report.record(ProviderConformanceTestResult::passed(
                ProviderConformanceProfile::ProviderCompute,
                format!("Compute operation family {}", family.id()),
            )),
            Err(error) => report.record(ProviderConformanceTestResult::failed(
                ProviderConformanceProfile::ProviderCompute,
                format!("Compute operation family {}", family.id()),
                error.to_string(),
            )),
        }
    }
}

fn validate_data_movement(
    provider: &dyn Provider,
    runtime: &Runtime,
    report: &mut ProviderConformanceReport,
) {
    let metadata = provider.metadata();
    let tensor = TensorDescriptor::materialized(
        ShapeDescriptor::new([1]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    for kind in metadata.compute_advertisement.data_movement.keys().copied() {
        let movement = match kind {
            ComputeDataMovementKind::Upload => ComputeDataMovementDescriptor::upload(
                HostBufferDescriptor::new(4, HostBufferEncoding::RawBytes),
                tensor.clone(),
            ),
            _ => {
                let resource = TensorResourceDescriptor::new(
                    TensorResourceId::new("conformance-resource"),
                    tensor.clone(),
                    ResourceAffinity::new(FallbackClass::ProviderPinned)
                        .with_provider(ProviderBinding::new(&metadata.name)),
                );
                match kind {
                    ComputeDataMovementKind::Download => {
                        ComputeDataMovementDescriptor::download(resource, tensor.clone())
                    }
                    ComputeDataMovementKind::Copy => {
                        ComputeDataMovementDescriptor::copy(resource, tensor.clone())
                    }
                    ComputeDataMovementKind::Materialize => {
                        ComputeDataMovementDescriptor::materialize(resource, tensor.clone())
                    }
                    ComputeDataMovementKind::Transfer => {
                        ComputeDataMovementDescriptor::transfer(resource, tensor.clone())
                    }
                    ComputeDataMovementKind::DTypeConversion => {
                        ComputeDataMovementDescriptor::dtype_conversion(resource, tensor.clone())
                    }
                    ComputeDataMovementKind::PlacementConversion => {
                        ComputeDataMovementDescriptor::placement_conversion(
                            resource,
                            tensor.clone(),
                        )
                    }
                    ComputeDataMovementKind::Upload => unreachable!(),
                }
            }
        };
        match runtime.validate_compute_data_movement(&metadata.name, &[movement]) {
            Ok(()) => report.record(ProviderConformanceTestResult::passed(
                ProviderConformanceProfile::ProviderDataMovement,
                format!("data movement {}", kind.id()),
            )),
            Err(error) => report.record(ProviderConformanceTestResult::failed(
                ProviderConformanceProfile::ProviderDataMovement,
                format!("data movement {}", kind.id()),
                error.to_string(),
            )),
        }
    }
}

fn validate_cancellation(provider: &dyn Provider, report: &mut ProviderConformanceReport) {
    let metadata = provider.metadata();
    if provider.execution_api().is_none() {
        report.record(ProviderConformanceTestResult::skipped(
            ProviderConformanceProfile::ProviderCancellation,
            "cancellation unsupported error",
            "provider does not expose ProviderExecutionApi",
        ));
        return;
    }
    report.record(ProviderConformanceTestResult::passed(
        ProviderConformanceProfile::ProviderCancellation,
        format!("cancellation capability explicit for {}", metadata.name),
    ));
}

fn validate_observability(provider: &dyn Provider, report: &mut ProviderConformanceReport) {
    let status = provider.status_snapshot();
    check(
        report,
        ProviderConformanceProfile::ProviderObservability,
        "status observation fields",
        !status.provider.as_str().trim().is_empty(),
        "provider status snapshot does not identify provider",
    );
    check(
        report,
        ProviderConformanceProfile::ProviderObservability,
        "diagnostics redacted",
        !status.diagnostics.iter().any(|diagnostic| {
            diagnostic
                .message
                .as_deref()
                .is_some_and(|message| contains_native_handle_text(&[message]))
        }),
        "status diagnostics appear to expose a native handle",
    );
}

fn check(
    report: &mut ProviderConformanceReport,
    profile: ProviderConformanceProfile,
    requirement: impl Into<String>,
    passed: bool,
    diagnostic: impl Into<String>,
) {
    let requirement = requirement.into();
    if passed {
        report.record(ProviderConformanceTestResult::passed(profile, requirement));
    } else {
        report.record(ProviderConformanceTestResult::failed(
            profile,
            requirement,
            diagnostic,
        ));
    }
}

fn is_stable_identifier(value: &str) -> bool {
    let value = value.trim();
    !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, ':' | '/' | '.' | '_' | '-'))
}

fn contains_native_handle_text(values: &[&str]) -> bool {
    values.iter().any(|value| {
        let lower = value.to_ascii_lowercase();
        lower.contains("0x") || lower.contains("native_handle") || lower.contains("raw handle")
    })
}

pub fn provider_conformance_report_json(
    report: &ProviderConformanceReport,
) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(report)
}

pub fn provider_conformance_profile_ids(
    profiles: impl IntoIterator<Item = ProviderConformanceProfile>,
) -> BTreeMap<String, bool> {
    profiles
        .into_iter()
        .map(|profile| (profile.id().into(), profile.required_by_default()))
        .collect()
}
