//! Post-baseline Provider roadmap contract (see
//! `openspec/changes/define-post-baseline-provider-roadmap`).
//!
//! The first Magnetar baseline is CPU-only and correctness-first (Reference
//! CPU Provider). This module does not implement CUDA, Metal, OpenVINO, QNN,
//! WebGPU, optimized CPU kernels, quantized numerics, flash/paged attention,
//! or benchmark numbers -- the change's proposal "Non-Goals" section rules
//! all of that out explicitly. Instead it defines, as executable Rust types
//! and validation functions, the roadmap **contract** that any future
//! hardware-specific or optimized Provider work must satisfy:
//!
//! - [`ProviderRoadmapPhase`]: the ordered introduction phases (optimized CPU
//!   -> CUDA -> Metal -> OpenVINO -> QNN -> WebGPU -> quantized execution
//!   expansion -> advanced attention kernels -> production performance
//!   profiles) and the [`crate::ProviderConformanceProfile`] gates each phase
//!   declares before it is production-ready.
//! - [`reject_model_family_provider_name`]: an executable, regression-proof
//!   rejection of model-family Provider names (`QwenProvider`,
//!   `LlamaProvider`, `GemmaProvider`-shaped names), implementing "No
//!   Model-Family Providers" (`specs/provider-roadmap/spec.md`) and
//!   "Providers Advertise Capabilities, Not Model Families"
//!   (`specs/provider/spec.md`).
//! - [`ProviderRoadmapHardwareFamily`]: per-hardware-family scope (CUDA,
//!   Metal, OpenVINO, QNN, WebGPU) covering Device metadata templates,
//!   supported [`crate::KernelMemoryClass`]es, the native-handle-kind names
//!   Runtime must never expose, and the primary fallback edge to Reference
//!   CPU (or a browser-CPU-like path for WebGPU).
//! - [`ProviderRoadmapFeature`]: the optional (`MAY`, never `SHALL`)
//!   per-phase features named in the proposal (SIMD/BLAS/thread
//!   pools/cache-aware/fused kernels for optimized CPU; device
//!   memory/streams/cuBLAS/flash attention/... for CUDA; buffers/command
//!   queues/pipelines/... for Metal; and the OpenVINO/QNN/WebGPU
//!   equivalents), each tagged with its phase and always optional.
//! - `KernelQuantizationMethod`, [`KernelQuantizationMetadata`] (defined in
//!   [`crate::kernel`], re-exported here) and
//!   [`validate_quantization_declaration`] /
//!   [`reject_hidden_dequantization`]: the roadmap-specific quantization
//!   metadata `kernel.rs` did not already carry, plus the checks that make
//!   hidden dequantization mechanically impossible to pass.
//! - [`AdvancedAttentionVariant`] and [`validate_advanced_attention_declaration`]:
//!   validation over the *existing* [`crate::KernelKvCacheMetadata`],
//!   [`crate::KernelPrecisionMetadata`], [`crate::KernelDeterminism`], and
//!   [`crate::KernelFallbackClass`] metadata `kernel.rs` already defines,
//!   rather than a duplicate parallel metadata struct.
//! - [`validate_fused_kernel_declaration`][]: composes
//!   [`crate::KernelFusionMetadata`] to require an explicit
//!   semantic-equivalence declaration before a fused Kernel is accepted.
//! - [`ProviderRoadmapError`]: the 19 structured error categories from the
//!   proposal's "Error Model" section.
//! - [`ProviderRoadmapObservationKind`] / [`ProviderRoadmapObservation`]: the
//!   14 observation categories from the proposal's "Observability" section,
//!   with redacted metadata only (no raw tensor/weight/prompt/cache/handle
//!   fields exist on the type, and free-form values are passed through
//!   `redact_backend_diagnostic`).
//! - [`ProviderRoadmapFallbackEdge`] / [`evaluate_provider_roadmap_fallback`]:
//!   a roadmap-level wrapper over [`crate::reference_cpu::evaluate_fallback`]
//!   that adds the privacy/precision gates the proposal's "Fallback Policy"
//!   section requires, remaining deny-by-default.
//! - [`ProviderRoadmapBenchmarkCategory`] / [`ProviderRoadmapBenchmarkResult`]:
//!   benchmark categories kept structurally separate from conformance -- no
//!   function in this module accepts a benchmark result as input to a
//!   conformance-pass decision.
//! - [`ProviderRoadmapConformanceReport`] / [`run_provider_roadmap_conformance`]:
//!   a small conformance report, in the shape of
//!   [`crate::CliBoundaryConformanceReport`], asserting the roadmap
//!   guarantees above hold.

use crate::{
    ComputeDType, DeviceMetadata, DeviceType, KernelDeterminism, KernelFallbackClass,
    KernelFusionMetadata, KernelKvCacheMetadata, KernelMemoryClass, KernelPrecisionMetadata,
    KernelQuantizationMetadata, OperatorId, ProviderConformanceProfile,
    provider_conformance_profile_ids,
};
use crate::{FallbackClass, ResourceAffinity, reference_cpu::FallbackPolicyContext};
use crate::{TensorLayoutKind, reference_cpu::evaluate_fallback};
use crate::{compute::redact_backend_diagnostic, inference_api::validate_inference_scope};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

pub const PROVIDER_ROADMAP_VERSION: &str = "0.1.0";

// ---------------------------------------------------------------------
// Roadmap phases
// ---------------------------------------------------------------------

/// Provider introduction phases from the proposal's "Provider Introduction
/// Phases" section. Ordering is `SHOULD`, not `SHALL` (platform priorities
/// may vary), but every phase declares the conformance gates it needs before
/// being production-ready.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProviderRoadmapPhase {
    OptimizedCpu,
    Cuda,
    Metal,
    OpenVino,
    Qnn,
    WebGpu,
    QuantizedExecutionExpansion,
    AdvancedAttentionKernels,
    ProductionPerformanceProfiles,
}

/// The nine roadmap phases in the proposal's suggested order.
pub const PROVIDER_ROADMAP_PHASES: &[ProviderRoadmapPhase] = &[
    ProviderRoadmapPhase::OptimizedCpu,
    ProviderRoadmapPhase::Cuda,
    ProviderRoadmapPhase::Metal,
    ProviderRoadmapPhase::OpenVino,
    ProviderRoadmapPhase::Qnn,
    ProviderRoadmapPhase::WebGpu,
    ProviderRoadmapPhase::QuantizedExecutionExpansion,
    ProviderRoadmapPhase::AdvancedAttentionKernels,
    ProviderRoadmapPhase::ProductionPerformanceProfiles,
];

impl ProviderRoadmapPhase {
    pub const fn id(self) -> &'static str {
        match self {
            Self::OptimizedCpu => "optimized-cpu-provider",
            Self::Cuda => "cuda-provider",
            Self::Metal => "metal-provider",
            Self::OpenVino => "openvino-provider",
            Self::Qnn => "qnn-provider",
            Self::WebGpu => "webgpu-provider",
            Self::QuantizedExecutionExpansion => "quantized-execution-expansion",
            Self::AdvancedAttentionKernels => "advanced-attention-kernels",
            Self::ProductionPerformanceProfiles => "production-performance-profiles",
        }
    }

    /// This phase's `SHOULD`-order position, 1-indexed as in the proposal.
    pub const fn ordinal(self) -> u8 {
        match self {
            Self::OptimizedCpu => 1,
            Self::Cuda => 2,
            Self::Metal => 3,
            Self::OpenVino => 4,
            Self::Qnn => 5,
            Self::WebGpu => 6,
            Self::QuantizedExecutionExpansion => 7,
            Self::AdvancedAttentionKernels => 8,
            Self::ProductionPerformanceProfiles => 9,
        }
    }

    /// [`ProviderConformanceProfile`]s this phase SHALL declare as required
    /// conformance gates before it is considered production-ready. Every
    /// phase requires `ProviderCore` at minimum; hardware and advanced-
    /// feature phases add their own profile.
    pub fn required_conformance_gates(self) -> BTreeSet<ProviderConformanceProfile> {
        let mut gates = BTreeSet::from([ProviderConformanceProfile::ProviderCore]);
        match self {
            Self::OptimizedCpu => {
                gates.insert(ProviderConformanceProfile::ProviderCompute);
            }
            Self::Cuda => {
                gates.insert(ProviderConformanceProfile::Cuda);
                gates.insert(ProviderConformanceProfile::ProviderDataMovement);
            }
            Self::Metal => {
                gates.insert(ProviderConformanceProfile::Metal);
                gates.insert(ProviderConformanceProfile::ProviderDataMovement);
            }
            Self::OpenVino => {
                gates.insert(ProviderConformanceProfile::OpenVino);
            }
            Self::Qnn => {
                gates.insert(ProviderConformanceProfile::Qnn);
            }
            Self::WebGpu => {
                gates.insert(ProviderConformanceProfile::WebGpu);
                gates.insert(ProviderConformanceProfile::Browser);
            }
            Self::QuantizedExecutionExpansion => {
                gates.insert(ProviderConformanceProfile::Quantized);
            }
            Self::AdvancedAttentionKernels => {
                gates.insert(ProviderConformanceProfile::AdvancedAttention);
            }
            Self::ProductionPerformanceProfiles => {
                gates.insert(ProviderConformanceProfile::ProviderObservability);
                gates.insert(ProviderConformanceProfile::FusedKernel);
            }
        }
        gates
    }
}

/// The roadmap's readiness gate: whether `phase` is production-ready given
/// the conformance profiles that have actually passed for a Provider,
/// implementing "each Provider must pass the required conformance gates
/// before being considered production-ready". Kept distinct from
/// [`ProviderRoadmapPhase::required_conformance_gates`] (which only
/// *declares* what is required) so "declared" and "actually satisfied" are
/// two different, both-checkable things -- registering a phase's gates
/// never by itself implies this returns `true`.
pub fn phase_is_production_ready(
    phase: ProviderRoadmapPhase,
    passed_profiles: &BTreeSet<ProviderConformanceProfile>,
) -> bool {
    phase
        .required_conformance_gates()
        .iter()
        .all(|gate| passed_profiles.contains(gate))
}

// ---------------------------------------------------------------------
// Provider feature metadata
// ---------------------------------------------------------------------

/// Optional (`MAY`) per-phase Provider features named in the proposal: none
/// of these are ever required. `is_optional()` is `true` for every variant,
/// and no conformance gate in this module treats any of them as mandatory --
/// this is what "allow SIMD/BLAS/thread pools/cache-aware kernels/fused
/// kernels as optional" and each hardware family's "feature placeholders"
/// tasks mean at the contract level: named, phase-tagged, and structurally
/// non-mandatory.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProviderRoadmapFeature {
    // Optimized CPU
    Simd,
    Blas,
    ThreadPoolExecution,
    CacheAwareKernels,
    OptimizedCpuFusedKernels,
    // CUDA
    CudaDeviceMemory,
    CudaPinnedHostMemory,
    CudaStreams,
    CudaKernels,
    CudaBlasMatmul,
    CudaFusedKernels,
    CudaFlashAttention,
    CudaQuantizedKernels,
    // Metal
    MetalBuffers,
    MetalCommandQueues,
    MetalComputePipelines,
    MetalOptimizedMatmul,
    MetalFusedKernels,
    // OpenVINO
    OpenVinoGraphCompilation,
    OpenVinoStaticShapeProfiles,
    OpenVinoDynamicShapeProfiles,
    OpenVinoQuantizedExecution,
    // QNN
    QnnMobileInference,
    QnnNpuKernels,
    QnnQuantizedExecution,
    QnnStaticShapeCompilation,
    // WebGPU
    WebGpuBrowserBuffers,
    WebGpuWgslKernels,
    WebGpuCommandSubmission,
    WebGpuReducedDtypeSet,
    WebGpuReducedLayoutSet,
}

impl ProviderRoadmapFeature {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Simd => "simd",
            Self::Blas => "blas",
            Self::ThreadPoolExecution => "thread-pool-execution",
            Self::CacheAwareKernels => "cache-aware-kernels",
            Self::OptimizedCpuFusedKernels => "optimized-cpu-fused-kernels",
            Self::CudaDeviceMemory => "cuda-device-memory",
            Self::CudaPinnedHostMemory => "cuda-pinned-host-memory",
            Self::CudaStreams => "cuda-streams",
            Self::CudaKernels => "cuda-kernels",
            Self::CudaBlasMatmul => "cuda-blas-matmul",
            Self::CudaFusedKernels => "cuda-fused-kernels",
            Self::CudaFlashAttention => "cuda-flash-attention",
            Self::CudaQuantizedKernels => "cuda-quantized-kernels",
            Self::MetalBuffers => "metal-buffers",
            Self::MetalCommandQueues => "metal-command-queues",
            Self::MetalComputePipelines => "metal-compute-pipelines",
            Self::MetalOptimizedMatmul => "metal-optimized-matmul",
            Self::MetalFusedKernels => "metal-fused-kernels",
            Self::OpenVinoGraphCompilation => "openvino-graph-compilation",
            Self::OpenVinoStaticShapeProfiles => "openvino-static-shape-profiles",
            Self::OpenVinoDynamicShapeProfiles => "openvino-dynamic-shape-profiles",
            Self::OpenVinoQuantizedExecution => "openvino-quantized-execution",
            Self::QnnMobileInference => "qnn-mobile-inference",
            Self::QnnNpuKernels => "qnn-npu-kernels",
            Self::QnnQuantizedExecution => "qnn-quantized-execution",
            Self::QnnStaticShapeCompilation => "qnn-static-shape-compilation",
            Self::WebGpuBrowserBuffers => "webgpu-browser-buffers",
            Self::WebGpuWgslKernels => "webgpu-wgsl-kernels",
            Self::WebGpuCommandSubmission => "webgpu-command-submission",
            Self::WebGpuReducedDtypeSet => "webgpu-reduced-dtype-set",
            Self::WebGpuReducedLayoutSet => "webgpu-reduced-layout-set",
        }
    }

    /// The roadmap phase this feature belongs to.
    pub const fn phase(self) -> ProviderRoadmapPhase {
        match self {
            Self::Simd
            | Self::Blas
            | Self::ThreadPoolExecution
            | Self::CacheAwareKernels
            | Self::OptimizedCpuFusedKernels => ProviderRoadmapPhase::OptimizedCpu,
            Self::CudaDeviceMemory
            | Self::CudaPinnedHostMemory
            | Self::CudaStreams
            | Self::CudaKernels
            | Self::CudaBlasMatmul
            | Self::CudaFusedKernels
            | Self::CudaFlashAttention
            | Self::CudaQuantizedKernels => ProviderRoadmapPhase::Cuda,
            Self::MetalBuffers
            | Self::MetalCommandQueues
            | Self::MetalComputePipelines
            | Self::MetalOptimizedMatmul
            | Self::MetalFusedKernels => ProviderRoadmapPhase::Metal,
            Self::OpenVinoGraphCompilation
            | Self::OpenVinoStaticShapeProfiles
            | Self::OpenVinoDynamicShapeProfiles
            | Self::OpenVinoQuantizedExecution => ProviderRoadmapPhase::OpenVino,
            Self::QnnMobileInference
            | Self::QnnNpuKernels
            | Self::QnnQuantizedExecution
            | Self::QnnStaticShapeCompilation => ProviderRoadmapPhase::Qnn,
            Self::WebGpuBrowserBuffers
            | Self::WebGpuWgslKernels
            | Self::WebGpuCommandSubmission
            | Self::WebGpuReducedDtypeSet
            | Self::WebGpuReducedLayoutSet => ProviderRoadmapPhase::WebGpu,
        }
    }

    /// Every roadmap feature is `MAY`, never `SHALL`: this is always `true`.
    pub const fn is_optional(self) -> bool {
        true
    }
}

/// All roadmap features belonging to `phase`, in declaration order.
pub fn provider_roadmap_features_for_phase(
    phase: ProviderRoadmapPhase,
) -> Vec<ProviderRoadmapFeature> {
    PROVIDER_ROADMAP_FEATURES
        .iter()
        .copied()
        .filter(|feature| feature.phase() == phase)
        .collect()
}

/// Every roadmap feature, across every phase.
pub const PROVIDER_ROADMAP_FEATURES: &[ProviderRoadmapFeature] = &[
    ProviderRoadmapFeature::Simd,
    ProviderRoadmapFeature::Blas,
    ProviderRoadmapFeature::ThreadPoolExecution,
    ProviderRoadmapFeature::CacheAwareKernels,
    ProviderRoadmapFeature::OptimizedCpuFusedKernels,
    ProviderRoadmapFeature::CudaDeviceMemory,
    ProviderRoadmapFeature::CudaPinnedHostMemory,
    ProviderRoadmapFeature::CudaStreams,
    ProviderRoadmapFeature::CudaKernels,
    ProviderRoadmapFeature::CudaBlasMatmul,
    ProviderRoadmapFeature::CudaFusedKernels,
    ProviderRoadmapFeature::CudaFlashAttention,
    ProviderRoadmapFeature::CudaQuantizedKernels,
    ProviderRoadmapFeature::MetalBuffers,
    ProviderRoadmapFeature::MetalCommandQueues,
    ProviderRoadmapFeature::MetalComputePipelines,
    ProviderRoadmapFeature::MetalOptimizedMatmul,
    ProviderRoadmapFeature::MetalFusedKernels,
    ProviderRoadmapFeature::OpenVinoGraphCompilation,
    ProviderRoadmapFeature::OpenVinoStaticShapeProfiles,
    ProviderRoadmapFeature::OpenVinoDynamicShapeProfiles,
    ProviderRoadmapFeature::OpenVinoQuantizedExecution,
    ProviderRoadmapFeature::QnnMobileInference,
    ProviderRoadmapFeature::QnnNpuKernels,
    ProviderRoadmapFeature::QnnQuantizedExecution,
    ProviderRoadmapFeature::QnnStaticShapeCompilation,
    ProviderRoadmapFeature::WebGpuBrowserBuffers,
    ProviderRoadmapFeature::WebGpuWgslKernels,
    ProviderRoadmapFeature::WebGpuCommandSubmission,
    ProviderRoadmapFeature::WebGpuReducedDtypeSet,
    ProviderRoadmapFeature::WebGpuReducedLayoutSet,
];

// ---------------------------------------------------------------------
// No model-family Providers
// ---------------------------------------------------------------------

/// Model-family name fragments the roadmap's "No Model-Family Providers"
/// requirement forbids as a Provider identity, matching the proposal's
/// `QwenProvider` / `LlamaProvider` / `GemmaProvider` examples plus other
/// well-known model families ("such as" in the proposal is explicitly
/// non-exhaustive).
const KNOWN_MODEL_FAMILY_NAME_FRAGMENTS: &[&str] = &[
    "qwen", "llama", "gemma", "mistral", "mixtral", "phi", "falcon", "yi", "baichuan", "deepseek",
    "gpt", "bert", "t5", "claude", "grok",
];

/// Rejects a Provider identity that names a model family instead of a
/// capability or hardware target, implementing "No Model-Family Providers"
/// (`specs/provider-roadmap/spec.md`) and "Providers Advertise Capabilities,
/// Not Model Families" (`specs/provider/spec.md`) as an executable,
/// regression-proof check. Hardware/optimized names such as `CudaProvider`,
/// `MetalProvider`, `OptimizedCpuProvider`, or `ReferenceCpuProvider` are not
/// rejected.
pub fn reject_model_family_provider_name(name: &str) -> Result<(), ProviderRoadmapError> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err(ProviderRoadmapError::InternalProviderRoadmapError {
            reason: "provider name must not be empty".into(),
        });
    }
    let lower = trimmed.to_ascii_lowercase();
    if let Some(family_part) = lower.strip_suffix("provider")
        && KNOWN_MODEL_FAMILY_NAME_FRAGMENTS
            .iter()
            .any(|fragment| family_part.contains(fragment))
    {
        return Err(ProviderRoadmapError::ProviderRoadmapUnsupported {
            reason: format!(
                "'{trimmed}' names a model-family Provider; Providers must advertise \
                 capabilities and Kernels, not model-family ownership"
            ),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Hardware families
// ---------------------------------------------------------------------

/// The five post-baseline hardware Provider families named in the proposal.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProviderRoadmapHardwareFamily {
    Cuda,
    Metal,
    OpenVino,
    Qnn,
    WebGpu,
}

impl ProviderRoadmapHardwareFamily {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::OpenVino => "openvino",
            Self::Qnn => "qnn",
            Self::WebGpu => "webgpu",
        }
    }

    pub const fn conformance_profile(self) -> ProviderConformanceProfile {
        match self {
            Self::Cuda => ProviderConformanceProfile::Cuda,
            Self::Metal => ProviderConformanceProfile::Metal,
            Self::OpenVino => ProviderConformanceProfile::OpenVino,
            Self::Qnn => ProviderConformanceProfile::Qnn,
            Self::WebGpu => ProviderConformanceProfile::WebGpu,
        }
    }

    /// Native handle/resource kind names Runtime SHALL keep internal for
    /// this family ("CUDA Provider SHALL keep native handles internal",
    /// "Metal Provider SHALL keep native handles internal", "QNN Provider
    /// SHALL not expose native QNN handles", "OpenVINO compiled graph
    /// internals SHALL remain opaque"). WebGPU has no native-handle boundary
    /// of its own: it runs under browser sandboxing constraints instead (see
    /// [`Self::requires_no_native_provider_loading`]).
    pub const fn native_handle_kinds(self) -> &'static [&'static str] {
        match self {
            Self::Cuda => &[
                "cuda-stream",
                "cuda-device-pointer",
                "cuda-module",
                "cuda-event",
                "cuda-kernel-handle",
            ],
            Self::Metal => &[
                "metal-device-handle",
                "metal-buffer",
                "metal-command-queue",
                "metal-pipeline",
                "metal-event",
            ],
            Self::OpenVino => &["openvino-compiled-graph"],
            Self::Qnn => &["qnn-native-handle"],
            Self::WebGpu => &[],
        }
    }

    /// [`KernelMemoryClass`]es this family MAY use, drawn from the existing
    /// baseline enum -- no parallel memory-class type is introduced.
    pub fn memory_classes(self) -> BTreeSet<KernelMemoryClass> {
        match self {
            Self::Cuda => {
                BTreeSet::from([KernelMemoryClass::Device, KernelMemoryClass::PinnedHost])
            }
            Self::Metal => BTreeSet::from([KernelMemoryClass::Device, KernelMemoryClass::Unified]),
            Self::OpenVino => BTreeSet::from([KernelMemoryClass::Device, KernelMemoryClass::Host]),
            Self::Qnn => {
                BTreeSet::from([KernelMemoryClass::ProviderOwned, KernelMemoryClass::Device])
            }
            Self::WebGpu => BTreeSet::from([
                KernelMemoryClass::BrowserLinearMemory,
                KernelMemoryClass::FutureWebGpuBuffer,
            ]),
        }
    }

    /// The primary post-baseline fallback edge for this family.
    pub const fn primary_fallback_edge(self) -> ProviderRoadmapFallbackEdge {
        match self {
            Self::Cuda => ProviderRoadmapFallbackEdge::CudaToReferenceCpu,
            Self::Metal => ProviderRoadmapFallbackEdge::MetalToReferenceCpu,
            Self::OpenVino => ProviderRoadmapFallbackEdge::OpenVinoToReferenceCpu,
            Self::Qnn => ProviderRoadmapFallbackEdge::QnnToReferenceCpu,
            Self::WebGpu => ProviderRoadmapFallbackEdge::WebGpuToBrowserCpuLike,
        }
    }

    /// WebGPU Provider SHALL be compatible with browser constraints and
    /// SHALL NOT require Wasmtime or native Provider loading; every other
    /// family may use native/dynamic Provider loading.
    pub const fn requires_no_native_provider_loading(self) -> bool {
        matches!(self, Self::WebGpu)
    }

    /// A representative [`DeviceMetadata`] template for this family, reusing
    /// the existing generic `DeviceMetadata`/`DeviceType` contract instead of
    /// a family-specific parallel struct.
    pub fn device_metadata_template(self) -> DeviceMetadata {
        let (device_type, vendor, architecture) = match self {
            Self::Cuda => (DeviceType::Gpu, "NVIDIA", "cuda"),
            Self::Metal => (DeviceType::Gpu, "Apple", "metal"),
            Self::OpenVino => (DeviceType::Cpu, "Intel", "openvino"),
            Self::Qnn => (DeviceType::Npu, "Qualcomm", "qnn"),
            Self::WebGpu => (DeviceType::Gpu, "browser", "webgpu"),
        };
        let mut metadata = DeviceMetadata::new(
            crate::DeviceId::new(format!("{}-roadmap-template", self.id())),
            format!("{} roadmap template device", self.id()),
            device_type,
            format!("{}-provider", self.id()),
        );
        metadata.vendor = vendor.into();
        metadata.architecture = architecture.into();
        metadata.memory_class_support = self.memory_classes();
        metadata
    }
}

/// Denies a native handle/resource access request for a post-baseline
/// hardware family, implementing "Native Handles Remain Hidden"
/// (`specs/provider-roadmap/spec.md`), "Runtime Rejects Native Handle
/// Exposure" (`specs/runtime/spec.md`), and "Post-Baseline Tensor Handles
/// Remain Opaque" (`specs/tensor/spec.md`). Every native handle kind for
/// every hardware family is denied unconditionally: this Runtime never
/// exposes native Provider/Device/Kernel/memory handles through public APIs.
pub fn reject_native_handle_exposure(
    family: ProviderRoadmapHardwareFamily,
    handle_kind: &str,
) -> Result<(), ProviderRoadmapError> {
    let _ = family;
    Err(ProviderRoadmapError::ProviderNativeHandleExposureDenied {
        handle_kind: handle_kind.to_string(),
    })
}

// ---------------------------------------------------------------------
// Kernel fusion
// ---------------------------------------------------------------------

/// Everything a fused Kernel must declare before Runtime accepts it: the
/// equivalent portable Operator sequence / graph fragment
/// ([`KernelFusionMetadata`]), a precision tolerance
/// ([`KernelPrecisionMetadata`]), and explicit fallback behavior
/// ([`KernelFallbackClass`]). Reuses existing Kernel-contract metadata types
/// instead of duplicating them.
pub struct FusedKernelDeclaration<'a> {
    pub fusion: Option<&'a KernelFusionMetadata>,
    pub precision: &'a KernelPrecisionMetadata,
    pub fallback_hints: &'a BTreeSet<KernelFallbackClass>,
}

/// Validates a fused Kernel's semantic-equivalence declaration, implementing
/// "Fused Kernels Declare Semantic Equivalence" (`specs/kernel/spec.md`) and
/// "Kernel Fusion" (proposal). Rejects: no fusion metadata at all, an empty
/// equivalent-Operator group, `preserves_graph_semantics == false`, a
/// missing precision tolerance profile, or no declared fallback behavior.
pub fn validate_fused_kernel_declaration(
    declaration: FusedKernelDeclaration<'_>,
) -> Result<(), ProviderRoadmapError> {
    let Some(fusion) = declaration.fusion else {
        return Err(ProviderRoadmapError::ProviderFusionInvalid {
            reason: "fused kernel must declare semantic-equivalence metadata".into(),
        });
    };
    if fusion.operator_group.is_empty() {
        return Err(ProviderRoadmapError::ProviderFusionInvalid {
            reason: "fused kernel must declare at least one equivalent portable Operator".into(),
        });
    }
    if !fusion.preserves_graph_semantics {
        return Err(ProviderRoadmapError::ProviderFusionInvalid {
            reason: "fused kernel does not preserve portable Operator/graph semantics".into(),
        });
    }
    if declaration.precision.tolerance_profile.is_none() {
        return Err(ProviderRoadmapError::ProviderFusionInvalid {
            reason: "fused kernel must declare a precision tolerance profile".into(),
        });
    }
    if declaration.fallback_hints.is_empty() {
        return Err(ProviderRoadmapError::ProviderFusionInvalid {
            reason: "fused kernel must declare fallback behavior".into(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Advanced attention
// ---------------------------------------------------------------------

/// Advanced attention variants from the proposal's "Advanced Attention"
/// section.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AdvancedAttentionVariant {
    FlashAttention,
    PagedAttention,
    SlidingWindowAttention,
    BlockSparseAttention,
    GqaMqaOptimizedAttention,
    KvCacheAwareAttention,
}

impl AdvancedAttentionVariant {
    pub const fn id(self) -> &'static str {
        match self {
            Self::FlashAttention => "flash-attention",
            Self::PagedAttention => "paged-attention",
            Self::SlidingWindowAttention => "sliding-window-attention",
            Self::BlockSparseAttention => "block-sparse-attention",
            Self::GqaMqaOptimizedAttention => "gqa-mqa-optimized-attention",
            Self::KvCacheAwareAttention => "kv-cache-aware-attention",
        }
    }

    /// Whether this variant inherently requires KV cache layout metadata.
    pub const fn requires_kv_cache_metadata(self) -> bool {
        matches!(self, Self::PagedAttention | Self::KvCacheAwareAttention)
    }
}

/// Everything an advanced attention path must declare, per the proposal:
/// supported Operator variant, tensor layout requirements, memory class
/// requirements, KV cache layout support, dtype support, precision
/// tolerance, determinism metadata, and fallback behavior. All fields reuse
/// existing Kernel-contract metadata types.
pub struct AdvancedAttentionDeclaration<'a> {
    pub variant: AdvancedAttentionVariant,
    pub operator: &'a OperatorId,
    pub layouts: &'a BTreeSet<TensorLayoutKind>,
    pub memory_classes: &'a BTreeSet<KernelMemoryClass>,
    pub dtypes: &'a BTreeSet<ComputeDType>,
    pub kv_cache: Option<&'a KernelKvCacheMetadata>,
    pub precision: &'a KernelPrecisionMetadata,
    pub determinism: &'a KernelDeterminism,
    pub fallback_hints: &'a BTreeSet<KernelFallbackClass>,
}

/// Validates an advanced attention declaration, implementing "Advanced
/// Attention Is Explicit" (`specs/provider-roadmap/spec.md`) and "Advanced
/// Kernels Declare Specialized Requirements" (`specs/kernel/spec.md`).
/// "Unsupported advanced attention SHALL fail explicitly" is implemented by
/// [`reject_unsupported_advanced_attention`] for the case where a variant is
/// not implemented at all; this function validates the declaration of a
/// variant that *is* implemented.
pub fn validate_advanced_attention_declaration(
    declaration: AdvancedAttentionDeclaration<'_>,
) -> Result<(), ProviderRoadmapError> {
    let fail = |reason: &str| {
        Err(ProviderRoadmapError::ProviderAdvancedAttentionUnsupported {
            variant: format!("{}: {reason}", declaration.variant.id()),
        })
    };
    if declaration.layouts.is_empty() {
        return fail("must declare at least one required tensor layout");
    }
    if declaration.memory_classes.is_empty() {
        return fail("must declare at least one required memory class");
    }
    if declaration.dtypes.is_empty() {
        return fail("must declare at least one supported dtype");
    }
    if declaration.precision.tolerance_profile.is_none() {
        return fail("must declare a precision tolerance profile");
    }
    if declaration.fallback_hints.is_empty() {
        return fail("must declare fallback behavior");
    }
    if declaration.variant.requires_kv_cache_metadata() && declaration.kv_cache.is_none() {
        return fail("must declare KV cache layout support");
    }
    let _ = declaration.determinism;
    let _ = declaration.operator;
    Ok(())
}

/// "Unsupported advanced attention SHALL fail explicitly": returns the
/// structured error for a graph that requires an advanced attention variant
/// the Provider does not implement at all.
pub fn reject_unsupported_advanced_attention(
    variant: AdvancedAttentionVariant,
) -> ProviderRoadmapError {
    ProviderRoadmapError::ProviderAdvancedAttentionUnsupported {
        variant: variant.id().to_string(),
    }
}

// ---------------------------------------------------------------------
// Quantized execution
// ---------------------------------------------------------------------

/// Validates a quantization declaration, implementing "Quantized Execution
/// Is Explicit" (`specs/provider-roadmap/spec.md`): quantized support SHALL
/// declare at least one supported Operator and a conformance tolerance
/// profile. Method, storage/compute/accumulation dtype, scale/zero-point
/// dtype, group size, packing layout, and dequantization behavior are all
/// mandatory (non-`Option`) fields on [`KernelQuantizationMetadata`] itself,
/// so a caller cannot construct one without declaring them.
pub fn validate_quantization_declaration(
    metadata: &KernelQuantizationMetadata,
) -> Result<(), ProviderRoadmapError> {
    if metadata.supported_operators.is_empty() {
        return Err(ProviderRoadmapError::ProviderQuantizationUnsupported {
            reason: "quantized execution must declare at least one supported Operator".into(),
        });
    }
    if metadata.conformance_tolerance_profile.trim().is_empty() {
        return Err(ProviderRoadmapError::ProviderQuantizationUnsupported {
            reason: "quantized execution must declare a conformance tolerance profile".into(),
        });
    }
    Ok(())
}

/// "No hidden quantization or dequantization SHALL occur": dequantization is
/// rejected unless the caller can attest it was explicitly declared in the
/// graph plan (via [`KernelDequantizationBehavior`][crate::kernel::KernelDequantizationBehavior]
/// on a validated [`KernelQuantizationMetadata`]).
pub fn reject_hidden_dequantization(declared: bool) -> Result<(), ProviderRoadmapError> {
    if declared {
        Ok(())
    } else {
        Err(ProviderRoadmapError::ProviderQuantizationUnsupported {
            reason: "dequantization occurred without an explicit graph-plan declaration".into(),
        })
    }
}

// ---------------------------------------------------------------------
// Layout expansion
// ---------------------------------------------------------------------

/// Post-baseline specialized layouts from the proposal's "Layout Expansion"
/// section, all already representable through the existing
/// [`TensorLayoutKind`] enum -- no parallel layout type is introduced.
pub const POST_BASELINE_LAYOUTS: &[TensorLayoutKind] = &[
    TensorLayoutKind::Blocked,
    TensorLayoutKind::Paged,
    TensorLayoutKind::QuantizedPacked,
    TensorLayoutKind::AttentionSpecific,
    TensorLayoutKind::ProviderOpaque,
    TensorLayoutKind::BrowserCompatible,
];

/// "Unsupported layouts SHALL fail or require explicit conversion":
/// implements "Layout Expansion Is Explicit"
/// (`specs/provider-roadmap/spec.md`) and "Post-Baseline Layouts Use Tensor
/// Layout Contract" (`specs/tensor/spec.md`).
pub fn require_explicit_layout_conversion(
    required_layout: TensorLayoutKind,
    available_layout: TensorLayoutKind,
    conversion_declared: bool,
) -> Result<(), ProviderRoadmapError> {
    if required_layout == available_layout || conversion_declared {
        Ok(())
    } else {
        Err(ProviderRoadmapError::ProviderLayoutUnsupported {
            layout: format!(
                "{available_layout:?} requires an explicit conversion to {required_layout:?}"
            ),
        })
    }
}

// ---------------------------------------------------------------------
// Memory expansion
// ---------------------------------------------------------------------

/// Post-baseline memory classes from the proposal's "Memory Expansion"
/// section, all already representable through the existing
/// [`KernelMemoryClass`] enum -- no parallel memory-class type is
/// introduced.
pub const POST_BASELINE_MEMORY_CLASSES: &[KernelMemoryClass] = &[
    KernelMemoryClass::Device,
    KernelMemoryClass::PinnedHost,
    KernelMemoryClass::Unified,
    KernelMemoryClass::Shared,
    KernelMemoryClass::ProviderOwned,
    KernelMemoryClass::BrowserLinearMemory,
    KernelMemoryClass::FutureWebGpuBuffer,
];

/// "Memory Manager SHALL track residency, Resource Affinity, transfer,
/// conversion, and cleanup": implements "Memory Expansion Is Tracked"
/// (`specs/provider-roadmap/spec.md`) and "Post-Baseline Memory Classes Are
/// Tracked" (`specs/memory/spec.md`).
pub fn require_memory_manager_tracking(
    memory_class: KernelMemoryClass,
    tracked: bool,
) -> Result<(), ProviderRoadmapError> {
    if tracked {
        Ok(())
    } else {
        Err(ProviderRoadmapError::ProviderMemoryClassUnsupported {
            memory_class: format!("{memory_class:?} is not tracked by Memory Manager"),
        })
    }
}

// ---------------------------------------------------------------------
// Conformance profile reporting
// ---------------------------------------------------------------------

/// The roadmap-introduced [`ProviderConformanceProfile`]s (beyond the
/// baseline core/compute/data-movement/cancellation/observability/dynamic-abi
/// set), in the shape [`provider_conformance_profile_ids`] expects.
/// "Provider registration SHALL not imply production readiness": every one
/// of these profiles is `required_by_default() == false`.
pub fn provider_roadmap_conformance_profile_ids() -> BTreeMap<String, bool> {
    provider_conformance_profile_ids([
        ProviderConformanceProfile::Cuda,
        ProviderConformanceProfile::Metal,
        ProviderConformanceProfile::OpenVino,
        ProviderConformanceProfile::Qnn,
        ProviderConformanceProfile::WebGpu,
        ProviderConformanceProfile::Quantized,
        ProviderConformanceProfile::AdvancedAttention,
        ProviderConformanceProfile::FusedKernel,
        ProviderConformanceProfile::Browser,
    ])
}

// ---------------------------------------------------------------------
// Benchmarks (kept structurally separate from conformance)
// ---------------------------------------------------------------------

/// Performance benchmark categories from the proposal's "Performance
/// Benchmarks" section. This change does not define benchmark numbers (see
/// the proposal's "Non-Goals"); it only names the categories and keeps the
/// result type out of every conformance-pass decision in this module.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProviderRoadmapBenchmarkCategory {
    PrefillLatency,
    DecodeLatency,
    TokensPerSecond,
    MemoryFootprint,
    BatchingThroughput,
    CacheHitBehavior,
    TransferOverhead,
    KernelDispatchOverhead,
}

impl ProviderRoadmapBenchmarkCategory {
    pub const fn id(self) -> &'static str {
        match self {
            Self::PrefillLatency => "prefill-latency",
            Self::DecodeLatency => "decode-latency",
            Self::TokensPerSecond => "tokens-per-second",
            Self::MemoryFootprint => "memory-footprint",
            Self::BatchingThroughput => "batching-throughput",
            Self::CacheHitBehavior => "cache-hit-behavior",
            Self::TransferOverhead => "transfer-overhead",
            Self::KernelDispatchOverhead => "kernel-dispatch-overhead",
        }
    }
}

/// A single benchmark measurement. Deliberately: no function anywhere in
/// this module accepts a [`ProviderRoadmapBenchmarkResult`] and returns or
/// influences a conformance pass/fail decision --
/// [`ProviderRoadmapConformanceReport::is_conformant`] only ever reads
/// [`ProviderRoadmapConformanceResult`]s, so "a Provider SHALL not pass
/// correctness merely because it is faster" holds mechanically, not just by
/// convention.
#[derive(Clone, Debug, PartialEq)]
pub struct ProviderRoadmapBenchmarkResult {
    pub category: ProviderRoadmapBenchmarkCategory,
    pub provider: String,
    pub value: f64,
    pub unit: String,
}

// ---------------------------------------------------------------------
// Fallback policy
// ---------------------------------------------------------------------

/// Post-baseline fallback edges from the proposal's "Fallback Policy"
/// section.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProviderRoadmapFallbackEdge {
    OptimizedCpuToReferenceCpu,
    CudaToOptimizedCpu,
    CudaToReferenceCpu,
    MetalToReferenceCpu,
    OpenVinoToReferenceCpu,
    QnnToReferenceCpu,
    WebGpuToBrowserCpuLike,
}

impl ProviderRoadmapFallbackEdge {
    pub const fn id(self) -> &'static str {
        match self {
            Self::OptimizedCpuToReferenceCpu => "optimized-cpu-to-reference-cpu",
            Self::CudaToOptimizedCpu => "cuda-to-optimized-cpu",
            Self::CudaToReferenceCpu => "cuda-to-reference-cpu",
            Self::MetalToReferenceCpu => "metal-to-reference-cpu",
            Self::OpenVinoToReferenceCpu => "openvino-to-reference-cpu",
            Self::QnnToReferenceCpu => "qnn-to-reference-cpu",
            Self::WebGpuToBrowserCpuLike => "webgpu-to-browser-cpu-like",
        }
    }
}

/// Roadmap-level fallback policy gates, layered on top of
/// [`FallbackPolicyContext`] (Resource Affinity, dtype/layout conversion):
/// memory, privacy, and precision policy, per "Fallback Remains Explicit"
/// (`specs/provider-roadmap/spec.md`) and "Validate memory policy before
/// fallback" / "Validate privacy and precision policy before fallback"
/// (tasks). All gates default to denying fallback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProviderRoadmapFallbackContext {
    pub cpu: FallbackPolicyContext,
    pub memory_policy_allows_fallback: bool,
    pub privacy_policy_allows_fallback: bool,
    pub precision_policy_allows_fallback: bool,
}

impl ProviderRoadmapFallbackContext {
    /// Deny-by-default: every gate must be explicitly opened by the caller.
    pub const fn deny_by_default() -> Self {
        Self {
            cpu: FallbackPolicyContext::new(false),
            memory_policy_allows_fallback: false,
            privacy_policy_allows_fallback: false,
            precision_policy_allows_fallback: false,
        }
    }
}

/// Evaluates whether a post-baseline fallback edge is permitted, composing
/// [`evaluate_fallback`] (Resource Affinity, dtype/layout conversion policy)
/// with the roadmap's additional memory, privacy, and precision policy
/// gates. Deny-by-default: fallback is denied unless every gate explicitly
/// allows it, implementing "Fallback Remains Explicit" and "Prevent silent
/// fallback".
pub fn evaluate_provider_roadmap_fallback(
    edge: ProviderRoadmapFallbackEdge,
    affinity: &ResourceAffinity,
    context: &ProviderRoadmapFallbackContext,
) -> Result<(), ProviderRoadmapError> {
    evaluate_fallback(affinity, &context.cpu).map_err(|error| {
        ProviderRoadmapError::ProviderFallbackDenied {
            reason: format!("{}: {error}", edge.id()),
        }
    })?;
    if !context.memory_policy_allows_fallback {
        return Err(ProviderRoadmapError::ProviderFallbackDenied {
            reason: format!("{}: memory policy forbids fallback", edge.id()),
        });
    }
    if !context.privacy_policy_allows_fallback {
        return Err(ProviderRoadmapError::ProviderFallbackDenied {
            reason: format!("{}: privacy policy forbids fallback", edge.id()),
        });
    }
    if !context.precision_policy_allows_fallback {
        return Err(ProviderRoadmapError::ProviderFallbackDenied {
            reason: format!("{}: precision policy forbids fallback", edge.id()),
        });
    }
    Ok(())
}

/// [`evaluate_provider_roadmap_fallback`], additionally producing the
/// "fallback considered" / "fallback used" / "fallback denied" observations
/// so fallback decisions are observable rather than silent.
pub fn evaluate_provider_roadmap_fallback_observed(
    edge: ProviderRoadmapFallbackEdge,
    affinity: &ResourceAffinity,
    context: &ProviderRoadmapFallbackContext,
) -> (
    Vec<ProviderRoadmapObservation>,
    Result<(), ProviderRoadmapError>,
) {
    let mut observations = vec![
        ProviderRoadmapObservation::new(ProviderRoadmapObservationKind::FallbackConsidered)
            .with_redacted_metadata("edge", edge.id()),
    ];
    let outcome = evaluate_provider_roadmap_fallback(edge, affinity, context);
    match &outcome {
        Ok(()) => observations.push(
            ProviderRoadmapObservation::new(ProviderRoadmapObservationKind::FallbackUsed)
                .with_redacted_metadata("edge", edge.id()),
        ),
        Err(error) => observations.push(
            ProviderRoadmapObservation::new(ProviderRoadmapObservationKind::FallbackDenied)
                .with_redacted_metadata("edge", edge.id())
                .with_redacted_metadata("error", error.id()),
        ),
    }
    (observations, outcome)
}

// ---------------------------------------------------------------------
// Runtime API / CLI boundary stability
// ---------------------------------------------------------------------

/// Capability/scope names shaped like a Provider-native handle. Composed
/// into [`reject_provider_specific_handle_capability`] rather than
/// duplicated per hardware family.
pub const PROVIDER_ROADMAP_FORBIDDEN_API_HANDLE_SCOPES: &[&str] = &[
    "cuda-stream",
    "cuda-device-pointer",
    "cuda-module",
    "cuda-event",
    "cuda-kernel-handle",
    "metal-device-handle",
    "metal-buffer",
    "metal-command-queue",
    "metal-pipeline",
    "metal-event",
    "openvino-compiled-graph",
    "qnn-native-handle",
];

/// Rejects a caller-supplied capability/scope name that would expose a
/// Provider-specific native handle through the Runtime Inference API or the
/// `magnetar-cli` boundary, implementing "Runtime API Stability" and "CLI
/// Boundary Stability" (`specs/provider-roadmap/spec.md`), "Runtime Owns
/// Optimized Provider Selection" (`specs/runtime/spec.md`). This does not
/// replace [`validate_inference_scope`] (which governs CLI-owned
/// authority); it is the roadmap-specific handle-exposure check the
/// baseline scope list does not cover.
pub fn reject_provider_specific_handle_capability(
    capability: &str,
) -> Result<(), ProviderRoadmapError> {
    let normalized = capability.trim().to_ascii_lowercase();
    if PROVIDER_ROADMAP_FORBIDDEN_API_HANDLE_SCOPES
        .iter()
        .any(|forbidden| normalized.contains(forbidden))
    {
        return Err(ProviderRoadmapError::ProviderNativeHandleExposureDenied {
            handle_kind: capability.to_string(),
        });
    }
    Ok(())
}

/// A `magnetar-cli`-supplied, non-authoritative Provider preference.
/// Runtime still owns Provider selection through Kernel Registry, Resource
/// Affinity, Memory Manager, readiness, and policy: nothing about receiving
/// this preference grants `magnetar-cli` selection authority.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProviderRoadmapPolicyPreference {
    pub preferred_provider: Option<String>,
    pub allow_optimized_provider_fallback: bool,
}

/// Redacts a Provider diagnostic before `magnetar-cli` displays it, reusing
/// `redact_backend_diagnostic` rather than a parallel redaction path.
pub fn cli_redacted_provider_diagnostic(message: &str) -> String {
    redact_backend_diagnostic(message)
}

/// Accepts a CLI-supplied Provider policy preference as advisory input only
/// (Runtime remains free to ignore or override it), implementing "CLI
/// Boundary Stability": `magnetar-cli` "may ... allow user-facing policy
/// preferences" but "Runtime still owns Provider selection".
pub fn cli_may_pass_policy_preference(
    preference: &ProviderRoadmapPolicyPreference,
) -> ProviderRoadmapPolicyPreference {
    preference.clone()
}

/// Rejects `magnetar-cli` selecting a raw Provider handle, delegating to
/// [`reject_provider_specific_handle_capability`] so the CLI-facing and
/// Runtime-facing checks share one rule set.
pub fn reject_cli_raw_provider_handle_selection(
    capability: &str,
) -> Result<(), ProviderRoadmapError> {
    reject_provider_specific_handle_capability(capability)
}

// ---------------------------------------------------------------------
// Error Model
// ---------------------------------------------------------------------

/// Structured Provider roadmap error, covering every error category from
/// the proposal's "Error Model" section.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProviderRoadmapError {
    ProviderRoadmapUnsupported { reason: String },
    OptimizedCpuProviderUnavailable { reason: String },
    CudaProviderUnavailable { reason: String },
    MetalProviderUnavailable { reason: String },
    OpenVinoProviderUnavailable { reason: String },
    QnnProviderUnavailable { reason: String },
    WebGpuProviderUnavailable { reason: String },
    ProviderFeatureUnsupported { feature: String },
    ProviderLayoutUnsupported { layout: String },
    ProviderDTypeUnsupported { dtype: String },
    ProviderMemoryClassUnsupported { memory_class: String },
    ProviderAdvancedAttentionUnsupported { variant: String },
    ProviderQuantizationUnsupported { reason: String },
    ProviderFusionInvalid { reason: String },
    ProviderConformanceFailed { report: String },
    ProviderBenchmarkFailed { reason: String },
    ProviderFallbackDenied { reason: String },
    ProviderNativeHandleExposureDenied { handle_kind: String },
    InternalProviderRoadmapError { reason: String },
}

impl ProviderRoadmapError {
    pub const fn id(&self) -> &'static str {
        match self {
            Self::ProviderRoadmapUnsupported { .. } => "provider-roadmap-unsupported",
            Self::OptimizedCpuProviderUnavailable { .. } => "optimized-cpu-provider-unavailable",
            Self::CudaProviderUnavailable { .. } => "cuda-provider-unavailable",
            Self::MetalProviderUnavailable { .. } => "metal-provider-unavailable",
            Self::OpenVinoProviderUnavailable { .. } => "openvino-provider-unavailable",
            Self::QnnProviderUnavailable { .. } => "qnn-provider-unavailable",
            Self::WebGpuProviderUnavailable { .. } => "webgpu-provider-unavailable",
            Self::ProviderFeatureUnsupported { .. } => "provider-feature-unsupported",
            Self::ProviderLayoutUnsupported { .. } => "provider-layout-unsupported",
            Self::ProviderDTypeUnsupported { .. } => "provider-dtype-unsupported",
            Self::ProviderMemoryClassUnsupported { .. } => "provider-memory-class-unsupported",
            Self::ProviderAdvancedAttentionUnsupported { .. } => {
                "provider-advanced-attention-unsupported"
            }
            Self::ProviderQuantizationUnsupported { .. } => "provider-quantization-unsupported",
            Self::ProviderFusionInvalid { .. } => "provider-fusion-invalid",
            Self::ProviderConformanceFailed { .. } => "provider-conformance-failed",
            Self::ProviderBenchmarkFailed { .. } => "provider-benchmark-failed",
            Self::ProviderFallbackDenied { .. } => "provider-fallback-denied",
            Self::ProviderNativeHandleExposureDenied { .. } => {
                "provider-native-handle-exposure-denied"
            }
            Self::InternalProviderRoadmapError { .. } => "internal-provider-roadmap-error",
        }
    }
}

impl fmt::Display for ProviderRoadmapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProviderRoadmapUnsupported { reason }
            | Self::OptimizedCpuProviderUnavailable { reason }
            | Self::CudaProviderUnavailable { reason }
            | Self::MetalProviderUnavailable { reason }
            | Self::OpenVinoProviderUnavailable { reason }
            | Self::QnnProviderUnavailable { reason }
            | Self::WebGpuProviderUnavailable { reason }
            | Self::ProviderQuantizationUnsupported { reason }
            | Self::ProviderFusionInvalid { reason }
            | Self::ProviderBenchmarkFailed { reason }
            | Self::ProviderFallbackDenied { reason }
            | Self::InternalProviderRoadmapError { reason } => {
                write!(f, "{}: {reason}", self.id())
            }
            Self::ProviderFeatureUnsupported { feature } => {
                write!(f, "{}: {feature}", self.id())
            }
            Self::ProviderLayoutUnsupported { layout } => write!(f, "{}: {layout}", self.id()),
            Self::ProviderDTypeUnsupported { dtype } => write!(f, "{}: {dtype}", self.id()),
            Self::ProviderMemoryClassUnsupported { memory_class } => {
                write!(f, "{}: {memory_class}", self.id())
            }
            Self::ProviderAdvancedAttentionUnsupported { variant } => {
                write!(f, "{}: {variant}", self.id())
            }
            Self::ProviderConformanceFailed { report } => write!(f, "{}: {report}", self.id()),
            Self::ProviderNativeHandleExposureDenied { handle_kind } => {
                write!(f, "{}: {handle_kind}", self.id())
            }
        }
    }
}

impl Error for ProviderRoadmapError {}

// ---------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------

/// Observation categories from the proposal's "Observability" section.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProviderRoadmapObservationKind {
    RoadmapFeatureDiscovered,
    CapabilityAdvertised,
    CapabilityRejected,
    OptimizedProviderSelected,
    AdvancedAttentionSelected,
    QuantizedKernelSelected,
    FusedKernelSelected,
    FallbackConsidered,
    FallbackUsed,
    FallbackDenied,
    ConformancePassed,
    ConformanceFailed,
    BenchmarkExecuted,
    BenchmarkSkipped,
}

/// A single Provider roadmap observation. Structurally guaranteed to never
/// carry raw tensor values, raw model weights, raw prompts, raw KV cache
/// contents, native Provider/Device/Kernel handles, memory pointers,
/// secrets, or filesystem paths by default: the only fields are an enum
/// `kind`, an optional Provider name, and a `redacted_metadata` string map
/// whose values are always passed through
/// `redact_backend_diagnostic` before being stored -- there is no field
/// through which an unredacted value could reach this type. Implements
/// "Provider Roadmap Observability" (`specs/provider-roadmap/spec.md`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderRoadmapObservation {
    pub kind: ProviderRoadmapObservationKind,
    pub provider: Option<String>,
    pub redacted_metadata: BTreeMap<String, String>,
}

impl ProviderRoadmapObservation {
    pub fn new(kind: ProviderRoadmapObservationKind) -> Self {
        Self {
            kind,
            provider: None,
            redacted_metadata: BTreeMap::new(),
        }
    }

    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Stores `key` -> `value`, redacting `value` through
    /// `redact_backend_diagnostic` first so a native-handle-shaped or
    /// path-shaped value can never survive into the observation.
    pub fn with_redacted_metadata(
        mut self,
        key: impl Into<String>,
        value: impl AsRef<str>,
    ) -> Self {
        self.redacted_metadata
            .insert(key.into(), redact_backend_diagnostic(value.as_ref()));
        self
    }
}

// ---------------------------------------------------------------------
// Conformance
// ---------------------------------------------------------------------

/// A single Provider roadmap conformance check result, mirroring
/// [`crate::CliBoundaryConformanceResult`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderRoadmapConformanceResult {
    pub requirement: String,
    pub passed: bool,
    pub diagnostic: Option<String>,
}

/// A collected set of [`ProviderRoadmapConformanceResult`]s.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderRoadmapConformanceReport {
    pub results: Vec<ProviderRoadmapConformanceResult>,
}

impl ProviderRoadmapConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.results.iter().all(|result| result.passed)
    }
}

fn record(
    results: &mut Vec<ProviderRoadmapConformanceResult>,
    requirement: impl Into<String>,
    passed: bool,
    diagnostic: impl Into<String>,
) {
    let diagnostic = diagnostic.into();
    results.push(ProviderRoadmapConformanceResult {
        requirement: requirement.into(),
        passed,
        diagnostic: (!passed).then_some(diagnostic),
    });
}

/// Runs the Provider roadmap conformance checks described in this module's
/// doc comment: model-family names rejected (hardware/optimized names
/// allowed); native handle exposure denied for every hardware family's
/// handle kinds; fused kernels require an explicit semantic-equivalence
/// declaration; quantized paths require explicit metadata and reject hidden
/// dequantization; unsupported advanced attention fails explicitly;
/// fallback is denied by default; the Runtime/CLI API surface rejects
/// Provider-specific handle capabilities while still accepting ordinary
/// inference scopes; the new conformance profiles are declared without
/// being required by default; and benchmarks stay separate from
/// conformance.
pub fn run_provider_roadmap_conformance() -> ProviderRoadmapConformanceReport {
    let mut results = Vec::new();

    for name in [
        "QwenProvider",
        "LlamaProvider",
        "GemmaProvider",
        "qwen-provider",
    ] {
        let outcome = reject_model_family_provider_name(name);
        let passed = outcome.is_err();
        record(
            &mut results,
            format!("model-family Provider name '{name}' is rejected"),
            passed,
            format!("unexpected outcome: {outcome:?}"),
        );
    }
    for name in [
        "CudaProvider",
        "MetalProvider",
        "OptimizedCpuProvider",
        "ReferenceCpuProvider",
    ] {
        let outcome = reject_model_family_provider_name(name);
        let passed = outcome.is_ok();
        record(
            &mut results,
            format!("hardware/optimized Provider name '{name}' is allowed"),
            passed,
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    for family in [
        ProviderRoadmapHardwareFamily::Cuda,
        ProviderRoadmapHardwareFamily::Metal,
        ProviderRoadmapHardwareFamily::OpenVino,
        ProviderRoadmapHardwareFamily::Qnn,
    ] {
        for handle_kind in family.native_handle_kinds() {
            let outcome = reject_native_handle_exposure(family, handle_kind);
            let passed = matches!(
                outcome,
                Err(ProviderRoadmapError::ProviderNativeHandleExposureDenied { .. })
            );
            record(
                &mut results,
                format!("{} native handle '{handle_kind}' is denied", family.id()),
                passed,
                format!("unexpected outcome: {outcome:?}"),
            );
        }
    }

    {
        let missing = validate_fused_kernel_declaration(FusedKernelDeclaration {
            fusion: None,
            precision: &KernelPrecisionMetadata::default(),
            fallback_hints: &BTreeSet::new(),
        });
        record(
            &mut results,
            "fused kernel without semantic-equivalence metadata is rejected",
            missing.is_err(),
            format!("unexpected outcome: {missing:?}"),
        );

        let valid_fusion = KernelFusionMetadata {
            operator_group: vec![OperatorId::magnetar(
                "matmul",
                1,
                crate::OperatorFamily::LinearAlgebra,
            )],
            preserves_graph_semantics: true,
        };
        let valid_precision = KernelPrecisionMetadata {
            tolerance_profile: Some("operator-default".into()),
            ..KernelPrecisionMetadata::default()
        };
        let valid_fallback = BTreeSet::from([KernelFallbackClass::AlternateKernel]);
        let present = validate_fused_kernel_declaration(FusedKernelDeclaration {
            fusion: Some(&valid_fusion),
            precision: &valid_precision,
            fallback_hints: &valid_fallback,
        });
        record(
            &mut results,
            "fused kernel with a complete semantic-equivalence declaration is accepted",
            present.is_ok(),
            format!("unexpected outcome: {present:?}"),
        );
    }

    {
        let hidden = reject_hidden_dequantization(false);
        record(
            &mut results,
            "hidden dequantization is rejected",
            hidden.is_err(),
            format!("unexpected outcome: {hidden:?}"),
        );
        let explicit = reject_hidden_dequantization(true);
        record(
            &mut results,
            "explicitly declared dequantization is accepted",
            explicit.is_ok(),
            format!("unexpected outcome: {explicit:?}"),
        );

        let empty_metadata = KernelQuantizationMetadata {
            method: crate::kernel::KernelQuantizationMethod::Int8,
            storage_dtype: ComputeDType::SInt8,
            compute_dtype: ComputeDType::Float32,
            accumulation_dtype: ComputeDType::Float32,
            scale_dtype: ComputeDType::Float32,
            zero_point_dtype: None,
            group_size: None,
            packing_layout: TensorLayoutKind::QuantizedPacked,
            dequantization: crate::kernel::KernelDequantizationBehavior::ExplicitBeforeOperator,
            supported_operators: BTreeSet::new(),
            conformance_tolerance_profile: String::new(),
        };
        let missing_metadata = validate_quantization_declaration(&empty_metadata);
        record(
            &mut results,
            "quantized path without a supported Operator or tolerance profile is rejected",
            missing_metadata.is_err(),
            format!("unexpected outcome: {missing_metadata:?}"),
        );
    }

    {
        let outcome =
            reject_unsupported_advanced_attention(AdvancedAttentionVariant::FlashAttention);
        record(
            &mut results,
            "unsupported advanced attention fails explicitly",
            matches!(
                outcome,
                ProviderRoadmapError::ProviderAdvancedAttentionUnsupported { .. }
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let context = ProviderRoadmapFallbackContext::deny_by_default();
        let affinity = ResourceAffinity::new(FallbackClass::Transparent);
        let outcome = evaluate_provider_roadmap_fallback(
            ProviderRoadmapFallbackEdge::CudaToReferenceCpu,
            &affinity,
            &context,
        );
        record(
            &mut results,
            "fallback is denied by default",
            matches!(
                outcome,
                Err(ProviderRoadmapError::ProviderFallbackDenied { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        for capability in PROVIDER_ROADMAP_FORBIDDEN_API_HANDLE_SCOPES {
            let outcome = reject_provider_specific_handle_capability(capability);
            record(
                &mut results,
                format!("Runtime API rejects Provider handle capability '{capability}'"),
                outcome.is_err(),
                format!("unexpected outcome: {outcome:?}"),
            );
        }
        let generic = validate_inference_scope("generation");
        record(
            &mut results,
            "Runtime API still accepts an ordinary inference scope",
            generic.is_ok(),
            format!("unexpected outcome: {generic:?}"),
        );
    }

    {
        let ids = provider_roadmap_conformance_profile_ids();
        let none_required = !ids.is_empty() && ids.values().all(|required| !required);
        record(
            &mut results,
            "post-baseline conformance profiles are declared without implying readiness",
            none_required,
            format!("some roadmap profile is required by default: {ids:?}"),
        );
    }

    {
        let benchmark_only_failure = ProviderRoadmapConformanceReport {
            results: vec![ProviderRoadmapConformanceResult {
                requirement: "optimized kernel matches Reference CPU output".into(),
                passed: false,
                diagnostic: Some("output differs beyond declared tolerance".into()),
            }],
        };
        record(
            &mut results,
            "a fast-but-incorrect Provider does not pass conformance",
            !benchmark_only_failure.is_conformant(),
            "benchmark separation invariant violated",
        );
    }

    ProviderRoadmapConformanceReport { results }
}
