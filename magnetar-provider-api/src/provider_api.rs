use crate::binding::*;
use crate::capability::*;
use crate::compute_support::*;
use crate::compute_types::*;
use crate::device::*;
use crate::execution::*;
use crate::kernel::*;
use crate::provider_abi::*;
pub use crate::provider_abi::*;
pub use crate::provider_status::ProviderError;
use crate::provider_status::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    path::PathBuf,
    sync::Arc,
};

pub const PROVIDER_API_VERSION: u32 = 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderMetadata {
    pub name: String,
    pub version: String,
    pub vendor: String,
    pub api_version: u32,
    pub description: String,
    pub capabilities: BTreeSet<Capability>,
    pub compute_advertisement: ProviderComputeAdvertisement,
    pub compute_operation_schema_support: BTreeMap<ComputeOperationId, ComputeOperationSupport>,
    pub compute_operation_support: BTreeMap<ComputeOperationFamily, ComputeOperationSupport>,
    pub compute_data_movement_support:
        BTreeMap<ComputeDataMovementKind, ComputeDataMovementSupport>,
}
impl ProviderMetadata {
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        vendor: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            vendor: vendor.into(),
            api_version: PROVIDER_API_VERSION,
            description: description.into(),
            capabilities: BTreeSet::new(),
            compute_advertisement: ProviderComputeAdvertisement::default(),
            compute_operation_schema_support: BTreeMap::new(),
            compute_operation_support: BTreeMap::new(),
            compute_data_movement_support: BTreeMap::new(),
        }
    }
}

/// A discovered provider library and the metadata it declares.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderDescriptor {
    pub metadata: ProviderMetadata,
    pub artifact_path: PathBuf,
}

/// A globally unique, package-qualified capability identifier.
/// Trusted native extension contract for provider-owned execution capabilities.
pub trait Provider: Send + Sync {
    fn metadata(&self) -> ProviderMetadata;
    fn register(&self, registry: &mut ProviderRegistry) -> Result<(), ProviderError>;
    fn health(&self) -> ProviderHealth {
        ProviderHealth::Available
    }
    fn health_report(&self) -> ProviderHealthReport {
        ProviderHealthReport::new(ProviderBinding::new(self.metadata().name), self.health())
    }
    fn status_snapshot(&self) -> ProviderStatusSnapshot {
        let metadata = self.metadata();
        let mut snapshot = ProviderStatusSnapshot::from_health_report(self.health_report());
        for family in metadata.compute_operation_support.keys() {
            snapshot = snapshot.with_operation_family_status(OperationFamilyStatus::available(
                ProviderBinding::new(&metadata.name),
                *family,
            ));
        }
        snapshot
    }
    fn capability_health(&self, capability: &CapabilityBinding) -> Option<CapabilityHealth> {
        Some(CapabilityHealth::new(
            ProviderBinding::new(self.metadata().name),
            capability.clone(),
            self.health(),
        ))
    }
    /// Discovers the execution devices owned by this provider.
    fn devices(&self) -> Vec<Arc<dyn Device>> {
        Vec::new()
    }
    fn kernel_advertisements(&self) -> Vec<KernelAdvertisement> {
        Vec::new()
    }
    fn initialize(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    fn shutdown(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    /// Optional Runtime-to-Provider Kernel execution boundary. Returns an
    /// owned, cloneable handle (not a borrow tied to this Provider's own
    /// lifetime) so callers can resolve it once from Runtime's provider
    /// registration and hold it independently -- e.g. across a later `&mut
    /// Runtime` borrow for Memory Manager access -- without downcasting
    /// `&dyn Provider` to a concrete type to recover an owned reference.
    fn execution_api(&self) -> Option<Arc<dyn ProviderExecutionApi>> {
        None
    }
    /// Optional Kernel Compilation Capability advertisement, implementing
    /// "Capability Discovery" (`define-provider-kernel-compilation-capability`):
    /// "A Provider MAY advertise a Kernel Compilation Capability... Compilation
    /// support SHALL NOT be assumed merely because the Provider can execute
    /// Kernels." Defaults to `None` so every existing Provider implementation
    /// remains valid without change.
    fn kernel_compilation_capability(
        &self,
    ) -> Option<crate::kernel_compilation::KernelCompilationCapabilityDescriptor> {
        None
    }
    /// Optional Runtime-to-Provider Kernel Compilation boundary, mirroring
    /// [`Self::execution_api`]'s optional-capability shape.
    fn kernel_compilation_api(
        &self,
    ) -> Option<&dyn crate::kernel_compilation::ProviderKernelCompilationApi> {
        None
    }
    /// Whether this Provider can service a generation request needing more
    /// than one decode step (`close-tachyon-scope-audit-gaps` task group
    /// 4). Defaults to `true` so every existing implementor's behavior is
    /// unchanged unless it explicitly overrides this -- the same pattern
    /// this trait already uses for [`Self::health`]/[`Self::devices`]. A
    /// Provider whose KV history is not host-readable (a genuinely
    /// device-resident-only Provider, e.g. `CudaProvider`) SHALL override
    /// this to `false`, so the generic first-native generation entry point
    /// can fail a multi-step decode request early and explicitly with
    /// `magnetar_runtime::InferenceApiError::Unsupported` before any
    /// real prefill work runs, instead of the request failing several
    /// layers deeper inside KV-history concatenation once it is already
    /// too late to avoid paying for that work.
    fn supports_multi_step_decode(&self) -> bool {
        true
    }
}

/// A Provider-agnostic tensor value (`define-provider-prepared-kernel-execution-contract`):
/// the value [`ProviderExecutionApi::read_tensor_value`]/[`ProviderExecutionApi::write_tensor_value`]
/// exchange, additive alongside the pre-existing `HostTensor`-typed
/// [`ProviderExecutionApi::read_tensor`]/[`ProviderExecutionApi::write_tensor`]
/// (see that trait's documentation for why both exist). Unlike those, a
/// Provider that holds a resource only device-resident can answer honestly
/// with [`Self::Opaque`] instead of being structurally required to produce
/// host bytes it does not have.
#[derive(Clone, Debug)]
pub enum TensorValue {
    /// Host-visible bytes. What every current Provider and test double
    /// produces and consumes today.
    Host(crate::tensor::HostTensor),
    /// This Provider holds the resource privately and declines to expose
    /// host-visible bytes for it. Never produced by Reference CPU today;
    /// exists so a genuinely device-resident Provider has a truthful
    /// answer instead of being forced to fabricate or panic.
    Opaque,
}

impl TensorValue {
    /// Unwraps a host-visible value, or a structured
    /// [`crate::compute_types::TensorError::ResidencyUnavailable`] naming `id` if
    /// this value declined host materialization. The three call sites that
    /// genuinely need host bytes (weight binding, KV-history concatenation,
    /// final logits extraction) use this rather than matching `Host`/`Opaque`
    /// themselves, so the failure is reported uniformly.
    pub fn into_host(
        self,
        id: &TensorResourceId,
    ) -> Result<crate::tensor::HostTensor, crate::compute_types::TensorError> {
        match self {
            Self::Host(tensor) => Ok(tensor),
            Self::Opaque => Err(crate::compute_types::TensorError::residency_unavailable(
                format!("resource '{id}' is device-resident and declined host materialization"),
            )),
        }
    }
}

/// The failure shape for [`ProviderExecutionApi::write_tensor_value_admitted`]:
/// a Memory Manager admission failure and a genuine Provider-native write
/// failure are distinct categories that SHALL NOT be conflated into one
/// error type (`generalize-first-native-provider-dispatch`'s "Provider
/// TensorValue Mutation Error Channel"). `Memory` means admission itself
/// never succeeded -- no write was attempted. `Provider` means admission
/// succeeded but the Provider's own write failed; the allocation that was
/// admitted has already been released by the time this variant is
/// returned.
#[derive(Clone, Debug)]
pub enum TensorValueAdmissionError {
    Memory(crate::memory::MemoryError),
    Provider(ProviderExecutionError),
}

impl fmt::Display for TensorValueAdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Memory(error) => write!(f, "memory admission failed: {error:?}"),
            Self::Provider(error) => write!(f, "Provider tensor value write failed: {error}"),
        }
    }
}

impl Error for TensorValueAdmissionError {}

/// Native Runtime-to-Provider execution boundary for validated planned work.
pub trait ProviderExecutionApi: Send + Sync {
    fn submit(
        &self,
        request: ProviderExecutionRequest,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError>;
    fn status(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionStatus, ProviderExecutionError>;
    fn cancel(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderCancellationOutcome, ProviderExecutionError>;
    fn complete(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionResult, ProviderExecutionError>;
    fn release(&self, handle: ProviderExecutionHandle) -> Result<(), ProviderExecutionError>;

    /// Optional Kernel-level execution boundary: submits one Runtime-created
    /// [`KernelInvocation`] for causal execution, mirroring [`Self::submit`]
    /// but carrying the actually-executable Kernel work directly rather than
    /// a `ComputeExecutionPlan`. A Provider that dispatches first-native
    /// Kernel work overrides this and [`Self::complete_kernel`] together;
    /// the default reports the capability as unsupported so a Provider that
    /// only implements the coarser plan-shaped API above remains valid
    /// without change.
    fn submit_kernel(
        &self,
        advertisement: &KernelAdvertisement,
        operator: &crate::operator::OperatorSpec,
        invocation: &KernelInvocation,
        memory: &mut crate::memory::MemoryManager,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError> {
        let _ = (advertisement, operator, invocation, memory);
        Err(ProviderExecutionError::new(
            ProviderExecutionErrorCode::UnsupportedOperation,
            ProviderExecutionPhase::Submit,
            ProviderBinding::new("unknown"),
            None,
            "this Provider does not implement Kernel-level execution",
        ))
    }

    /// Observes the result of a Kernel invocation previously submitted
    /// through [`Self::submit_kernel`]. See that method's documentation.
    fn complete_kernel(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<KernelResult, ProviderExecutionError> {
        Err(ProviderExecutionError::new(
            ProviderExecutionErrorCode::UnsupportedOperation,
            ProviderExecutionPhase::Complete,
            handle.provider.clone(),
            handle.device.clone(),
            "this Provider does not implement Kernel-level execution",
        ))
    }

    /// Writes one host-visible tensor into this Provider's opaque storage,
    /// keyed by [`TensorResourceId`]. Optional, alongside
    /// [`Self::submit_kernel`]: a Provider that implements Kernel-level
    /// execution through host-visible tensor data overrides this,
    /// [`Self::read_tensor`], and [`Self::allocate_workspace`] together.
    ///
    /// [`crate::tensor::HostTensor`] is host-CPU-shaped data movement,
    /// not a portable Runtime resource representation; a device-only
    /// Provider (CUDA, Metal, ...) is not expected to implement this
    /// meaningfully. It exists on this generic trait -- rather than only as
    /// a `ReferenceCpuExecutor` inherent method -- solely so first-native
    /// dispatch can reach the Runtime-registered Provider without
    /// downcasting `&dyn Provider` to a concrete type. Replacing it with a
    /// genuinely portable Resource-based transport, so this method can be
    /// removed, is tracked separately (task group 5).
    ///
    /// Returns a structured [`ProviderExecutionError`] on failure (GitHub
    /// issue "Provider tensor resource mutations lack structured failure
    /// channels"): a real Provider's write can fail (out of device memory,
    /// a transfer error), and this is the Provider Execution API's own
    /// existing structured-error contract, `ProviderExecutionErrorCode::
    /// MaterializationFailed` fitting this failure exactly. The default
    /// implementation is infallible (`Ok(())`), matching a Provider that
    /// does not implement this optional method at all.
    fn write_tensor(
        &self,
        id: TensorResourceId,
        tensor: crate::tensor::HostTensor,
    ) -> Result<(), ProviderExecutionError> {
        let _ = (id, tensor);
        Ok(())
    }

    /// Writes a Model Instance weight: like [`Self::write_tensor`], but the
    /// resource is never written again while the instance lives, so the
    /// Provider may keep it in any representation that reproduces every
    /// value exactly (`execute-native-half-precision`: a CUDA Provider
    /// stores a weight whose values are all bfloat16 values as bfloat16).
    fn write_weight_tensor(
        &self,
        id: TensorResourceId,
        tensor: crate::tensor::HostTensor,
    ) -> Result<(), ProviderExecutionError> {
        self.write_tensor(id, tensor)
    }

    /// Offers a Model Instance weight in its packed, block-quantized form
    /// (`execute-direct-quantized-weights`). A Provider that computes
    /// directly from `weight`'s format stores it under `id` and answers
    /// [`crate::packed_weight::PackedWeightWrite::Stored`]; every Kernel
    /// reading `id` must then accept it, dequantizing it itself where it
    /// has no direct path. Any other Provider answers `Declined` and stores
    /// nothing, and Model Loading writes the dequantized `f32` weight
    /// through [`Self::write_weight_tensor`] instead.
    fn write_packed_weight_tensor(
        &self,
        id: TensorResourceId,
        weight: crate::packed_weight::PackedWeight,
    ) -> Result<crate::packed_weight::PackedWeightWrite, ProviderExecutionError> {
        let _ = (id, weight);
        Ok(crate::packed_weight::PackedWeightWrite::Declined)
    }

    /// Reads back a tensor previously written through [`Self::write_tensor`]
    /// or produced by a completed Kernel invocation. See that method's
    /// documentation for why this is host-tensor-shaped and provisional.
    fn read_tensor(&self, id: &TensorResourceId) -> Option<crate::tensor::HostTensor> {
        let _ = id;
        None
    }

    /// Rows `first_row..first_row + rows` of a rank-2 tensor, shaped
    /// `[rows, columns]`: what a caller needing only part of a large output
    /// (a prefill's last logits row) downloads. The default reads the whole
    /// tensor and slices it; a Device Provider reads only those rows.
    fn read_tensor_rows(
        &self,
        id: &TensorResourceId,
        first_row: u64,
        rows: u64,
    ) -> Option<crate::tensor::HostTensor> {
        let tensor = self.read_tensor(id)?;
        let [_, columns] = tensor.shape[..] else {
            return None;
        };
        let start = usize::try_from(first_row.checked_mul(columns)?).ok()?;
        let end = usize::try_from(first_row.checked_add(rows)?.checked_mul(columns)?).ok()?;
        let data = tensor.data.get(start..end)?.to_vec();
        crate::tensor::HostTensor::new([rows, columns], data).ok()
    }

    /// Drops a tensor previously written through [`Self::write_tensor`] from
    /// this Provider's opaque storage. `Ok(true)` if a resource was
    /// actually present and removed, `Ok(false)` if `id` was already absent
    /// (idempotent: dropping an already-dropped or never-written resource is
    /// not an error). Runtime memory accounting (`MemoryManager::release`)
    /// is a separate concern from this Provider-side storage release (task
    /// group 9 / Correctif 11): releasing an allocation's accounting does
    /// not by itself free the Provider-owned bytes, and vice versa -- both
    /// must be released for a resource to be fully gone. See
    /// [`Self::write_tensor`]'s documentation for the same provisional-API
    /// rationale, and for why this returns a structured
    /// [`ProviderExecutionError`] on genuine failure -- distinct from
    /// `Ok(false)`'s "nothing to release," a real Provider's release can
    /// itself fail (e.g. Device lost mid-release), which `Ok`/`Err` can now
    /// distinguish where a bare `bool` could not.
    fn release_tensor(&self, id: &TensorResourceId) -> Result<bool, ProviderExecutionError> {
        let _ = id;
        Ok(false)
    }

    /// The [`Self::write_tensor_admitted`] counterpart to [`Self::release_tensor`]:
    /// drops `id` from this Provider's opaque storage (same as
    /// [`Self::release_tensor`]) *and* releases the `MemoryManager`
    /// allocation `write_tensor_admitted` admitted for it, if this Provider
    /// is still holding one -- a later `write_tensor_admitted` call to the
    /// same `id` already releases and replaces the previous allocation
    /// itself, so calling this after that is a no-op for the allocation
    /// half, not a double-release. `Ok(true)` if a resource was actually
    /// present and removed, `Ok(false)` if `id` was already absent.
    /// Resources written through plain [`Self::write_tensor`] were never
    /// admitted in the first place, so this releases nothing extra for
    /// them beyond what [`Self::release_tensor`] already does.
    fn release_admitted_tensor(
        &self,
        memory: &mut crate::memory::MemoryManager,
        id: &TensorResourceId,
    ) -> Result<bool, ProviderExecutionError> {
        let _ = memory;
        self.release_tensor(id)
    }

    /// Admits `tensor`'s byte size through `memory` (Correctif 1: admission
    /// SHALL precede Provider materialization) and writes it into this
    /// Provider's storage under `resource_id`, exactly like
    /// [`Self::write_tensor`] -- except that if this Provider already holds
    /// an allocation it admitted for the *same* `resource_id` (from an
    /// earlier call), that prior allocation is released as part of this
    /// call, after the new one succeeds. This is for resource ids a caller
    /// writes to repeatedly across calls under a *stable* id -- e.g. a
    /// first-native graph's per-node intermediate edges, rewritten once per
    /// generation step -- where plain [`Self::write_tensor`] plus a
    /// caller-side `memory.allocate` would either never account the write
    /// at all, or mint a new allocation every call with nothing releasing
    /// the one it superseded, growing the memory ledger unboundedly over a
    /// long-running session even though Provider storage itself does not
    /// grow (each write overwrites the same entry). Unlike
    /// [`Self::write_tensor`]/[`Self::release_tensor`], a Provider tracking
    /// this replacement itself is what makes that safe without the caller
    /// having to hold onto allocation ids across calls.
    fn write_tensor_admitted(
        &self,
        memory: &mut crate::memory::MemoryManager,
        resource_id: TensorResourceId,
        tensor: crate::tensor::HostTensor,
        class: crate::memory::MemoryAllocationClass,
        owner: crate::memory::MemoryAllocationOwner,
    ) -> Result<(), crate::memory::MemoryError> {
        let _ = (memory, resource_id, tensor, class, owner);
        Err(crate::memory::MemoryError::AllocationDenied {
            reason: "this Provider does not implement admitted tensor writes".into(),
        })
    }

    /// Reads back a value previously written through [`Self::write_tensor_value`]
    /// or [`Self::write_tensor_value_admitted`], as a Provider-agnostic
    /// [`TensorValue`] rather than the `HostTensor`-typed [`Self::read_tensor`].
    /// The default implementation reports nothing found; a Provider that
    /// only implements the `HostTensor`-typed pathway is still valid
    /// without change.
    fn read_tensor_value(&self, id: &TensorResourceId) -> Option<TensorValue> {
        let _ = id;
        None
    }

    /// The shape of the Tensor Resource `id` currently holds, without
    /// requiring host-visible bytes. Shape is structural metadata, not
    /// private tensor content -- unlike [`Self::read_tensor_value`], a
    /// Provider that declines host materialization for `id`
    /// ([`TensorValue::Opaque`]) can still answer this directly from its
    /// own device allocation record. `None` when `id` is not currently
    /// held by this Provider. Needed by a caller (KV-history
    /// concatenation, `implement-device-resident-multi-step-cuda-decode`)
    /// that holds only a `TensorResourceId` for a resource from a prior
    /// step and needs its real shape to build a dispatch input, without
    /// downloading it merely to learn that shape.
    fn resident_shape(&self, id: &TensorResourceId) -> Option<Vec<u64>> {
        let _ = id;
        None
    }

    /// Writes a Provider-agnostic tensor value, unadmitted -- the
    /// [`TensorValue`] counterpart to [`Self::write_tensor`]. The default
    /// implementation does nothing and succeeds; a Provider that only
    /// implements the `HostTensor`-typed pathway is still valid without
    /// change.
    ///
    /// Returns a structured [`ProviderExecutionError`] on failure, mirroring
    /// [`Self::write_tensor`]'s existing contract (`generalize-first-native-
    /// provider-dispatch`'s "Provider TensorValue Mutation Error Channel"):
    /// a real Provider's write can fail (allocation, H2D/D2D copy, lost
    /// context, an async error surfaced at sync time), and a bare `()`
    /// cannot represent that.
    fn write_tensor_value(
        &self,
        id: TensorResourceId,
        value: TensorValue,
    ) -> Result<(), ProviderExecutionError> {
        let _ = (id, value);
        Ok(())
    }

    /// Writes `id` as an f32 tensor of `shape` whose values are all zero,
    /// exactly as [`Self::write_tensor_value`] would write a host tensor of
    /// zeros, but without the caller building or transferring one. Returns
    /// `Ok(false)` when this Provider does not offer it (the default); the
    /// caller then writes the zeros itself.
    fn write_zeroed_tensor(
        &self,
        id: &TensorResourceId,
        shape: &[u64],
    ) -> Result<bool, ProviderExecutionError> {
        let _ = (id, shape);
        Ok(false)
    }

    /// The [`TensorValue`] counterpart to [`Self::write_tensor_admitted`]:
    /// admits `value`'s byte size (when it is [`TensorValue::Host`]; a
    /// Provider whose [`TensorValue::Opaque`] resources have their own
    /// accounting model may admit however is appropriate for them) before
    /// writing, replacing and releasing whatever allocation this Provider
    /// previously admitted for the same `resource_id`, exactly like
    /// [`Self::write_tensor_admitted`].
    ///
    /// Returns [`TensorValueAdmissionError`] rather than a bare
    /// [`crate::memory::MemoryError`]: admission and the Provider-native
    /// write are genuinely different failure categories (Memory Manager
    /// ledger vs. Provider execution), and conflating them into one
    /// `MemoryError`-typed result would force a real write failure to be
    /// misreported as a memory-admission failure or silently dropped
    /// (`generalize-first-native-provider-dispatch`'s "Provider TensorValue
    /// Mutation Error Channel").
    ///
    /// The default implementation fails closed, matching
    /// [`Self::write_tensor_admitted`]'s default: a generic default has no
    /// way to know this Provider's own identity for
    /// [`crate::memory::MemoryPlacement::ProviderOwnedOpaque`], so it cannot
    /// safely admit on a Provider's behalf. A concrete Provider that knows
    /// its own identity -- e.g. `ReferenceCpuExecutor`, which delegates to
    /// its own already-correct [`Self::write_tensor_admitted`] for the
    /// `Host` case -- overrides this with the real admit-then-write-with-
    /// rollback sequence `WeightMaterializationTransaction::stage_weight`
    /// already establishes for the `HostTensor`-typed pair.
    fn write_tensor_value_admitted(
        &self,
        memory: &mut crate::memory::MemoryManager,
        resource_id: TensorResourceId,
        value: TensorValue,
        class: crate::memory::MemoryAllocationClass,
        owner: crate::memory::MemoryAllocationOwner,
    ) -> Result<(), TensorValueAdmissionError> {
        let _ = (memory, resource_id, value, class, owner);
        Err(TensorValueAdmissionError::Memory(
            crate::memory::MemoryError::AllocationDenied {
                reason: "this Provider does not implement admitted tensor value writes".into(),
            },
        ))
    }

    /// Duplicates `from`'s current bytes to a fresh, caller-chosen identity
    /// `to`, without requiring host-visible bytes to exist at any point
    /// during the copy for a Provider that holds `from` device-resident
    /// (`implement-device-resident-multi-step-cuda-decode`). Replaces (and
    /// releases) whatever allocation `to` previously held, exactly like
    /// [`Self::write_tensor_value_admitted`] already does for a fresh
    /// write. Closes the KV-history commit/pending round-trip that would
    /// otherwise download `from` to host and re-upload it as `to` for no
    /// reason other than changing its identity.
    ///
    /// Returns the fresh [`crate::memory::MemoryAllocationId`] this copy
    /// admitted `to` under -- unlike [`Self::write_tensor_value_admitted`],
    /// a caller here has no `TensorValue` of its own to have already
    /// derived a byte size or allocation from, and no Runtime-side
    /// `TensorResidency` record generically maps an arbitrary resource id
    /// back to the allocation a Provider-internal admit-then-copy created
    /// for it (that bookkeeping is Provider-private, e.g. a `TensorResourceId
    /// -> MemoryAllocationId` table each concrete Provider already keeps
    /// for its own admitted writes) -- so a caller that needs the
    /// allocation id (e.g. to release it later, or to record it in a
    /// binding of its own) can only get it back as this call's own return
    /// value.
    ///
    /// The default implementation fails closed, matching
    /// [`Self::write_tensor_value_admitted`]'s own default: a generic
    /// default has no way to know this Provider's own identity for
    /// [`crate::memory::MemoryPlacement::ProviderOwnedOpaque`]/`Device`, so
    /// it cannot safely admit on a Provider's behalf. A concrete Provider
    /// that knows its own identity and storage (Reference CPU, CUDA)
    /// overrides this with a real copy -- a `HostTensor` clone for
    /// Reference CPU, a real device-to-device copy for CUDA.
    fn copy_tensor_admitted(
        &self,
        memory: &mut crate::memory::MemoryManager,
        from: &TensorResourceId,
        to: TensorResourceId,
        class: crate::memory::MemoryAllocationClass,
        owner: crate::memory::MemoryAllocationOwner,
    ) -> Result<crate::memory::MemoryAllocationId, TensorValueAdmissionError> {
        let _ = (memory, from, to, class, owner);
        Err(TensorValueAdmissionError::Memory(
            crate::memory::MemoryError::AllocationDenied {
                reason: "this Provider does not implement admitted tensor resource copies".into(),
            },
        ))
    }

    /// Requests scratch-space workspace through the Runtime's
    /// [`crate::memory::MemoryManager`] for a Kernel that advertises a
    /// required workspace. See [`Self::write_tensor`]'s documentation for
    /// why this lives on the generic trait provisionally.
    fn allocate_workspace(
        &self,
        memory: &mut crate::memory::MemoryManager,
        size_bytes: u64,
    ) -> Result<crate::memory::MemoryAllocationId, crate::memory::MemoryError> {
        let _ = (memory, size_bytes);
        Err(crate::memory::MemoryError::AllocationDenied {
            reason: "this Provider does not implement Kernel-level workspace allocation".into(),
        })
    }

    /// This Provider's own Kernel-dispatch observation trail (see
    /// [`Self::write_tensor`]'s documentation for the same provisional-API
    /// rationale). Lets a caller confirm dispatch reached the same
    /// Runtime-registered Provider instance it resolved, without
    /// downcasting to a concrete type to call an inherent method.
    fn observations(&self) -> Vec<KernelObservation> {
        Vec::new()
    }

    /// Whether this Provider can capture a generation step and replay it
    /// (`capture-and-replay-cuda-graphs`). The other step capture methods
    /// are only called when this returns `true`.
    fn supports_step_capture(&self) -> bool {
        false
    }

    /// Takes the Runtime's placement of the Model Instance's decoder layers
    /// across this Provider's Devices (`enforce-local-multigpu-and-expert-
    /// placement` G05), before any weight is written: every later weight
    /// goes to its layer's Device and every Kernel runs where its weights
    /// are, the Provider moving activations across segment boundaries
    /// explicitly. The default accepts a single-segment placement (nothing
    /// to do) and rejects any other: such a Provider has one Device.
    fn apply_layer_placement(
        &self,
        placement: &crate::layer_placement::LayerPlacement,
    ) -> Result<(), ProviderExecutionError> {
        if placement.segments().len() <= 1 {
            return Ok(());
        }
        Err(ProviderExecutionError::new(
            ProviderExecutionErrorCode::UnsupportedOperation,
            ProviderExecutionPhase::Submit,
            ProviderBinding::new("unknown"),
            None,
            "this Provider does not place layers across Devices",
        ))
    }

    /// Takes the Runtime's tensor-parallel plan (which weights are split
    /// across this Provider's Devices, and how) before any weight is
    /// written: every later weight is stored as each Device's shard, and
    /// every invocation runs on every Device over its shard. The default
    /// accepts a plan that splits nothing and rejects any other.
    fn apply_tensor_parallel(
        &self,
        plan: &crate::tensor_parallel::TensorParallelPlan,
    ) -> Result<(), ProviderExecutionError> {
        if !plan.splits() {
            return Ok(());
        }
        Err(ProviderExecutionError::new(
            ProviderExecutionErrorCode::UnsupportedOperation,
            ProviderExecutionPhase::Submit,
            ProviderBinding::new("unknown"),
            None,
            "this Provider does not split weights across Devices",
        ))
    }

    /// Starts recording every Kernel invocation and resource access until
    /// [`Self::end_step_capture`] or [`Self::abort_step_capture`]. Every
    /// integer attribute named in `request.advancing_attributes` that a
    /// captured invocation carries must be replayable increased by the
    /// replay offset; a Provider that cannot do so for an invocation fails
    /// that invocation, and the capture with it.
    fn begin_step_capture(
        &self,
        request: &StepCaptureRequest,
    ) -> Result<(), ProviderExecutionError> {
        let _ = request;
        Err(step_capture_unsupported(ProviderExecutionPhase::Submit))
    }

    /// Ends the capture [`Self::begin_step_capture`] started. When this
    /// returns, the captured step's results are available exactly as after
    /// an uncaptured step: a Provider that only records during capture runs
    /// the recorded work once here.
    fn end_step_capture(&self) -> Result<StepCaptureId, ProviderExecutionError> {
        Err(step_capture_unsupported(ProviderExecutionPhase::Complete))
    }

    /// Discards a capture in progress. The captured step's work may not
    /// have run.
    fn abort_step_capture(&self) {}

    /// Writes each host input into the resource it was captured with, then
    /// repeats the captured work with every advancing attribute increased
    /// by `offset`. Fails with [`StepReplayError::Invalidated`], running
    /// nothing, when a resource the capture touched no longer has the
    /// storage it had at capture time, or when `offset` leaves a bound the
    /// capture recorded.
    fn replay_step(
        &self,
        id: &StepCaptureId,
        offset: u64,
        inputs: Vec<(TensorResourceId, crate::tensor::HostTensor)>,
    ) -> Result<(), StepReplayError> {
        let _ = (id, offset, inputs);
        Err(StepReplayError::Failed(step_capture_unsupported(
            ProviderExecutionPhase::Submit,
        )))
    }

    /// Frees a capture. Unknown ids are ignored.
    fn release_step_capture(&self, id: &StepCaptureId) {
        let _ = id;
    }

    /// Greedy next-token selection on the Device (`keep-sampling-on-device`
    /// B10): the column of `logits`' row `row` with the highest score
    /// divided by `temperature` (left as is when it is 0), among the
    /// columns outside `excluded` whose raw score is finite, ties going to
    /// the lowest column -- exactly the Runtime's host greedy selection.
    /// Only the chosen column crosses back to the host.
    ///
    /// `None` when this Provider does not select on the Device, or no
    /// column is eligible: the caller then downloads the logits and selects
    /// on the host, which also reports why nothing was eligible.
    fn select_greedy_column(
        &self,
        logits: &TensorResourceId,
        row: u64,
        excluded: &[std::ops::Range<u64>],
        temperature: f32,
    ) -> Option<u64> {
        let _ = (logits, row, excluded, temperature);
        None
    }
}

fn step_capture_unsupported(phase: ProviderExecutionPhase) -> ProviderExecutionError {
    ProviderExecutionError::new(
        ProviderExecutionErrorCode::UnsupportedOperation,
        phase,
        ProviderBinding::new("unknown"),
        None,
        "this Provider does not capture generation steps",
    )
}

/// What a step capture records (`capture-and-replay-cuda-graphs`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StepCaptureRequest {
    /// Integer attributes that advance by one per step, and by the replay
    /// offset when a captured step is replayed.
    pub advancing_attributes: Vec<String>,
}

/// An opaque, Provider-chosen identity for one captured step.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StepCaptureId(String);

impl StepCaptureId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for StepCaptureId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Why [`ProviderExecutionApi::replay_step`] ran nothing or failed.
#[derive(Clone, Debug)]
pub enum StepReplayError {
    /// The capture can no longer be replayed: a resource it touched has
    /// other storage now, or the offset leaves one of its bounds. Nothing
    /// ran; the step must run uncaptured.
    Invalidated { reason: String },
    /// The replay itself failed.
    Failed(ProviderExecutionError),
}

impl fmt::Display for StepReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalidated { reason } => write!(f, "step capture invalidated: {reason}"),
            Self::Failed(error) => write!(f, "step replay failed: {error}"),
        }
    }
}

impl Error for StepReplayError {}

/// Receives provider contributions.
#[derive(Clone, Default)]
pub struct ProviderRegistry {
    devices: BTreeMap<DeviceId, Arc<dyn Device>>,
    device_providers: BTreeMap<DeviceId, String>,
    capabilities: BTreeMap<CapabilityId, BTreeMap<CapabilityVersion, Capability>>,
    capability_providers: BTreeMap<(CapabilityId, CapabilityVersion), BTreeSet<String>>,
}
impl ProviderRegistry {
    pub fn register_devices(
        &mut self,
        provider: &str,
        devices: impl IntoIterator<Item = Arc<dyn Device>>,
    ) -> Result<(), ProviderError> {
        let devices = devices.into_iter().collect::<Vec<_>>();
        let mut ids = BTreeSet::new();
        for device in &devices {
            let metadata = device.metadata();
            if metadata.id.as_str().trim().is_empty() {
                return Err(ProviderError::InvalidDevice(
                    "identifier must not be empty".into(),
                ));
            }
            if metadata.provider != provider {
                return Err(ProviderError::DeviceProviderMismatch {
                    device: metadata.id.clone(),
                    expected: provider.into(),
                    found: metadata.provider.clone(),
                });
            }
            if self.devices.contains_key(&metadata.id) || !ids.insert(metadata.id.clone()) {
                return Err(ProviderError::DeviceAlreadyRegistered(metadata.id.clone()));
            }
        }
        for device in devices {
            let metadata = device.metadata();
            self.device_providers
                .insert(metadata.id.clone(), provider.into());
            self.devices.insert(metadata.id.clone(), device);
        }
        Ok(())
    }
    pub fn device(&self, id: &DeviceId) -> Option<&dyn Device> {
        self.devices.get(id).map(AsRef::as_ref)
    }
    pub fn devices(&self) -> impl Iterator<Item = &dyn Device> {
        self.devices.values().map(AsRef::as_ref)
    }
    pub fn provider_for_device(&self, id: &DeviceId) -> Option<&str> {
        self.device_providers.get(id).map(String::as_str)
    }
    pub fn devices_for_provider<'a>(
        &'a self,
        provider: &'a str,
    ) -> impl Iterator<Item = &'a dyn Device> {
        self.devices.iter().filter_map(move |(id, device)| {
            (self.provider_for_device(id) == Some(provider)).then_some(device.as_ref())
        })
    }
    pub fn register_capabilities(
        &mut self,
        provider: &str,
        capabilities: impl IntoIterator<Item = Capability>,
    ) -> Result<(), ProviderError> {
        for capability in capabilities {
            self.register_capability(capability.clone())?;
            self.capability_providers
                .entry((capability.id, capability.version))
                .or_default()
                .insert(provider.into());
        }
        Ok(())
    }
    pub fn register_capability(&mut self, capability: Capability) -> Result<(), ProviderError> {
        if capability.id.as_str().trim().is_empty() {
            return Err(ProviderError::InvalidCapability(
                "identifier must not be empty".into(),
            ));
        }
        if capability.descriptor.contracts.is_empty() {
            return Err(ProviderError::InvalidCapability(format!(
                "capability '{}' must declare at least one WIT contract",
                capability.id
            )));
        }
        let versions = self.capabilities.entry(capability.id.clone()).or_default();
        match versions.get(&capability.version) {
            Some(existing) if existing != &capability => {
                Err(ProviderError::ConflictingCapability {
                    id: capability.id,
                    version: capability.version,
                })
            }
            Some(_) => Ok(()),
            None => {
                versions.insert(capability.version, capability);
                Ok(())
            }
        }
    }
    pub fn capabilities(&self) -> impl Iterator<Item = &Capability> {
        self.capabilities
            .values()
            .flat_map(|versions| versions.values())
    }
    pub fn capability(&self, id: &CapabilityId, version: CapabilityVersion) -> Option<&Capability> {
        self.capabilities.get(id)?.get(&version)
    }
    /// Returns the latest registered version that satisfies `required`.
    pub fn resolve_capability(
        &self,
        id: &CapabilityId,
        required: CapabilityVersion,
    ) -> Option<&Capability> {
        self.capabilities
            .get(id)?
            .iter()
            .rev()
            .find_map(|(_, capability)| {
                capability
                    .version
                    .is_compatible_with(required)
                    .then_some(capability)
            })
    }
    /// Returns one exact Capability only when `provider` advertises it.
    pub fn capability_for_provider(
        &self,
        id: &CapabilityId,
        version: CapabilityVersion,
        provider: &str,
    ) -> Option<&Capability> {
        let capability = self.capability(id, version)?;
        self.providers_for(capability)
            .any(|candidate| candidate == provider)
            .then_some(capability)
    }
    /// Returns the latest compatible version advertised by one Provider.
    pub fn resolve_capability_for_provider(
        &self,
        id: &CapabilityId,
        required: CapabilityVersion,
        provider: &str,
    ) -> Option<&Capability> {
        self.capabilities
            .get(id)?
            .iter()
            .rev()
            .find_map(|(_, capability)| {
                (capability.version.is_compatible_with(required)
                    && self
                        .providers_for(capability)
                        .any(|candidate| candidate == provider))
                .then_some(capability)
            })
    }
    pub fn validate_dependencies(&self) -> Result<(), ProviderError> {
        for capability in self.capabilities() {
            for (dependency, version) in &capability.descriptor.dependencies {
                if self.resolve_capability(dependency, *version).is_none() {
                    return Err(ProviderError::MissingCapabilityDependency {
                        capability: capability.id.clone(),
                        dependency: dependency.clone(),
                        required: *version,
                    });
                }
            }
        }
        Ok(())
    }
    /// Returns implementations for an exact, resolved capability version in deterministic order.
    pub fn providers_for(&self, capability: &Capability) -> impl Iterator<Item = &str> {
        self.capability_providers
            .get(&(capability.id.clone(), capability.version))
            .into_iter()
            .flat_map(|providers| providers.iter().map(String::as_str))
    }
    pub fn has_capability(&self, capability: &Capability) -> bool {
        self.capability(&capability.id, capability.version)
            .is_some()
    }
    #[doc(hidden)]
    pub fn clear(&mut self) {
        self.devices.clear();
        self.device_providers.clear();
        self.capabilities.clear();
        self.capability_providers.clear();
    }
}

impl From<ProviderError> for ComputeError {
    fn from(error: ProviderError) -> Self {
        let message = error.to_string();
        match error {
            ProviderError::NoCompatibleProvider(capability) => ComputeError::new(
                ComputeErrorCode::NoCompatibleProvider,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_capability(capability)),
            ProviderError::PolicyRejectedProvider {
                capability,
                policy: _,
            } => ComputeError::new(
                ComputeErrorCode::PolicyRejectedProvider,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_capability(capability)),
            ProviderError::DeviceAlreadyRegistered(device) => ComputeError::new(
                ComputeErrorCode::DeviceUnavailable,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_device(DeviceBinding::new(device))),
            ProviderError::DeviceProviderMismatch {
                device,
                expected,
                found,
            } => ComputeError::new(
                ComputeErrorCode::DeviceBoundResource,
                ComputeErrorPhase::AffinityValidation,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(
                ComputeDiagnostic::new()
                    .with_device(DeviceBinding::new(device))
                    .with_provider(ProviderBinding::new(found))
                    .with_rejected_candidate(ProviderBinding::new(expected)),
            ),
            ProviderError::IncompatibleApiVersion { provider, .. } => ComputeError::new(
                ComputeErrorCode::ProviderUnavailable,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(
                ComputeDiagnostic::new().with_provider(ProviderBinding::new(provider)),
            ),
            ProviderError::InvalidCapabilityVersion(version) => ComputeError::new(
                ComputeErrorCode::CapabilityVersionMismatch,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_backend_message(version)),
            ProviderError::Lifecycle(message_text) => ComputeError::new(
                ComputeErrorCode::ExecutionInterrupted,
                ComputeErrorPhase::Interruption,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_backend_message(message_text))
            .with_recovery_hint(RecoveryHint::RetryBeforeState),
            ProviderError::Load {
                path: _,
                message: backend_message,
            } => ComputeError::new(
                ComputeErrorCode::ProviderUnavailable,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_backend_message(backend_message))
            .with_recovery_hint(RecoveryHint::RetryBeforeState),
            ProviderError::ProviderPathDenied { path } => ComputeError::new(
                ComputeErrorCode::ProviderUnavailable,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_backend_message(path.to_string_lossy()))
            .with_recovery_hint(RecoveryHint::RetryBeforeState),
            ProviderError::UnsupportedDynamicAbi {
                path: _,
                expected_symbol,
                expected_descriptor,
            } => ComputeError::new(
                ComputeErrorCode::ProviderUnavailable,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_backend_message(format!(
                "{expected_symbol} v{}.{}",
                expected_descriptor.abi_version.major, expected_descriptor.abi_version.minor
            )))
            .with_recovery_hint(RecoveryHint::RetryBeforeState),
            ProviderError::InvalidAbiDescriptor(_)
            | ProviderError::UnsupportedAbiVersion { .. }
            | ProviderError::InvalidMetadata(_)
            | ProviderError::InvalidAdvertisement(_)
            | ProviderError::InvalidDeviceMetadata(_)
            | ProviderError::InitializationFailure(_)
            | ProviderError::ProviderNotReady(_)
            | ProviderError::ProviderDraining(_)
            | ProviderError::ProviderSaturated(_)
            | ProviderError::ExecutionRejected(_)
            | ProviderError::ExecutionFailed(_)
            | ProviderError::CancellationUnsupported(_)
            | ProviderError::ResourceInvalid(_)
            | ProviderError::PanicOrUnwindViolation(_)
            | ProviderError::InternalProviderError(_)
            | ProviderError::CancellationFailed { .. } => ComputeError::new(
                ComputeErrorCode::ProviderUnavailable,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message.clone(),
            )
            .with_diagnostic(ComputeDiagnostic::new().with_backend_message(message)),
            other => ComputeError::new(
                ComputeErrorCode::Internal,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_backend_message(other.to_string())),
        }
    }
}
