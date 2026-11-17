//! The Reference CPU Provider with every value held "on the Device": its
//! executor answers every `read_tensor_value` with [`TensorValue::Opaque`],
//! as a device Provider (CUDA) does, and forwards everything else to the
//! Reference CPU executor. A Model Component's graph that runs here never
//! asks the engine for host bytes of an intermediate value, so it does not
//! fail on a device Provider for that reason. A head-wise `concat` that did
//! made DeepSeek fail at graph planning on CUDA.

use magnetar_runtime::compute_types::TensorResourceId;
use magnetar_runtime::execution::{
    ProviderCancellationOutcome, ProviderExecutionHandle, ProviderExecutionRequest,
    ProviderExecutionResult, ProviderExecutionStatus,
};
use magnetar_runtime::kernel::{KernelAdvertisement, KernelInvocation, KernelResult};
use magnetar_runtime::memory::{
    MemoryAllocationClass, MemoryAllocationId, MemoryAllocationOwner, MemoryError, MemoryManager,
};
use magnetar_runtime::operator::OperatorSpec;
use magnetar_runtime::packed_weight::{PackedWeight, PackedWeightWrite};
use magnetar_runtime::tensor::HostTensor;
use magnetar_runtime::{
    Device, KernelObservation, Provider, ProviderError, ProviderExecutionApi,
    ProviderExecutionError, ProviderHealth, ProviderMetadata, ProviderRegistry,
    ReferenceCpuProvider, TensorValue, TensorValueAdmissionError,
};
use std::sync::Arc;

/// See the module documentation.
pub(crate) struct ResidentReferenceCpuProvider {
    inner: ReferenceCpuProvider,
    executor: Arc<ResidentExecutor>,
}

impl ResidentReferenceCpuProvider {
    pub(crate) fn new() -> Self {
        let inner = ReferenceCpuProvider::new();
        let executor = Arc::new(ResidentExecutor {
            inner: inner
                .execution_api()
                .expect("the Reference CPU Provider executes Kernels"),
        });
        Self { inner, executor }
    }
}

impl Provider for ResidentReferenceCpuProvider {
    fn metadata(&self) -> ProviderMetadata {
        self.inner.metadata()
    }
    fn register(&self, registry: &mut ProviderRegistry) -> Result<(), ProviderError> {
        self.inner.register(registry)
    }
    fn health(&self) -> ProviderHealth {
        self.inner.health()
    }
    fn devices(&self) -> Vec<Arc<dyn Device>> {
        self.inner.devices()
    }
    fn kernel_advertisements(&self) -> Vec<KernelAdvertisement> {
        self.inner.kernel_advertisements()
    }
    fn execution_api(&self) -> Option<Arc<dyn ProviderExecutionApi>> {
        Some(self.executor.clone())
    }
}

/// The Reference CPU executor, with every value reported resident.
struct ResidentExecutor {
    inner: Arc<dyn ProviderExecutionApi>,
}

impl ProviderExecutionApi for ResidentExecutor {
    fn submit(
        &self,
        request: ProviderExecutionRequest,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError> {
        self.inner.submit(request)
    }
    fn status(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionStatus, ProviderExecutionError> {
        self.inner.status(handle)
    }
    fn cancel(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderCancellationOutcome, ProviderExecutionError> {
        self.inner.cancel(handle)
    }
    fn complete(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionResult, ProviderExecutionError> {
        self.inner.complete(handle)
    }
    fn release(&self, handle: ProviderExecutionHandle) -> Result<(), ProviderExecutionError> {
        self.inner.release(handle)
    }
    fn submit_kernel(
        &self,
        advertisement: &KernelAdvertisement,
        operator: &OperatorSpec,
        invocation: &KernelInvocation,
        memory: &mut MemoryManager,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError> {
        self.inner
            .submit_kernel(advertisement, operator, invocation, memory)
    }
    fn complete_kernel(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<KernelResult, ProviderExecutionError> {
        self.inner.complete_kernel(handle)
    }
    fn write_tensor(
        &self,
        id: TensorResourceId,
        tensor: HostTensor,
    ) -> Result<(), ProviderExecutionError> {
        self.inner.write_tensor(id, tensor)
    }
    fn write_weight_tensor(
        &self,
        id: TensorResourceId,
        tensor: HostTensor,
    ) -> Result<(), ProviderExecutionError> {
        self.inner.write_weight_tensor(id, tensor)
    }
    fn write_packed_weight_tensor(
        &self,
        id: TensorResourceId,
        weight: PackedWeight,
    ) -> Result<PackedWeightWrite, ProviderExecutionError> {
        self.inner.write_packed_weight_tensor(id, weight)
    }
    fn read_tensor(&self, id: &TensorResourceId) -> Option<HostTensor> {
        self.inner.read_tensor(id)
    }
    fn read_tensor_rows(
        &self,
        id: &TensorResourceId,
        first_row: u64,
        rows: u64,
    ) -> Option<HostTensor> {
        self.inner.read_tensor_rows(id, first_row, rows)
    }
    fn release_tensor(&self, id: &TensorResourceId) -> Result<bool, ProviderExecutionError> {
        self.inner.release_tensor(id)
    }
    fn release_admitted_tensor(
        &self,
        memory: &mut MemoryManager,
        id: &TensorResourceId,
    ) -> Result<bool, ProviderExecutionError> {
        self.inner.release_admitted_tensor(memory, id)
    }
    fn write_tensor_admitted(
        &self,
        memory: &mut MemoryManager,
        resource_id: TensorResourceId,
        tensor: HostTensor,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<(), MemoryError> {
        self.inner
            .write_tensor_admitted(memory, resource_id, tensor, class, owner)
    }
    /// The one difference from the Reference CPU: a value it holds is
    /// reported resident, never as host bytes.
    fn read_tensor_value(&self, id: &TensorResourceId) -> Option<TensorValue> {
        self.inner
            .read_tensor_value(id)
            .map(|_| TensorValue::Opaque)
    }
    fn resident_shape(&self, id: &TensorResourceId) -> Option<Vec<u64>> {
        self.inner.resident_shape(id)
    }
    fn write_tensor_value(
        &self,
        id: TensorResourceId,
        value: TensorValue,
    ) -> Result<(), ProviderExecutionError> {
        self.inner.write_tensor_value(id, value)
    }
    fn write_zeroed_tensor(
        &self,
        id: &TensorResourceId,
        shape: &[u64],
    ) -> Result<bool, ProviderExecutionError> {
        self.inner.write_zeroed_tensor(id, shape)
    }
    fn write_tensor_value_admitted(
        &self,
        memory: &mut MemoryManager,
        resource_id: TensorResourceId,
        value: TensorValue,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<(), TensorValueAdmissionError> {
        self.inner
            .write_tensor_value_admitted(memory, resource_id, value, class, owner)
    }
    fn copy_tensor_admitted(
        &self,
        memory: &mut MemoryManager,
        from: &TensorResourceId,
        to: TensorResourceId,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<MemoryAllocationId, TensorValueAdmissionError> {
        self.inner
            .copy_tensor_admitted(memory, from, to, class, owner)
    }
    fn allocate_workspace(
        &self,
        memory: &mut MemoryManager,
        size_bytes: u64,
    ) -> Result<MemoryAllocationId, MemoryError> {
        self.inner.allocate_workspace(memory, size_bytes)
    }
    fn observations(&self) -> Vec<KernelObservation> {
        self.inner.observations()
    }
    fn select_greedy_column(
        &self,
        logits: &TensorResourceId,
        row: u64,
        excluded: &[std::ops::Range<u64>],
        temperature: f32,
    ) -> Option<u64> {
        self.inner
            .select_greedy_column(logits, row, excluded, temperature)
    }
}
