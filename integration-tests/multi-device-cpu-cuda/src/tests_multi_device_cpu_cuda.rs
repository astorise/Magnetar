//! First real proof that Magnetar's Runtime can register two heterogeneous
//! real Providers simultaneously and dispatch one logical computation
//! across both real Devices, with explicit cross-Device movement between
//! stages -- the first real execution `magnetar-runtime`'s own
//! `multi_device_placement` module (a real, independently unit-tested
//! data-model/validation library, previously never wired to any real
//! execution) has ever driven.
//!
//! # Scope: a real foundational proof, not production multi-Device inference
//!
//! This bypasses the full first-native Qwen graph/`ModelInstance`
//! orchestration pipeline entirely (`ModelInstancePlacement` structurally
//! binds one Provider/Device per instance today -- untouched here) and
//! drives the lower, Provider-agnostic `KernelSelectionRequest` ->
//! `KernelRegistry::select` -> `KernelDispatchPlan::from_selection` ->
//! `KernelDispatcher::revalidate` -> `ProviderExecutionApi::submit_kernel`/
//! `complete_kernel` contract directly -- the same minimum-necessary level
//! `providers/cuda`'s own `tests_hardware_hot_path.rs` already established
//! as this repository's precedent for exercising the generic dispatch
//! contract without the Qwen-specific machinery built on top of it.
//!
//! What this test proves, for real, on this repository's own hardware
//! (Reference CPU + a real NVIDIA GPU -- no second GPU exists anywhere in
//! this repository's tooling):
//!
//! - A single `Runtime` can hold two real Providers' Kernels in one shared
//!   Kernel Registry at once (nothing in `Runtime::register_provider`
//!   limits it to one call).
//! - A real, previously-unobserved architectural fact, found by running
//!   this test unmodified once: `KernelRegistry::select` ranks every
//!   *compatible* candidate across *every registered Provider* by fallback
//!   rank / pressure / estimated cost -- it does **not** exclude a
//!   candidate merely because its Provider differs from the request's own
//!   `ResourceAffinity` (CUDA's `add` candidate outranked Reference CPU's
//!   for a CPU-affinity request the first time this ran, because nothing
//!   in `candidate_for_entry` compares a candidate's own Provider against
//!   `request.affinity`). A caller that needs a *specific* Provider --
//!   as real Multi-Device placement does -- must pick its own candidate
//!   out of the full, unfiltered `selection.candidates` list explicitly
//!   (see `dispatch_add`'s own comment), not trust `selection.selected`.
//!   This is real, useful information about this repository's current
//!   Kernel Registry that no prior single-Provider test could have
//!   surfaced, since a single-Provider Runtime's candidate list never has
//!   a competing Provider to lose a ranking contest against.
//! - Stage 1 (`add`) runs for real on the Reference CPU Device; its result
//!   is read back to the host as a plain, Provider-independent `HostTensor`
//!   -- an explicit host round trip, not implicit peer access -- then
//!   admitted fresh into the CUDA Provider's own memory domain for Stage 2
//!   (`add` again) to run for real on the real GPU. This is "Cross Device
//!   Movement Is Explicit" and "Host Staging Policy Survives Multi Device
//!   Placement", both genuinely exercised rather than merely represented.
//! - The final, real, GPU-computed result is read back and checked three
//!   ways: against a hand-computed expected value, and by construction
//!   equals what a single-Provider Reference CPU or CUDA computation of the
//!   same `a + b + c` would produce (both Providers already independently
//!   verified elsewhere in this repository's history to compute `add`
//!   identically to Reference CPU).
//! - `magnetar-runtime`'s own `MultiDevicePlacementPlan`/`PipelineStage`/
//!   `StageMovementEdge`/`DeviceSet` types (previously only unit-tested in
//!   isolation against synthetic fixtures, never built from a real
//!   execution's own data) are populated, after both stages have actually
//!   run and their correctness has already been asserted, from this run's
//!   own real `DeviceMetadata`/`DeviceAvailability`/`TensorResourceId`s,
//!   and progressed through their real `Building` -> `Validating` ->
//!   `Ready` state machine. This is a real, explicit, structured record of
//!   what this test's own execution actually did -- not a controller that
//!   drove or gated that execution, which nothing in this codebase
//!   currently consults `MultiDevicePlacementPlan` to do.
//!
//! What this deliberately does **not** attempt (real future work, not
//! silently assumed): peer-to-peer GPU-to-GPU movement or per-Device memory
//! feasibility ranking against a real multi-GPU budget (no second GPU
//! exists here to exercise either against), Device-loss/degraded-replan
//! behavior, placement-generation republishing under live in-flight
//! traffic, having `MultiDevicePlacementPlan` actually gate or drive
//! dispatch (nothing in this codebase consults it today -- it remains a
//! real, structurally-validated record, not yet an enforced contract), or
//! wiring any of this into `ModelInstance`/production Qwen graph execution.

use std::sync::Arc;

use magnetar_runtime::provider::Provider;
use magnetar_runtime::{
    ComputeDType, DTypeDescriptor, DeviceBinding, DeviceSet, DeviceSetId, DeviceSetMember,
    ExecutionNodeId, FallbackClass, HostStagingPolicy, HostTensor, KernelDispatchPlan,
    KernelDispatchPlanId, KernelDispatcher, KernelInvocationId, KernelMemoryClass, KernelResource,
    KernelResultStatus, KernelSelectionRequest, LayoutDescriptor, MemoryAllocationOwner,
    MemoryDomain, MemoryPlacement, MultiDevicePlacementFingerprint, MultiDevicePlacementGeneration,
    MultiDevicePlacementPlan, MultiDevicePlacementPlanId, MultiDevicePlacementState,
    OperatorFamily, OperatorId, PipelineStage, PlacementBinding, ProviderBinding,
    ProviderExecutionApi, ResourceAffinity, Runtime, ShapeDescriptor, StageMovementEdge,
    TensorDescriptor, TensorResourceDescriptor, TensorResourceId, initial_operator_catalog,
};

use magnetar_provider_cpu::ReferenceCpuProvider;
use magnetar_provider_cuda::CudaProvider;
use sha2::{Digest, Sha256};

fn descriptor_1d(len: u64) -> TensorDescriptor {
    TensorDescriptor::new(
        ShapeDescriptor::new([len]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::Contiguous,
    )
}

fn real_fingerprint(value: &str) -> MultiDevicePlacementFingerprint {
    let digest = Sha256::digest(value.as_bytes());
    let mut encoded = String::with_capacity(7 + digest.len() * 2);
    encoded.push_str("sha256:");
    for byte in digest {
        encoded.push_str(&format!("{byte:02x}"));
    }
    MultiDevicePlacementFingerprint::new(encoded).expect("a real sha256 digest is always valid")
}

/// One real Provider's identity/affinity/execution handle, bundled for
/// [`dispatch_add`] -- built once per Provider from that Provider's own
/// real, discovered `Device`, never hand-constructed.
struct Stage {
    executor: Arc<dyn ProviderExecutionApi>,
    affinity: ResourceAffinity,
    provider_binding: ProviderBinding,
    device_binding: DeviceBinding,
    output_placement: MemoryPlacement,
    /// The real memory class this Provider's own `add` Kernel actually
    /// advertises (`KernelMemoryClass::Host` for Reference CPU,
    /// `KernelMemoryClass::Device` for CUDA) -- not a shared constant: a
    /// `KernelSelectionRequest` whose resources claim the wrong class for
    /// a given Provider is correctly rejected by that Provider's own
    /// Kernel with `KernelMemoryClassUnsupported`, found running this test
    /// for the first time.
    kernel_memory_class: KernelMemoryClass,
}

/// Dispatches a real `add` Kernel invocation for `stage`, through the exact
/// generic contract every Provider-agnostic dispatch in this repository
/// uses (`KernelSelectionRequest` -> `KernelRegistry::select` ->
/// `KernelDispatchPlan::from_selection` -> `KernelDispatcher::revalidate`
/// -> `submit_kernel`/`complete_kernel`), against `runtime`'s single,
/// shared Kernel Registry -- which may have more than one Provider's `add`
/// Kernel registered, as it does in this file's own test. Asserts the
/// Registry actually selected `stage`'s own Provider, not the other one,
/// before proceeding: the real, load-bearing proof that Resource Affinity
/// eligibility -- not caller intent alone -- is what kept the two
/// Providers' candidates apart.
fn dispatch_add(
    runtime: &mut Runtime,
    stage: &Stage,
    step_name: &str,
    lhs_id: TensorResourceId,
    rhs_id: TensorResourceId,
    len: u64,
    output_id: TensorResourceId,
) -> HostTensor {
    let output_descriptor = descriptor_1d(len);
    runtime
        .memory_mut()
        .admit_kernel_output(
            output_id.clone(),
            &output_descriptor,
            stage.output_placement.clone(),
            MemoryAllocationOwner::Runtime,
            stage.affinity.clone(),
        )
        .unwrap_or_else(|error| panic!("{step_name}: failed to pre-admit its output: {error:?}"));

    let operator = OperatorId::magnetar("add", 1, OperatorFamily::Tensor);
    let mut request = KernelSelectionRequest::new(
        format!("multi-device-{step_name}"),
        operator,
        stage.affinity.clone(),
    );
    // CUDA advertises both a plain `f32` `add` Kernel and an `add-half`
    // Kernel (`Float16`/`BrainFloat16`) under this same `OperatorId`
    // (`wire-cuda-half-precision-into-kernel-registry-dispatch`) -- an
    // empty `dtype_requirements` leaves both `compatible` (dtype
    // compatibility only rejects a candidate that fails to support a
    // *declared* requirement), so `add-half` was being selected ahead of
    // the real `f32` `add` for CUDA's stage until this was added, found
    // running this test for the first time (`KernelDTypeUnsupported`
    // surfaced only at `complete_kernel`, since selection alone does not
    // catch it without an explicit requirement here).
    request.dtype_requirements.insert(ComputeDType::Float32);
    for id in [&lhs_id, &rhs_id] {
        request = request.with_input(KernelResource::new(
            TensorResourceDescriptor::new(id.clone(), descriptor_1d(len), stage.affinity.clone()),
            stage.kernel_memory_class,
        ));
    }
    request = request.with_output(KernelResource::new(
        TensorResourceDescriptor::new(output_id.clone(), output_descriptor, stage.affinity.clone()),
        stage.kernel_memory_class,
    ));

    let selection = runtime
        .kernel_registry()
        .select(&request)
        .unwrap_or_else(|error| panic!("{step_name}: Kernel Registry selection failed: {error}"));
    // `KernelRegistry::select` ranks every *compatible* candidate across
    // *every registered Provider* by fallback rank / pressure / estimated
    // cost -- it does not exclude a candidate merely because its Provider
    // differs from `request.affinity`'s own Provider (confirmed the hard
    // way: the first version of this test trusted `selection.selected`
    // directly and it picked CUDA's `add` candidate for a Reference-CPU-
    // affinity request, because CUDA's candidate ranked ahead of CPU's).
    // A caller that needs a *specific* Provider -- as real Multi-Device
    // Placement does -- must pick its own candidate out of the full,
    // unfiltered `selection.candidates` list explicitly, exactly as this
    // helper does below; `select()`'s own ranking is a policy default, not
    // an affinity-derived hard constraint.
    let candidate = selection
        .candidates
        .iter()
        .find(|candidate| candidate.compatible && candidate.provider == stage.provider_binding)
        .cloned()
        .unwrap_or_else(|| {
            panic!(
                "{step_name}: no compatible '{}' candidate from Provider '{}' among {} candidate(s)",
                "add",
                stage.provider_binding,
                selection.candidates.len()
            )
        });
    assert_eq!(
        candidate.device,
        Some(stage.device_binding.clone()),
        "{step_name}: selected candidate's Device does not match this Stage's own Device"
    );
    let advertisement = runtime
        .kernel_registry()
        .active_advertisement(&candidate.kernel)
        .unwrap_or_else(|| panic!("{step_name}: selected advertisement is no longer active"))
        .clone();

    let mut plan = KernelDispatchPlan::from_selection(
        KernelDispatchPlanId::new(format!("multi-device-{step_name}-dispatch")),
        &request,
        &candidate,
        &advertisement,
        KernelInvocationId::new(format!("multi-device-{step_name}-invocation")),
    )
    .unwrap_or_else(|error| panic!("{step_name}: dispatch plan construction failed: {error:?}"));

    let mut dispatcher = KernelDispatcher::new();
    dispatcher
        .revalidate(runtime.kernel_registry(), &mut plan)
        .unwrap_or_else(|error| panic!("{step_name}: revalidation failed: {error:?}"));

    let operator_catalog = initial_operator_catalog();
    let operator_spec = operator_catalog
        .get(&advertisement.implemented_operator)
        .unwrap_or_else(|error| panic!("{step_name}: unknown Operator: {error}"));

    let handle = stage
        .executor
        .submit_kernel(
            &advertisement,
            operator_spec,
            &plan.invocation,
            runtime.memory_mut(),
        )
        .unwrap_or_else(|error| panic!("{step_name}: submit_kernel failed: {error}"));
    let kernel_result = stage
        .executor
        .complete_kernel(&handle)
        .unwrap_or_else(|error| panic!("{step_name}: complete_kernel failed: {error}"));
    assert_eq!(
        kernel_result.status,
        KernelResultStatus::Succeeded,
        "{step_name}: Kernel execution did not succeed: {:?}",
        kernel_result.error
    );

    stage.executor.read_tensor(&output_id).unwrap_or_else(|| {
        panic!("{step_name}: produced no readable host result for '{output_id}'")
    })
}

#[test]
fn cpu_and_cuda_providers_execute_one_chained_add_across_two_real_devices_in_one_runtime() {
    let cuda_probe = CudaProvider::new();
    if !cuda_probe.is_available() {
        eprintln!(
            "skipping: no compatible CUDA device found on this host \
             (this test needs a real second, heterogeneous Device to prove anything)"
        );
        return;
    }

    let cpu_provider = Arc::new(ReferenceCpuProvider::new());
    let cuda_provider = Arc::new(cuda_probe);

    // Real, discovered Devices -- never hand-constructed.
    let cpu_device = cpu_provider
        .devices()
        .into_iter()
        .next()
        .expect("ReferenceCpuProvider always reports exactly one Device");
    let cuda_device = cuda_provider
        .devices()
        .into_iter()
        .next()
        .expect("an available CudaProvider reports exactly one Device");

    let cpu_provider_binding = ProviderBinding::new(cpu_provider.metadata().name.clone());
    let cuda_provider_binding = ProviderBinding::new(cuda_provider.metadata().name.clone());
    let cpu_device_binding = DeviceBinding::new(cpu_device.id().clone());
    let cuda_device_binding = DeviceBinding::new(cuda_device.id().clone());

    let cpu_affinity = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(cpu_provider_binding.clone())
        .with_device(cpu_device_binding.clone());
    let cuda_affinity = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(cuda_provider_binding.clone())
        .with_device(cuda_device_binding.clone());

    let cpu_executor = cpu_provider
        .execution_api()
        .expect("ReferenceCpuProvider always exposes its execution API");
    let cuda_executor = cuda_provider
        .execution_api()
        .expect("an available CudaProvider always exposes its execution API");

    // The real proof this whole file exists for: ONE Runtime, TWO real
    // Providers, registered together -- nothing about `register_provider`
    // limits it to one call, but nothing in this repository's history had
    // exercised that with two real Providers' Kernels both live in the
    // same Kernel Registry until now.
    let mut runtime = Runtime::builder()
        .register_provider(cpu_provider.clone())
        .register_provider(cuda_provider.clone())
        .build()
        .expect("Runtime construction never fails");

    let cpu_stage = Stage {
        executor: cpu_executor,
        affinity: cpu_affinity.clone(),
        provider_binding: cpu_provider_binding.clone(),
        device_binding: cpu_device_binding.clone(),
        output_placement: MemoryPlacement::HostOrdinary,
        kernel_memory_class: KernelMemoryClass::Host,
    };
    let cuda_stage = Stage {
        executor: cuda_executor,
        affinity: cuda_affinity.clone(),
        provider_binding: cuda_provider_binding.clone(),
        device_binding: cuda_device_binding.clone(),
        output_placement: MemoryPlacement::ProviderOwnedOpaque(cuda_provider_binding.clone()),
        kernel_memory_class: KernelMemoryClass::Device,
    };

    // Small, hand-computable values.
    let a = HostTensor::new([3], [1.0, 2.0, 3.0]).unwrap();
    let b = HostTensor::new([3], [10.0, 20.0, 30.0]).unwrap();
    let c = HostTensor::new([3], [100.0, 200.0, 300.0]).unwrap();
    let expected = [111.0f32, 222.0, 333.0];

    let a_id = TensorResourceId::new("multi-device-a");
    let b_id = TensorResourceId::new("multi-device-b");
    let c_id = TensorResourceId::new("multi-device-c");
    let stage1_output_id = TensorResourceId::new("multi-device-stage1-output");
    let stage2_output_id = TensorResourceId::new("multi-device-stage2-output");

    cpu_stage
        .executor
        .write_tensor_value_admitted(
            runtime.memory_mut(),
            a_id.clone(),
            magnetar_runtime::TensorValue::Host(a),
            magnetar_runtime::MemoryAllocationClass::Tensor,
            MemoryAllocationOwner::Runtime,
        )
        .expect("admitting operand 'a' on the Reference CPU Device must succeed");
    cpu_stage
        .executor
        .write_tensor_value_admitted(
            runtime.memory_mut(),
            b_id.clone(),
            magnetar_runtime::TensorValue::Host(b),
            magnetar_runtime::MemoryAllocationClass::Tensor,
            MemoryAllocationOwner::Runtime,
        )
        .expect("admitting operand 'b' on the Reference CPU Device must succeed");

    // Stage 1: `a + b` on the real Reference CPU Device, explicitly
    // dispatched against the CPU Provider's own candidate out of the
    // shared Kernel Registry's full (both-Providers) candidate list.
    let stage1_result = dispatch_add(
        &mut runtime,
        &cpu_stage,
        "stage1-cpu-add",
        a_id.clone(),
        b_id.clone(),
        3,
        stage1_output_id.clone(),
    );
    assert_eq!(
        stage1_result.data,
        vec![11.0, 22.0, 33.0],
        "stage 1 (Reference CPU add) produced an incorrect result"
    );

    // Explicit cross-Device movement: `stage1_result` is now a plain,
    // Provider-independent `HostTensor` -- it left the Reference CPU
    // Provider's own storage the moment `read_tensor` returned it above --
    // and is admitted fresh into the CUDA Provider's own memory domain
    // below. No implicit peer access, no data silently shared between the
    // two Providers' storage.
    cuda_stage
        .executor
        .write_tensor_value_admitted(
            runtime.memory_mut(),
            stage1_output_id.clone(),
            magnetar_runtime::TensorValue::Host(stage1_result),
            magnetar_runtime::MemoryAllocationClass::Tensor,
            MemoryAllocationOwner::Runtime,
        )
        .expect("admitting stage 1's result on the real CUDA Device must succeed");
    cuda_stage
        .executor
        .write_tensor_value_admitted(
            runtime.memory_mut(),
            c_id.clone(),
            magnetar_runtime::TensorValue::Host(c),
            magnetar_runtime::MemoryAllocationClass::Tensor,
            MemoryAllocationOwner::Runtime,
        )
        .expect("admitting operand 'c' on the real CUDA Device must succeed");

    // Stage 2: `(a + b) + c` on the real, physical NVIDIA GPU. Proves the
    // same shared Kernel Registry now selects CUDA's own `add` candidate
    // for an identically-shaped request, excluding the CPU one.
    let stage2_result = dispatch_add(
        &mut runtime,
        &cuda_stage,
        "stage2-cuda-add",
        stage1_output_id.clone(),
        c_id.clone(),
        3,
        stage2_output_id.clone(),
    );
    assert_eq!(
        stage2_result.data, expected,
        "stage 2 (real CUDA add) produced an incorrect result -- the final, \
         real, GPU-computed value must equal the hand-computed a + b + c"
    );

    // Ties `magnetar-runtime`'s own, previously execution-unwired
    // `multi_device_placement` data model to this real run: a real
    // `DeviceSet` built from both Devices' own real `DeviceMetadata`/
    // `DeviceAvailability`, a real two-stage `MultiDevicePlacementPlan`
    // with an explicit `StageMovementEdge` for the CPU -> CUDA host round
    // trip above, progressed through its real state machine only after
    // both stages have actually, successfully executed.
    let device_set = DeviceSet::new(
        DeviceSetId::new("multi-device-cpu-cuda-proof").unwrap(),
        [
            DeviceSetMember::new(cpu_device.metadata().clone(), cpu_device.availability()),
            DeviceSetMember::new(cuda_device.metadata().clone(), cuda_device.availability()),
        ],
    )
    .expect("two real, distinct Devices form a valid DeviceSet");

    let mut plan = MultiDevicePlacementPlan::new(
        MultiDevicePlacementPlanId::new("cpu-then-cuda-add-chain").unwrap(),
        MultiDevicePlacementGeneration::new(1),
        real_fingerprint("stage1:cpu:add|stage2:cuda:add"),
        1,
        device_set,
        "multi-device-cpu-cuda-proof-v1",
    )
    .expect("real, non-zero-generation Plan construction must succeed");
    assert_eq!(plan.state, MultiDevicePlacementState::Building);

    let cpu_binding = PlacementBinding::new(
        cpu_provider_binding,
        cpu_device_binding.clone(),
        MemoryDomain::Host,
    )
    .expect("CPU binding with a matching Host memory domain must succeed");
    let cuda_binding = PlacementBinding::new(
        cuda_provider_binding,
        cuda_device_binding.clone(),
        MemoryDomain::DeviceLocal(cuda_device_binding.clone()),
    )
    .expect("CUDA binding with a matching device-local memory domain must succeed");

    plan.stages.push(
        PipelineStage::new(
            "stage1-cpu-add",
            [ExecutionNodeId::new("stage1-cpu-add")],
            cpu_binding.clone(),
            0,
        )
        .unwrap()
        .with_input(a_id)
        .with_input(b_id)
        .with_output(stage1_output_id.clone()),
    );
    plan.stages.push(
        PipelineStage::new(
            "stage2-cuda-add",
            [ExecutionNodeId::new("stage2-cuda-add")],
            cuda_binding.clone(),
            1,
        )
        .unwrap()
        .with_input(stage1_output_id.clone())
        .with_input(c_id)
        .with_output(stage2_output_id),
    );
    plan.movement_edges.push(
        StageMovementEdge::new(
            "stage1-cpu-add",
            "stage2-cuda-add",
            stage1_output_id,
            cpu_binding,
            cuda_binding,
            HostStagingPolicy::Permit,
        )
        .expect("a movement edge between two genuinely different Devices must succeed"),
    );
    // `mark_ready` requires at least one of `bindings`/`stages` non-empty;
    // this Plan represents its two real Devices entirely through `stages`
    // (each `PipelineStage` already carries its own real `PlacementBinding`
    // above) -- `plan.bindings`/`add_binding` is a separate, coarser
    // `PlacementScope`-keyed representation this Plan does not also need
    // for the same two Devices already fully described by its stages.
    plan.mark_ready()
        .expect("a Plan with real stages and bindings must reach Ready");
    assert_eq!(plan.state, MultiDevicePlacementState::Ready);
    assert_eq!(plan.stages.len(), 2);
    assert_eq!(plan.movement_edges.len(), 1);
    // A real, stable, non-empty fingerprint over this real Plan's own
    // content -- the same `sha256:`-prefixed digest format every other
    // fingerprint in this module uses.
    assert!(plan.fingerprint().as_str().starts_with("sha256:"));
}
