## Context

- The CUDA Provider owns one `CudaContext` per Device. It compiles `KERNEL_SOURCE` with NVRTC at initialization, plus Tensor Core modules for `compute_80` and later, and loads them with cudarc (`context.load_module`).
- Every operator the first-native engine dispatches resolves through the Kernel Registry. The Provider receives a `KernelInvocation` whose kernel name selects a built-in launch (`executor.rs`).
- Step capture records those launches into CUDA graphs (`capture-and-replay-cuda-graphs`). Pooled decode captures are reused across requests (`reuse-decode-captures-across-requests`).
- The GPU runner (`arc-gpu-magnetar`) has two RTX 3060 (`sm_86`).

## Goals / Non-Goals

**Goals:**

- Prepare and dispatch a precompiled CUBIN through the registry, with the built-in kernel as fallback, on a real GPU.
- A launch contract that is data, validated before any launch.
- Measure the whole cost: preparation, the first launch, steady state, memory.

**Non-Goals:**

- A general CUDA ABI for every kernel shape, PTX JIT, hot swap.

## Decisions

### D1. Preparation is its own capability

```text
trait ProviderKernelPreparationApi {
    fn capability(&self) -> KernelPreparationCapability;        // formats, targets, limits
    fn prepare(&self, request: KernelPreparationRequest)
        -> Result<PreparedKernelHandle, KernelPreparationError>; // cold path
    fn release(&self, id: &PreparedKernelId) -> Result<(), KernelPreparationError>;
}

struct KernelPreparationRequest {
    bundle: KernelBundleIdentity,          // lot 0
    blob: KernelBlobDigest,
    bytes: Arc<[u8]>,                      // staged, verified bytes
    launch: LaunchDescriptor,              // parsed extension
    semantic: KernelSemanticBinding,
    specialization: KernelManifestSpecialization,
    device: DeviceBinding,                 // Runtime-selected
}

struct PreparedKernelHandle { id: PreparedKernelId, advertisement: KernelAdvertisement }
```

- `KernelCompilationCapabilityDescriptor` describes source formats and compilers. Preparation describes the loadable formats only (`nvidia:cubin` with its architectures). Keeping them apart means an AOT-only Provider can prepare without pretending to compile.
- The returned advertisement carries a kernel identity derived from the bundle identity and the entry symbol, distinct from every built-in kernel name.

### D2. `magnetar:cuda-launch@1`

```json
{
  "namespace": "magnetar:cuda-launch@1",
  "required": true,
  "data": {
    "entry": "silu_mul_f32",
    "parameters": [
      { "input": 0 }, { "input": 1 }, { "output": 0 },
      { "element_count": { "input": 0 } }
    ],
    "block": [256, 1, 1],
    "grid": { "ceil_div": { "element_count": { "input": 0 } }, "by": 256 },
    "dynamic_shared_bytes": 0
  }
}
```

- **Parameters** form a closed vocabulary:
  - `input`, `output`: a device pointer to that tensor;
  - `attribute`: an `i32`, `i64` or `f32` taken from the invocation;
  - `dimension`: a tensor axis length;
  - `element_count`: a tensor's element count;
  - `constant`: a literal `i32` or `f32`.

  No pointer arithmetic, no user expressions.
- **Geometry.** The block is constant. Each grid axis is a constant or the ceil-division of one derived value by a constant. Dynamic shared memory is a constant, checked against the Device limit at preparation.
- **Checks.** A descriptor that names an input the semantic binding does not have, an attribute of the wrong type, a block over the Device limit, or an `i32` value that could overflow for the declared specialization fails at preparation, never at launch.
- **Tensors.** The POC supports contiguous row-major `f32` and `bf16` only, matching the specialization's dtype and layout.

### D3. Loading and launching on the CUDA Provider

- The architecture check compares the artifact's `target.architecture` (`sm_86`) with the Device's compute capability. A mismatch is a structured `IncompatibleTarget`. No driver JIT or fallback compilation happens.
- The image is loaded from memory (the driver's `cuModuleLoadData`, through cudarc where it exposes it) and the entry is resolved with `cuModuleGetFunction`. A missing symbol fails preparation.
- Launch: the executor builds the parameter array from the invocation and the descriptor and launches on the Provider stream. Inside a step capture, the launch is recorded like any built-in launch, and the capture records a reference to the prepared kernel.
- `release` fails with `InUse` while a live capture references the kernel. Modules are unloaded on release or at Provider shutdown.

### D4. Registry and selection

- After `prepare`, the Runtime registers the returned advertisement under `KernelRegistrationAuthority::Provider`, bound to the bundle identity, as generation 1 of that kernel.
- It competes as a candidate for the same operator as the built-in kernel. Ranking follows `kernel-selection-policy`: filter before rank, no hidden cross-Provider fallback.
- For the experiment, the Runtime takes an explicit pin `(operator, bundle identity)` from its policy configuration. The pin is recorded in dispatch observations and refused if the candidate is ineligible. When C09 lands, its pin replaces this one.

### D5. Fixture production without downloaded binaries

- The CUBIN is built on the GPU runner by a test tool from a CUDA source checked into the repository, with NVRTC targeting the real architecture `sm_86` (CUBIN output).
- The Provider still loads bytes it did not compile itself, which is what this lot must exercise. No binary obtained from outside the build is executed.
- The tool writes the bundle (manifest, blobs, launch extension) and signs the manifest digest with an ephemeral key generated by the job. Only the job's test trust store trusts that key, and the job's permission set grants `nvidia:cubin` on the CUDA Provider to that publisher only.

### D6. Qualification and measurement

- **Qualification** is differential against the Reference CPU kernel over the specialization's shapes, including tails (element counts that are not a multiple of the block). It uses the explicit tolerance of `generated-kernel-qualification` and runs on the GPU before preparation is offered to selection.
- **Measurement.** Qwen2.5-0.5B on the CUDA Provider, greedy, built-in versus pinned external `silu-mul` on the same job. Recorded: TTFT, TPOT, p50/p95, peak device memory, preparation time and the first launch. The generated tokens must match.

## Risks / Trade-offs

- **A wrong descriptor launches a kernel with wrong pointers.** Validation at preparation and differential qualification are the guard. A memory fault still takes down the CUDA context, which is why execution permission is explicit (lot 0).
- **An elementwise POC proves the path, not a speedup.** The built-in `silu-mul` is already memory-bound. `rmsnorm` adds a reduction (shared memory, a block per row) to exercise the descriptor.
- **cudarc may not expose module loading from raw image bytes** in the pinned version. Then the driver call is wrapped in the Provider, keeping the unsafe surface to that wrapper.

## Open Questions

- None for the threshold: RFC question 9 is decided in lot 2 (`serve-kernel-bundles-and-swap-generations`, design D6). A significant gain of at least 10% with no significant regression promotes automatically per hardware target; from 5% to 10%, promotion is manual. Until lot 2 lands, this lot's experiment uses the explicit pin and records the gain, interval and guarded metrics that the policy will use.
