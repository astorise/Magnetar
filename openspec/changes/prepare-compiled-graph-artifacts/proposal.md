## Why

Lot 4 of the [accelerator packages RFC](../../../docs/rfcs/accelerator-packages-2026-11-23.md) extends lots 1 to 3 in two directions.

- **More Providers for operator kernels**, once each has a verified compute path:
  - WGPU runs one real WGSL kernel today (`add`);
  - ROCm and QNN discover devices without computing;
  - the Metal Provider is a structural placeholder;
  - the Intel NPU Provider (the `openvino` submodule) discovers devices through Level Zero.
- **Compiled graphs.** Accelerators such as OpenVINO NPU and the QNN HTP execute graphs compiled ahead of time for a specific SDK, driver, accelerator generation and often a specific model's weights, not isolated kernels. Such an artifact cannot be swapped like an attention kernel: it replaces a segment of the execution graph, or a whole Model Instance, and may own its weights' memory.

The RFC asks not to inject a `CompiledGraph` as an `OperatorKernel`, and to keep Kernel Exchange Bundle v1 intact.

## What Changes

- **The `magnetar:accelerator-package@1` envelope.** It is a canonical JSON document, content-addressed like manifest v1, that groups variants of one semantic capability under a name and version. Each variant has:
  - a kind (`operator-kernel`, `fused-kernel` or `compiled-graph`);
  - a Provider compatibility, a format and requirements;
  - an artifact reference. Operator and fused kernels reference a Kernel Exchange Bundle by manifest digest, unchanged. A compiled graph references its own blob and binding.

  The envelope groups and describes. It grants nothing, and each referenced artifact goes through lots 0 to 2.
- **The `CompiledGraph` binding:**
  - a segment signature: the ordered operator semantics, attributes, and the shapes, dtypes and layouts at the segment's inputs and outputs;
  - a model binding: the Model Artifact digest, or the digests of the weight tensors baked in;
  - the SDK, compiler, driver and runtime versions and the accelerator generation;
  - the memory it allocates itself.
- **Segment or instance substitution:**
  - the Runtime matches a compiled graph to a contiguous segment of a prepared execution graph whose signature and model binding match exactly, or to a whole Model Instance;
  - the Provider prepares it as a new instance and the Memory Manager admits its declared memory;
  - a swap prepares the new instance and drains the old one (lot 2 leases), never patching a running graph.
- **A per-Provider plan, each step gated on its own change:**

  | Provider | Kind | Format | Gated on |
  | --- | --- | --- | --- |
  | WGPU | operator kernels | `wgsl` (pipelines) | `integrate-wgpu-production-compute` |
  | ROCm | operator kernels | HSACO | `qualify-rocm-compute-profiles` |
  | Metal | operator kernels | `metallib` | `evaluate-native-metal-prefill` |
  | Intel NPU, QNN | compiled graphs | — | `evaluate-hardware-gated-npu-compute` (G04) and real hardware |
  | IREE (optional) | producer of compiled graphs | — | `evaluate-iree-provider-compilation` (C07) |

- **A support matrix**, `docs/architecture/accelerator-support-matrix.md`. Its axes are Provider × artifact kind × replacement granularity × status (contract, fixture, hardware-qualified). A cell moves only on recorded evidence.

## Capabilities

### New Capabilities

- `accelerator-package`: the envelope and its validation.
- `compiled-graph-artifact`: the compiled graph binding, segment and instance substitution, and memory admission.
- `accelerator-support-matrix`: the evidence-gated support matrix.

### Modified Capabilities

None. Provider-specific requirements land in each gating change.

## Impact

- `magnetar-kernel`: envelope parsing and validation; compiled graph binding types and the segment signature.
- `magnetar-runtime`: segment matching on prepared execution graphs; instance-level substitution through the Model Instance lifecycle; Memory Manager admission of declared memory.
- The Provider repositories (`wgpu`, `rocm`, `metal`, `openvino`, `qnn`): only through their gating changes.
- Documentation: the support matrix.

## Dependencies

- Lots 0 to 2 ([`define-kernel-artifact-sources`](../define-kernel-artifact-sources/proposal.md), [`prepare-prebuilt-cuda-kernels`](../prepare-prebuilt-cuda-kernels/proposal.md), [`serve-kernel-bundles-and-swap-generations`](../serve-kernel-bundles-and-swap-generations/proposal.md)). Lot 3 for the CPU variants it lists.
- Per Provider: the gating changes listed above.

## Compatibility and authority

- Kernel Exchange Bundle v1 is unchanged and remains the source of truth for operator and fused kernels.
- A compiled graph never runs without a matching segment signature and model binding. A mismatch is a structured incompatibility, never a best-effort substitution.
- Components still describe graphs; the Runtime decides substitution. No WIT change.
- The RFC's question 8 is not decided here: whether Intel gets an OpenVINO Provider separate from a Level Zero one is decided in G04.

## Non-goals

- Implementing any Provider's compute path (each gating change does).
- Compiling graphs inside the Runtime.
- Partial matches, or substitution across model revisions.
