## Context

- Lots 0 to 2 bring an operator kernel from a source to an active generation on CUDA and, in lot 3, on the CPU.
- The other Providers have no compute path to host a kernel yet, and compiled-graph accelerators need a different unit of replacement.
- The Runtime already has an execution graph per phase with a semantic fingerprint (`ExecutionGraphSemanticFingerprint`), prepared plans, Model Instances with a lifecycle, and a Memory Manager that admits allocations.

## Goals / Non-Goals

**Goals:**

- One envelope to describe a capability's variants across Providers and kinds, without changing v1 bundles.
- An exact, verifiable condition under which a compiled graph may replace part or all of an execution.
- A truthful support matrix.

**Non-Goals:**

- Provider compute implementations, Runtime-side graph compilation, fuzzy matching.

## Decisions

### D1. Envelope, not a manifest v2

```json
{
  "schema": "magnetar:accelerator-package@1",
  "name": "optimized-attention",
  "version": "1.0.0",
  "semantic_binding": { "operator": "magnetar:operator/attention", "version": 1 },
  "variants": [
    { "id": "cuda-sm86", "kind": "operator-kernel", "provider": "nvidia-cuda",
      "format": "nvidia:cubin", "requirements": { "architecture": "sm_86" },
      "bundle": "sha256:<manifest-digest>" },
    { "id": "cpu-avx512", "kind": "operator-kernel", "provider": "cpu-native",
      "format": "magnetar:cpu-elf@1", "requirements": { "features": ["avx512f"] },
      "bundle": "sha256:<manifest-digest>" },
    { "id": "npu-graph", "kind": "compiled-graph", "provider": "intel-npu",
      "format": "openvino:compiled-model", "blob": "sha256:<blob-digest>",
      "binding": { "...": "see D2" } }
  ]
}
```

- Canonical JSON, the same duplicate-key and size limits as manifest v1, and a digest over the canonical bytes.
- `bundle` references are resolved through lots 0 to 2. The envelope's own signature (lot 5) authenticates the grouping, not the bundles: each bundle keeps its own envelope and trust decision.
- `requirements` and `provider` are inputs to eligibility, never decisions. The registry filters as usual.

### D2. Compiled graph binding

```text
CompiledGraphBinding {
  granularity: Segment | Instance,
  segment: SegmentSignature {          // Segment only
    operators: [ (operator id, version, attributes) ... ],  // in graph order
    inputs:  [ (shape, dtype, layout) ... ],
    outputs: [ (shape, dtype, layout) ... ],
  },
  model: { artifact_digest } | { weight_digests: [ (tensor name, digest) ... ] },
  toolchain: { sdk, compiler, driver, runtime },   // exact versions
  accelerator: { vendor, generation },
  memory: { device_bytes, host_bytes },            // declared, admitted
}
```

- **Segment matching** looks for a contiguous region of a prepared execution graph whose node sequence, attributes and boundary tensors equal the signature, and whose weights' digests equal the binding. Matching is exact. Dynamic dimensions must be in the declared shape ranges.
- **Instance granularity** replaces every graph of a Model Instance for the bound Model Artifact.
- **Toolchain and accelerator** are compared with what the Provider reports. A mismatch makes the variant ineligible.

### D3. Preparation and swap

- The Provider prepares a compiled graph as a new native instance. The Memory Manager admits its declared memory before preparation and reconciles it with what the Provider reports after.
- The Runtime builds a plan in which the matched segment is one node dispatched to the prepared graph, and everything else is unchanged.
- A swap is a new plan revision with lot 2 leases. The old compiled instance drains and is released at zero leases. A running plan is never patched.

### D4. Gating per Provider

| Provider | Kind | Format | Preparation | Gated on |
| --- | --- | --- | --- | --- |
| CUDA | operator, fused | `nvidia:cubin` | lot 1 | — |
| CPU (`cpu-native`) | operator, fused | `magnetar:cpu-elf@1` | lot 3 | — |
| WGPU | operator | `wgsl` → pipeline | compile the pipeline (cold path) | `integrate-wgpu-production-compute` |
| ROCm | operator | HSACO | `hipModuleLoadData` | `qualify-rocm-compute-profiles` |
| Metal | operator | `metallib` | `newLibraryWithData` | `evaluate-native-metal-prefill` |
| Intel NPU, QNN | compiled graph | vendor compiled model or context | vendor runtime | G04 plus hardware |

WGSL is source: a WGPU "preparation" compiles a pipeline in the cold path, and is recorded as compilation (`provider-kernel-compilation`), not as loading a prebuilt binary.

### D5. The support matrix

A cell is `contract` (types and tests), `fixture` (a simulated or fixture Device), or `hardware-qualified` (a recorded real-hardware run with evidence). A cell never moves on discovery alone.

### D6. Answers to RFC questions

- **Q7, CompiledGraph:** a separate envelope with an exact model and weight binding. Manifest v1 is unchanged.
- **Q8, Intel:** deferred to G04.

## Risks / Trade-offs

- **Exact matching may rarely match** a graph the Runtime fused differently. That is the intended failure mode: a compiled graph that does not match runs nowhere.
- **Compiled graphs that own weights duplicate memory** with the Model Instance's resident weights. Instance granularity can release the instance's own copies; segment granularity cannot. The admission accounts for both.

## Open Questions

- Whether segment signatures should include the Runtime's fusion passes' version, so a compiled graph can target pre- or post-fusion graphs.
