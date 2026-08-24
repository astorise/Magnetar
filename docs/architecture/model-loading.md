# Model Loading Contract

Model Loading is the Runtime-owned transition from a validated Model Artifact to
a loaded model context. A Model Artifact remains data: manifest, weights,
configuration, tokenizer associations, quantization metadata, shards, trust, and
provenance. Loading performs the policy-controlled work that makes that data
resident and usable for inference.

Loading is separate from:

- Model Artifact validation, which proves bytes and metadata are acceptable.
- Model Residency, which records where materialized model data lives.
- future Model Instance lifecycle, which may expose a richer public object.
- Provider execution, which consumes loaded state but does not own policy.
- Session creation, which may reference or request loading by policy.
- KV cache creation, which occurs during generation prefill/decode.

## Preconditions

Loading checks manifest schema, artifact digest, required parts, shards,
architecture metadata, tokenizer references, dtype metadata, quantization
metadata, trust, revocation, and license policy before memory allocation. Failed
preconditions return structured `ModelLoadingError` values and do not allocate
model residency.

## Lifecycle

Loaded contexts use explicit states:

```text
requested -> validating -> planning -> allocating -> materializing -> ready
ready -> active -> draining -> unloading -> unloaded
validating/planning/allocating/materializing -> failed
failed/invalid -> unloading
```

Invalid transitions are rejected.

## Architecture, Provider, And Device

Architecture implementation is resolved separately from Provider identity.
Implementations may be Runtime-native, Component-based, Provider-assisted, or a
test fixture. Provider and Device bindings can appear in a residency plan only
after Runtime policy, Resource Affinity, Resolution Policy, and capability
compatibility have been evaluated. A Model Artifact does not directly select a
Provider or Device.

## Memory And Residency

The Memory Manager owns model loading feasibility, allocation, pressure, pending
allocation, and residency accounting. Residency plans include artifact identity,
architecture, storage and compute dtype, quantization handling, shard placement,
memory placement, temporary workspace, expected resident bytes, fallback options,
unload policy, and diagnostics. Plans expose stable metadata only, not raw
native handles or loaded weight pointers.

Residency locations include host memory, pinned host memory, device memory,
unified/shared memory, Provider-owned opaque memory, browser linear memory,
future WebGPU buffers, sharded residency, mixed residency, and pending residency.

## DType, Quantization, And Shards

Loading distinguishes storage dtype from compute dtype. Conversions must be
explicit in the residency plan and must account for temporary workspace.
Quantized artifacts require explicit policy: direct quantized execution,
load-time dequantization, lazy dequantization, Provider-specific transform, or
rejection. Sharded artifacts can be loaded sequentially, in parallel, as
single-Device placement, future multi-Device split, host-lazy placement, or
rejected.

## Lazy, Partial, Unload, And Reload

Lazy loading and partial loading are explicit policies. Partial loading cannot
produce a ready context when required parts are missing for the target usage.
Unload drains or rejects active use according to policy, releases Memory Manager
and Provider-owned resources, and may invalidate associated KV caches. Reload is
a new validated loading process unless policy explicitly allows context
mutation.

## Browser Compatibility

The contract is platform-neutral and does not require Wasmtime. Browser targets
can use browser linear memory and future WebGPU-buffer residency metadata, while
native-only pinned memory or native Provider loading returns structured
unsupported-feature errors.

## Non-Goals

This contract does not define sampling, logits processing, continuous batching,
remote model download, full adapter activation, full Model Instance public
lifecycle, KV cache layout, out-of-process Provider memory, or any requirement
for GPU hardware.
