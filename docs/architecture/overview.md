# Magnetar Architecture Overview

This document is the canonical conceptual entry point for Magnetar architecture.
Specialized architecture documents may add detail, but they must not contradict
the relationships and ownership boundaries described here without a dedicated
architecture change.

## Canonical Execution Model

```text
Component
    |
    | imports Capability
    v
Runtime
    |
    | Resolution Policy plus Resource Affinity
    v
Provider
    |
    v
Device
```

Components request Capabilities. They do not select CUDA, Metal, CPU, ROCm,
OpenVINO, QNN, Providers, or Devices directly. The Runtime resolves compatible
execution targets according to mandatory compatibility constraints, Resource
Affinity, Provider and Device health, and the active Resolution Policy.

## Responsibility Matrix

| Concept | Responsibility | Stable now |
| --- | --- | --- |
| Runtime | Local orchestration, Capability resolution, Provider management, Device discovery coordination, scheduling, planning, observability, and recovery policy | Partially |
| Component | Portable WebAssembly Component Model code using WIT contracts | Planned |
| Capability | Portable WIT contract imported or exported by Components | Partially |
| Provider | Trusted native implementation of one or more Capabilities | Partially |
| Device | Provider-owned physical or logical execution target | Partially |
| Resource Affinity | Authoritative binding metadata for live resources, artifacts, execution contexts, Providers, Devices, models, adapters, caches, and future resource identities | Partially |
| Resolution Policy | Candidate selection after compatibility and affinity constraints | Partially |
| Model Artifact | Weights and associated model data | Planned |
| Component Artifact | Executable WASM Component code | Planned |

## Terminology

| Term | Canonical meaning |
| --- | --- |
| Runtime | The local-node authority for Magnetar execution. |
| Component | Portable WASM application or Runtime extension code. |
| Capability | A portable WIT contract that describes an ability. |
| Provider | A native trusted Runtime extension that implements Capabilities. |
| Device | A Provider-owned physical or logical execution target. |
| Resource Affinity | Runtime metadata constraining where a live resource may be used. |
| Resolution Policy | Runtime policy for choosing among compatible candidates. |
| Artifact | Versioned content consumed by Magnetar, such as Component code or model data. |
| Model | Future Runtime-managed AI model identity and execution state. |
| Agent | Client-owned behavior that may call Magnetar for inference and execute tools externally. |
| Tool | Client-owned operation outside Magnetar inference Runtime authority. |

Deprecated or non-canonical primary terms:

| Term | Replacement |
| --- | --- |
| Backend | Provider when describing native implementation; Device when describing execution target. |
| Plugin | Provider for trusted native code; Component for portable WASM code. |
| Host | Runtime, Component, Provider, or environment depending on the exact role. |

## Component Boundary

Components are portable and sandboxable. They consume portable WIT contracts and
may expose inference-related behavior such as model architecture logic,
tokenization, prompt formatting, sampling, logits processing, generation
helpers, inference diagnostics, or observability emission.

Components must not receive:

- native Runtime handles
- Provider-native handles
- Device-native handles
- raw pointers or GPU pointers
- queues, streams, kernel objects, allocator objects, or backend storage
- Rust trait objects or process-local object references

Component-to-Runtime calls should be coarse-grained: graph, batch, model,
session, or equivalent execution units. Magnetar must not require one WIT
transition for every eager tensor primitive.

## Provider Boundary

Providers are native trusted Runtime extensions. They implement Capabilities,
expose Devices, and own native details such as kernels, allocators, queues,
streams, native contexts, driver APIs, and Device resources.

Provider internals remain invisible to Components. Providers do not perform
global Provider resolution; the Runtime does.

Examples:

| Integration | Classification |
| --- | --- |
| CUDA compute implementation | Provider |
| CPU compute implementation | Provider |
| Metal compute implementation | Provider |
| OpenVINO or QNN implementation | Provider |
| OpenTelemetry exporter written as WASM | Component |
| Prometheus or Jaeger portable integration | Component |
| Llama or Qwen model architecture logic | Model, Component, or Runtime module; not a Provider solely because it is model-specific |

## AI Runtime Scope

Magnetar is intended to become a standalone AI inference Runtime. Future
Magnetar responsibilities include model loading, model residency,
tokenization, prompt templates, generation, streaming, continuous batching, KV
cache, prefix cache, adapters, LoRA, quantization, multi-device execution, and
service/API usage.

Magnetar does not own general-purpose workspace, filesystem, Git, network,
secret, process, shell, or agent tool authority. A client such as
`magnetar-cli` may read files, inspect Git, call network APIs, manage secrets,
execute tools, orchestrate an agent workflow, and call Magnetar for inference.
Magnetar receives prompt/context input, loads authorized models, tokenizes,
generates, streams tokens, manages inference cache state, executes compute, and
returns inference results.

These are roadmap responsibilities until their dedicated changes are completed.
Current documentation must not describe future model or generation WIT
contracts as stable.

## Magnetar and Tachyon

Magnetar owns local AI execution.

Tachyon owns distributed service orchestration: cluster membership, inter-node
discovery, routing, deployment, GitOps, node selection, and cluster-level
availability.

Tachyon may distribute Magnetar-compatible WASM Components and model artifacts.
Magnetar remains responsible for Component validation, compatibility validation,
Capability linking, authority enforcement, sandbox execution, and local runtime
semantics.

Magnetar must not depend on Tachyon for standalone operation.

```text
Tachyon
   |
   v
Magnetar
```

The reverse architectural dependency is not allowed.

## Scheduling Boundary

Magnetar owns local Runtime scheduling and future intra-node inference
scheduling, including continuous batching when implemented. Tachyon may route a
request to a node, but Magnetar owns execution after the request reaches that
node.

This prevents duplicate model-specific intra-node inference scheduling between
Magnetar and Tachyon after migration.

## Artifact Boundary

A Component Artifact is executable WASM Component code.

A Model Artifact is model data such as weights, configuration, tokenizer data,
and model metadata.

A future resident model may combine a model architecture implementation, Model
Artifact, optional Component Artifact, Provider, Device or Device group,
execution resources, and Resource Affinity. Those identities must remain
distinguishable.

## Consumption Modes

Future embedded, CLI, service, and Tachyon-integrated Magnetar usage must share
the same Runtime semantics. A frontend must not redefine Provider selection,
Resource Affinity, scheduling, or model execution rules.

`magnetar-cli` is planned as a first-party client of Runtime services. It must
not contain a separate inference engine.

## Stability

Canonical now:

- Runtime
- Component
- Capability
- Provider
- Device
- Resource Affinity
- Resolution Policy
- Magnetar/Tachyon responsibility boundary

Future work:

- concrete model, generation, agent, and tool WIT contracts
- concrete Component distribution protocol
- concrete Provider ABI stabilization
- concrete Magnetar service transport

## Related Documents

- [Capability taxonomy](capability-taxonomy.md)
- [Component runtime boundary](component-runtime.md)
- [Resource affinity](resource-affinity.md)
- [Resolution policy](resolution-policy.md)
- [Provider health](provider-health.md)
- [Provider status](provider-status.md)
- [Provider compute advertisement](provider-compute-advertisement.md)
- [Compute operation catalog](compute-operation-catalog.md)
- [Compute operation schemas](compute-operation-schemas.md)
- [Compute graph submission](compute-graph-submission.md)
- [Compute execution planning](compute-execution-planning.md)
- [Compute data movement](compute-data-movement.md)
- [Memory planning](memory-planning.md)
- [Scheduler](scheduler.md)
- [Runtime observability](runtime-observability.md)
- [Observability exporter components](observability-exporter-components.md)
