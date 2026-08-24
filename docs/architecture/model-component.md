# Model Component Contract

Model Component is Magnetar's portable model architecture layer.

It sits between Model Artifact data and Runtime-validated Execution Graphs. A
Model Component understands architecture metadata such as Qwen, Llama, Gemma,
Mistral, or test fixture families, and describes the portable work the Runtime
can validate, plan, and dispatch through Operator, Kernel Registry, and Provider
contracts.

## Boundary

A Model Component is not a Provider. Providers advertise and execute portable
Operator or Kernel capabilities for hardware targets such as CPU, CUDA, Metal,
OpenVINO, or QNN. Model Components do not select Providers, receive Provider
handles, or call Provider APIs.

A Model Component is not a Kernel. Kernels are selected later by Runtime Kernel
Registry and Dispatch after graph validation and planning. Model Components may
declare portable Operator requirements, but Provider-specific Kernel names are
not authoritative requirements.

A Model Component is not a Model Artifact. Model Artifacts are data: weights,
config, tokenizer files, generation defaults, quantization metadata, and
related manifests. A Model Component is executable or declarative architecture
logic with separate identity, trust, versioning, and compatibility metadata.

A Model Component is not the Generation, Sampling, Scheduler, agent, tool, or
workspace runtime. Request lifecycle, streaming, stop conditions, sampling
decisions, sessions, batching, and workspace/tool authority remain Runtime-owned
or contract-owned outside the Model Component.

## Identity And Versioning

Model Component identity includes:

- stable `ModelComponentId`
- Model Component contract version
- implementation kind
- supported architecture families and revisions
- supported Model Artifact schema versions
- supported Runtime Capability versions
- supported Operator catalog version
- supported Execution Graph contract version
- trust status
- provenance metadata
- signature state

Breaking compatibility requires explicit version negotiation or rejection. A
component with an unsupported major contract version, graph contract version, or
Operator catalog major version is rejected before use.

## Architecture Compatibility

A Model Component validates architecture metadata before Runtime uses it for
loading or graph production. Compatibility includes architecture family, model
type, hidden size, layer count, attention head count, KV head count, head
dimension, intermediate size, vocabulary size, context length, position
encoding, normalization, activation, attention variant, quantization metadata,
tokenizer compatibility, and adapter target modules.

Metadata validation is structural and portable. It rejects invalid shapes, zero
dimensions, unsupported architecture families, and incompatible quantization or
tokenizer expectations without using Provider or Device selectors.

## Target Modules

When adapters are supported, a Model Component exposes target module metadata.
The canonical roles are:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`
- `lm_head`
- `embedding`
- `norm`
- `attention`
- `mlp`

Adapter Loading uses this metadata for compatibility checks. Adapter activation
and lifecycle remain Runtime-owned.

## Graph Production

A Model Component may produce Execution Graphs for phases such as model-load,
warmup, prefill, decode, adapter activation, adapter merge, sampling helper, and
test.

Component-produced graphs are untrusted until Runtime validates them. Runtime
validates graph schema, version, phase, Operator identities, Operator
attributes, tensor edges, shape rules, dtype rules, layout rules, Resource
Affinity, memory behavior, adapter metadata, KV cache metadata, and policy
constraints before planning or execution.

## Operator And Capability Requirements

Model Components declare portable Operator requirements: Operator IDs,
families, versions, alternatives, and shape, dtype, and layout constraints.
Requirements reference the portable Operator catalog, not Provider-specific
Kernel symbols.

Model Components also declare inference-scoped Runtime Capability requirements,
including model metadata validation, graph production, operator catalog read,
tensor descriptor creation, KV cache metadata, adapter metadata, tokenizer
metadata, generation defaults validation, diagnostics, and observability emit.

## Authority Model

Allowed authority is inference-scoped:

- `model-artifact-read`
- `tokenizer-artifact-read`
- `prompt-template-read`
- `adapter-artifact-read`
- `quantization-artifact-read`
- `inference-session-state`
- `generation-session-state`
- `kv-cache-access`
- `prefix-cache-access`
- `compute-capability`
- `generation-capability`
- `sampling-capability`
- `observability-emit`
- `runtime-diagnostics`
- `graph-production`
- `operator-catalog-read`

Forbidden authority includes filesystem, network, environment, process, shell,
secrets, workspace, Git, source-control, tool execution, and external services.
Trusted status does not override forbidden authority.

## Runtime Relationships

Model Loading may use Model Component metadata for architecture validation,
config validation, target module declaration, graph metadata preparation, and
warmup graph construction. It must not let a Model Component bypass artifact
trust, memory admission, or Provider resolution.

Model Instance metadata may reference the Model Component identity and version
used to create it. Instance lifecycle and readiness remain Runtime-owned.

Generation may request prefill and decode graphs where available. Generation
still owns request lifecycle, stop conditions, streaming delivery, and Sampling
boundary.

Adapter Loading uses target modules and adapter compatibility metadata. Adapter
activation, deactivation, merge policy, and cache invalidation remain
Runtime-owned.

KV Cache metadata may include layer count, head count, KV head count, head
dimension, cache dtype, layout preferences, paged support, append semantics, and
position behavior. KV cache allocation and lifecycle remain Runtime-owned.

Prefix Cache may use architecture metadata in fingerprints and compatibility
checks, but lookup, sharing, invalidation, eviction, privacy, and policy remain
Runtime-owned.

Tokenizer compatibility metadata may include vocabulary size, special tokens,
tokenizer family, chat template compatibility, and added-token behavior.
Tokenizer encode/decode remains owned by the Tokenizer Contract.

Quantization compatibility metadata may include quantization method, tensor
grouping, scale metadata, zero-point metadata, packed layout expectations,
dequantization Operator requirements, and quantized Operator requirements.
Provider and Kernel support are resolved later by Runtime.

## Browser Compatibility

The contract is platform-neutral. Browser-compatible paths may be implemented
through WebAssembly Components, Runtime-native browser implementations,
JavaScript-mediated host bindings, or test fixtures. Browser targets do not
require Wasmtime or native Provider loading. Unsupported native-only features
return structured browser-feature-unsupported errors.

## Observability

Runtime observations for Model Component lifecycle and graph production are
redacted. They may include registration, validation, rejection, architecture
compatibility checks, config validation failures, graph production, exposed
target modules, adapter metadata, KV cache metadata, Operator requirements,
authority denial, Component-to-Provider access denial, and conformance results.

Observability does not expose raw prompts, raw weights, raw adapter tensors, raw
KV cache contents, raw Provider handles, raw Device handles, raw Kernel handles,
or memory pointers by default.

## Non-Goals

This contract does not implement concrete Qwen, Llama, Gemma, or Mistral math.
It does not define Provider Kernel ABI, native kernels, graph optimization,
training, fine-tuning, model download, tool calling, agent orchestration,
workspace authority, filesystem/network authority, or direct raw handle access.
