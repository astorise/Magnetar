# Component Runtime Boundary

Magnetar separates Component Runtime semantics from concrete WebAssembly engine
mechanics.

The Component Runtime is Magnetar-owned. It owns Component registration,
contract validation, import authorization, Link Plan construction, definition
and instance identity, lifecycle policy, error normalization, resource-limit
policy, and observability correlation.

The Component Engine is an internal adapter. It prepares executable Component
definitions, creates engine instances, invokes exported interfaces through
contract-specific adapters, handles interruption, extracts traps, and releases
engine state. Engine-native objects such as Stores, Linkers, compiled
Component handles, traps, fuel, epochs, and resource tables remain behind this
boundary.

## Definitions and Instances

A Component definition describes reusable Component code and its WIT contracts.
It has Runtime-owned definition identity and may be prepared once before
creating one or more live instances.

A Component instance has separate Runtime-owned identity. Multiple instances may
originate from the same definition, but they must not implicitly share mutable
engine Store state. Instance identity is assigned by the Runtime, not by
Component code.

## Linking

Component dependencies are WIT imports, not direct Component names.

The Runtime builds an immutable Link Plan before instantiation. The plan maps
each authorized import to a Runtime endpoint such as a Capability endpoint or a
Runtime service endpoint. A Component export does not automatically become a
global import source for other Components, and matching imports and exports are
not automatically composed without explicit Runtime policy.

Linking a Capability does not select a permanent Provider. For example,
`magnetar:compute/run` links the Component to the Runtime Compute endpoint.
Provider and Device resolution still occurs later according to Capability
compatibility, Resolution Policy, Resource Affinity, Provider health, and
execution state.

## Authority

The Component Runtime is fail-closed and scoped to inference. An interface
absent from the authorized Link Plan is unavailable to the Component instance.

Magnetar Component manifests may request only inference-scoped authority:

- model-artifact-read
- tokenizer-artifact-read
- prompt-template-read
- adapter-artifact-read
- quantization-artifact-read
- inference-session-state
- generation-session-state
- kv-cache-access
- prefix-cache-access
- compute-capability
- generation-capability
- sampling-capability
- observability-emit
- runtime-diagnostics

Filesystem, network, environment variables, process execution, secrets,
sockets, Git, workspace access, host command execution, and broad WASI
interfaces are outside Magnetar inference scope. They belong to clients or
orchestrators such as `magnetar-cli`, not to the Magnetar Runtime.

The first Wasmtime adapter does not install broad default WASI. Authorized
inference interfaces will be added only through explicit scoped Link Plan
entries.

The target Component Artifact manifest shape for an inference Component is:

```yaml
schema: magnetar-component-artifact
schema_version: 1

artifact:
  kind: component
  digest:
    algorithm: sha256
    value: "sha256:0123456789abcdef..."

component:
  name: "magnetar.examples.tokenizer"
  version: "0.1.0"
  description: "Tokenizer Component fixture"
  role: "tokenizer"

runtime:
  magnetar:
    min_version: "0.1.0"

wit:
  imports:
    - package: "magnetar:compute"
      interface: "run"
      version: "2.0.0"
  exports:
    - package: "magnetar:tokenizer"
      interface: "tokenize"
      version: "1.0.0"

capabilities:
  requires:
    - id: "magnetar:compute/run"
      version: "2.0.0"

authority:
  requires:
    - tokenizer-artifact-read
    - compute-capability
    - observability-emit

publisher:
  id: "local-dev"
  name: "Local Development"

source:
  kind: "local"
  uri: "./fixtures/tokenizer.component.wasm"

signatures: []
```

This authority block is invalid for Magnetar:

```yaml
authority:
  requires:
    - filesystem
    - network
    - secrets
    - git
    - workspace
```

## Resource Limits and Concurrency

Runtime policy can require memory limits, cap live Component instances, and
cap concurrent invocations per instance. The Wasmtime adapter maps
`max_memory_bytes` to a private `StoreLimits` limiter. If a required memory
limit cannot be represented, preparation fails closed.

The current invocation API is synchronous and receives `&mut self`, so one
manager cannot mutate the same engine Store concurrently. Multiple Component
instances receive distinct engine Store state. Broader async host-call
concurrency and resource-table ownership fixtures remain part of the host
adapter work.

## Host Adapter Scope

The first Wasmtime host adapter translates approved Link Plan entries for
unit-shaped Component imports, such as `() -> ()` test hooks, into private
Wasmtime linker entries. The fixture path covers unit exports, primitive `u32`
returns, host-call round trips, traps, and deadline-triggered interruption.
More complex WIT signatures, resources, and async host operations fail closed
until typed Runtime adapters are added.

## Lifecycle

Generic Components are not required to export universal `start` or `stop`
functions. Runtime lifecycle covers definition registration, validation,
preparation, instance creation, ready state, failure, destruction, and shutdown.
Application-specific initialization or shutdown belongs in explicit WIT
contracts when needed.

Runtime shutdown prevents new Component invocations and destroys live Component
instances according to policy without depending on a portable `stop` export.

## Errors

Component Engine failures are normalized before crossing canonical Magnetar
APIs. Stable Component Runtime errors distinguish unresolved imports,
unauthorized imports, preparation failure, instantiation failure, traps,
interruption, resource-limit failure, invalid lifecycle transitions, and engine
failure.

Component traps are not Provider failures. Provider failures are not engine
failures. Compute cancellation and Component interruption remain separate
domains unless an explicit Runtime mapping is applied.

## Engine Independence

The first production engine may use Wasmtime, but Wasmtime is an implementation
choice rather than an architectural dependency. The public Component Runtime
model must remain valid if a future Component Engine implementation replaces
Wasmtime.

## Wasmtime Feature Policy

The concrete Wasmtime adapter is available behind the
`wasmtime-component-engine` Cargo feature. The feature is disabled by default
so engine-neutral Runtime contracts remain cheap to build and test. Enabling
the feature adds Wasmtime with Component Model and async Component support, but
Wasmtime-native types remain contained in the adapter module and are not part
of canonical Magnetar APIs.

Use this local check for the concrete adapter:

```text
cargo check -p magnetar-runtime --features wasmtime-component-engine
```
