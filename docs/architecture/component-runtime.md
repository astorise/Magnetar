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

The Component Runtime is fail-closed. An interface absent from the authorized
Link Plan is unavailable to the Component instance.

Filesystem, network, environment variables, process execution, secrets,
sockets, host command execution, and WASI interfaces are not ambient. Each
interface must be explicitly authorized and linked by Runtime policy.

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
