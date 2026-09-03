# model-component-graph-contract Specification

## Purpose
TBD - created by archiving change reach-architecture-freeze-1. Update Purpose after archive.
## Requirements
### Requirement: Model Component Graph Contract

A Model Component SHALL export prefill and decode graph semantics to the
Runtime through the `model-component-graph-contract` Capability instead of
the Runtime synthesizing model-family-specific graphs internally.

This Capability is the concrete mechanism that satisfies `model-component`'s
existing `Graph Production` requirement; it does not replace or relax that
requirement.

#### Scenario: Component builds a graph through the contract

Given a Model Component that imports the `model-component-graph-contract`
Capability

When the Runtime requests a prefill or decode graph for that Component

Then the Runtime obtains an `ExecutionGraph` produced through calls the
Component made against this Capability, not through Runtime-internal,
model-family-specific graph construction code.

---

### Requirement: Graph Contract Is a Runtime-Owned Builder, Not a Serialized Descriptor

The Runtime SHALL expose graph construction as a Capability whose operations
the Component invokes to incrementally describe the graph (a builder), and
the Runtime SHALL own the resulting `ExecutionGraph` value and its
validation types.

The contract SHALL NOT require the Component to serialize a complete graph
descriptor (for example as an opaque JSON or CBOR blob) for the Runtime to
parse.

#### Scenario: Component describes a graph incrementally

Given a Component producing a decode graph

When the Component calls the graph-builder Capability's operations to add
nodes, edges, and outputs

Then the Runtime constructs the `ExecutionGraph` directly from those calls

And no intermediate serialized graph document exists that the Runtime must
independently parse.

---

### Requirement: Graph Builder Covers Prefill and Decode Phases

The graph-builder Capability SHALL support producing both prefill-phase and
decode-phase graphs, matching `execution-graph`'s `Prefill And Decode
Graphs` requirement.

#### Scenario: Decode-only request rejected without prefill support

Given a Component that only implements prefill-graph production

When the Runtime requests a decode graph

Then the Runtime treats decode-graph absence as a structured capability gap,
not as an implicit fallback to a Runtime-synthesized decode graph.

---

### Requirement: Graph Builder Node and Operator Identity

Every node the Component adds through the graph-builder Capability SHALL
carry a stable node identity and reference a portable Operator identity and
version, consistent with `execution-graph`'s `Execution Graph Identity` and
`model-component`'s `Operator Requirements`.

The Component SHALL NOT reference Provider-specific Kernel names as
authoritative through this Capability.

#### Scenario: Component references a Provider-specific kernel name

Given a Component attempts to add a node whose declared requirement is a
Provider-specific Kernel identifier (for example `cuda.flash_attention_v2`)

When the Runtime validates the node

Then the Runtime rejects the node or treats the Provider-specific identifier
as non-authoritative, per `model-component`'s existing `Operator
Requirements` scenario for invalid kernel requirements.

---

### Requirement: Graph Builder Tensor Descriptors and Weight References

The graph-builder Capability SHALL require every input, output, and
weight/constant reference the Component supplies to use portable `Tensor
Descriptor` values and Runtime-recognized weight/constant reference
identities; the Component SHALL NOT supply raw native buffers or
process-local pointers.

#### Scenario: Component references a weight not present in the loaded artifact

Given a Component adds a node whose weight reference does not resolve
against the `Model Artifact` bytes validated by `Model Loading`

When the Runtime validates the produced graph

Then the Runtime rejects the graph before planning, and no Kernel is
dispatched for it.

---

### Requirement: Graph Builder KV Logical Resources

Where a node participates in incremental decode, the Component SHALL
describe its KV involvement using Runtime-owned KV logical resource
identities exposed by the graph-builder Capability, consistent with
`kv-cache`'s `Execution Graphs Represent KV Cache Use` requirement.

#### Scenario: Component attempts to name a private KV layout

Given a Component attempts to describe KV usage using a Component-invented
naming convention instead of the Runtime-issued KV logical resource identity

When the Runtime validates the graph

Then the Runtime rejects the KV reference as unrecognized.

---

### Requirement: Graph Builder Does Not Grant Provider or Device Authority

The graph-builder Capability SHALL NOT allow a Component to select or pin a
Provider or Device for any node it describes, consistent with
`model-component`'s `Provider Boundary` requirement and this repository's
architectural invariant that Components request Capabilities, not Providers
or Devices.

#### Scenario: Component requests a specific Provider

Given a Component attempts to request execution on a named Provider through
the graph-builder Capability

When the Runtime processes the request

Then the Runtime rejects the request; Provider and Device selection remain
exclusively Runtime-owned through Resolution Policy and Resource Affinity.

---

### Requirement: Component-Produced Graphs Remain Untrusted Until Validated

A graph produced through the graph-builder Capability SHALL be treated as
untrusted Component output until the Runtime validates it, consistent with
`model-component`'s `Component-Produced Graphs Are Untrusted Until
Validated` requirement.

#### Scenario: Structurally valid but semantically wrong graph

Given two Components produce graphs with the same node count but different
Operator sequences for the same requested phase

When the Runtime validates each graph

Then the Runtime treats them as distinct graphs with distinct fingerprints,
and neither is accepted as equivalent to a Runtime-expected reference graph
by node count alone.

#### Scenario: Same operators, different attributes

Given a Component produces a graph using the expected Operators but with
different Operator attributes than a prior graph from the same Component

When the Runtime validates and executes the graph

Then the resulting `ExecutionGraph` reflects the Component-supplied
attributes exactly

And no Runtime-internal fallback substitutes different attribute values.

---

### Requirement: Graph Contract Version

The graph-builder Capability SHALL be versioned independently of any single
Component, consistent with `capability`'s `Capability Versioning`
requirement.

#### Scenario: Component requires unsupported contract version

Given a Component requires a graph-builder Capability version the Runtime
does not implement

When the Runtime attempts to link the Component

Then the Runtime fails linking with a structured
capability-version-mismatch error before any graph production is attempted.

---

### Requirement: Strict First-Native Requires Contract-Produced Graphs

Under the strict first-native profile, the Runtime SHALL require every
executed graph to originate from a Component's use of the
`model-component-graph-contract` Capability.

The Runtime SHALL NOT substitute a Runtime-internal, model-family-specific
graph builder in the strict profile, consistent with
`first-native-execution-profile`'s `Simplification Is Allowed But Bypass Is
Not` requirement.

#### Scenario: Component or Engine unavailable under strict profile

Given strict first-native execution is requested and no Component Engine
capable of producing a graph through this Capability is available

When the Runtime attempts to build the execution graph

Then the Runtime fails with a structured `component-engine-unavailable` (or
equivalent) error

And no Rust-synthesized graph is substituted for the missing Component
output.

