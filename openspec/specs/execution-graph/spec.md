# execution-graph Specification

## Purpose
TBD - created by archiving change define-execution-graph-and-operator-contract. Update Purpose after archive.
## Requirements
### Requirement: Execution Graph

Magnetar SHALL define Execution Graph as a Runtime-understandable composition
of operators and tensor/resource edges for inference execution.

#### Scenario: Build decode graph

Given a Model Instance is ready

When Runtime prepares decode execution

Then it may use an Execution Graph describing decode operators and edges.

---

### Requirement: Execution Graph Is Not Provider

An Execution Graph SHALL NOT be treated as a Provider or Provider-specific
implementation.

#### Scenario: CUDA execution

Given a graph contains attention and matmul operators

When Runtime plans execution on CUDA Provider

Then the graph remains portable and Provider-specific kernels are selected
later.

---

### Requirement: Execution Graph Is Not Kernel

An Execution Graph SHALL describe computation but SHALL NOT define concrete
kernel implementation.

#### Scenario: Graph attention node

Given a graph contains an attention operator

When execution is planned

Then Runtime selects a compatible kernel later through kernel dispatch.

---

### Requirement: Graph Producer Boundary

An Execution Graph SHALL support production by Runtime-native architecture code,
Model Component, Provider-assisted builder, or test fixture.

Runtime SHALL validate the graph regardless of producer.

#### Scenario: Component-produced graph

Given a Model Component emits a graph

When Runtime receives it

Then Runtime validates it before planning or execution.

---

### Requirement: Component Graph Producer Has No Provider Handles

A Component producing a graph SHALL NOT receive raw Provider handles, Device
handles, memory pointers, or raw tensor storage.

#### Scenario: Component emits graph

Given a Component describes model forward

When Runtime links it

Then the Component uses authorized graph contracts and not Provider handles.

---

### Requirement: Execution Graph Identity

An Execution Graph SHALL have identity and version metadata.

Graph identity MAY include graph phase, producer metadata, model instance
compatibility, adapter compatibility, tokenizer dependency, and graph
fingerprint.

#### Scenario: Adapter changes graph

Given adapter activation changes graph semantics

When Runtime builds the graph

Then graph identity reflects the active adapter set.

---

### Requirement: Graph Phases

Execution Graph phase metadata SHALL be able to represent model-load, warmup, prefill, decode,
adapter-activation, adapter-merge, sampling-helper, and test.

#### Scenario: Prefill graph

Given generation performs prefill

When Runtime prepares execution

Then the graph phase is prefill or equivalent metadata is present.

---

### Requirement: Tensor Edges

Execution Graph edges SHALL describe tensor/resource flow without exposing raw
memory pointers.

#### Scenario: Tensor edge

Given operator A produces tensor T consumed by operator B

When Runtime validates the graph

Then T is represented as a logical tensor edge.

---

### Requirement: Graph Validation

Runtime SHALL validate Execution Graphs before execution.

Validation SHALL include operator identities, attributes, arity, tensor edge
consistency, shape, dtype, layout, Resource Affinity, memory behavior, aliasing,
Provider Capability feasibility, and policy constraints.

#### Scenario: Invalid dtype edge

Given operator A outputs FP16

And operator B requires INT8 without conversion

When Runtime validates the graph

Then validation fails or planning inserts explicit conversion where policy
allows.

---

### Requirement: Graph Planning

Runtime SHALL plan graph execution before Provider submission.

Planning SHALL determine execution order, memory needs, workspace, data
movement, layout conversion, dtype conversion, KV cache use, adapter paths,
Provider/Device compatibility, kernel selection placeholder, batching
compatibility, and failure handling.

#### Scenario: Layout conversion required

Given an operator requires layout X

And input tensor has layout Y

When planning runs

Then Runtime inserts explicit layout conversion or rejects the graph.

---

### Requirement: No Silent Data Movement

Graph planning SHALL NOT silently move data across Resource Affinity boundaries.

#### Scenario: Device-bound tensor

Given tensor T is Device-bound to Device A

When an operator is planned for Device B

Then Runtime performs explicit authorized movement or rejects planning.

---

### Requirement: Graph Execution Boundary

Graph execution SHALL run through Runtime-owned execution paths.

Components SHALL NOT call Providers directly.

#### Scenario: Execute graph

Given graph validation and planning succeed

When Runtime executes the graph

Then Provider interaction occurs through Runtime-managed dispatch.

---

### Requirement: Prefill And Decode Graphs

Execution Graphs SHALL support distinct prefill and decode phase metadata.

#### Scenario: Decode graph

Given a generation operation is in decode phase

When Runtime schedules it

Then decode graph metadata can differ from prefill graph metadata.

---

### Requirement: Adapter-Aware Graph

Adapter changes that affect semantics SHALL be explicit in graph metadata and
identity.

#### Scenario: LoRA active

Given LoRA adapter is active

When Runtime builds model forward graph

Then the graph includes adapter path or fused adapter metadata.

---

### Requirement: KV-Cache-Aware Graph

Graphs that use KV cache SHALL represent KV cache inputs, outputs, append
behavior, and compatibility metadata explicitly.

#### Scenario: Decode with KV cache

Given decode uses existing KV cache

When Runtime validates the graph

Then KV cache input and append behavior are represented.

---

### Requirement: Prefix-Cache-Aware Graph

Prefix Cache reuse SHALL alter graph planning explicitly and SHALL not bypass
validation.

#### Scenario: Prefix hit

Given Prefix Cache provides reused prefix length

When prefill graph is planned

Then Runtime adjusts prefill boundary explicitly.

---

### Requirement: Graph Error Categories

Graph failures SHALL use structured error categories.

#### Scenario: Graph validation failed

Given graph contains invalid operator arity

When validation runs

Then Runtime returns graph-validation-failed with operator details.

---

### Requirement: Graph Observability

Runtime SHALL define observations for graph creation, validation, planning,
execution, operator execution, inserted conversions, data movement, workspace,
and failures.

Observability SHALL not expose raw tensor values, prompts, weights, cache
contents, or native handles by default.

#### Scenario: Graph planned

Given graph planning succeeds

When observability records it

Then Runtime emits redacted planning metadata.

### Requirement: Graph Planning Produces Kernel Requirements

Execution Graph planning SHALL produce Kernel requirements for operator
invocations.

#### Scenario: Plan operator

Given a graph contains matmul

When Runtime plans the graph

Then it produces Kernel requirements derived from Operator and tensor metadata.

---

### Requirement: Graph Execution Does Not Directly Bind Native Kernels

Execution Graphs SHALL not expose raw native Kernel function pointers.

#### Scenario: Graph inspected

Given a graph is inspected by a Component

When metadata is returned

Then raw Kernel pointers are not exposed.

---

### Requirement: Graph Fusion Requires Kernel Semantic Validation

If graph planning considers fused Kernels, Runtime SHALL validate that fusion
preserves graph semantics.

#### Scenario: Fused attention path

Given a fused Kernel replaces multiple graph operators

When Runtime validates the plan

Then Operator semantics are preserved.

### Requirement: Graph Planning Uses Kernel Registry

Execution Graph planning SHALL use Kernel Registry to resolve operator
implementation candidates.

#### Scenario: Plan attention

Given graph contains attention Operator

When Runtime plans execution

Then Kernel Registry provides compatible Kernel Candidates.

---

### Requirement: Graph Plan Contains Dispatch Requirements Not Raw Kernels

Execution Graph plans SHALL contain Runtime-managed Kernel requirements or
Dispatch Plans, not raw native function pointers.

#### Scenario: Inspect graph plan

Given graph plan is inspected

When metadata is returned

Then raw native Kernel function pointers are absent.

---

### Requirement: Graph Execution Uses Runtime Dispatch

Graph execution SHALL execute operators through Runtime Kernel Dispatch.

#### Scenario: Execute planned graph

Given graph planning selects Kernel dispatch plans

When execution runs

Then Runtime dispatches Kernel Invocations through owning Providers.

### Requirement: Model Component May Produce Execution Graph

Execution Graphs SHALL support production by Model Components.

Runtime SHALL validate all Component-produced graphs before planning or
execution.

#### Scenario: Component graph

Given Model Component emits prefill graph

When Runtime receives it

Then graph validation runs before graph planning.

---

### Requirement: Model Component Graphs Must Use Portable Operators

Model Component-produced graphs SHALL use portable Operator identities.

#### Scenario: Provider-specific graph node

Given a Model Component graph contains `cuda.flash_attention`

When Runtime validates the graph

Then validation rejects Provider-specific node identity as portable Operator.

---

### Requirement: Model Component Graphs Do Not Embed Kernel Handles

Model Component-produced graphs SHALL not embed raw Kernel handles, Provider
handles, Device handles, or function pointers.

#### Scenario: Raw kernel pointer

Given a graph includes native function pointer metadata

When Runtime validates it

Then validation fails.

---

### Requirement: Execution Graph May Run On Reference CPU

When it executes through Reference CPU Kernels, a validated Execution Graph SHALL be treated as a normal graph subject to standard validation and dispatch.
A validated Execution Graph MAY execute through Reference CPU Kernels where compatible and policy allows.

#### Scenario: CPU graph execution

Given graph operators are supported by Reference CPU Kernels

When Runtime dispatches the graph

Then graph execution may complete on Reference CPU.

---

### Requirement: CPU Execution Does Not Bypass Graph Planning

Execution through Reference CPU SHALL still use graph validation, planning,
Kernel Registry, Kernel Dispatch, and Memory Manager.

#### Scenario: Direct CPU path

Given a graph is valid

When CPU execution is requested

Then Runtime does not bypass normal planning and dispatch.

### Requirement: Graph Planning Applies First Operator Scope

Execution Graph planning SHALL apply first operator implementation scope when
running the initial executable baseline.

#### Scenario: Unsupported graph node

Given graph contains unsupported MoE dispatch

When first baseline planning runs

Then graph planning fails with operator-explicitly-unsupported.

---

### Requirement: Graph Planning Avoids Hidden Substitutions

Graph planning SHALL not silently replace unsupported operators with unrelated
operators.

#### Scenario: Quantized matmul

Given graph requires quantized-matmul

When no implementation exists

Then planning rejects it instead of silently using f32 matmul.

---

### Requirement: First Decoder Graph Is In Scope

A decoder-only graph using the required-now operator set SHALL be considered
valid for first baseline planning if all metadata is compatible.

#### Scenario: Decoder graph valid

Given graph uses embedding, RMSNorm, matmul, RoPE, attention, softmax, SiLU,
add, mul, residual-add, and logits matmul

When first scope validation runs

Then graph operators pass scope validation.

### Requirement: Graph Edges Use Tensor Descriptors

Execution Graph edges SHALL use Tensor Descriptors before materialization and Tensor Resources after planning where applicable.

#### Scenario: Graph edge

Given graph operator A produces edge E

When graph is validated

Then E has Tensor Descriptor metadata.

---

### Requirement: Graph Planning Materializes Tensor Resources

Graph planning SHALL materialize Tensor Resources through Runtime and Memory Manager where execution requires storage.

#### Scenario: Planned output

Given graph output requires storage

When planning runs

Then Runtime asks Memory Manager to plan Tensor Resource allocation.

---

### Requirement: Graph Planning Makes Tensor Conversion Explicit

Graph planning SHALL make dtype conversion, layout conversion, memory movement, and host staging explicit.

#### Scenario: DType mismatch

Given producer outputs f16

And consumer requires f32

When graph planning runs

Then explicit dtype conversion is inserted or planning fails.

