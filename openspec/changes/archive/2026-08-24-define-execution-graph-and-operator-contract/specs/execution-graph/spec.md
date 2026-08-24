## ADDED Requirements
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
