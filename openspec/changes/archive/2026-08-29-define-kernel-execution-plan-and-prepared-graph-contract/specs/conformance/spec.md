## ADDED Requirements
### Requirement: Graph Semantic Authority Conformance

Conformance SHALL prove Prepared Execution Plan cannot redefine Execution Graph
semantics.

#### Scenario: Kernel binding substitution

Given Plan attempts to bind Kernel for different Operator semantics

When validated

Then Plan preparation fails.

---

### Requirement: Exact Binding Conformance

Conformance SHALL prove ready Plan uses the exact Kernel/specialization it
validated.

#### Scenario: Registry preference changes

Given Plan references Kernel A

When Registry later prefers B

Then executing existing Plan still uses A until safe replacement.

---

### Requirement: Native Handle Isolation Conformance

Conformance SHALL prove Plan contains no native executable pointer semantics.

#### Scenario: Provider returns opaque IDs

Given native CUDA/Metal state exists

When Plan/debug representation is inspected

Then native addresses are absent.

---

### Requirement: Session Resource Isolation Conformance

Conformance SHALL prove reusable Plan does not capture one Session's Tensor
Resources as global state.

#### Scenario: Two Sessions

Given both share Plan

When they execute concurrently

Then KV resources remain distinct.

---

### Requirement: Dynamic Guard Conformance

Conformance SHALL reject invocation outside Plan shape/workload envelope.

#### Scenario: Batch too large

Given Plan supports batch <=8

When batch=16

Then Kernel dispatch does not occur through that Plan.

---

### Requirement: Stale Versus Invalid Conformance

Conformance SHALL prove stale Plan may remain policy-safe while invalid Plan
cannot receive new work.

#### Scenario: Performance drift

Given Kernel remains qualified/trusted

When performance evidence becomes stale

Then Plan may continue temporarily.

#### Scenario: Kernel revoked

Given same Kernel is revoked

When invocation begins

Then Plan cannot execute.

---

### Requirement: Kernel Promotion Plan Isolation Conformance

Conformance SHALL prove Kernel hot swap does not mutate in-flight Plan bindings.

#### Scenario: Kernel generation changes

Given old Plan invocation active

When new Kernel promoted

Then new Plan is prepared and old invocation remains coherent.

---

### Requirement: Prepared Plan Memory Authority Conformance

Conformance SHALL prove Plan cannot override Memory Manager.

#### Scenario: Planned workspace unavailable

Given Plan requests workspace

When Memory Manager denies it

Then Plan does not force allocation.

---

### Requirement: No Hot-Path Full Planning Conformance

Conformance SHALL prove compatible ready Plan execution does not perform full
Registry/selection/autotuning/compilation pipeline.

#### Scenario: Repeated decode

Given Plan guards pass

When decode runs

Then bounded Plan execution path is used.

---

### Requirement: Atomic Plan Replacement Conformance

Conformance SHALL prove concurrent dispatch sees complete Plan generation.

#### Scenario: Replacement race

Given new Plan is published during request arrival

When each request binds Plan

Then each sees complete old or complete new generation.

---

### Requirement: In-Flight Lifetime Conformance

Conformance SHALL prove retiring Plan resources remain available until
quiescence.

#### Scenario: Old Provider segment in use

Given reference exists

When Plan retired

Then Provider destroys segment only after use completes.

---

### Requirement: Plan Cache Eligibility Conformance

Conformance SHALL prove cached Plan cannot bypass current revocation/trust/
qualification/readiness.

#### Scenario: Persisted Plan references revoked Kernel

Given Runtime restarts

When cache loads

Then Plan is not marked ready.

---

### Requirement: Provider Prepared Segment Optionality Conformance

Conformance SHALL prove Provider without graph-capture capability can execute a
Prepared Plan.

#### Scenario: Reference CPU

Given no native prepared-segment support

When Plan executes

Then individual prepared Kernel bindings are dispatched.

---

### Requirement: Adaptive Replan Conformance

Conformance SHALL prove adaptive feedback requests a new Plan rather than
mutating current Plan binding.

#### Scenario: Performance model prefers candidate B

Given current Plan uses A

When adaptation occurs

Then replacement generation is constructed.

---

### Requirement: Plan Observability Redaction Conformance

Conformance SHALL prove Plan diagnostics do not expose native handles, tensor
addresses, model weights, KV data, prompts or secrets.

#### Scenario: Plan invalidation report

Given detailed state exists

When report is exported

Then only safe logical identifiers are present.
