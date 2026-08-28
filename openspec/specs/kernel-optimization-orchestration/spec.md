# kernel-optimization-orchestration Specification

## Purpose
This specification defines the boundary between the Kernel Optimization Plane and the Runtime Inference Plane: Optimization Campaign identity and lifecycle, optimization triggers, the external generator boundary, optimization worker capability, composition with existing compilation/qualification/benchmark contracts, campaign budgets and cancellation, evidence bundles, non-authoritative recommendations, artifact transport using stable identity, security/credential/privacy boundaries, and Runtime's exclusive authority over promotion. It keeps Magnetar Runtime a pure inference runtime while allowing arbitrarily advanced external systems — AI agents, CI, vendor tooling, or Tachyon-managed infrastructure — to propose improved Kernel implementations.
## Requirements
### Requirement: Optimization Plane Is Separate From Inference Plane

Magnetar SHALL distinguish Kernel Optimization Plane from Runtime Inference
Plane.

#### Scenario: AI optimizer exists

Given an AI agent generates Kernel source

When Runtime executes inference

Then the agent is not part of the token-generation execution path.

---

### Requirement: Optimization Campaign Is Explicit

Kernel optimization SHALL be represented as an explicit campaign or equivalent
bounded workflow.

#### Scenario: New GPU architecture

Given optimization is requested for new architecture

When work begins

Then it has campaign identity, target, workload profile and budget.

---

### Requirement: Hot Path Cannot Start Campaign

Normal inference decode SHALL NOT synchronously launch an Optimization
Campaign.

#### Scenario: No optimized Kernel available

Given decode has no generated optimized candidate

When Runtime executes

Then it uses fallback/admission/failure policy rather than starting AI search.

---

### Requirement: Workload Profile Avoids Raw User Data By Default

Optimization workload profile SHALL describe execution characteristics without
raw user content by default.

#### Scenario: Sequence histogram

Given production workloads are analyzed

When optimization profile is created

Then sequence-length distribution may be included but raw prompts are absent.

---

### Requirement: Generator Is External Producer

Kernel generator SHALL be treated as an external artifact producer relative to
Runtime.

#### Scenario: KernelEvolve-like generator

Given generator produces Triton candidate

When candidate enters Magnetar

Then it enters as Kernel Artifact rather than executable Runtime authority.

---

### Requirement: Generator Cannot Directly Promote

Kernel generator SHALL NOT directly activate production Kernel.

#### Scenario: Agent reports fastest candidate

Given generated candidate is fastest in campaign

When campaign completes

Then Runtime still performs normal trust, qualification, selection and
promotion validation.

---

### Requirement: Optimization Worker Capability Is Explicit

Optimization worker SHALL expose compatibility/capability metadata.

#### Scenario: sm90 campaign

Given worker has only AMD GPU

When target is NVIDIA sm90

Then worker is not selected as compatible benchmark worker.

---

### Requirement: Candidate Failure Isolated

Failure of one candidate SHALL NOT corrupt other candidates or active
production Kernel.

#### Scenario: Candidate compiler crashes

Given ten candidates exist

When one compilation crashes

Then other candidates may continue according to campaign policy.

---

### Requirement: Campaign Budget

Optimization Campaign SHALL have bounded resources.

#### Scenario: Candidate limit reached

Given campaign maximum is 500 candidates

When 500 candidates have been generated

Then additional generation is denied unless policy expands budget.

---

### Requirement: Campaign Cancellation

Optimization Campaign SHALL support cancellation without affecting active
production Kernel.

#### Scenario: Operator cancels optimization

Given active Kernel v12 is serving inference

When campaign for v13 candidates is cancelled

Then v12 remains active.

---

### Requirement: Evidence Bundle

Campaign SHALL produce or reference evidence required for recommendation where
applicable.

#### Scenario: Candidate recommended

Given candidate is recommended for latency profile

When recommendation is inspected

Then qualification and benchmark evidence are identifiable.

---

### Requirement: Recommendation Is Non-Authoritative

Optimization Recommendation SHALL NOT itself cause production promotion.

#### Scenario: External service recommends candidate

Given recommendation is received

When Runtime processes it

Then current production eligibility is re-evaluated.

---

### Requirement: Artifact Transport Uses Stable Identity

Optimization/Runtime boundary SHALL use artifact identity/digests rather than
native execution handles.

#### Scenario: Compiled artifact transferred

Given CUBIN moves from optimization worker to production deployment

When Runtime receives it

Then its digest/metadata identify it, not worker-local CUfunction pointer.

---

### Requirement: Orchestrator Is Vendor-Neutral

Magnetar SHALL not require one specific orchestrator implementation.

#### Scenario: CI instead of Tachyon

Given organization uses CI for optimization

When campaign executes

Then Magnetar artifact/evidence contracts remain valid.

---

### Requirement: Offline Inference Does Not Depend On Optimizer

Runtime inference SHALL not require connectivity to Kernel optimization
service when compatible required artifacts are locally available.

#### Scenario: Optimization service unavailable

Given production has valid Prepared Kernels

When service is offline

Then inference continues.

---

### Requirement: Optimization Credentials Stay Outside Runtime

Optimization credentials SHALL NOT become ambient Runtime Inference API
authority.

#### Scenario: Agent API key

Given external AI generator requires API key

When Runtime serves inference

Then Runtime session does not possess that key.

---

### Requirement: Production Promotion Is Runtime-Policy Controlled

Production promotion SHALL remain governed by Runtime/deployment policy.

#### Scenario: Candidate submitted for promotion

Given qualification evidence is valid

But candidate violates current Resource Affinity

When promotion is considered

Then promotion is denied.

---

### Requirement: Runtime Revalidates Evidence

Runtime SHALL revalidate current production-relevant state before promotion.

#### Scenario: Qualification revoked after campaign

Given candidate was qualified during optimization

But evidence is now revoked

When promotion is requested

Then candidate is denied.

---

### Requirement: Optimization Observability Is Redacted

Optimization events SHALL not expose sensitive inference or native Runtime
state by default.

#### Scenario: Campaign fails

Given diagnostic is exported

Then secrets, raw prompts, raw model weights and native handles are absent.
