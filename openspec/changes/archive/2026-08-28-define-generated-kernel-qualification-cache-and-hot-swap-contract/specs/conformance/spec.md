## ADDED Requirements

### Requirement: Compilation Does Not Imply Qualification

Conformance SHALL prove successful compilation alone does not make candidate
eligible when qualification is required.

#### Scenario: Compiled-only candidate

Given artifact compiles but has no qualification

When production selection runs

Then candidate is rejected.

---

### Requirement: Qualification Does Not Imply Trust

Conformance SHALL prove qualified but untrusted Kernel is rejected where
production policy requires trust.

#### Scenario: Unknown source

Given candidate passes correctness tests but trust fails

When production policy requires both

Then candidate is ineligible.

---

### Requirement: Differential Mismatch Rejects Kernel

Conformance SHALL validate incorrect generated Kernel fails qualification.

#### Scenario: Broken MatMul

Given generated MatMul changes one result

When differential suite runs

Then candidate is rejected.

---

### Requirement: Explicit Tolerance Conformance

Conformance SHALL validate tolerance profile is explicit and enforced.

#### Scenario: Error outside tolerance

Given candidate exceeds declared tolerance

When compared

Then qualification fails.

---

### Requirement: Shape Envelope Conformance

Qualification SHALL not silently exceed tested compatibility envelope.

#### Scenario: Untested sequence length

Given qualification covers <=4096

When execution requests 8192

Then candidate is not considered qualified by that evidence.

---

### Requirement: Determinism Claim Conformance

Conformance SHALL reject Kernel that falsely advertises deterministic behavior.

#### Scenario: Repeated output differs

Given deterministic flag is true

When repeated runs differ unexpectedly

Then qualification fails.

---

### Requirement: Performance Cannot Override Correctness

Conformance SHALL validate faster incorrect Kernel never wins selection.

#### Scenario: Fastest candidate wrong

Given candidate A is incorrect but fastest and B is correct

When selection runs

Then B remains preferred/eligible.

---

### Requirement: Cache Hit Does Not Grant Eligibility

Conformance SHALL validate cached artifact is re-evaluated according to current
trust, qualification and compatibility policy.

#### Scenario: Revoked cache hit

Given revoked artifact is cached

When resolved

Then it is rejected.

---

### Requirement: Cache Corruption Fails Closed

Conformance SHALL validate corrupt cache entry is never prepared.

#### Scenario: Digest mismatch

Given cached bytes are modified

When read

Then integrity error occurs.

---

### Requirement: Atomic Promotion Conformance

Conformance SHALL validate dispatch never observes partially promoted Registry
state.

#### Scenario: Concurrent dispatch

Given promotion races with request

When Kernel resolves

Then request uses complete old or complete new generation.

---

### Requirement: In-Flight Generation Safety

Conformance SHALL validate old Prepared Kernel remains valid for in-flight work
after new generation promotion.

#### Scenario: Promotion during invocation

Given old generation is executing

When new one is promoted

Then old invocation completes safely.

---

### Requirement: Safe Retirement Conformance

Conformance SHALL validate retiring Kernel is destroyed only after quiescence.

#### Scenario: Active references

Given retiring Kernel has reference count greater than zero

When cleanup runs

Then Provider destruction does not occur.

---

### Requirement: Rollback Conformance

Conformance SHALL validate rollback can restore known-good eligible generation.

#### Scenario: New candidate fails after promotion

Given previous generation remains available

When rollback occurs

Then new dispatches use previous generation.

---

### Requirement: Revocation Conformance

Conformance SHALL validate revoked Kernel receives no new work.

#### Scenario: Active Kernel revoked

Given Kernel is revoked

When next dispatch occurs

Then another eligible Kernel is selected or structured failure is returned.

---

### Requirement: Provider Lifetime Independence Conformance

Conformance SHALL validate Kernel hot swap does not unload Provider.

#### Scenario: CUDA kernel replacement

Given new PreparedKernel generation is installed

When swap completes

Then CUDA Provider instance remains active.

---

### Requirement: Candidate Failure Atomicity

Conformance SHALL validate failure of candidate qualification, benchmark,
preparation or promotion leaves current active Kernel intact.

#### Scenario: Candidate preparation crashes

Given v1 active and v2 preparation fails

When failure completes

Then v1 remains active.