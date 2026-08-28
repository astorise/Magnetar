## ADDED Requirements

### Requirement: Runtime Coordinates Generated Kernel Qualification Policy

Runtime or authorized tooling SHALL evaluate qualification evidence before
production eligibility.

#### Scenario: Unqualified candidate

Given policy requires qualification

When Runtime plans execution

Then unqualified candidate is excluded.

---

### Requirement: Runtime Separates Correctness From Performance

Runtime SHALL not accept performance evidence as replacement for correctness
qualification.

#### Scenario: Benchmark winner is incorrect

Given fastest candidate failed differential tests

When selection runs

Then candidate remains rejected.

---

### Requirement: Runtime Coordinates Promotion

Runtime SHALL control candidate promotion according to Registry and policy.

#### Scenario: New qualified candidate

Given candidate is qualified and prepared

When policy approves promotion

Then Runtime atomically publishes new active generation.

---

### Requirement: Runtime Preserves In-Flight Generation

Runtime SHALL preserve acquired Prepared Kernel generation for in-flight
execution unless explicit safe migration exists.

#### Scenario: Promotion during generation

Given token operation has acquired old generation

When new generation is promoted

Then current operation continues safely on old generation.

---

### Requirement: Runtime Supports Safe Rollback

Rollback SHALL NOT be treated as available for a generation that is no longer retained or compatible.

Runtime SHOULD support rollback to compatible known-good generation when
available.

#### Scenario: Replacement starts failing

Given new Kernel is active and known-good previous generation exists

When rollback policy triggers

Then new work returns to previous generation.

---

### Requirement: Runtime Applies Revocation

Runtime SHALL prevent new work from using revoked Kernel.

#### Scenario: Qualification revoked

Given active Kernel qualification is revoked

When new invocation begins

Then Runtime does not dispatch it.

---

### Requirement: Runtime Does Not Depend On Kernel Generator

Runtime SHALL consume generic qualification and artifact metadata regardless of
generator.

#### Scenario: Human kernel and AI kernel

Given both satisfy identical qualification policy

When selected

Then origin alone does not alter execution semantics.