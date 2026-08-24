## ADDED Requirements

### Requirement: Runtime Owns Model Instance Registry

Runtime SHALL own the Model Instance registry, lookup, authorization, lifecycle,
readiness, and cleanup.

#### Scenario: Lookup instance

Given a caller references a ModelInstanceId

When Runtime resolves it

Then Runtime validates identity, authorization, lifecycle, and readiness.

---

### Requirement: Runtime Prevents Model Instance Forgery

Runtime SHALL reject forged ModelInstanceId values and non-authoritative
instance metadata.

#### Scenario: Forged instance

Given a caller claims a Model Instance exists on Device A

When Runtime validates the claim

Then Runtime rejects or ignores the non-authoritative metadata.

---

### Requirement: Runtime Coordinates Instance Lifecycle

Runtime SHALL coordinate Model Loading, Memory Manager, Provider, Device,
Session, Generation, Adapter, KV Cache, Prefix Cache, and Scheduler constraints
for Model Instance lifecycle.

#### Scenario: Instance unload

Given unload is requested

When Runtime performs unload

Then it coordinates dependent subsystems before releasing resources.

---

### Requirement: Runtime Does Not Expose Raw Instance Internals

Runtime SHALL not expose raw model weights, raw Provider handles, raw Device
handles, raw memory pointers, raw KV cache contents, or raw prompts through
Model Instance APIs by default.

#### Scenario: Instance status

Given status is requested

When Runtime returns status

Then it returns redacted stable metadata only.

---

### Requirement: Runtime Observes Model Instance Lifecycle

Runtime SHALL define structured observations for Model Instance lifecycle without
exposing raw weights, prompts, cache contents, or native handles.

#### Scenario: Instance failed

Given an instance transitions to failed

When Runtime emits observability

Then it records stable error and state metadata.
