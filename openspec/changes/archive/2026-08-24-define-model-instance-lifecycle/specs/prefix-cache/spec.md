## ADDED Requirements

### Requirement: Prefix Cache Binds To Model Instance Compatibility

Prefix Cache entries SHALL include Model Instance identity or compatible model
context metadata where required for safe reuse.

#### Scenario: Instance mismatch

Given prefix entry was created for Model Instance A

When request uses incompatible Model Instance B

Then Runtime rejects reuse.

---

### Requirement: Model Instance Changes Invalidate Prefix Cache

Runtime SHALL invalidate dependent Prefix Cache entries according to policy on
Model Instance unload, invalidation, adapter mutation, or incompatible reload.

#### Scenario: Instance reload

Given a prefix entry depends on old instance state

When instance reload changes compute dtype or adapter state

Then Runtime invalidates or rejects reuse.
