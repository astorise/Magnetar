## ADDED Requirements
### Requirement: Memory Manager Allocates Kernel Workspace

Kernel workspace SHALL be allocated through Memory Manager.

#### Scenario: Kernel workspace

Given Kernel dispatch requires temporary workspace

When Runtime plans dispatch

Then Memory Manager admits, queues, or rejects the workspace allocation.

---

### Requirement: Memory Manager Validates Kernel Memory Classes

Memory Manager SHALL validate that input, output, and workspace memory classes
are compatible with Kernel requirements.

#### Scenario: Pinned host required

Given Kernel requires pinned host memory

When Memory Manager cannot provide it

Then Runtime rejects the Kernel or chooses fallback.

---

### Requirement: Memory Manager Tracks Kernel Resource Effects

Memory Manager SHALL track resource metadata changes caused by Kernel execution.

#### Scenario: Kernel writes output

Given Kernel writes an output tensor

When execution completes

Then Memory Manager records output readiness, residency, and Resource Affinity.