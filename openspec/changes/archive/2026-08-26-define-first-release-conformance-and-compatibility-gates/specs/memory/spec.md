## ADDED Requirements

### Requirement: Memory Release Gate

Memory Manager baseline SHALL have release gate coverage.

#### Scenario: Untracked allocation

Given Runtime-visible allocation is not tracked

When release validation runs

Then stable release is blocked.

---

### Requirement: Cache Is Not Memory Residency Gate

Release gates SHALL validate cache storage is distinct from memory residency.

#### Scenario: Cached model

Given model is cached but not loaded

When Memory Manager is inspected

Then model tensors are not resident.