## ADDED Requirements
### Requirement: Memory Manager Tracks Adapter Residency

Memory Manager SHALL track adapter residency separately from base model
residency.

#### Scenario: Adapter resident

Given adapter tensors are loaded into Device memory

When Runtime reports memory usage

Then adapter residency is accounted separately.

---

### Requirement: Memory Manager Enforces Adapter Memory Budget

Memory Manager SHALL enforce adapter memory budgets according to Runtime and
session policy.

#### Scenario: Adapter budget exceeded

Given a session adapter memory budget is exceeded

When adapter loading is requested

Then Memory Manager rejects, queues, or delays according to policy.

---

### Requirement: Memory Manager Accounts For Adapter Merge Workspace

Memory Manager SHALL account for workspace required to merge, unmerge, or
transform adapters.

#### Scenario: Merge workspace

Given merge-on-activation requires temporary workspace

When Runtime plans activation

Then Memory Manager accounts for merge workspace.