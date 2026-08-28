## ADDED Requirements

### Requirement: Compilation Workspace Is Distinct From Inference Tensor Memory

Compiler temporary/workspace memory SHALL not be confused with Runtime Tensor
Resource residency.

#### Scenario: Compiler uses 2 GiB host memory

Given compilation is active

When inference tensor accounting is inspected

Then compiler workspace is classified separately.

---

### Requirement: Compilation Resource Pressure May Affect Admission

Runtime SHALL account for compilation resource pressure when deciding whether to start additional cold-path work.

#### Scenario: Host pressure high

Given compilation would exceed configured resource policy

When job is submitted

Then Runtime queues or rejects cold-path compilation.

---

### Requirement: Compilation Never Owns Runtime Tensor Resources

Compiler SHALL NOT obtain ownership of inference Tensor Resources as part of
normal compilation.

#### Scenario: Shape specialization

Given compiler needs tensor shapes

When request is built

Then metadata is provided rather than mutable inference tensor ownership.