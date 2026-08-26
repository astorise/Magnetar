## ADDED Requirements

### Requirement: First Operator Scope Implemented Before Qwen Baseline

The first operator implementation scope SHALL be implemented before Qwen
baseline graph execution.

#### Scenario: Qwen graph validates

Given Qwen graph uses attention and RMSNorm

When graph validation runs

Then required-now operator metadata exists.

---

### Requirement: Operator Fixtures Support CPU Baseline

Required-now Operators SHALL have fixtures usable by Reference CPU conformance.

#### Scenario: Softmax fixture

Given softmax is required-now

When CPU conformance runs

Then softmax fixture exists.