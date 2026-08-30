## ADDED Requirements
### Requirement: Runtime Owns Complete Model Forward Execution

Runtime SHALL own the path that turns loaded Model Instance inputs into model
logits.

#### Scenario: Generate next token

Given Model Instance and Session exist

When Runtime performs generation

Then logits originate from execution of the loaded model through its Prepared
Execution Plan.

### Requirement: Runtime Does Not Require Caller Next Logits Function

Ordinary Runtime inference SHALL not require caller-provided model-forward
callback.

#### Scenario: CLI generation

Given CLI supplies prompt text

When generation starts

Then CLI does not provide `next_logits`.

### Requirement: First Profile Uses One Provider Device

Runtime SHALL allow a simple single Reference CPU Provider/Device placement for
first-profile execution.

#### Scenario: Plan construction

Given first profile model loads

When placement resolves

Then one Reference CPU Device may satisfy all graph nodes.

### Requirement: Runtime Uses Prepared Plan

First-profile Qwen execution SHALL use PreparedExecutionPlan.

#### Scenario: Decode step

Given Plan is ready and compatible

When next token executes

Then Runtime reuses prepared Kernel/resource decisions.

### Requirement: Deferred Capability Absence Is Valid

Runtime first-profile conformance SHALL not require advanced optimization
subsystems to be enabled.

#### Scenario: Autotuning disabled

Given static Reference CPU Kernel selection works

When model executes

Then Runtime remains profile-conformant.
