## ADDED Requirements
### Requirement: First Profile Builds Prepared Execution Plan

Qwen fixture SHALL execute through PreparedExecutionPlan.

#### Scenario: Model becomes ready

Given graph and Reference CPU Kernels are available

When preparation completes

Then a ready Plan contains Kernel/resource execution decisions.

### Requirement: Single Device Plan Is Valid

First profile Plan SHALL allow all nodes to bind to one Reference CPU Device.

#### Scenario: Graph has all mandatory Operators

Given CPU provides all Kernels

When Plan is built

Then no multi-Device machinery is required.

### Requirement: Plan Contains Kernel Bindings

Ready Plan SHALL identify selected prepared Kernels rather than rediscovering
them through direct model-specific calls.

#### Scenario: Decode Plan

Given Attention/RMSNorm/MatMul bindings are prepared

When token executes

Then Plan references those bindings.

### Requirement: Decode Reuses Compatible Plan

Repeated incremental decode SHALL reuse compatible Plan generation when graph
and shape guards remain valid.

#### Scenario: Ten generated tokens

Given graph/shape guards remain valid

When tokens execute

Then complete Plan construction is not repeated for every token.
