## ADDED Requirements
### Requirement: First Qwen Architecture Is WASM Component

Mandatory profile SHALL execute Qwen architecture implementation through a WASM
Component.

#### Scenario: Model loads

Given Qwen Component Artifact is available

When Model Instance initializes

Then Component Engine instantiates the WASM Component.

### Requirement: Qwen Component Is Provider Neutral

Qwen Component SHALL not select Reference CPU Provider as part of model
semantics.

#### Scenario: Graph creation

Given Component emits MatMul Operator

When Runtime later selects Kernel

Then Provider selection is Runtime-owned.

### Requirement: Qwen Component Is Device Neutral

Qwen Component SHALL not name CPU Device.

#### Scenario: Model graph emitted

Given first profile happens to run CPU

When graph is inspected

Then concrete CPU placement is absent from model semantics.

### Requirement: Qwen Component Does Not Execute Kernels Directly

WASM Component SHALL use portable Operator/graph boundary.

#### Scenario: MatMul needed

Given Qwen layer requires projection

When Component describes computation

Then it emits/uses MatMul Operator rather than invoking Reference CPU function.

### Requirement: Qwen Component Has No Ambient Inference Authority

Component SHALL not require arbitrary filesystem or network access for normal
model execution.

#### Scenario: Weight required

Given weight is external

When Component needs model tensor

Then Runtime-provided logical Resource is used.