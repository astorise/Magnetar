## ADDED Requirements
### Requirement: CLI Is Runtime Inference Client

`magnetar-cli` SHALL invoke RuntimeInferenceApi for first-profile generation.

#### Scenario: `magnetar run`

Given user supplies model reference and prompt

When command executes

Then CLI delegates inference to Runtime.

### Requirement: CLI Does Not Execute Reference Kernels

CLI SHALL not link model execution logic directly to Reference CPU Kernel
functions.

#### Scenario: MatMul required

Given model generation is running

When MatMul executes

Then call originates through Runtime Plan/Provider path rather than CLI.

### Requirement: CLI Does Not Own KV Cache

CLI SHALL not maintain model KV tensors across generated tokens.

#### Scenario: Streaming generation

Given tokens are emitted incrementally

When next token is requested

Then Runtime Session owns cached model state.

### Requirement: CLI Does Not Fabricate Logits

CLI SHALL not provide placeholder or deterministic fake logits to Generation.

#### Scenario: First-profile fixture

Given CLI asks Runtime to generate

When Sampling receives logits

Then they originate from Qwen model execution.