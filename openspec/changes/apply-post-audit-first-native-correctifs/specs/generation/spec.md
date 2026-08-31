## ADDED Requirements

### Requirement: First-Native Logits Come From Runtime Model Execution
First-native Generation SHALL obtain logits only from Runtime model execution of a ready ModelInstance through a compatible PreparedExecutionPlan.

#### Scenario: Runtime execution produces logits
- **WHEN** first-native Generation needs next-token logits
- **THEN** Runtime executes the selected ModelInstance through a PreparedExecutionPlan and passes those logits to Sampling.

#### Scenario: No external logits producer
- **WHEN** a caller attempts to provide logits or a per-request forward callback for normal first-native Generation
- **THEN** Runtime rejects the request or the API shape does not expose that capability.

### Requirement: First-Native Prefill Initializes Runtime KV
First-native prefill SHALL create or initialize Runtime-owned KV cache state when KV cache is enabled.

#### Scenario: Prefill commits KV
- **WHEN** Runtime executes prefill for a prompt token sequence
- **THEN** Runtime records KV state associated with the Session, ModelInstance, tokenizer/template compatibility, and Resource Affinity.

### Requirement: First-Native Decode Is Incremental
First-native decode SHALL consume existing Runtime-owned KV state and submit only newly admitted token input for the baseline greedy decode step.

#### Scenario: One-token decode
- **WHEN** baseline first-native decode generates the next token after prefill
- **THEN** decode input to model execution contains exactly the newly admitted token and references prior context through KV state.

#### Scenario: Decode cannot use invalid KV
- **WHEN** required KV state is missing, invalidated, released, or belongs to another Session or ModelInstance
- **THEN** Runtime rejects the decode step with a structured generation or KV compatibility error.

#### Scenario: Incremental logits match oracle
- **WHEN** tests compare full-sequence oracle logits with prefill plus incremental decode logits
- **THEN** the values match within an explicit tolerance.
