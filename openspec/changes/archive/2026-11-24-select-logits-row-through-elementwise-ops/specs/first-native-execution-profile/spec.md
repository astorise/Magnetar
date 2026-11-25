## ADDED Requirements

### Requirement: The sampled logits row is selected through row-wise nodes

When a prefill samples one row, the Runtime SHALL compute the LM head on that row only. This applies when the head is a plain or bias matmul producing `"logits"` directly, or through row-wise unary nodes that are each the only reader of their input.

#### Scenario: Logit softcap after the head
- **WHEN** a bucket graph's `"logits"` is a softcap over its LM head's output
- **THEN** the head reads the gathered row only, and the logits equal the uncapped graph's logits capped
