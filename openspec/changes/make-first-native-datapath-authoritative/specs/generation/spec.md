## ADDED Requirements

### Requirement: Decode Uses Correct Absolute Position
Generation decode SHALL pass the absolute position of the token being decoded.

#### Scenario: Prompt length four
- **WHEN** a prompt has length 4 and the first, second, and third generated tokens are decoded
- **THEN** the decode positions are 4, 5, and 6 respectively.

#### Scenario: Real loop oracle
- **WHEN** the real generation loop performs multi-step incremental decode
- **THEN** its logits match the full-sequence oracle at the same absolute positions.

### Requirement: Token Commit Follows Causal Execution
Generation SHALL commit a token only after graph execution, plan binding execution, provider completion, logits production, sampling, and KV commit succeed.

#### Scenario: KV commit fails
- **WHEN** KV commit fails after sampling
- **THEN** the generated token is not committed as a successful generation step.
