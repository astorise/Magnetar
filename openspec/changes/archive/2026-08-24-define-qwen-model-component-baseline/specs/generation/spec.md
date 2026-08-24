## ADDED Requirements

### Requirement: Generation May Use Qwen Graphs

Generation SHALL be permitted to use Qwen prefill and decode graphs when Model
Instance is Qwen-compatible and ready.

#### Scenario: Qwen decode

Given Qwen Model Instance is ready

When Generation performs decode

Then Runtime may request Qwen decode graph.

---

### Requirement: Qwen Graphs Do Not Own Sampling

Qwen graphs SHALL produce logits, while Sampling Contract owns token selection.

#### Scenario: Next token

Given Qwen decode graph returns logits

When next token is selected

Then Sampling Contract performs selection.