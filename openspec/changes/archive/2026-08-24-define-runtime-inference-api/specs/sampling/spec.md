## ADDED Requirements

### Requirement: Sampling Parameters Are Inference API Inputs

Runtime Inference API SHALL allow Sampling parameters to be provided as validated inputs to Generation.

#### Scenario: Temperature provided

Given caller sets temperature

When generation request is validated

Then Sampling Contract validates temperature semantics.

---

### Requirement: Inference API Does Not Expose Raw Logits By Default

Runtime Inference API SHALL not expose raw logits by default.

#### Scenario: Streaming output

Given decode produces logits

When streaming events are emitted

Then raw logits are not included unless a future explicit diagnostic policy
allows it.