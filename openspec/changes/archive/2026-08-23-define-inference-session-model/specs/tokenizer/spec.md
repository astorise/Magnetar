## ADDED Requirements
### Requirement: Session References Tokenizer

An Inference Session SHALL reference a tokenizer compatible with its model
context.

#### Scenario: Session tokenizer

Given session creation requests model M and tokenizer T

When T is incompatible with M

Then session creation fails.

---

### Requirement: Session May Own Tokenizer Streaming State

A session operation SHALL be able to own tokenizer streaming decode state.

#### Scenario: Streaming decode state

Given generated tokens are being decoded incrementally

When a token produces partial output

Then the session operation preserves tokenizer decode state.