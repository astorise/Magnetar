## ADDED Requirements

### Requirement: Tokenizer Is Exposed Through Inference API

Runtime Inference API SHALL expose encode, decode, and streaming decode through the Tokenizer Contract.

#### Scenario: Tokenize prompt

Given prompt text is supplied

When API tokenization runs

Then Tokenizer Contract produces token IDs.

---

### Requirement: Inference API Tokenization Is Redacted By Default

Raw prompt logging SHALL be disabled by default for tokenization API use.

#### Scenario: Tokenization failed

Given tokenization fails

When diagnostics are emitted

Then raw prompt text is not logged by default.