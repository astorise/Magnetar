## ADDED Requirements

### Requirement: Qwen Tokenizer Compatibility

Tokenizer compatibility for Qwen baseline SHALL validate vocabulary and special
token metadata required by the Qwen Model Component.

#### Scenario: EOS missing

Given Qwen generation metadata requires EOS token

When tokenizer compatibility is checked

Then Runtime rejects tokenizer if EOS metadata is unavailable.

---

### Requirement: Qwen Component Does Not Own Tokenization

Qwen Model Component SHALL not own tokenization execution.

#### Scenario: Encode prompt

Given user prompt needs tokenization

When Runtime processes it

Then Tokenizer Contract performs encoding before Qwen graph execution.