## ADDED Requirements

### Requirement: Tokenizer Metadata Supports Sampling

Tokenizer metadata SHALL provide token ID validity, vocabulary size, special
token metadata, and stop token preparation used by Sampling.

#### Scenario: Vocabulary size

Given logits length differs from tokenizer vocabulary size

When Sampling validates input

Then Sampling reports vocabulary-mismatch.

---

### Requirement: Text Constraints Become Token Constraints

Textual constraints SHALL be converted through Tokenizer before affecting
Sampling.

#### Scenario: Text stop sequence

Given a textual stop sequence is configured

When Sampling or Generation needs token constraints

Then Runtime uses Tokenizer-derived token metadata.