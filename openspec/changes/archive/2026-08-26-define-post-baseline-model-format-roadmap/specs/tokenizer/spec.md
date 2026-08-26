## ADDED Requirements

### Requirement: Tokenizer Formats Normalize Into Tokenizer Artifact

tokenizer.json, tokenizer_config, SentencePiece, and embedded tokenizer metadata SHALL normalize into Tokenizer Artifact metadata.

#### Scenario: tokenizer.json normalized

Given tokenizer.json is parsed

When normalization completes

Then Tokenizer Contract receives normalized Tokenizer Artifact metadata.

---

### Requirement: Generation Config Does Not Override Tokenizer Policy Silently

Tokenizer-related metadata from generation_config or tokenizer_config SHALL not
silently override Tokenizer or Runtime policy.

#### Scenario: PAD token mismatch

Given tokenizer_config and generation_config disagree on PAD token

When compatibility validation runs

Then Runtime resolves according to policy or reports conflict.