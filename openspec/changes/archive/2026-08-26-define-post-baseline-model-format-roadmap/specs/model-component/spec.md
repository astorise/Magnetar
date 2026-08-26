## ADDED Requirements

### Requirement: Model Components Interpret Normalized Architecture Metadata

Model Components SHALL interpret normalized architecture metadata rather than raw
external format files.

#### Scenario: Qwen config normalized

Given Hugging Face-style config is normalized

When Qwen Model Component validates architecture

Then it reads normalized metadata.

---

### Requirement: Format Parsers Do Not Produce Execution Graphs

Format parsers SHALL not produce authoritative execution graphs.

#### Scenario: Parser emits graph

Given format parser attempts to emit Provider-specific graph

When Runtime validates it

Then Runtime rejects or ignores graph behavior outside Model Component contract.