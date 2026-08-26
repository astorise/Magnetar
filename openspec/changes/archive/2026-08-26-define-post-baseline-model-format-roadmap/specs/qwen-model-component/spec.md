## ADDED Requirements

### Requirement: Qwen Component Uses Normalized Config Metadata

Qwen Model Component SHALL use normalized config metadata from Model Artifact.

#### Scenario: Qwen hidden size

Given config normalization extracts hidden size

When Qwen validation runs

Then Qwen Component validates hidden size through normalized metadata.

---

### Requirement: Qwen Component Does Not Parse Raw Format Files

Qwen Model Component SHALL not parse raw safetensors, GGUF, tokenizer.json, or
local file structures directly.

#### Scenario: Raw file parse requested

Given Qwen Component attempts to parse raw GGUF file

When Runtime validates authority

Then access is denied or rejected as boundary violation.