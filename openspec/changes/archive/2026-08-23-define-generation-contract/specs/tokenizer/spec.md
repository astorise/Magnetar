## ADDED Requirements

### Requirement: Tokenizer Provides Generation Input

Tokenizer output SHALL provide token IDs and prompt token count usable by the
Generation Contract.

#### Scenario: Prompt encoded

Given a prompt is encoded

When generation request is constructed

Then token IDs and token count come from tokenizer output.

---

### Requirement: Tokenizer Provides Streaming Decode For Generation

Tokenizer streaming decode SHALL be used to convert generated token streams into
valid text chunks.

#### Scenario: Generated token stream

Given generation emits token IDs

When text streaming is requested

Then tokenizer streaming decode produces text chunks.

---

### Requirement: Tokenizer Provides Stop Sequence Preparation

Tokenizer SHALL define token pattern preparation for textual stop sequences
where feasible.

#### Scenario: Text stop sequence

Given a user configures a textual stop sequence

When generation validates stop conditions

Then Runtime asks tokenizer to prepare token matching metadata where possible.
