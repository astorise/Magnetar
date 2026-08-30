## ADDED Requirements
### Requirement: KV Implementation Has Independent Gate

KV append/read/position semantics SHALL pass focused tests before incremental
Qwen decode is declared complete.

#### Scenario: Append two positions

Given layer cache receives positions 0 and 1

When read occurs

Then correct K/V values and sequence length are returned.

### Requirement: Incremental Decode Gate Measures KV Growth

Final model tests SHALL prove KV state grows incrementally.

#### Scenario: Prompt length N

Given one decode token is processed

When step completes

Then logical KV length becomes N+1 rather than being recreated as unrelated
state.

### Requirement: Session KV Is Isolated

Two independent inference Sessions SHALL not share mutable KV state accidentally.

#### Scenario: Sessions A and B

Given same Model Instance

When each decodes different token sequences

Then their KV positions/state remain independent.

### Requirement: KV Failure Is Structured

Out-of-range position/context failure SHALL not panic or corrupt memory.

#### Scenario: Context limit reached

Given append exceeds fixture context

When decode runs

Then structured KV/context failure is returned.