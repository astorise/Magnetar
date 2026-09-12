## ADDED Requirements

### Requirement: Production Generation Delivers Ordered Incremental Stream Events

A production generation entry point SHALL be able to deliver each produced token to a caller-supplied callback as it is produced, in production order, rather than only after the entire generation completes. Each token event SHALL carry the produced token id and an incremental, tokenizer-decoded text delta.

#### Scenario: Multi-token CPU generation delivers ordered events before completion

- **WHEN** a streaming production generation request produces more than one token on Reference CPU
- **THEN** the caller's callback receives one `Token` event per produced token, in the exact order the tokens were produced, before the terminal `Finished` event

#### Scenario: Streamed text reconstructs the non-streaming result exactly

- **WHEN** the text deltas from every `Token` event delivered during a streaming generation are concatenated in delivery order
- **THEN** the result is exactly equal to the same request's non-streaming decoded output text, with no duplicated or missing text

---

### Requirement: Production Generation Streaming Terminates With Exactly One Finished Event

A streaming production generation entry point SHALL deliver exactly one `Finished` event carrying the real finish reason and usage, after the last `Token` event, regardless of whether generation ended by reaching a stop condition, the token budget, or caller-requested cancellation.

#### Scenario: Finished event carries real usage

- **WHEN** a streaming production generation completes
- **THEN** exactly one `Finished` event is delivered, carrying the same finish reason and usage the entry point's own non-streaming return value reports

#### Scenario: A provider or runtime error does not emit a synthetic Finished event

- **WHEN** a decode step fails with a Provider or Runtime error before generation would otherwise complete
- **THEN** the entry point returns that error exactly as every non-streaming entry point does, and does not invoke the callback with a `Finished` event describing a completion that did not happen

---

### Requirement: Production Generation Streaming Supports Caller-Requested Cancellation

A streaming production generation entry point SHALL let the caller's callback request that generation stop after the token just delivered, and SHALL then terminate generation cleanly -- releasing resources the same way any other generation completion does -- rather than continuing to decode or propagating a hard error.

#### Scenario: Callback-requested cancellation stops decode promptly

- **WHEN** a caller's callback signals cancellation upon receiving a `Token` event
- **THEN** no further decode steps run, a `Finished` event carrying a cancelled finish reason is delivered, and the entry point's own resource cleanup (session close, model instance unload) still runs exactly as it does for any other completion

#### Scenario: A cancelled stream still flushes pending partial text

- **WHEN** caller-requested cancellation occurs while the tokenizer's incremental decode still holds pending partial bytes (a multi-byte character split across the last two tokens)
- **THEN** those bytes are flushed as text before the `Finished` event, never silently dropped

---

### Requirement: Streaming Is Purely Additive To Existing Production Generation

Introducing streaming delivery SHALL NOT change the signature or observable behavior of any pre-existing production generation entry point, and SHALL NOT change the behavior of the underlying generation loop for any caller that does not supply a streaming callback.

#### Scenario: Pre-existing non-streaming entry points are unaffected

- **WHEN** a caller uses `run_production_qwen_generation`, `_for_provider`, `_for_provider_with_prompt`, or `_for_provider_with_request`
- **THEN** generation runs exactly as it did before streaming delivery existed, with no per-token callback invoked
