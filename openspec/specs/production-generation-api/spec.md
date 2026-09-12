# production-generation-api Specification

## Purpose
TBD - created by archiving change expose-production-generation-parameters. Update Purpose after archive.
## Requirements
### Requirement: Production Generation Accepts Caller-Supplied Generation Parameters

The production generation entry point SHALL accept a caller-supplied `GenerationParameters` value and forward it, unmodified in meaning, to the same Generation Contract every other Runtime generation path uses -- not substitute an internally hardcoded value.

#### Scenario: Temperature and top_p reach sampling

- **WHEN** a caller requests production generation with `temperature` and `top_p` set to non-default, non-greedy values
- **THEN** the resulting `GenerationRequest.parameters` carries those exact values into the sampling contract, and the sampling decision is not forced to greedy selection

#### Scenario: Seed reaches the sampling contract

- **WHEN** a caller requests production generation with a `seed` set
- **THEN** the resulting `GenerationRequest.parameters.seed` carries that value into the sampling contract

#### Scenario: Legacy entry points preserve greedy defaults

- **WHEN** a caller uses `run_production_qwen_generation`, `run_production_qwen_generation_for_provider`, or `run_production_qwen_generation_for_provider_with_prompt`
- **THEN** generation runs exactly as it did before this requirement existed: greedy decoding with default stop conditions and the checkpoint manifest's own token budget

---

### Requirement: Production Generation Accepts Caller-Supplied Stop Conditions

The production generation entry point SHALL accept a caller-supplied `StopConditions` value, including plain-text stop sequences, and prepare those text sequences against the real tokenizer so generation actually stops on them -- a caller SHALL NOT be required to construct the tokenizer-aware prepared form itself.

#### Scenario: A text stop sequence stops generation early

- **WHEN** a caller requests production generation with a `stop_text_sequences` entry that the model is expected to produce before its natural token budget is exhausted
- **THEN** generation stops at that sequence, and the produced text does not extend past it

#### Scenario: Token-id stop conditions are still honored

- **WHEN** a caller requests production generation with `stop_conditions.stop_token_ids` set
- **THEN** generation stops when one of those token ids is produced, exactly as the underlying Generation Contract already does for any other caller

---

### Requirement: Production Generation Fails Closed On Unsupported Requests

When a caller-supplied generation parameter, stop condition, or token budget cannot be honored by the selected execution path, the production generation entry point SHALL return a structured `Unsupported` error naming the reason, and SHALL NOT silently ignore the request or proceed as if it had been honored.

#### Scenario: A token budget exceeding a non-multi-step Provider's capability is rejected

- **WHEN** a caller requests a resolved token budget (manifest default or caller override) greater than one, against a Provider that does not support multi-step decode
- **THEN** the entry point returns `InferenceApiError::Unsupported` naming the Provider and the unsupported decode-step count, and no generation is attempted

---

### Requirement: Production Generation Request Overrides Are Optional And Additive

`ProductionGenerationRequest`'s token-budget override SHALL be optional; when absent, the checkpoint manifest's own configured default SHALL apply exactly as it did before this capability existed.

#### Scenario: No override preserves manifest-driven token budget

- **WHEN** a caller supplies `ProductionGenerationRequest.max_new_tokens: None`
- **THEN** the resolved token budget is the checkpoint manifest's own configured default, unchanged from prior behavior

#### Scenario: An explicit override replaces the manifest default

- **WHEN** a caller supplies `ProductionGenerationRequest.max_new_tokens: Some(n)`
- **THEN** the resolved token budget is `n`, regardless of the checkpoint manifest's own configured default

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

