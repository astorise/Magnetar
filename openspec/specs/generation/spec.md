# generation Specification

## Purpose
TBD - created by archiving change define-generation-contract. Update Purpose after archive.
## Requirements
### Requirement: Generation Contract

Magnetar SHALL define a Generation Contract for producing output token IDs from
validated input token IDs.

Generation SHALL be part of Magnetar inference Runtime.

#### Scenario: Generate tokens

Given validated input token IDs and a compatible model context

When generation runs

Then Magnetar produces output token IDs according to generation parameters and
stop conditions.

---

### Requirement: Generation Is Token-Based

Generation input and output SHALL be token-based.

Raw structured chat messages SHALL be rendered and tokenized before generation.

#### Scenario: Chat request

Given a client submits structured chat messages

When Runtime prepares generation

Then messages are rendered and tokenized before the Generation Contract receives
input.

---

### Requirement: Generation Does Not Own Tokenization

Generation SHALL NOT own text-to-token or token-to-text conversion.

#### Scenario: Output text

Given generation emits token IDs

When text output is needed

Then Runtime uses Tokenizer decode or streaming decode.

---

### Requirement: Generation Request

A GenerationRequest SHALL include model reference, tokenizer reference, input
token IDs, prompt token count, max token limits, generation parameters, stop
conditions, streaming mode, and cancellation metadata.

#### Scenario: Missing model reference

Given a generation request lacks a model reference

When Runtime validates it

Then validation fails.

---

### Requirement: Generation Parameters

Generation parameters SHALL be explicit and validated.

Parameters MAY include temperature, top-p, top-k, penalties, seed,
deterministic mode, greedy mode, sampling flag, banned token IDs, allowed token
IDs, logits processors, and EOS behavior.

#### Scenario: Invalid temperature

Given a generation request specifies an invalid temperature

When validation runs

Then Runtime returns a parameter-invalid error.

---

### Requirement: Prefill Stage

Generation SHALL distinguish the prefill stage from the decode stage.

Prefill consumes input tokens and initializes generation state.

#### Scenario: Prefill starts

Given a valid generation request

When generation begins

Then Runtime enters prefill before iterative decode.

---

### Requirement: Decode Stage

Generation SHALL define iterative decode semantics.

Each decode step produces or selects the next token ID until a stop condition is
met.

#### Scenario: Decode token

Given prefill has completed

When one decode step runs

Then one next-token decision is produced unless generation stops or fails.

---

### Requirement: Stop Conditions

Generation SHALL support explicit stop conditions.

Stop conditions MAY include max new tokens, max total tokens, EOS token, stop
token, stop token pattern, stop text sequence, cancellation, memory policy,
runtime shutdown, and Provider failure.

#### Scenario: Max new tokens reached

Given max new tokens is 10

When 10 tokens have been generated

Then generation stops with max-new-tokens finish reason.

---

### Requirement: Finish Reason

Generation SHALL report a finish reason.

Finish reasons SHOULD distinguish length, EOS, stop sequence, cancellation,
interruption, memory limit, shutdown, Provider error, model error, policy denial,
and generic error.

#### Scenario: EOS reached

Given a generated token matches configured EOS behavior

When generation stops

Then finish reason is EOS-token.

---

### Requirement: Streaming Generation

Generation SHALL support streaming token output.

Streaming token events SHALL preserve token order.

#### Scenario: Stream tokens

Given streaming mode is enabled

When generation produces tokens

Then token-generated events are emitted in generation order.

---

### Requirement: Text Streaming Uses Tokenizer

Text chunks in streaming responses SHALL be produced by Tokenizer streaming
decode.

Generation itself SHALL stream token IDs.

#### Scenario: Partial text token

Given a generated token does not form a complete text chunk

When text streaming is enabled

Then Tokenizer streaming decode holds pending partial state.

---

### Requirement: Usage Accounting

Generation SHALL report usage accounting.

Usage SHOULD include prompt tokens, generated tokens, total tokens, durations
where available, and finish reason.

#### Scenario: Usage report

Given generation completes after 5 output tokens

When usage is reported

Then generated token count is 5.

---

### Requirement: Context Window Validation

Generation SHALL validate prompt and output length against model and Runtime
limits.

Generation SHALL NOT silently truncate input tokens.

#### Scenario: Context exceeded

Given prompt tokens plus max new tokens exceed model context length

When generation request is validated

Then Runtime rejects the request unless explicit policy allows adjustment.

---

### Requirement: EOS Behavior

EOS behavior SHALL be explicit and support multiple EOS IDs where applicable.

#### Scenario: Ignore EOS

Given policy says ignore EOS

When EOS token is generated

Then generation does not stop solely because EOS appeared.

---

### Requirement: Stop Text Sequence Boundary

Stop text sequence handling SHALL use tokenizer-prepared information where
needed.

Generation owns stopping behavior; Tokenizer owns text/token mapping.

#### Scenario: Stop text sequence

Given a stop sequence is textual

When generation prepares stopping

Then Runtime uses tokenizer support to derive stop matching information where
feasible.

---

### Requirement: Deterministic Generation

Generation SHALL define deterministic mode semantics.

If deterministic mode is requested but unsupported, Runtime SHALL return a
structured unsupported error.

#### Scenario: Determinism unsupported

Given deterministic generation is requested

And selected execution path does not support it

When validation runs

Then Runtime reports deterministic-mode-unsupported.

---

### Requirement: Generation Cancellation

Generation SHALL support cancellation semantics.

#### Scenario: Cancel during decode

Given generation is in decode stage

When cancellation is requested

Then generation stops with cancelled or interrupted result according to policy.

---

### Requirement: Generation Memory Admission

Generation SHALL use Memory Manager for memory admission.

#### Scenario: Memory unavailable

Given generation requires workspace memory

And Memory Manager rejects admission

When generation is requested

Then Runtime returns memory-admission-failed or queues according to policy.

---

### Requirement: Generation Does Not Select Provider Or Device Directly

Generation requests SHALL NOT contain authoritative Provider or Device
selection.

Provider and Device selection remain Runtime-owned.

#### Scenario: Request attempts Provider pin

Given a generation request attempts to select Provider `cuda`

When Runtime validates the request

Then the request is rejected or the field is ignored as non-authoritative policy
metadata.

---

### Requirement: Generation Error Categories

Generation errors SHALL be structured.

#### Scenario: Provider execution failure

Given Provider execution fails during generation

When Runtime reports the failure

Then the error maps to a stable provider-execution-failed category.

---

### Requirement: Generation Observability

Runtime SHALL define Generation observations.

Observability SHALL not log raw prompts by default.

#### Scenario: Token generated observation

Given a token is generated

When observability records the event

Then it may record token index and timing

And does not record raw prompt text by default.

### Requirement: Generation May Reference Session

A GenerationRequest SHALL be able to reference an Inference Session.

When a session is referenced, Generation SHALL use session model binding,
tokenizer binding, policy, memory budget, cancellation state, and observability
correlation.

#### Scenario: Generate with session

Given a ready session

When generation runs inside it

Then session bindings are applied.

---

### Requirement: Generation Supports One-Shot Session

Generation SHALL support Runtime one-shot requests through implicit short-lived session semantics.

#### Scenario: One-shot generation

Given a caller submits a one-shot generation request

When Runtime executes it

Then Generation uses session semantics and cleans up session-scoped resources
after completion.

---

### Requirement: Generation Respects Session Concurrency

Generation SHALL respect session concurrency policy.

#### Scenario: Session active

Given a session allows only one active operation

And one generation is active

When a second generation is requested

Then Runtime queues or rejects according to session policy.

---

### Requirement: Generation Respects Session Cancellation

Generation SHALL observe session cancellation state.

#### Scenario: Session cancelled

Given a session is cancelled

When generation is active

Then generation stops or fails according to cancellation policy.

---

### Requirement: Generation Uses Session Streaming State

When streaming generation runs inside a session, streaming state SHALL be
associated with the session operation.

#### Scenario: Streaming chunks

Given generation streams token IDs

When tokenizer streaming decode has partial state

Then the session operation preserves that state.

### Requirement: Generation Uses KV Cache

Generation SHALL use KV cache through Runtime-managed cache references where
cache is enabled.

#### Scenario: Decode after prefill

Given prefill has populated a ready KV cache

When decode runs

Then Generation uses the cache to continue token production.

---

### Requirement: Prefill May Populate KV Cache

Prefill SHALL route KV cache creation or population through Runtime policy when
KV cache is enabled.

#### Scenario: Prefill prompt

Given a generation request includes prompt tokens

When prefill executes

Then Runtime may allocate and populate KV cache entries for those tokens.

---

### Requirement: Decode May Append KV Cache

Decode SHALL route key/value state append for newly generated tokens through
Runtime policy when KV cache is enabled.

#### Scenario: Append generated token

Given decode produces a token

When model state advances

Then Runtime may append corresponding KV state to the cache.

---

### Requirement: Generation Validates Cache Compatibility

Generation SHALL validate KV cache compatibility before reuse.

#### Scenario: Prompt mismatch

Given a cache was created for prompt prefix A

When generation for prompt prefix B attempts reuse

Then Runtime rejects reuse or rebuilds according to policy.

---

### Requirement: Generation Handles Cache Invalidation

If KV cache becomes invalid during generation, Runtime policy SHALL determine
whether generation fails, rebuilds, retries, or cancels.

#### Scenario: Device reset

Given Device reset invalidates cache residency

When generation continues

Then Runtime handles the invalid cache according to policy.

### Requirement: Generation Uses Sampling For Next Token

Generation SHALL use Sampling or equivalent Runtime-owned selection logic to
choose the next token from logits.

#### Scenario: Decode step

Given model forward returns logits

When decode step needs a token

Then Generation invokes Sampling to select the next token.

---

### Requirement: Generation Owns Stop Conditions

Generation SHALL remain responsible for stop condition evaluation.

Sampling may mask or select tokens according to policy, but Generation decides
whether generation stops.

#### Scenario: EOS selected

Given Sampling selects EOS

When Generation receives it

Then Generation applies EOS stop policy.

---

### Requirement: Generation Owns KV Cache Advance

Generation SHALL remain responsible for updating generation state and KV cache
after Sampling returns a token.

#### Scenario: Token selected

Given Sampling selects a token

When Generation accepts it

Then Generation updates state and appends KV cache where applicable.

---

### Requirement: Generation Validates Sampling Parameters

Generation request validation SHALL include Sampling parameter validation or
delegation to the Sampling Contract.

#### Scenario: Invalid top-p

Given generation request includes invalid top-p

When Runtime validates the request

Then validation fails before decode execution.

