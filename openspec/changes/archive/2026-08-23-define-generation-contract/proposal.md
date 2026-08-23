# Define Generation Contract

## Why

Magnetar is an inference Runtime.

The Runtime now has contracts for:

- Model Artifacts
- Tokenizer behavior
- Memory Manager
- Providers
- Devices
- Component execution
- inference-scoped authority

The next missing inference contract is Generation.

Generation is the Runtime behavior that consumes input token IDs and produces
new token IDs.

Generation must be separate from tokenization.

Generation must also be separate from sampling implementation, KV cache
implementation, Provider execution, and client-side agent orchestration.

Without a Generation Contract, Magnetar cannot reliably define:

- text completion
- chat completion after template/tokenization
- streaming tokens
- prefill versus decode
- max new tokens
- context length validation
- stop token handling
- stop sequence handling
- EOS handling
- generation parameters
- deterministic generation
- sampling controls
- logits processing boundary
- cancellation
- partial failure behavior
- output token accounting
- memory feasibility
- Provider execution boundary
- future batching
- future KV cache

This change defines the stable Generation Contract.

## What Changes

This change introduces a Generation domain contract.

The Generation Contract SHALL define how Magnetar produces output tokens from
validated input tokens and a loaded model context.

The contract SHALL support:

- generation request
- generation parameters
- input token sequence
- output token stream
- prefill stage
- decode stage
- stop conditions
- EOS behavior
- max token limits
- streaming events
- deterministic seed where supported
- cancellation
- generation diagnostics
- structured errors
- observability

This change does not fully define KV cache internals.

It prepares for a later KV cache model.

## Generation Input

Generation input SHALL be token-based.

A generation request SHALL consume token IDs produced by the Tokenizer Contract
or otherwise validated against the model tokenizer.

Input MAY include:

- input token IDs
- prompt token count
- optional attention mask
- optional token type IDs
- model reference
- tokenizer reference
- generation parameters
- stop conditions
- requested output format
- streaming preference
- cancellation handle or request identity
- optional session reference

Generation SHALL NOT accept raw chat messages as its core input.

Structured messages must be rendered and tokenized before generation.

## Generation Output

Generation output SHALL be token-based.

Output MAY include:

- generated token IDs
- output token count
- finish reason
- streaming events
- optional logits metadata where policy permits
- optional token probabilities where policy permits
- diagnostics
- usage accounting

Decoded text is produced by Tokenizer decode or streaming decode.

Generation SHALL NOT own text detokenization.

## Generation Request

A GenerationRequest SHALL describe a single generation operation.

It SHOULD include:

- request ID
- model instance or model reference
- tokenizer reference
- input token IDs
- prompt token count
- max new tokens
- max total tokens where applicable
- generation parameters
- stop conditions
- streaming mode
- priority or scheduling metadata
- memory admission metadata
- cancellation metadata
- observability correlation ID

The exact Rust names are implementation-defined.

## Generation Parameters

Generation parameters SHOULD include:

- temperature
- top-p
- top-k
- min-p where supported
- typical-p where supported
- repetition penalty
- frequency penalty
- presence penalty
- seed where supported
- deterministic mode
- greedy mode
- sampling enabled flag
- logits processors
- banned token IDs
- allowed token IDs where supported
- stop token IDs
- stop text sequences through tokenizer-prepared patterns
- EOS behavior

This change defines parameter structure and validation boundaries.

Detailed sampling and logits processor behavior is defined later by:

```text
define-sampling-and-logits-processing-contract
```

## Prefill Stage

Generation SHALL distinguish prefill from decode.

Prefill consumes the input token sequence to initialize generation state.

Prefill may produce:

- initial logits
- initial hidden execution state
- KV cache state placeholder
- prompt token accounting
- memory usage diagnostics

This change defines prefill as a stage but does not define KV cache internals.

## Decode Stage

Decode produces new tokens iteratively.

A decode step conceptually performs:

```text
current state
    |
    v
model forward for next token logits
    |
    v
logits processing
    |
    v
sampling / selection
    |
    v
next token ID
    |
    v
state update
```

Generation SHALL represent decode progress and completion.

## Stop Conditions

Generation SHALL support explicit stop conditions.

Stop conditions MAY include:

- max new tokens reached
- max total tokens reached
- EOS token reached
- stop token ID reached
- stop token pattern reached
- stop text sequence reached through tokenizer state
- cancellation requested
- memory policy stopped
- runtime shutdown
- Provider failure
- policy refusal

Finish reason SHALL be explicit.

## Finish Reasons

Generation finish reasons SHOULD include:

```text
max-new-tokens
max-total-tokens
eos-token
stop-token
stop-sequence
cancelled
interrupted
length-limit
memory-limit
runtime-shutdown
provider-error
model-error
policy-denied
error
```

The exact serialized names are implementation-defined.

## Streaming

Generation SHALL support streaming token output.

Streaming events MAY include:

- generation-started
- prefill-started
- prefill-completed
- token-generated
- decode-step-completed
- stop-condition-met
- generation-completed
- generation-cancelled
- generation-failed
- usage-updated

Streaming token events SHALL contain token IDs.

Text chunks MAY be produced by Runtime integration with Tokenizer streaming
decode, but Generation itself remains token-based.

## Usage Accounting

Generation SHALL report usage accounting.

Usage SHOULD include:

- prompt token count
- generated token count
- total token count
- prefill duration where available
- decode duration where available
- tokens per second where available
- memory admission result where available
- finish reason

Usage accounting SHALL not require logging raw prompt text.

## Context Window Validation

Generation SHALL validate context length.

Context validation SHALL consider:

- prompt token count
- max new tokens
- model context length
- tokenizer model max length
- runtime policy
- truncation policy already applied by tokenizer stage
- KV cache capacity where known

If the request exceeds limits, Generation SHALL fail before execution unless
explicit policy allows truncation or reduction.

Generation SHALL not silently truncate input tokens.

## EOS Behavior

EOS behavior SHALL be explicit.

Policy MAY include:

- stop on EOS
- ignore EOS
- include EOS in output tokens
- exclude EOS from output tokens
- stop only on configured EOS IDs
- allow multiple EOS IDs

Generation SHALL not assume every model has exactly one EOS token.

## Stop Text Sequences

Stop text sequences require tokenizer participation.

Generation MAY consume tokenizer-prepared stop token patterns or streaming decode
state to detect stop text.

The Tokenizer Contract owns text/token mapping.

Generation owns stopping behavior.

## Determinism

Generation MAY support deterministic mode.

If a seed is provided and deterministic mode is supported, repeated generation
with the same inputs, model state, parameters, and Provider behavior SHOULD be
reproducible according to the declared determinism level.

If determinism is unsupported, Runtime SHALL report that explicitly when
requested.

## Cancellation

Generation SHALL support cancellation semantics.

Cancellation MAY occur:

- before prefill
- during prefill
- between decode steps
- during decode step if Provider supports interruption
- after completion

Cancellation SHALL produce a stable finish reason or error.

Cancellation SHALL release or preserve resources according to Runtime policy.

## Memory Manager Relationship

Generation SHALL use Memory Manager for memory feasibility and admission.

Memory-related needs MAY include:

- input token buffers
- output token buffers
- temporary logits buffers
- sampling workspace
- prefill workspace
- decode workspace
- future KV cache
- future prefix cache
- batch collation buffers

If memory admission fails, generation SHALL fail, queue, or retry according to
Runtime policy.

## Provider Relationship

Generation SHALL not select Providers directly from user input.

Runtime Resolution and Planning select Providers and Devices for the underlying
model execution.

Generation may require Capabilities such as Compute or future Generation
Capabilities.

Provider and Device selection remain Runtime-owned.

## Model Relationship

Generation SHALL require a validated and loaded model context or model
reference.

A Model Artifact alone is not necessarily executable.

A future Model Instance contract defines loaded model lifecycle.

This change allows generation requests to reference a model context while
deferring full Model Instance semantics.

## Tokenizer Relationship

Generation SHALL consume tokenizer-validated input token IDs.

Generation SHALL use tokenizer metadata for:

- EOS IDs
- stop token preparation
- model max length
- special token compatibility
- streaming decode integration

Generation SHALL not own tokenizer artifact validation.

## Browser Relationship

Generation Contract SHALL be platform-neutral.

Browser targets MAY support a subset of generation features depending on:

- available Component Engine
- available Provider model
- Memory Manager constraints
- browser execution limits
- WebGPU or future browser execution support

Unavailable features SHALL be reported explicitly.

## Error Model

Generation errors SHALL be structured.

Error categories SHOULD include:

- model not loaded
- model artifact invalid
- model instance unavailable
- tokenizer incompatible
- input tokens invalid
- prompt too long
- max tokens invalid
- parameter invalid
- stop condition invalid
- deterministic mode unsupported
- sampling mode unsupported
- logits processor unsupported
- memory admission failed
- provider resolution failed
- provider execution failed
- provider not ready
- provider saturated
- cancellation requested
- cancellation unsupported
- streaming consumer failed
- runtime shutdown
- generation interrupted
- internal generation error

## Observability

Runtime SHOULD emit observations for:

- generation requested
- generation admitted
- generation rejected
- prefill started
- prefill completed
- decode started
- token generated
- stop condition met
- generation completed
- generation cancelled
- generation failed
- memory admission failed
- Provider execution failed
- streaming backpressure
- usage reported

Observability SHALL not log raw prompts by default.

Token IDs and text chunks SHALL be redacted or policy-controlled when needed.

## Non-Goals

This change does not:

- define full KV cache model
- define prefix cache model
- define continuous batching
- define full sampling implementation
- define logits processor registry
- define model loading lifecycle fully
- define Model Instance lifecycle fully
- define chat template rendering
- define client agent orchestration
- define filesystem access
- define Git access
- define network access
- define secret access
- define remote generation protocol
- define Tachyon distributed routing
- require GPU hardware
- require browser generation implementation

## Impact

Magnetar gains a stable Generation Contract.

The inference pipeline becomes:

```text
Model Artifact / future Model Instance
        |
        v
Tokenizer.encode
        |
        v
GenerationRequest
        |
        v
Generation Runtime
        |
        +-- prefill
        +-- decode loop
        +-- stop conditions
        +-- streaming token events
        |
        v
Tokenizer.streaming_decode
        |
        v
client-visible text
```

This prepares later changes:

- inference session model
- KV cache model
- model loading contract
- sampling and logits processing contract
- continuous batching contract