# Define Sampling And Logits Processing Contract

## Why

Magnetar now has contracts for:

- Model Artifacts
- Model Loading
- Tokenizer
- Generation
- Inference Sessions
- KV Cache
- Memory Manager
- Providers and Devices

Generation defines the token production loop.

However, the choice of the next token must be separated from the generation
loop.

The model forward pass produces logits.

Sampling and logits processing transform those logits into a next token
decision.

Without a dedicated contract, sampling behavior may become hidden inside:

- Generation
- Provider execution
- model architecture code
- tokenizer logic
- client policy
- Scheduler
- session state

That would make reproducibility, parameter validation, policy enforcement,
banned tokens, stop token handling, deterministic mode, numerical behavior,
and observability difficult to reason about.

This change defines a stable Sampling and Logits Processing Contract.

## What Changes

This change introduces Sampling as a first-class inference Runtime contract.

Sampling SHALL consume logits or equivalent next-token scores and return a
selected token ID or a structured refusal/error.

The contract SHALL define:

- logits input
- logits processor chain
- sampling parameters
- greedy selection
- temperature scaling
- top-k filtering
- top-p filtering
- min-p / typical-p placeholders
- repetition penalty
- frequency penalty
- presence penalty
- banned token IDs
- allowed token IDs
- required token masks
- stop token masking where applicable
- deterministic seed behavior where supported
- probability metadata where policy permits
- structured sampling errors
- observability

This change defines contract boundaries.

It does not require implementing every sampler immediately.

## Sampling Is Token-Based

Sampling operates on token IDs and token scores.

Sampling SHALL NOT operate on raw text.

Textual constraints must be converted through the Tokenizer Contract before
they affect logits or token selection.

## Sampling Input

A SamplingRequest SHOULD include:

- request ID
- logits or score reference
- vocabulary size
- current step index
- prompt token history or generated token history where needed
- tokenizer metadata reference
- generation parameters
- logits processor configuration
- RNG seed or RNG state where supported
- deterministic mode flag
- allowed token mask
- banned token set
- stop token metadata
- policy metadata
- observability correlation ID

The exact Rust names are implementation-defined.

## Sampling Output

A SamplingResult SHOULD include:

- selected token ID
- selection mode
- token rank where available
- token probability where policy permits
- log probability where policy permits
- finish hint where applicable
- diagnostics
- updated RNG state where applicable

Sampling SHALL not produce decoded text.

## Raw Logits

Raw logits are model output before sampling processors.

Raw logits MAY be represented as:

- Runtime tensor reference
- Provider-owned tensor reference
- host-accessible score buffer
- opaque logits handle
- test fixture vector

Raw logits SHALL not expose raw Provider handles to Components.

Runtime policy controls whether logits can be materialized to host memory.

## Logits Processor Chain

Sampling SHALL support an ordered logits processor chain.

Processors MAY include:

- invalid token masking
- vocabulary range masking
- special token masking
- banned token masking
- allowed token masking
- repetition penalty
- frequency penalty
- presence penalty
- temperature scaling
- top-k filtering
- top-p filtering
- min-p filtering placeholder
- typical-p filtering placeholder
- stop token preparation
- policy filters
- custom inference-scoped processors

Processor order SHALL be deterministic and documented.

A request SHALL not rely on unspecified processor order.

## Processor Ownership

Logits processors are inference-scoped.

They SHALL NOT receive filesystem, network, Git, secrets, workspace, or process
authority.

A processor MAY be implemented as:

- Runtime-native code
- Component-based inference processor
- Provider-assisted operation
- test fixture

If a processor is Component-based, it remains a Component Artifact separate from
sampling configuration and model data.

## Greedy Selection

Greedy selection SHALL choose the highest valid score after logits processing.

Tie handling SHALL be deterministic or explicitly unspecified by policy.

If greedy mode is requested, stochastic sampling parameters SHALL be ignored or
rejected according to policy.

## Temperature

Temperature scaling SHALL be explicit.

Temperature validation SHALL reject invalid values.

A temperature of zero SHALL not ambiguously mean both greedy and divide-by-zero.

Policy SHALL define whether temperature zero maps to greedy mode or is invalid.

## Top-K

Top-k filtering SHALL keep only the highest-k eligible tokens.

Validation SHALL reject invalid k values.

If k exceeds vocabulary size, Runtime policy SHALL define whether it is clamped
or accepted as no-op.

No silent behavior is allowed without policy.

## Top-P

Top-p filtering SHALL keep the smallest eligible token set whose cumulative
probability reaches the threshold.

Validation SHALL reject invalid top-p values.

Top-p behavior SHALL be defined after required previous processors.

## Min-P And Typical-P

The contract MAY reserve min-p and typical-p parameters.

If requested but unsupported, Runtime SHALL return structured unsupported errors.

## Penalties

Sampling SHALL support penalty metadata.

Penalties MAY include:

- repetition penalty
- frequency penalty
- presence penalty

Penalties SHALL be applied using token history.

Penalty behavior SHALL be deterministic for the same input, parameters, and
history.

Invalid penalty values SHALL be rejected.

## Banned And Allowed Tokens

Sampling SHALL support banned token IDs and allowed token IDs.

Banned tokens SHALL be removed from eligibility.

Allowed token IDs, if present, SHALL restrict eligibility to the allowed set.

If both banned and allowed sets are provided, policy SHALL define precedence.

If no valid token remains, sampling SHALL fail with a structured error.

## Special Token Policy

Sampling SHALL respect special token policy.

Special token behavior MAY include:

- allow EOS
- disallow EOS until minimum length
- allow PAD never
- allow BOS only at beginning
- suppress UNK
- suppress additional special tokens
- allow tool-specific tokens only outside Magnetar core if policy permits

Special token metadata comes from the Tokenizer Contract.

## Stop Tokens

Sampling may receive stop token metadata.

Stop token behavior belongs to Generation stop conditions, but sampling may mask
or prefer certain tokens according to policy.

For example:

- EOS may be allowed and then Generation stops
- EOS may be suppressed until minimum token count
- stop token IDs may be allowed but recognized by Generation
- some stop tokens may be forbidden from output

The boundary SHALL be explicit.

## Minimum Length

Sampling MAY support minimum generation length.

If minimum length is active, EOS and certain stop tokens may be masked until the
minimum is reached.

The behavior SHALL be policy-controlled.

## Determinism And RNG

Sampling MAY support deterministic stochastic sampling.

If a seed is provided and deterministic mode is supported, repeated sampling
with the same inputs, parameters, processor order, and RNG state SHOULD produce
the same token selection.

If deterministic behavior cannot be guaranteed, Runtime SHALL report the
declared determinism level.

Provider or hardware nondeterminism SHALL be surfaced in diagnostics where
relevant.

## RNG State

Sampling MAY expose Runtime-owned RNG state.

RNG state SHALL be opaque to clients unless policy allows inspection.

A session may carry RNG state for deterministic continuation.

RNG state SHALL not encode secrets.

## Probability Metadata

Sampling MAY report token probability or log probability.

Probability metadata SHALL be policy-controlled because it can expose model
behavior and increase memory cost.

If probabilities are requested but unavailable, Runtime SHALL return a
structured unsupported error.

## Logits Materialization Policy

Raw logits may be large and sensitive.

Runtime SHALL control whether logits are materialized to host memory.

A sampling implementation may operate:

- on Provider-owned logits
- on Device-resident logits
- through Compute operations
- through host materialized scores
- through a test vector

Materialization through host memory SHALL respect Memory Manager policy and
HostStagingPolicy.

## Memory Manager Relationship

Sampling SHALL use Memory Manager for temporary buffers.

Buffers MAY include:

- logits buffer
- probability buffer
- mask buffer
- sorted token buffer
- top-k workspace
- top-p workspace
- RNG state
- history buffer
- penalty workspace

Sampling SHALL not allocate unbounded memory outside Runtime policy.

## Provider Relationship

Sampling SHALL not select Providers or Devices directly.

If sampling is Provider-assisted, Runtime Resolution determines the Provider and
Device.

Provider-assisted sampling SHALL preserve Resource Affinity for logits tensors.

If logits are Provider-owned, sampling must either execute compatibly on that
Provider/Device or perform explicit authorized data movement.

## Tokenizer Relationship

Tokenizer provides:

- vocabulary size
- special token IDs
- added token metadata
- stop token preparation
- token ID validity

Sampling SHALL use tokenizer metadata but SHALL not decode text.

## Generation Relationship

Generation owns the decode loop and stop behavior.

Sampling owns next-token selection.

Conceptual boundary:

```text
Generation decode step
    |
    v
model forward -> logits
    |
    v
Sampling and logits processing
    |
    v
next token ID
    |
    v
Generation updates state and checks stop condition
```

Generation SHALL call Sampling or equivalent Runtime-owned selection logic.

Sampling SHALL not advance KV cache by itself.

## Session Relationship

An Inference Session may carry default sampling parameters and RNG state.

Session policy may restrict allowed sampling parameters.

Generation requests inside the session SHALL be validated against session
policy.

## Browser Relationship

Sampling contract SHALL be platform-neutral.

Browser targets may implement a reduced feature set.

Unsupported sampling features SHALL return structured errors.

Sampling on browser SHALL not require Wasmtime or native Provider loading.

## Error Model

Sampling errors SHALL be structured.

Error categories SHOULD include:

- logits unavailable
- logits invalid
- vocabulary mismatch
- invalid token ID
- invalid sampling parameter
- temperature invalid
- top-k invalid
- top-p invalid
- min-p unsupported
- typical-p unsupported
- repetition penalty invalid
- frequency penalty invalid
- presence penalty invalid
- banned token invalid
- allowed token invalid
- no eligible token
- deterministic mode unsupported
- RNG unavailable
- probability metadata unsupported
- logits materialization denied
- logits materialization failed
- memory allocation failed
- Provider-assisted sampling unavailable
- Provider execution failed
- Resource Affinity conflict
- tokenizer metadata missing
- processor unsupported
- processor failed
- policy denied
- browser feature unsupported
- internal sampling error

## Observability

Runtime SHOULD emit observations for:

- sampling requested
- processor chain built
- processor applied
- token selected
- sampling failed
- no eligible token
- deterministic seed used
- probability metadata requested
- logits materialization requested
- logits materialization denied
- Provider-assisted sampling used
- memory allocation failed
- policy denied

Observability SHALL not log raw logits by default.

Observability SHALL not log raw prompt text by default.

Token IDs and probabilities SHALL be redacted or policy-controlled where needed.

## Non-Goals

This change does not:

- define full generation loop
- define tokenizer execution
- define KV cache internals
- define continuous batching
- define model architecture implementation
- define client agent behavior
- define safety/moderation policy
- define tool calling
- define grammar-constrained decoding fully
- define JSON schema constrained decoding fully
- define speculative decoding
- define beam search
- require GPU sampling implementation
- require browser sampling implementation
- expose raw logits to clients by default
- allow Components arbitrary access to logits
- allow sampling to select Provider/Device directly

## Impact

Magnetar gains a clean boundary for next-token selection.

The decode loop becomes:

```text
Generation
    |
    v
model forward
    |
    v
raw logits
    |
    v
Sampling / Logits Processing
    |
    v
next token ID
    |
    v
Generation state update
```

This prepares later changes:

- prefix cache model
- continuous batching contract
- constrained decoding contract
- speculative decoding contract
- model serving API