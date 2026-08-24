# Tasks

## 1. Sampling Scope

- [x] Define Sampling as next-token selection from logits.
- [x] Document Sampling versus Generation.
- [x] Document Sampling versus Tokenizer.
- [x] Document Sampling versus Provider execution.
- [x] Document Sampling versus KV cache.
- [x] Document Sampling versus safety/moderation policy.
- [x] Document Sampling versus client agent behavior.

## 2. Sampling Module

- [x] Create first-class `sampling` module or equivalent.
- [x] Export canonical sampling types from crate root.
- [x] Keep sampling platform-neutral.
- [x] Keep sampling independent from direct Provider selection.
- [x] Add module-level documentation.

## 3. Sampling Request

- [x] Define SamplingRequest.
- [x] Include request ID.
- [x] Include logits or score reference.
- [x] Include vocabulary size.
- [x] Include current step index.
- [x] Include token history where needed.
- [x] Include tokenizer metadata reference.
- [x] Include generation parameters.
- [x] Include processor configuration.
- [x] Include RNG seed or state where supported.
- [x] Include deterministic mode flag.
- [x] Include allowed token mask.
- [x] Include banned token set.
- [x] Include stop token metadata.
- [x] Include policy metadata.
- [x] Include observability correlation ID.

## 4. Sampling Result

- [x] Define SamplingResult.
- [x] Include selected token ID.
- [x] Include selection mode.
- [x] Include token rank where available.
- [x] Include token probability where policy permits.
- [x] Include log probability where policy permits.
- [x] Include finish hint where applicable.
- [x] Include diagnostics.
- [x] Include updated RNG state where applicable.
- [x] Ensure decoded text is not part of core SamplingResult.

## 5. Logits Representation

- [x] Define raw logits reference.
- [x] Define Provider-owned logits reference.
- [x] Define Device-resident logits reference.
- [x] Define host materialized logits buffer.
- [x] Define test fixture logits vector.
- [x] Prevent raw Provider handle exposure.
- [x] Define logits shape validation.
- [x] Define vocabulary-size compatibility.
- [x] Add logits representation tests.

## 6. Processor Chain

- [x] Define ordered logits processor chain.
- [x] Define deterministic processor order.
- [x] Define invalid token masking.
- [x] Define vocabulary range masking.
- [x] Define special token masking.
- [x] Define banned token masking.
- [x] Define allowed token masking.
- [x] Define repetition penalty processor.
- [x] Define frequency penalty processor.
- [x] Define presence penalty processor.
- [x] Define temperature processor.
- [x] Define top-k processor.
- [x] Define top-p processor.
- [x] Define min-p placeholder.
- [x] Define typical-p placeholder.
- [x] Define policy filter placeholder.
- [x] Add processor order tests.

## 7. Processor Ownership

- [x] Ensure processors are inference-scoped.
- [x] Prevent filesystem authority.
- [x] Prevent network authority.
- [x] Prevent Git authority.
- [x] Prevent secrets authority.
- [x] Prevent workspace authority.
- [x] Prevent process authority.
- [x] Allow Runtime-native processors.
- [x] Allow Component-based processors.
- [x] Allow Provider-assisted processors.
- [x] Add processor authority tests.

## 8. Greedy Selection

- [x] Define greedy selection.
- [x] Define tie behavior.
- [x] Define interaction with sampling flag.
- [x] Define interaction with temperature zero.
- [x] Add greedy tests.

## 9. Temperature

- [x] Define valid temperature range.
- [x] Define temperature scaling semantics.
- [x] Define temperature zero policy.
- [x] Reject invalid temperature.
- [x] Add temperature tests.

## 10. Top-K

- [x] Define top-k parameter.
- [x] Validate top-k.
- [x] Define behavior when k exceeds vocabulary size.
- [x] Apply top-k after required earlier processors.
- [x] Add top-k tests.

## 11. Top-P

- [x] Define top-p parameter.
- [x] Validate top-p.
- [x] Define cumulative probability behavior.
- [x] Apply top-p after required earlier processors.
- [x] Add top-p tests.

## 12. Min-P And Typical-P

- [x] Reserve min-p parameter.
- [x] Reserve typical-p parameter.
- [x] Return unsupported error when requested but unavailable.
- [x] Add unsupported tests.

## 13. Penalties

- [x] Define repetition penalty.
- [x] Define frequency penalty.
- [x] Define presence penalty.
- [x] Validate penalty values.
- [x] Apply penalties using token history.
- [x] Ensure deterministic penalty behavior.
- [x] Add penalty tests.

## 14. Banned And Allowed Tokens

- [x] Define banned token IDs.
- [x] Define allowed token IDs.
- [x] Validate token IDs against tokenizer metadata.
- [x] Define precedence when both are provided.
- [x] Detect no eligible token.
- [x] Add banned token tests.
- [x] Add allowed token tests.
- [x] Add no eligible token tests.

## 15. Special Token Policy

- [x] Use tokenizer special token metadata.
- [x] Define EOS allowance.
- [x] Define PAD suppression.
- [x] Define BOS beginning-only policy.
- [x] Define UNK suppression policy.
- [x] Define additional special token behavior.
- [x] Add special token policy tests.

## 16. Stop Token Handling

- [x] Accept stop token metadata from Generation.
- [x] Define EOS suppression until minimum length where policy requires.
- [x] Define stop token mask behavior.
- [x] Keep stop decision in Generation.
- [x] Add stop token sampling tests.

## 17. Minimum Length

- [x] Define minimum generated token count.
- [x] Mask EOS before minimum length where policy requires.
- [x] Mask stop tokens before minimum length where policy requires.
- [x] Add minimum length tests.

## 18. Determinism

- [x] Define deterministic mode.
- [x] Define seed.
- [x] Define RNG state.
- [x] Define deterministic support declaration.
- [x] Define unsupported determinism error.
- [x] Define nondeterminism diagnostics.
- [x] Add deterministic fixture tests.
- [x] Add unsupported deterministic tests.

## 19. RNG State

- [x] Define Runtime-owned RNG state.
- [x] Keep RNG state opaque unless policy permits.
- [x] Allow session-carried RNG state.
- [x] Ensure RNG state does not encode secrets.
- [x] Add RNG state tests.

## 20. Probability Metadata

- [x] Define probability output.
- [x] Define log probability output.
- [x] Define token rank output.
- [x] Gate probability metadata by policy.
- [x] Return unsupported error when unavailable.
- [x] Add probability metadata tests.

## 21. Logits Materialization

- [x] Define logits materialization policy.
- [x] Allow Provider-owned logits processing.
- [x] Allow Device-resident logits processing.
- [x] Allow host materialized logits processing.
- [x] Enforce Memory Manager policy.
- [x] Enforce HostStagingPolicy.
- [x] Reject denied materialization.
- [x] Add materialization tests.

## 22. Memory Manager Integration

- [x] Account for logits buffer.
- [x] Account for probability buffer.
- [x] Account for mask buffer.
- [x] Account for sorted token buffer.
- [x] Account for top-k workspace.
- [x] Account for top-p workspace.
- [x] Account for RNG state.
- [x] Account for history buffer.
- [x] Account for penalty workspace.
- [x] Add memory integration tests.

## 23. Provider Integration

- [x] Allow Provider-assisted sampling where advertised.
- [x] Preserve Resource Affinity for logits.
- [x] Reject incompatible Provider/Device movement.
- [x] Use Runtime Resolution for Provider-assisted sampling.
- [x] Map Provider sampling errors.
- [x] Add Provider integration tests.

## 24. Tokenizer Integration

- [x] Use vocabulary size.
- [x] Use token ID range.
- [x] Use special token IDs.
- [x] Use added token metadata.
- [x] Use stop token preparation metadata.
- [x] Reject tokenizer metadata missing.
- [x] Add tokenizer integration tests.

## 25. Generation Integration

- [x] Have Generation call Sampling for next token.
- [x] Keep Generation responsible for decode loop.
- [x] Keep Generation responsible for stop conditions.
- [x] Prevent Sampling from advancing KV cache.
- [x] Add generation integration tests.

## 26. Session Integration

- [x] Allow session default sampling parameters.
- [x] Allow session RNG state.
- [x] Enforce session parameter policy.
- [x] Reject disallowed sampling modes.
- [x] Add session integration tests.

## 27. Browser Compatibility

- [x] Keep sampling contract platform-neutral.
- [x] Avoid Wasmtime dependency.
- [x] Avoid native Provider loading requirement.
- [x] Report unsupported browser features.
- [x] Add wasm32 check where feasible.

## 28. Error Model

- [x] Define logits-unavailable error.
- [x] Define logits-invalid error.
- [x] Define vocabulary-mismatch error.
- [x] Define invalid-token-id error.
- [x] Define invalid-sampling-parameter error.
- [x] Define temperature-invalid error.
- [x] Define top-k-invalid error.
- [x] Define top-p-invalid error.
- [x] Define min-p-unsupported error.
- [x] Define typical-p-unsupported error.
- [x] Define repetition-penalty-invalid error.
- [x] Define frequency-penalty-invalid error.
- [x] Define presence-penalty-invalid error.
- [x] Define banned-token-invalid error.
- [x] Define allowed-token-invalid error.
- [x] Define no-eligible-token error.
- [x] Define deterministic-mode-unsupported error.
- [x] Define RNG-unavailable error.
- [x] Define probability-metadata-unsupported error.
- [x] Define logits-materialization-denied error.
- [x] Define logits-materialization-failed error.
- [x] Define memory-allocation-failed error.
- [x] Define Provider-assisted-sampling-unavailable error.
- [x] Define Provider-execution-failed error.
- [x] Define Resource-Affinity-conflict error.
- [x] Define tokenizer-metadata-missing error.
- [x] Define processor-unsupported error.
- [x] Define processor-failed error.
- [x] Define policy-denied error.
- [x] Define browser-feature-unsupported error.
- [x] Define internal-sampling error.

## 29. Observability

- [x] Emit sampling requested observation.
- [x] Emit processor chain built observation.
- [x] Emit processor applied observation.
- [x] Emit token selected observation.
- [x] Emit sampling failed observation.
- [x] Emit no eligible token observation.
- [x] Emit deterministic seed used observation.
- [x] Emit probability metadata requested observation.
- [x] Emit logits materialization requested observation.
- [x] Emit logits materialization denied observation.
- [x] Emit Provider-assisted sampling used observation.
- [x] Emit memory allocation failed observation.
- [x] Emit policy denied observation.
- [x] Avoid raw logits logging by default.
- [x] Avoid raw prompt logging by default.

## 30. Tests

- [x] Test greedy selection.
- [x] Test invalid temperature.
- [x] Test temperature zero policy.
- [x] Test top-k filtering.
- [x] Test invalid top-k.
- [x] Test top-p filtering.
- [x] Test invalid top-p.
- [x] Test min-p unsupported.
- [x] Test typical-p unsupported.
- [x] Test repetition penalty.
- [x] Test frequency penalty.
- [x] Test presence penalty.
- [x] Test banned tokens.
- [x] Test allowed tokens.
- [x] Test no eligible token.
- [x] Test special token masking.
- [x] Test minimum length EOS masking.
- [x] Test deterministic seed fixture.
- [x] Test deterministic unsupported.
- [x] Test probability metadata disabled by policy.
- [x] Test logits materialization denied.
- [x] Test Resource Affinity conflict.
- [x] Test Sampling does not decode text.
- [x] Test Sampling does not advance KV cache.
- [x] Test Sampling does not select Provider/Device directly.
- [x] Test raw logits not logged by default.

## 31. Documentation

- [x] Document Sampling Contract.
- [x] Document logits processor chain.
- [x] Document processor order.
- [x] Document greedy mode.
- [x] Document temperature.
- [x] Document top-k.
- [x] Document top-p.
- [x] Document penalties.
- [x] Document banned/allowed tokens.
- [x] Document special token policy.
- [x] Document determinism.
- [x] Document RNG state.
- [x] Document probability metadata.
- [x] Document logits materialization policy.
- [x] Document Generation boundary.
- [x] Document Tokenizer boundary.
- [x] Document Memory Manager relationship.
- [x] Document Provider relationship.
- [x] Document browser compatibility.
- [x] Document non-goals.

## 32. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run wasm32 check where feasible.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Sampling tests.
- [x] Run Generation tests.
- [x] Run Tokenizer tests.
- [x] Run Memory Manager tests where impacted.
- [x] Run Provider conformance tests where impacted.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify Sampling is token/logits based.
- [x] Verify Sampling does not decode text.
- [x] Verify Sampling does not advance KV cache.
- [x] Verify Sampling does not select Provider/Device directly.
- [x] Verify raw logits and prompts are not logged by default.