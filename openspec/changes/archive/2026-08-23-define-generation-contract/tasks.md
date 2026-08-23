# Tasks

## 1. Generation Scope

- [x] Define Generation as token-based inference Runtime behavior.
- [x] Document Generation versus Tokenizer.
- [x] Document Generation versus Sampling.
- [x] Document Generation versus KV cache.
- [x] Document Generation versus Provider.
- [x] Document Generation versus Model Artifact.
- [x] Document Generation versus client chat/message rendering.

## 2. Generation Module

- [x] Create first-class `generation` module or equivalent.
- [x] Export canonical generation types from crate root.
- [x] Keep generation platform-neutral.
- [x] Keep generation independent from direct Provider selection.
- [x] Add module-level documentation.

## 3. Generation Request

- [x] Define GenerationRequest.
- [x] Include request ID.
- [x] Include model reference.
- [x] Include tokenizer reference.
- [x] Include input token IDs.
- [x] Include prompt token count.
- [x] Include max new tokens.
- [x] Include max total tokens where applicable.
- [x] Include generation parameters.
- [x] Include stop conditions.
- [x] Include streaming mode.
- [x] Include priority metadata.
- [x] Include cancellation metadata.
- [x] Include observability correlation ID.

## 4. Generation Output

- [x] Define GenerationOutput.
- [x] Include generated token IDs.
- [x] Include generated token count.
- [x] Include finish reason.
- [x] Include usage accounting.
- [x] Include diagnostics.
- [x] Keep decoded text outside core Generation output unless produced through
      tokenizer integration.
- [x] Add output tests.

## 5. Generation Parameters

- [x] Define temperature.
- [x] Define top-p.
- [x] Define top-k.
- [x] Define min-p placeholder.
- [x] Define typical-p placeholder.
- [x] Define repetition penalty.
- [x] Define frequency penalty.
- [x] Define presence penalty.
- [x] Define seed.
- [x] Define deterministic mode.
- [x] Define greedy mode.
- [x] Define sampling enabled flag.
- [x] Define banned token IDs.
- [x] Define allowed token IDs placeholder.
- [x] Define logits processor references placeholder.
- [x] Validate parameters.
- [x] Add parameter validation tests.

## 6. Prefill Stage

- [x] Define prefill stage.
- [x] Validate input tokens before prefill.
- [x] Record prompt token count.
- [x] Prepare model execution state placeholder.
- [x] Prepare KV cache placeholder without defining internals.
- [x] Emit prefill observations.
- [x] Add prefill tests.

## 7. Decode Stage

- [x] Define decode stage.
- [x] Define decode step.
- [x] Define next-token logits boundary.
- [x] Define sampling/logits processing boundary.
- [x] Define next token output.
- [x] Define state update placeholder.
- [x] Emit decode observations.
- [x] Add decode tests with mock model executor.

## 8. Stop Conditions

- [x] Define max-new-tokens stop.
- [x] Define max-total-tokens stop.
- [x] Define EOS stop.
- [x] Define stop token ID.
- [x] Define stop token pattern.
- [x] Define stop text sequence through tokenizer-prepared patterns.
- [x] Define cancellation stop.
- [x] Define memory policy stop.
- [x] Define runtime shutdown stop.
- [x] Add stop condition tests.

## 9. Finish Reasons

- [x] Define max-new-tokens finish reason.
- [x] Define max-total-tokens finish reason.
- [x] Define EOS-token finish reason.
- [x] Define stop-token finish reason.
- [x] Define stop-sequence finish reason.
- [x] Define cancelled finish reason.
- [x] Define interrupted finish reason.
- [x] Define length-limit finish reason.
- [x] Define memory-limit finish reason.
- [x] Define runtime-shutdown finish reason.
- [x] Define provider-error finish reason.
- [x] Define model-error finish reason.
- [x] Define policy-denied finish reason.
- [x] Define generic error finish reason.
- [x] Add finish reason tests.

## 10. Streaming Events

- [x] Define generation-started event.
- [x] Define prefill-started event.
- [x] Define prefill-completed event.
- [x] Define token-generated event.
- [x] Define decode-step-completed event.
- [x] Define stop-condition-met event.
- [x] Define generation-completed event.
- [x] Define generation-cancelled event.
- [x] Define generation-failed event.
- [x] Define usage-updated event.
- [x] Add streaming event tests.

## 11. Token Streaming

- [x] Stream generated token IDs.
- [x] Preserve token order.
- [x] Preserve request identity.
- [x] Include token index.
- [x] Include optional token probability only if policy permits.
- [x] Avoid decoded text as core Generation responsibility.
- [x] Add token streaming tests.

## 12. Tokenizer Streaming Integration

- [x] Integrate generated token stream with Tokenizer streaming decode.
- [x] Preserve pending partial decode state.
- [x] Emit valid text chunks only through tokenizer decode.
- [x] Handle stop text sequence detection with tokenizer support.
- [x] Add tokenizer/generation streaming integration tests.

## 13. Usage Accounting

- [x] Count prompt tokens.
- [x] Count generated tokens.
- [x] Count total tokens.
- [x] Record prefill duration where available.
- [x] Record decode duration where available.
- [x] Record tokens per second where available.
- [x] Record finish reason.
- [x] Avoid raw prompt logging.
- [x] Add usage tests.

## 14. Context Window Validation

- [x] Validate prompt token count.
- [x] Validate max new tokens.
- [x] Validate max total tokens.
- [x] Validate model context length.
- [x] Validate tokenizer model max length.
- [x] Validate runtime policy.
- [x] Reject over-limit request unless policy allows reduction.
- [x] Do not silently truncate.
- [x] Add context window tests.

## 15. EOS Behavior

- [x] Define stop-on-EOS policy.
- [x] Define ignore-EOS policy.
- [x] Define include-EOS-output policy.
- [x] Define exclude-EOS-output policy.
- [x] Support multiple EOS IDs.
- [x] Validate EOS token IDs against tokenizer metadata.
- [x] Add EOS tests.

## 16. Stop Text Sequences

- [x] Accept stop text sequences as request metadata.
- [x] Ask tokenizer to prepare stop token patterns where feasible.
- [x] Track stop sequence detection during decode.
- [x] Distinguish token stop from text stop.
- [x] Add stop sequence tests.

## 17. Determinism

- [x] Define deterministic mode request.
- [x] Define seed field.
- [x] Define unsupported determinism error.
- [x] Define determinism diagnostic.
- [x] Add deterministic mock tests.
- [x] Add unsupported determinism tests.

## 18. Cancellation

- [x] Define cancellation request identity.
- [x] Support cancellation before prefill.
- [x] Support cancellation during prefill where feasible.
- [x] Support cancellation between decode steps.
- [x] Support cancellation during Provider execution where supported.
- [x] Support cancellation after completion as no-op or stable result.
- [x] Release or preserve resources according to policy.
- [x] Add cancellation tests.

## 19. Memory Manager Integration

- [x] Request memory admission before generation.
- [x] Account for input token buffers.
- [x] Account for output token buffers.
- [x] Account for logits buffers.
- [x] Account for sampling workspace.
- [x] Account for prefill workspace.
- [x] Account for decode workspace.
- [x] Prepare KV cache placeholder memory requirement.
- [x] Prepare prefix cache placeholder memory requirement.
- [x] Add memory admission tests.

## 20. Provider And Planning Integration

- [x] Use Runtime Resolution for Provider selection.
- [x] Use Planning for execution plan.
- [x] Use Provider execution through Runtime-owned path.
- [x] Prevent GenerationRequest from selecting Provider directly.
- [x] Prevent GenerationRequest from selecting Device directly.
- [x] Add tests rejecting direct Provider/Device selection.

## 21. Model Relationship

- [x] Require validated model reference.
- [x] Prepare for Model Instance reference.
- [x] Reject generation when model is not loaded or unavailable.
- [x] Validate model architecture supports generation.
- [x] Validate generation parameters against model metadata.
- [x] Add model relationship tests.

## 22. Tokenizer Relationship

- [x] Require tokenizer compatibility.
- [x] Validate input token IDs.
- [x] Use tokenizer metadata for EOS.
- [x] Use tokenizer metadata for stop token preparation.
- [x] Use tokenizer metadata for max length where available.
- [x] Add tokenizer relationship tests.

## 23. Browser Compatibility

- [x] Keep Generation Contract platform-neutral.
- [x] Report unsupported features explicitly on browser targets.
- [x] Avoid Wasmtime dependency in Generation Contract.
- [x] Avoid native Provider loading requirement in Generation Contract.
- [x] Add wasm32 compile check where feasible.

## 24. Error Model

- [x] Define model-not-loaded error.
- [x] Define model-artifact-invalid error.
- [x] Define model-instance-unavailable error.
- [x] Define tokenizer-incompatible error.
- [x] Define input-tokens-invalid error.
- [x] Define prompt-too-long error.
- [x] Define max-tokens-invalid error.
- [x] Define parameter-invalid error.
- [x] Define stop-condition-invalid error.
- [x] Define deterministic-mode-unsupported error.
- [x] Define sampling-mode-unsupported error.
- [x] Define logits-processor-unsupported error.
- [x] Define memory-admission-failed error.
- [x] Define provider-resolution-failed error.
- [x] Define provider-execution-failed error.
- [x] Define provider-not-ready error.
- [x] Define provider-saturated error.
- [x] Define cancellation-requested result/error.
- [x] Define cancellation-unsupported error.
- [x] Define streaming-consumer-failed error.
- [x] Define runtime-shutdown error.
- [x] Define generation-interrupted error.
- [x] Define internal-generation error.

## 25. Observability

- [x] Emit generation requested observation.
- [x] Emit generation admitted observation.
- [x] Emit generation rejected observation.
- [x] Emit prefill started observation.
- [x] Emit prefill completed observation.
- [x] Emit decode started observation.
- [x] Emit token generated observation.
- [x] Emit stop condition met observation.
- [x] Emit generation completed observation.
- [x] Emit generation cancelled observation.
- [x] Emit generation failed observation.
- [x] Emit memory admission failed observation.
- [x] Emit Provider execution failed observation.
- [x] Emit streaming backpressure observation.
- [x] Emit usage reported observation.
- [x] Avoid raw prompt logging by default.

## 26. Tests

- [x] Test valid GenerationRequest.
- [x] Test invalid input tokens.
- [x] Test prompt too long.
- [x] Test invalid max new tokens.
- [x] Test invalid temperature.
- [x] Test greedy generation parameters.
- [x] Test sampling parameter validation.
- [x] Test EOS stop.
- [x] Test max-new-tokens stop.
- [x] Test stop-token stop.
- [x] Test stop-sequence stop with tokenizer fixture.
- [x] Test streaming token order.
- [x] Test usage accounting.
- [x] Test cancellation before prefill.
- [x] Test cancellation during decode.
- [x] Test memory admission failure.
- [x] Test Provider resolution failure.
- [x] Test Provider execution failure.
- [x] Test direct Provider/Device selection impossible.
- [x] Test raw prompt not logged by default.

## 27. Documentation

- [x] Document Generation Contract.
- [x] Document input tokens.
- [x] Document output tokens.
- [x] Document prefill.
- [x] Document decode.
- [x] Document streaming events.
- [x] Document stop conditions.
- [x] Document finish reasons.
- [x] Document generation parameters.
- [x] Document tokenizer boundary.
- [x] Document sampling boundary.
- [x] Document KV cache placeholder.
- [x] Document Memory Manager relationship.
- [x] Document Provider relationship.
- [x] Document browser compatibility.
- [x] Document non-goals.

## 28. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run wasm32 check where feasible.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Generation tests.
- [x] Run Tokenizer tests.
- [x] Run Model Artifact tests.
- [x] Run Memory Manager tests where impacted.
- [x] Run Provider conformance tests where impacted.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify Generation is token-based.
- [x] Verify Generation does not own tokenization.
- [x] Verify Generation does not select Provider/Device directly.
- [x] Verify raw prompts are not logged by default.