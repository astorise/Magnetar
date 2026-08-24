# Tasks

## 1. API Scope

- [x] Define Runtime Inference API.
- [x] Document allowed inference API areas.
- [x] Document forbidden non-inference API areas.
- [x] Document API boundary with `magnetar-cli`.
- [x] Document API boundary with Tachyon.
- [x] Document non-goals.

## 2. API Module

- [x] Create first-class `inference_api` module or equivalent.
- [x] Export canonical API request/result types from crate root.
- [x] Keep API platform-neutral.
- [x] Keep API independent from direct Provider/Device selection by clients.
- [x] Add module-level documentation.

## 3. Primary Types

- [x] Define model reference type.
- [x] Define model artifact reference type.
- [x] Define model instance reference type.
- [x] Define inference session reference type.
- [x] Define generation request type.
- [x] Define generation handle type.
- [x] Define generation event type.
- [x] Define generation result type.
- [x] Define tokenization request type.
- [x] Define tokenization result type.
- [x] Define streaming handle type.
- [x] Define cancellation token type.
- [x] Define runtime diagnostics type.
- [x] Define usage report type.
- [x] Ensure all IDs are opaque.
- [x] Add type tests.

## 4. Handle Safety

- [x] Prevent Provider handle exposure.
- [x] Prevent Device handle exposure.
- [x] Prevent Kernel handle exposure.
- [x] Prevent raw tensor pointer exposure.
- [x] Prevent memory pointer exposure.
- [x] Prevent raw KV cache exposure.
- [x] Prevent raw model weight exposure.
- [x] Prevent Provider-owned opaque resource exposure.
- [x] Add handle safety tests.

## 5. Model Resolution API

- [x] Define model resolution request.
- [x] Define model resolution result.
- [x] Support local Runtime registry.
- [x] Support client-provided artifact reference.
- [x] Support trusted cache reference.
- [x] Support development fixture.
- [x] Support future external source placeholder.
- [x] Support future Tachyon source placeholder.
- [x] Prevent arbitrary filesystem access.
- [x] Prevent arbitrary network access.
- [x] Add model resolution tests.

## 6. Model Loading API

- [x] Define model loading request.
- [x] Include model reference.
- [x] Include optional tokenizer reference.
- [x] Include optional adapter references.
- [x] Include target usage.
- [x] Include dtype policy.
- [x] Include layout policy.
- [x] Include memory budget.
- [x] Include cache policy.
- [x] Include Provider policy preferences as non-authoritative.
- [x] Include timeout.
- [x] Include observability correlation.
- [x] Add loading API tests.

## 7. Model Instance API

- [x] Define create instance operation.
- [x] Define inspect instance operation.
- [x] Define warm instance operation.
- [x] Define suspend instance operation.
- [x] Define resume instance operation.
- [x] Define drain instance operation.
- [x] Define unload instance operation.
- [x] Validate active sessions/generations.
- [x] Validate memory pressure.
- [x] Validate Provider/Device readiness.
- [x] Validate policy.
- [x] Add Model Instance API tests.

## 8. Session API

- [x] Define session creation request.
- [x] Include Model Instance reference or model reference.
- [x] Include tokenizer reference or policy.
- [x] Include generation defaults override.
- [x] Include memory budget.
- [x] Include KV cache policy.
- [x] Include Prefix Cache policy.
- [x] Include adapter activation policy.
- [x] Include streaming policy.
- [x] Include cancellation policy.
- [x] Include timeout.
- [x] Include idle TTL.
- [x] Include privacy/redaction policy.
- [x] Include observability correlation.
- [x] Define session close operation.
- [x] Add Session API tests.

## 9. One-Shot Inference

- [x] Define one-shot inference request.
- [x] Model one-shot as implicit session.
- [x] Ensure one-shot does not bypass Model Instance.
- [x] Ensure one-shot does not bypass Tokenizer.
- [x] Ensure one-shot does not bypass Generation.
- [x] Ensure one-shot does not bypass Sampling.
- [x] Ensure one-shot does not bypass Memory Manager.
- [x] Ensure one-shot does not bypass Provider/Kernel contracts.
- [x] Add one-shot tests.

## 10. Tokenization API

- [x] Define encode request.
- [x] Define decode request.
- [x] Define streaming decode request.
- [x] Support plain text prompt.
- [x] Support chat messages.
- [x] Support already-tokenized input.
- [x] Support test token sequence.
- [x] Apply chat template only through authorized contract.
- [x] Validate tokenizer/model compatibility.
- [x] Return token usage metadata.
- [x] Disable raw prompt logging by default.
- [x] Add tokenization API tests.

## 11. Prompt Input Boundary

- [x] Accept plain text where allowed.
- [x] Accept chat messages where allowed.
- [x] Accept already-tokenized input.
- [x] Accept test token sequence.
- [x] Prevent external retrieval.
- [x] Prevent workspace scanning.
- [x] Prevent file reading.
- [x] Prevent tool execution.
- [x] Add prompt boundary tests.

## 12. Generation API

- [x] Define generation request.
- [x] Include session or model reference.
- [x] Include input token IDs or prompt input.
- [x] Include generation parameters.
- [x] Include sampling parameters.
- [x] Include stop conditions.
- [x] Include streaming mode.
- [x] Include max new tokens.
- [x] Include max total tokens.
- [x] Include priority.
- [x] Include timeout/deadline.
- [x] Include cancellation token.
- [x] Include observability correlation.
- [x] Include privacy/redaction policy.
- [x] Add generation API tests.

## 13. Streaming API

- [x] Define generation-started event.
- [x] Define prefill-started event.
- [x] Define prefill-completed event.
- [x] Define decode-token event.
- [x] Define decoded-text event.
- [x] Define stop-reached event.
- [x] Define generation-completed event.
- [x] Define generation-failed event.
- [x] Define generation-cancelled event.
- [x] Define usage-updated event.
- [x] Define diagnostic event.
- [x] Define ordering guarantees.
- [x] Prevent raw logits by default.
- [x] Prevent raw KV cache by default.
- [x] Prevent raw tensor values by default.
- [x] Add streaming tests.

## 14. Generation Result

- [x] Include generated token IDs where policy allows.
- [x] Include decoded text where requested.
- [x] Include finish reason.
- [x] Include usage accounting.
- [x] Include timing metadata.
- [x] Include cache usage metadata.
- [x] Include Model Instance metadata.
- [x] Include structured diagnostics.
- [x] Include redaction status.
- [x] Include error information.
- [x] Prevent raw handle exposure.
- [x] Add result tests.

## 15. Cancellation API

- [x] Define cancellation token.
- [x] Support queued generation cancellation.
- [x] Support tokenization cancellation where applicable.
- [x] Support prefill cancellation.
- [x] Support decode cancellation.
- [x] Support sampling cancellation.
- [x] Support batching cancellation.
- [x] Support graph execution cancellation.
- [x] Support Kernel Dispatch cancellation.
- [x] Report Provider/Kernel cancellation limitations.
- [x] Add cancellation tests.

## 16. Backpressure

- [x] Define accepted admission state.
- [x] Define queued admission state.
- [x] Define rejected admission state.
- [x] Define delayed admission state.
- [x] Define cancelled admission state.
- [x] Define timed-out admission state.
- [x] Expose backpressure metadata.
- [x] Add backpressure tests.

## 17. Adapter Activation API

- [x] Define adapter activation request.
- [x] Support operation scope.
- [x] Support session scope.
- [x] Support model instance scope.
- [x] Validate Model Component compatibility.
- [x] Validate cache compatibility.
- [x] Validate memory budget.
- [x] Validate policy.
- [x] Add adapter API tests.

## 18. KV Cache Policy API

- [x] Define KV cache enabled/disabled policy.
- [x] Define KV cache scope.
- [x] Define KV cache budget.
- [x] Define KV cache reuse policy.
- [x] Define KV cache eviction policy.
- [x] Define KV cache privacy policy.
- [x] Define KV cache persistence placeholder.
- [x] Prevent raw KV cache mutation.
- [x] Add KV cache API tests.

## 19. Prefix Cache Policy API

- [x] Define Prefix Cache enabled/disabled policy.
- [x] Define Prefix Cache scope.
- [x] Define Prefix Cache sharing policy.
- [x] Define Prefix Cache TTL.
- [x] Define Prefix Cache budget.
- [x] Define Prefix Cache privacy policy.
- [x] Define Prefix Cache reuse policy.
- [x] Preserve Runtime-owned matching.
- [x] Add Prefix Cache API tests.

## 20. Diagnostics API

- [x] Define diagnostics request.
- [x] Include model resolution status.
- [x] Include model loading status.
- [x] Include Model Instance status.
- [x] Include Provider readiness summary.
- [x] Include Device readiness summary.
- [x] Include memory pressure summary.
- [x] Include Kernel availability summary.
- [x] Include operator missing summary.
- [x] Include tokenizer compatibility status.
- [x] Include cache status summary.
- [x] Include queue/admission status.
- [x] Apply default redaction.
- [x] Add diagnostics tests.

## 21. Usage Reporting

- [x] Track prompt token count.
- [x] Track generated token count.
- [x] Track total token count.
- [x] Track prefill duration.
- [x] Track decode duration.
- [x] Track tokens per second where available.
- [x] Track cache hit/miss metadata.
- [x] Track memory estimate.
- [x] Track queued time.
- [x] Track cancellation status.
- [x] Prevent raw prompt exposure by default.
- [x] Add usage tests.

## 22. Error Model

- [x] Define inference-api-unavailable error.
- [x] Define model-reference-invalid error.
- [x] Define model-resolution-failed error.
- [x] Define model-loading-failed error.
- [x] Define model-instance-not-ready error.
- [x] Define model-instance-unavailable error.
- [x] Define model-component-unavailable error.
- [x] Define tokenizer-unavailable error.
- [x] Define tokenizer-incompatible error.
- [x] Define tokenization-failed error.
- [x] Define session-creation-failed error.
- [x] Define session-not-found error.
- [x] Define session-closed error.
- [x] Define generation-rejected error.
- [x] Define generation-queued status.
- [x] Define generation-timeout error.
- [x] Define generation-cancelled error.
- [x] Define generation-failed error.
- [x] Define sampling-failed error.
- [x] Define stop-condition-invalid error.
- [x] Define adapter-activation-failed error.
- [x] Define KV-cache-unavailable error.
- [x] Define Prefix-Cache-unavailable error.
- [x] Define memory-admission-failed error.
- [x] Define Provider-unavailable error.
- [x] Define Device-unavailable error.
- [x] Define Kernel-unavailable error.
- [x] Define operator-unsupported error.
- [x] Define graph-planning-failed error.
- [x] Define policy-denied error.
- [x] Define cancellation-unsupported error.
- [x] Define streaming-unavailable error.
- [x] Define streaming-interrupted error.
- [x] Define diagnostics-redacted status.
- [x] Define browser-feature-unsupported error.
- [x] Define internal-inference-api error.

## 23. Observability

- [x] Emit inference request received observation.
- [x] Emit model resolved observation.
- [x] Emit model resolution failed observation.
- [x] Emit model loading requested observation.
- [x] Emit model loaded observation.
- [x] Emit model loading failed observation.
- [x] Emit model instance selected observation.
- [x] Emit session created observation.
- [x] Emit session closed observation.
- [x] Emit prompt tokenized observation.
- [x] Emit tokenization failed observation.
- [x] Emit generation accepted observation.
- [x] Emit generation queued observation.
- [x] Emit generation started observation.
- [x] Emit prefill started observation.
- [x] Emit prefill completed observation.
- [x] Emit decode started observation.
- [x] Emit token generated observation.
- [x] Emit generation completed observation.
- [x] Emit generation failed observation.
- [x] Emit generation cancelled observation.
- [x] Emit adapter activated observation.
- [x] Emit KV cache used observation.
- [x] Emit Prefix Cache hit observation.
- [x] Emit Prefix Cache miss observation.
- [x] Emit memory admission failed observation.
- [x] Emit Provider unavailable observation.
- [x] Emit Kernel unavailable observation.
- [x] Emit stream opened observation.
- [x] Emit stream closed observation.
- [x] Emit stream interrupted observation.
- [x] Avoid raw prompt/weight/tensor/cache/handle/path/secret logging.

Note: `InferenceApiObserver` is wired into the `*_observed` lifecycle
variants (resolve, load, create instance, create/close session, tokenize,
submit generation, activate adapter, cancel) and into
`run_generation_loop`, which drives prefill, per-token decode through the
Sampling Contract, and stop/cancellation, emitting the full 31-kind
Streaming/Generation/Cache/Provider/Kernel observation sequence -- each
path has a passing test. `run_generation_loop` takes Provider execution as
an explicit `next_logits` callback boundary (Runtime never computes logits
itself), so real per-token wiring against a concrete Provider is future
work; the orchestration and its observability are exercised end-to-end
today via caller-supplied logits.

## 24. Browser Compatibility

- [x] Keep Runtime Inference API platform-neutral.
- [x] Support reduced browser inference paths.
- [x] Avoid Wasmtime requirement.
- [x] Avoid native Provider loading requirement.
- [x] Avoid arbitrary filesystem access.
- [x] Avoid process execution.
- [x] Avoid shell execution.
- [x] Avoid native mmap requirement.
- [x] Return unsupported browser errors where needed.
- [x] Add wasm32 check where feasible.

## 25. Tachyon Boundary

- [x] Allow Tachyon adapter to call Runtime Inference API.
- [x] Preserve Tachyon ownership of distributed orchestration.
- [x] Preserve Magnetar ownership of local inference.
- [x] Prevent Tachyon bypass of Runtime validation.
- [x] Prevent Tachyon bypass of Model Instance lifecycle.
- [x] Prevent Tachyon bypass of Kernel Registry.
- [x] Prevent Tachyon bypass of Memory Manager.
- [x] Prevent Tachyon bypass of Provider contracts.
- [x] Add Tachyon boundary tests.

## 26. magnetar-cli Boundary

- [x] Allow `magnetar-cli` to call Runtime Inference API.
- [x] Keep workspace ownership in `magnetar-cli`.
- [x] Keep file access ownership in `magnetar-cli`.
- [x] Keep Git ownership in `magnetar-cli`.
- [x] Keep network ownership in `magnetar-cli`.
- [x] Keep secrets ownership in `magnetar-cli`.
- [x] Keep shell/process ownership in `magnetar-cli`.
- [x] Keep tools ownership in `magnetar-cli`.
- [x] Keep agent orchestration ownership in `magnetar-cli`.
- [x] Prevent Runtime API from absorbing CLI responsibilities.
- [x] Add CLI boundary tests.

## 27. Tests

- [x] Test model resolution.
- [x] Test invalid model reference.
- [x] Test explicit model loading.
- [x] Test implicit one-shot loading where policy allows.
- [x] Test Model Instance creation through API.
- [x] Test session creation.
- [x] Test session close.
- [x] Test tokenization.
- [x] Test chat template boundary.
- [x] Test generation accepted.
- [x] Test generation queued.
- [x] Test generation rejected.
- [x] Test streaming event order.
- [x] Test cancellation before dispatch.
- [x] Test cancellation during decode.
- [x] Test Provider cancellation unsupported.
- [x] Test KV cache policy accepted.
- [x] Test Prefix Cache policy accepted.
- [x] Test adapter activation denied when incompatible.
- [x] Test diagnostics redaction.
- [x] Test usage accounting.
- [x] Test no Provider/Device/Kernel handles exposed.
- [x] Test no filesystem access.
- [x] Test no network/tool/Git authority.

## 28. Documentation

- [x] Document Runtime Inference API.
- [x] Document primary types.
- [x] Document model resolution.
- [x] Document model loading.
- [x] Document Model Instance API.
- [x] Document Session API.
- [x] Document one-shot inference.
- [x] Document Tokenization API.
- [x] Document Prompt Input boundary.
- [x] Document Generation API.
- [x] Document Streaming API.
- [x] Document Cancellation API.
- [x] Document Backpressure.
- [x] Document Adapter API.
- [x] Document KV Cache API.
- [x] Document Prefix Cache API.
- [x] Document Diagnostics API.
- [x] Document Usage Reporting.
- [x] Document Tachyon boundary.
- [x] Document `magnetar-cli` boundary.
- [x] Document non-goals.

## 29. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run wasm32 check where feasible.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Runtime Inference API tests.
- [x] Run Model Loading tests.
- [x] Run Model Instance tests.
- [x] Run Session tests.
- [x] Run Tokenizer tests.
- [x] Run Generation tests.
- [x] Run Sampling tests.
- [x] Run KV Cache tests.
- [x] Run Prefix Cache tests.
- [x] Run Tensor tests.
- [x] Run Reference CPU tests.
- [x] Run Observability tests.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify API is inference-only.
- [x] Verify `magnetar-cli` responsibilities remain outside Runtime.
- [x] Verify no raw internal handles are exposed.
