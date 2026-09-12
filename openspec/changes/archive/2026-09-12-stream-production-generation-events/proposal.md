## Why

The production Qwen generation entry point family only returns a final `FirstNativeFixtureGeneration` after every decode step has already run. An embedder (Tachyon) implementing an OpenAI-shaped `"stream": true` request today has exactly one honest option through this surface: wait for the entire generation, then fake incremental delivery by chopping up the final text -- which collapses time-to-first-token into total generation time and moves a Runtime responsibility (incremental, byte-safe tokenizer decode) into a layer that has no business reimplementing it. The Runtime already tracks per-token progress internally (`InferenceApiObservationKind::TokenGenerated` fires once per decode step inside the generation loop) and already has an incremental, UTF-8-boundary-safe tokenizer decode primitive (`Tokenizer::streaming_decode`/`StreamingDecodeState`) -- neither is exposed live to a caller. Closing this is the second and final concrete integration gap a Tachyon-authored review of the OpenAI-compatible surface found (the first, caller-supplied generation parameters/stop conditions, closed in `expose-production-generation-parameters`).

## What Changes

- Add a minimal, tokenizer-agnostic per-token hook to the Runtime Inference API's generation loop (`inference_api.rs`): an optional callback invoked once per produced token, immediately after it is committed, that can request early, clean termination (reusing the existing `FinishReason::Cancelled` path) -- exposed additively; every existing caller of `run_generation_loop_with_execution_plans`/`run_generation_loop_inner` passes no callback and is unaffected.
- Add `run_production_qwen_generation_for_provider_streaming`, a new production entry point that wraps the above with the real tokenizer's incremental decode (`streaming_decode`), translating each produced token into an ordered `GenerationStreamEvent::Token { token_id, text_delta }` delivered to a caller-supplied `on_event` callback as it happens, followed by exactly one `GenerationStreamEvent::Finished { finish_reason, usage }` once decode completes (or is cancelled).
- `on_event` returning a cancellation signal stops generation cleanly (no panic, no partial/garbled output, resources released the same way any other generation completion releases them) rather than propagating a hard error.
- Every existing production entry point (`run_production_qwen_generation(_for_provider)(_with_prompt)(_with_request)`) is unaffected -- the streaming variant is additive, not a replacement.

## Capabilities

### New Capabilities
(none -- this extends `production-generation-api`, added by `expose-production-generation-parameters`)

### Modified Capabilities
- `production-generation-api`: adds incremental, per-token streaming delivery alongside the existing final-result and caller-supplied-parameters requirements.

## Impact

- `magnetar-runtime/src/inference_api.rs`: new optional per-token callback parameter threaded through `run_generation_loop_inner`, a new `run_generation_loop_with_execution_plans_streaming` wrapper, and a new `GenerationStreamEvent` type.
- `magnetar-runtime/src/first_native_runtime.rs`: new `run_production_qwen_generation_for_provider_streaming` entry point.
- No breaking changes: every existing public function signature is unchanged, and every existing caller's observable behavior is identical (no callback = no behavior change to the loop itself).
- Tests: ordered stream events for a real multi-token CPU generation, that concatenated streamed text deltas equal the non-streaming final decoded text exactly, that `Finished` is emitted exactly once with the real usage/finish reason, and that a consumer requesting cancellation mid-stream stops generation cleanly with resources released.
