## 1. Generic per-token hook in the generation loop

- [x] 1.1 Added `on_token: Option<&mut dyn FnMut(TokenId) -> std::ops::ControlFlow<()>>` to `run_generation_loop_inner` in `inference_api.rs`, invoked immediately after `generated.push(step.token_id)` and before the existing `step.finish_reason` break check.
- [x] 1.2 On `ControlFlow::Break(())`, the loop breaks with `FinishReason::Cancelled` (reusing the existing cancellation observations/path).
- [x] 1.3 `run_generation_loop`/`run_generation_loop_with_execution_plans` (every existing caller) pass `None`; added `run_generation_loop_with_execution_plans_streaming` accepting the callback and passing `Some(...)`.
- [x] 1.4 Confirmed every pre-existing call site compiles unchanged and every pre-existing test still passes (1250 + 173 `magnetar-runtime` tests, unaffected).

## 2. `GenerationStreamEvent` and the streaming production entry point

- [x] 2.1 Added `GenerationStreamEvent { Token { token_id, text_delta }, Finished { finish_reason, usage } }` in `first_native_runtime.rs`.
- [x] 2.2 Added `run_production_qwen_generation_for_provider_streaming`, sharing all setup with `run_production_qwen_generation_for_provider_with_request` through a new shared `prepare_production_generation` helper (extracted from the previously-duplicated setup, also used to build `PreparedProductionGeneration`) and a new shared `finish_production_generation` tail helper -- both non-streaming and streaming entry points now call the same two helpers, eliminating the risk of the two diverging.
- [x] 2.3 **Revised from the original plan** (see design.md): the per-token text delta is NOT computed via `Tokenizer::streaming_decode`/`StreamingDecodeState` -- implementing that first and testing it against a real ingested tokenizer revealed `HuggingFaceTokenizer::decode` (the tokenizer that actually backs production checkpoints) ignores `streaming_state` entirely, producing wrong text (`"!chi!"` instead of `"! c hi !"` for the same four tokens) when decoded one token at a time. Fixed by tracking the full `generated_so_far` token list and `previously_emitted_text`, re-decoding the whole accumulated sequence each step via the existing `decode_tokens_streaming`, and computing the delta as `full_text.strip_prefix(&previously_emitted_text)` -- correct for any `Tokenizer` implementation regardless of whether it carries genuine incremental state.
- [x] 2.4 No separate flush step needed (see design.md): because every step re-decodes the full sequence, the last `Token` event already contains everything a flush would have; `Finished { finish_reason, usage }` is delivered immediately after.
- [x] 2.5 Post-loop behavior (final `decoded_text`, session close, model instance unload) is the shared `finish_production_generation` helper, called identically by both entry points, so the streaming entry point's `Result<FirstNativeFixtureGeneration, _>` matches what the non-streaming entry point would return for the same request.

## 3. Tests

- [x] 3.1 Added `production_generation_request_streaming_delivers_ordered_events_then_finished`: ordered `Token` events (matching `generated_token_ids` exactly, in order) followed by exactly one `Finished` event as the last delivered event.
- [x] 3.2 Added `production_generation_request_streaming_text_deltas_reconstruct_the_non_streaming_result`: concatenating every delivered `text_delta` equals the non-streaming entry point's `outcome.text` for the same request/prompt/weights. This test is what caught the real `streaming_decode` defect above (it failed with the original implementation, passed after the fix).
- [x] 3.3 Added `production_generation_request_streaming_finished_event_matches_non_streaming_usage`: `Finished.finish_reason`/`usage` equal the entry point's own returned `finish_reason`/`usage`.
- [x] 3.4 Added `production_generation_request_streaming_callback_cancellation_stops_cleanly`: a callback breaking on the first `Token` event stops generation at exactly one token, delivers `Finished` with `FinishReason::Cancelled`, and the call still returns `Ok` (proving session close/instance unload cleanup succeeded, since either failing would itself return `Err`).
- [x] 3.5 Added `production_generation_request_streaming_unsupported_gate_never_invokes_the_callback`: the existing multi-step-decode `Unsupported` gate still applies via the streaming entry point, and the callback is never invoked when generation is rejected before it starts.
- [x] 3.6 Regression guard: full `magnetar-runtime` (1250 + 173) and `integration-tests/production-loading` (14 passed, 4 pre-existing `#[ignore]`d) suites pass unchanged.

## 4. Docs and cleanup

- [x] 4.1 Updated `README.md` (new "currently supports" bullet, updated Tachyon scope-charter note, and a new note under "not yet supported" naming the real `HuggingFaceTokenizer::decode`/`streaming_state` gap discovered during this change) and `docs/production-model-loading-integration.md` (new step 9 with a worked streaming example).
- [x] 4.2 `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings` (both `magnetar-runtime` and `integration-tests/production-loading`), the full test suites, and `openspec validate stream-production-generation-events --strict` all pass.
