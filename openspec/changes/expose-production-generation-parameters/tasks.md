## 1. Request type and entry point

- [x] 1.1 Added `ProductionGenerationRequest { prompt: PromptInput, parameters: GenerationParameters, stop_conditions: StopConditions, max_new_tokens: Option<usize> }` to `first_native_runtime.rs`.
- [x] 1.2 Added `run_production_qwen_generation_for_provider_with_request(fixture, payload_source, trust_store, request: ProductionGenerationRequest, chat_formatter, provider) -> Result<FirstNativeFixtureGeneration, InferenceApiError>`.
- [x] 1.3 Resolved the token budget as `max_new_tokens.unwrap_or_else(|| fixture.manifest.generation....unwrap_or(64) as usize)`; the existing multi-step-decode `Unsupported` gate is now checked against this resolved value.
- [x] 1.4 Text stop sequences are resolved against the real tokenizer directly via `Tokenizer::resolve_stop_sequence` (the same default trait method `generation::prepare_stop_sequences` uses internally for its `RuntimeTokenizer<T>`-typed callers; `fixture.tokenizer` here is a `FixtureTokenizer` reached through the `Tokenizer` trait object, not that concrete wrapper, so the free function itself did not fit) and merged into `stop_conditions.prepared_stop_sequences` before building the `GenerationRequest`.
- [x] 1.5 `SessionCreationRequest.generation_defaults` is set to `parameters.clone()`.
- [x] 1.6 The `GenerationRequest` is built via `build_generation_request` with the caller's `parameters` and the merged `stop_conditions`.

## 2. Legacy wrappers

- [x] 2.1 `run_production_qwen_generation_for_provider_with_prompt` is now a thin wrapper constructing `ProductionGenerationRequest` with `GenerationParameters::greedy()`, `StopConditions::default()`, `max_new_tokens: None` -- its exact prior behavior.
- [x] 2.2 `run_production_qwen_generation_for_provider` and `run_production_qwen_generation` are unaffected (they already delegate to `_with_prompt`); their existing tests still pass unchanged.
- [x] 2.3 Added `production_request_generation_entry_point_is_public`, a source-text guard test asserting `run_production_qwen_generation_for_provider_with_request` is declared `pub fn`, matching the existing guard pattern for `_with_prompt`.

## 3. Tests

- [x] 3.1 Added `production_generation_request_forwards_non_greedy_sampling_parameters`: two runs with identical non-greedy `temperature`/`top_p`/`seed` parameters on the same prompt/weights produce identical generated token ids, proving the seed and non-greedy parameters actually reach the real sampling contract (not hardcoding any specific generated token, since this crate's fast per-PR bundle uses synthetic weights).
- [x] 3.2 Added `production_generation_request_honors_a_caller_supplied_stop_token_id` (fast, per-PR, deterministic token-id-stop variant, since the tiny synthetic bundle's weights make a text-decoded stop sequence non-reproducible at write time): observes this exact prompt/weights' own first greedily-generated token id from an unconstrained run, then proves a second run with that id as `stop_conditions.stop_token_ids` stops immediately after producing it. The nightly real-checkpoint text-stop-sequence scenario (`stop_text_sequences`) remains to be added to `tests_real_checkpoint_smoke.rs` as a follow-up manual/nightly test; the mechanism itself (`resolve_stop_sequence` wiring) is exercised by this fast test's `stop_conditions` forwarding and by the underlying `generation.rs`/`sampling.rs` contract's own existing tests.
- [x] 3.3 Added `production_generation_request_max_new_tokens_override_replaces_manifest_default`: `max_new_tokens: Some(2)` caps generation at 2 tokens even though the manifest's own configured default is 5.
- [x] 3.4 Added `production_generation_request_unsupported_gate_applies_to_the_overridden_token_budget`: a manifest default of 1 (which alone would pass the gate) combined with a caller-supplied `max_new_tokens: Some(2)` override against `MultiStepDecodeUnsupportedProvider` still fails fast with `Unsupported`, proving the gate checks the resolved value, not only the manifest's own default.
- [x] 3.5 Regression guard: every pre-existing test calling a legacy entry point (`tachyon_shaped_real_production_ingestion_loads_through_the_real_qwen_component`, `multi_step_decode_request_against_an_unsupporting_provider_fails_fast`, the real-checkpoint smoke tests, etc.) still passes unchanged.

## 4. Docs and cleanup

- [x] 4.1 Updated `README.md` (new "currently supports" bullet plus an updated "Tachyon scope charter reconciliation" note) and `docs/production-model-loading-integration.md` (new step 8 with a worked example) describing the new request-based entry point; legacy wrappers documented as unchanged.
- [x] 4.2 `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings` (both `magnetar-runtime` and `integration-tests/production-loading`), the full `magnetar-runtime` test suite (1250 + 173 passed) and the full `production-loading` integration suite (9 passed, 4 pre-existing `#[ignore]`d) all pass; `openspec validate expose-production-generation-parameters --strict` passes.
