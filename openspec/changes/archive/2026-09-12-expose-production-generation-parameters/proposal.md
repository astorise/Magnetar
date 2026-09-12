## Why

The production Qwen generation entry point family (`run_production_qwen_generation_for_provider*` in `magnetar-runtime`) is the surface an external embedder (Tachyon) calls to run real generation against a production-loaded model. That surface currently hardcodes `GenerationParameters::greedy()` and `StopConditions::default()` internally, even though the underlying Generation Contract (`GenerationRequest`, `generation.rs`, `sampling.rs`) already fully implements temperature/top_p/top_k/min_p/typical_p/penalties/seed/banned-and-allowed tokens, and both token-id and text-level stop sequences (`prepare_stop_sequences`). An embedder translating an OpenAI-shaped request (`temperature`, `top_p`, `seed`, `stop`, ...) into a Magnetar generation request today has no way to pass those values through this entry point without either being silently ignored or reimplementing sampling/stop-matching outside Magnetar -- both of which cross the Magnetar/Tachyon ownership boundary (generation semantics belong to Magnetar). This is a real, narrowly-scoped plumbing gap, not a missing capability: closing it requires forwarding values the Runtime already knows how to honor, not building new generation logic.

## What Changes

- Add a new public production entry point, `run_production_qwen_generation_for_provider_with_request`, accepting a `ProductionGenerationRequest` (prompt, `GenerationParameters`, `StopConditions`, and an optional `max_new_tokens` override) alongside the existing chat-formatter/provider arguments.
- The new entry point calls `prepare_stop_sequences` against the fixture's real tokenizer so caller-supplied `stop_text_sequences` (e.g. OpenAI `stop: ["END"]`) are honored, not just token-id stops.
- Existing entry points (`run_production_qwen_generation`, `run_production_qwen_generation_for_provider`, `run_production_qwen_generation_for_provider_with_prompt`) become thin wrappers over the new one, constructing a `ProductionGenerationRequest` with exactly today's defaults (greedy, default stop conditions, manifest's own `max_tokens`) -- no existing caller's behavior changes.
- Any requested option the selected execution path cannot honor (e.g. more than one decode step against a Provider that does not support multi-step decode) SHALL surface as the existing structured `InferenceApiError::Unsupported`, never silently ignored.

## Capabilities

### New Capabilities
- `production-generation-api`: the production-facing generation entry point contract Tachyon (or any embedder) calls to run real generation without reimplementing sampling, stop-matching, or streaming -- this change covers the generation-parameters/stop-conditions half; a follow-up change extends it with incremental streaming.

### Modified Capabilities
(none -- `inference-api`'s abstract Generation API requirement is already satisfied by the underlying contract; this change closes a gap in a concrete production entry point, not in the abstract Runtime Inference API itself)

## Impact

- `magnetar-runtime/src/first_native_runtime.rs`: new `ProductionGenerationRequest` struct, new `run_production_qwen_generation_for_provider_with_request` function; existing entry points become thin wrappers.
- No breaking changes: existing public function signatures are unchanged, and their observable behavior (greedy decoding, default stop conditions, manifest-driven `max_tokens`) is preserved exactly.
- Tests: new coverage that temperature/top_p/seed reach the real sampling contract, that a text stop sequence actually stops generation early, that an unsupported combination (e.g. multi-step decode against a non-supporting Provider) still fails closed, and that the legacy helpers keep their exact prior behavior.
