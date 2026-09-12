## Context

`run_production_qwen_generation_for_provider_with_prompt` (`magnetar-runtime/src/first_native_runtime.rs`) is the production entry point that loads a real Qwen checkpoint, tokenizes/formats a prompt, and runs generation through a caller-supplied Provider. It is the surface an embedder like Tachyon calls today. Internally it:

- builds the session with `generation_defaults: GenerationParameters::greedy()`
- builds the `GenerationRequest` with `GenerationParameters::greedy()` and `StopConditions::default()`
- passes `|_generated_so_far| false` as the cancellation/stop closure

None of this is a limitation of the underlying Generation Contract. `generation.rs`'s per-step logic (`sampling_request.parameters = request.parameters.clone()`, EOS/stop-token/stop-pattern matching against `request.stop_conditions`) and `sampling.rs` (temperature, top_p, top_k, min_p, typical_p, penalties, seed, banned/allowed tokens) already consume `GenerationParameters`/`StopConditions` generically. `prepare_stop_sequences(tokenizer, stop_text_sequences) -> Vec<TokenStopPattern>` already exists to turn caller text stops into the tokenizer-aware form `StopConditions.prepared_stop_sequences` needs. The gap is purely that the production entry point never accepts or forwards these values -- it manufactures its own greedy/default request regardless of what a caller wants.

## Goals / Non-Goals

**Goals:**
- Let a caller supply real `GenerationParameters` and `StopConditions` (including text-level stop sequences) to production Qwen generation, reaching the same sampling/stop-matching contract every other Runtime generation path already uses.
- Preserve every existing public entry point's exact current behavior and signature.
- Fail closed (structured `Unsupported`, not silent ignoring or a panic) when a requested option cannot be honored by the selected execution path.

**Non-Goals:**
- Incremental/streaming token delivery (MAG-P1-2, a separate follow-up change).
- Any change to `GenerationParameters`, `StopConditions`, `sampling.rs`, or `generation.rs` semantics -- they already do the right thing.
- An HTTP/OpenAI-shaped request type. `ProductionGenerationRequest` is a Magnetar-native Rust struct; translating an OpenAI JSON body into it is Tachyon's job, matching the existing Magnetar/Tachyon boundary.
- Session-level generation defaults reuse across multiple requests (`SessionCreationRequest.generation_defaults` stays session-scoped; this change addresses the one-shot production entry point).

## Decisions

- **New request struct, new entry point, existing ones become wrappers.** `ProductionGenerationRequest { prompt: PromptInput, parameters: GenerationParameters, stop_conditions: StopConditions, max_new_tokens: Option<usize> }` plus `run_production_qwen_generation_for_provider_with_request(fixture, payload_source, trust_store, request, chat_formatter, provider)`. `run_production_qwen_generation_for_provider_with_prompt` becomes:
  ```rust
  run_production_qwen_generation_for_provider_with_request(
      fixture, payload_source, trust_store,
      ProductionGenerationRequest { prompt, parameters: GenerationParameters::greedy(), stop_conditions: StopConditions::default(), max_new_tokens: None },
      chat_formatter, provider,
  )
  ```
  identical to today's hardcoded values, so zero behavior change for existing callers. This mirrors the exact pattern `close-tachyon-scope-audit-gaps` already used for `run_production_qwen_generation_for_provider_with_prompt` itself (a `_with_request`-shaped superset function, with the narrower function becoming its thin wrapper) -- proven, not novel.

- **`max_new_tokens: Option<usize>` overrides the manifest default, `None` preserves it.** The existing behavior reads `fixture.manifest.generation.max_tokens.unwrap_or(64)`. Wrapping in `Option` lets a caller who wants a different token budget than the checkpoint's own manifest default (e.g. an OpenAI `max_tokens: 128`) get it, while every existing wrapper passes `None` and reproduces today's manifest-driven value exactly.

- **The existing multi-step-decode `Unsupported` gate (`close-tachyon-scope-audit-gaps`) is reused unchanged**, now checked against the resolved `max_new_tokens` (whichever of manifest-default or caller-override applies) rather than only the manifest value. This is the same fail-closed mechanism, just fed the right number.

- **Text stop sequences are prepared inside the entry point, not left to the caller.** `ProductionGenerationRequest.stop_conditions.stop_text_sequences` is caller-facing (plain `Vec<String>`); the entry point itself resolves each one against the real tokenizer (`Tokenizer::resolve_stop_sequence`, the same default trait method `generation::prepare_stop_sequences` uses for its `RuntimeTokenizer<T>`-typed callers -- `fixture.tokenizer` here is a `FixtureTokenizer` reached through the `Tokenizer` trait directly, not that concrete wrapper) and merges the result into `stop_conditions.prepared_stop_sequences` before building the `GenerationRequest`. A caller should never have to know that "prepared" tokenizer-aware form exists -- that plumbing detail is exactly what an embedder should not have to reimplement.

- **Session generation defaults track the request's own parameters, not a separate hardcoded greedy.** `SessionCreationRequest.generation_defaults` is set to `request.parameters.clone()` instead of `GenerationParameters::greedy()`, so a session created for a sampling request doesn't silently carry greedy defaults for any session-level fallback behavior.

## Risks / Trade-offs

- **Widening what a caller can request also widens what can go wrong** (e.g. a `seed` with `deterministic: true` requires reproducibility guarantees the underlying sampling contract already validates via `GenerationParameters::validate()` / `GenerationRequest::validate()`, invoked by the existing `prepare_generation` call this entry point already makes -- no new validation gap is introduced, the existing contract's own validation now actually gets exercised with non-default input for the first time in this call path).
- **One more public function to keep behaviorally pinned.** Mitigated the same way `close-tachyon-scope-audit-gaps` mitigated it for the prompt-input superset: a source-text guard test asserting the new function is `pub fn`, plus tests asserting each existing wrapper's exact prior defaults are still what it passes through.
