## Context

Two real gaps survived `implement-production-qwen-model-loading`'s closeout because neither was in that change's original scope, but both were found reconciling the shipped code against a Tachyon-authored Magnetar scope charter (`perimetre-magnetar.md`, §7 and §15) and confirmed by direct code inspection this session:

1. `loaders/huggingface` parses a real `chat_template` string from `tokenizer_config.json` (`tokenizer.rs`'s `chat_template_reference`) but `lib.rs::ingest()` never threads it into the returned `ModelManifest` -- it hardcodes `chat_template: None`. The only `ChatTemplateFormatter` wired into a real caller (`magnetar-cli`'s `CliChatTemplateFormatter`) is a fixed `"{role}: {content}\n"` join, not an interpreter of the artifact's own template. `run_production_qwen_generation`'s own real-checkpoint smoke test uses plain-text prompts for exactly this reason, and produced incoherent output on a real Instruct-tuned checkpoint.
2. A CUDA generation request needing more than one decode step fails today -- correctly, fail-closed, no wrong answer -- but only after a real (expensive) prefill runs, deep inside KV-history concatenation, as a generic `TensorError::ResidencyUnavailable`. There is no early, explicit signal a caller can check or catch as "this request shape is not supported by this Provider," and no way to know without first paying for a real prefill.

Both are independent of `implement-device-resident-multi-step-cuda-decode` (which will eventually make CUDA multi-step decode actually work); this change makes the current, correct limitation visible and cheap to detect instead of an internal error a caller has to reverse-engineer.

## Goals / Non-Goals

**Goals:**
- A real chat template, when a bundle declares one, is preserved through ingestion and is what actually renders `PromptInput::ChatMessages` before tokenization -- for the common real-world subset of Jinja2 chat templates (variable substitution, `{% for %}` over `messages`, `{% if %}` on loop/message state, the handful of filters/tests real templates use), not a hand-rolled full Jinja2 implementation.
- A caller requesting a decode shape a bound Provider cannot service gets a structured, named `Unsupported`-shaped error before real generation work starts, not an internal residency error after a real prefill already ran.
- `tokens_per_second` (an existing, currently-always-`None` contract field) reflects real measured generation time.

**Non-Goals:**
- Wiring `magnetar-cli` to run generation against a production-loaded instance at all. Investigation while starting task group 3 found `magnetar-cli`'s existing chat/run commands (`pipeline.rs`'s `ChatSession`/`one_shot`) are bound only to the separate `qwen-test` Component fixture (`run_first_native_generation`/`FirstNativeChatSession`); `magnetar model load --file` (the only CLI path that runs real production ingestion) only proves loading and has no follow-on generation command at all. There is therefore no existing CLI chat-formatter selection to extend, and building "run generation against a locally loaded production model" as a new CLI feature is a materially larger, separate scope than closing the two audit gaps this change targets -- it is not attempted here.
- Implementing real device-resident multi-step CUDA decode itself (`implement-device-resident-multi-step-cuda-decode`, tracked separately).
- A general-purpose, spec-complete Jinja2 engine. Templates using constructs outside the supported subset fail closed with a clear error naming the unsupported construct, not a silent mis-render.
- Any change to `ChatTemplateFormatter`'s existing trait shape, to Resource Affinity, to Memory Manager, or to Model Loading's lifecycle.
- Retrofitting `tokens_per_second` (or any other observability field) beyond what this change's own new measurement points naturally produce -- the wider observability gap the scope reconciliation also found (most of §20's metrics are contract fields with no producer yet) stays out of scope here.

## Decisions

### Chat template rendering lives in `loaders/huggingface`, not `magnetar-runtime`

Jinja2-style chat templates are a Hugging Face `tokenizer_config.json` convention, not a portable, model-neutral concept -- the same externalization boundary this session already used for real tokenizer/Safetensors/config parsing applies here. `magnetar-runtime`'s `ChatTemplateFormatter` trait already exists and already expresses exactly the right shape (`fn format(&self, messages: &[ChatMessage]) -> Result<String, InferenceApiError>`); this change adds a real implementation of it externally, gaining `magnetar-runtime` zero new dependency.

**Alternative considered**: implement rendering inside `magnetar-runtime` so every ingestor benefits automatically. Rejected: it would require either vendoring a template engine into Core (violating the "Core has no concrete-format dependency" invariant this session enforced repeatedly via the `submodule-integration` CI guard) or hand-rolling one there, for a concern that is fundamentally about one source format's convention.

### Use `minijinja` for the real subset, not a hand-rolled parser

`minijinja` is a small, actively maintained, MIT-licensed Jinja2-compatible engine already the de facto standard for this exact job in the Rust LLM-serving ecosystem. Real chat templates (Qwen, Llama 3, Mistral, Gemma, ...) use a narrow, consistent subset of Jinja2 -- `minijinja`'s default feature set covers it directly, with no need to enable its more exotic optional features.

**Alternative considered**: hand-roll a minimal template interpreter covering only the observed subset. Rejected: real templates vary enough (different filters, different loop/conditional nesting) that a hand-rolled interpreter would need to grow ad hoc for every new model family, re-solving a problem a mature crate already solves correctly; `minijinja` is external to Core either way, so its dependency weight is irrelevant to the invariant this session cares about.

### Explicit `Unsupported` signal via a new default `Provider` trait method, not a new capability-advertisement field

Add `fn supports_multi_step_decode(&self) -> bool { true }` as a new default method on the existing `Provider` trait (matching the trait's existing pattern of defaulted methods like `health()`/`devices()`), so every current implementor (Reference CPU, and any future Provider) keeps its exact current behavior with zero code change; `CudaProvider` (external, in `providers/cuda`) overrides it to `false`, matching its own real, already-documented, already-tested residency behavior. The generic first-native generation entry point checks this once, before building the Runtime or running prefill, when the request needs more than one decode step, and fails with a new structured `InferenceApiError::Unsupported { reason }` naming the real constraint.

**Alternative considered**: repurpose the existing `ProviderMetadata.compute_data_movement_support` matrix (`ComputeDataMovementKind::Download` support) as the signal. Rejected: that matrix is not currently populated by either Reference CPU or CUDA for this specific meaning (device-resident KV *history* read-back, not general download support), so reusing it would mean either auditing and correctly wiring a broader, more general-purpose mechanism as a side effect of this change, or trusting an unpopulated/loosely-related field -- a materially larger and riskier change for the same outcome as one small, explicit, purpose-built method.
**Alternative considered**: keep the existing deep error and only improve its message/type at the point it already occurs. Rejected: it does not solve the "pay for a real prefill before learning the request cannot complete" part of the gap, only the clarity part.

### `tokens_per_second` measured at the same generation-loop boundary for every Provider

Wall-clock timing wraps the existing prefill/decode execution-plan dispatch in the shared first-native generation loop (`run_generation_loop_with_execution_plans` and the production entry points built on it), so Reference CPU and CUDA get identical measurement semantics with no Provider-specific code. Real time, not a synthetic estimate from token count and an assumed rate.

## Risks / Trade-offs

- [`minijinja` accepts a template construct real templates don't use, silently producing a plausible-but-wrong render instead of failing] → Restrict evaluation to a documented, tested subset explicitly (a small allow-list of the Jinja2 constructs this change verifies against real checkpoints' templates). A syntax error in that subset is a named `MalformedMetadata`-shaped error at ingestion time, the same fail-closed posture `production_model_ingestion` already uses everywhere else. A syntactically valid template referencing a construct outside the subset (e.g. an unknown filter, which `minijinja` only resolves at call time, not compile time) cannot be caught that early -- it still fails closed, with a named `TokenizationFailed`-shaped error, no later than the first real render.
- [A future Provider forgets to override `supports_multi_step_decode` and silently ships a broken multi-step CUDA-shaped decode] → The default `true` is only ever wrong for a Provider whose KV tensors are genuinely not host-readable; this is the same class of "a well-behaved Provider must know its own real constraints" trust `magnetar-runtime` already places in every other `Provider` method (e.g. `health()`, `devices()`), not a new kind of risk.
- [Real chat template rendering changes generated output for every existing test/fixture path that used the generic placeholder] → Scoped to apply only when a manifest declares a real `chat_template`; every existing fixture/tiny-bundle test (which declares none) is unaffected, verified by keeping `CliChatTemplateFormatter` as the explicit fallback.

## Addendum: a second real defect this change's own real-checkpoint testing found

Verifying task 3.2 end to end against the real public checkpoint (not just the two pre-existing `real_public_checkpoint_*` tests, both of which happen to use short plain-text prompts under 32 tokens) surfaced a second, independent real defect: `prepare_first_native_plan_for_graph`'s `PlanGuard::SequenceRange` ceiling was unconditionally `E2E_FIXTURE_CONTEXT` (32 tokens -- the tiny synthetic E2E fixture's own declared context length), applied to every first-native Plan regardless of the real model actually being served. A real chat-rendered prompt (Qwen2.5-Instruct's own template adds `<|im_start|>`/`<|im_end|>` markup plus a system message) routinely exceeds 32 tokens, so production generation with a real chat template would fail with `PlanWorkloadIncompatible` even though the chat-template rendering itself was already correct -- a real bug this change's own more-realistic end-to-end test happened to expose, not one the two audit-identified gaps predicted. Fixed by scaling the guard's ceiling to `token_count.max(E2E_FIXTURE_CONTEXT)` (`first_native_runtime.rs`): the Plan's own node bindings and edge shapes are already built for exactly `token_count` real tokens, so a workload of that same size must always be self-consistent; `E2E_FIXTURE_CONTEXT` remains a floor, so every existing tiny-synthetic-fixture test (built for well under 32 tokens) is unaffected. Verified: the full `magnetar-runtime` test suite (1242 + 173 tests) and the real public checkpoint chat-message test both pass after this fix, with the checkpoint's own generated text changing from the audit's originally-reported incoherent `" 1000000"` (plain-text path) to a coherent `"The capital of France, France."` (chat-message path, same real weights, same real prompt intent).

## Migration Plan

Purely additive: no existing public API signature changes (the new `Provider` trait method is defaulted; `ChatTemplateFormatter` is unchanged; `tokens_per_second` was already `Option<u64>`, now sometimes `Some`). No data migration. Rollback is a plain revert -- no persisted state depends on any of this.
