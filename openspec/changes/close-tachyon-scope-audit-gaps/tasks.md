## 1. Real chat template threading

- [ ] 1.1 In `loaders/huggingface`, thread the already-parsed `chat_template_reference` (from `tokenizer_config.json`) into the `ModelManifest.chat_template` field produced by `HuggingFaceIngestor::ingest`, instead of hardcoding `None`.
- [ ] 1.2 Add a test proving a bundle whose `tokenizer_config.json` declares a `chat_template` produces a manifest carrying it, and a bundle that declares none produces `None`, unchanged from prior behavior.

## 2. Real chat template rendering

- [ ] 2.1 Add `minijinja` (or an equivalent minimal, actively maintained Jinja2-compatible crate) as a dependency of `loaders/huggingface` only -- never a `magnetar-runtime` dependency, enforced by the existing `submodule-integration` dependency guard.
- [ ] 2.2 Implement a real chat-template renderer in `loaders/huggingface` implementing `magnetar_runtime`'s existing `ChatTemplateFormatter` trait, rendering the artifact's real template against the standard message shape (`role`, `content`) real chat templates expect.
- [ ] 2.3 Define and document the supported rendering subset explicitly (variable substitution, `{% for %}` over `messages`, `{% if %}` on loop/message state, the specific filters/tests real checkpoints' templates use) and reject any other construct at ingestion time with a structured, named error rather than at first render.
- [ ] 2.4 Add tests against at least two materially different real chat template strings (e.g. a Qwen-style `<|im_start|>`/`<|im_end|>` template and a materially different real template from another checkpoint family) proving correct rendering byte-for-byte against the real template's own expected output.
- [ ] 2.5 Add a test proving an unsupported construct is rejected at ingestion with a structured error naming it, not silently mis-rendered or silently ignored.

## 3. Wiring the real formatter into production generation

- [ ] 3.1 `run_production_qwen_generation`/`_for_provider` (or the layer that currently applies `PromptInput::ChatMessages`) uses the ingested manifest's real chat template when present, falling back to the existing generic behavior only when none is declared.
- [ ] 3.2 `magnetar-cli`'s chat path selects the real formatter for a loaded production model whose manifest declares a real template, keeping `CliChatTemplateFormatter` as the explicit fallback for a manifest that declares none (including every existing tiny-fixture/`qwen-test` path, which must be unaffected).
- [ ] 3.3 Add a source guard (or extend the existing production-loading source guards) proving the real chat-template path is reachable through public APIs only, not through Runtime-internal primitives.

## 4. Explicit `Unsupported` signal for a decode shape a Provider cannot perform

- [ ] 4.1 Add `fn supports_multi_step_decode(&self) -> bool { true }` as a new default method on the `Provider` trait in `magnetar-runtime`, so every existing implementor's behavior is unchanged unless it explicitly overrides it.
- [ ] 4.2 Override `supports_multi_step_decode` to `false` on `CudaProvider` (`providers/cuda`), matching its already-documented, already-tested device-resident-only KV history behavior.
- [ ] 4.3 Add a new structured `InferenceApiError::Unsupported { reason }` (or extend an existing structured error enum) naming the real constraint.
- [ ] 4.4 In the generic first-native generation entry point, check `provider.supports_multi_step_decode()` once, before running prefill, when the request requires more than one decode step; fail with the new `Unsupported` error before any real execution work if the bound Provider declares it cannot perform it.
- [ ] 4.5 Add a test proving a >1-token generation request against a Provider declaring `supports_multi_step_decode() == false` fails fast (before prefill executes) with the new `Unsupported` error.
- [ ] 4.6 Add a real-hardware test proving a >1-token generation request against the real `CudaProvider` fails with this same `Unsupported` error (not the previous internal residency error), run via `gpu-runner-smoke.yml`.
- [ ] 4.7 Add a test proving a >1-token generation request against Reference CPU (or any Provider declaring support) is unaffected -- this requirement introduces no new restriction for a Provider capable of the requested shape.

## 5. Real `tokens_per_second` measurement

- [ ] 5.1 Measure real wall-clock time across prefill/decode dispatch in the shared first-native generation loop (`run_generation_loop_with_execution_plans` and the production entry points built on it).
- [ ] 5.2 Populate `tokens_per_second` in the returned generation usage metadata from that real measurement, for both Reference CPU and CUDA, replacing the always-`None` default when at least one token was produced.
- [ ] 5.3 Add a test proving `tokens_per_second` is `Some` and consistent with generated token count and measured duration after a real generation call.

## 6. Documentation and scope reconciliation

- [ ] 6.1 Record in `SUBMODULES.md`/README which parts of the Tachyon-authored Magnetar scope charter this change closes (chat template, `Unsupported` decode signaling, real `tokens_per_second`), and which remain explicitly deferred exactly as the charter itself frames them (multi-device, quantization, GGUF wired into Model Loading, native FP16/BF16 compute, additional Providers/Components, device-resident multi-step CUDA decode itself).
- [ ] 6.2 Update `docs/production-model-loading-integration.md` to describe the real chat-template behavior for embedders.
- [ ] 6.3 Document the new `Provider::supports_multi_step_decode` method and the `Unsupported` error shape for external Provider implementers.

## 7. Quality gates

- [ ] 7.1 Run `cargo fmt`/`clippy -D warnings`/`cargo doc -D warnings`/`cargo test` for `loaders/huggingface`, `magnetar-runtime` (with and without `wasmtime-component-engine`), `providers/cuda`, and `magnetar-cli` -- all clean.
- [ ] 7.2 Re-run the full `integration-tests/production-loading` suite, including the real public checkpoint tests (`tests_real_checkpoint_smoke.rs`), confirming no regression from the chat-template and `Unsupported`-signal changes.
- [ ] 7.3 Re-run `gpu-runner-smoke.yml` on real hardware and confirm the new CUDA `Unsupported` test and the unaffected Reference-CPU/CUDA prefill tests all pass.
- [ ] 7.4 Run `openspec validate --all --strict` and archive only after every task above is complete with linked evidence (commit SHA, `Quality` and `GPU Runner Smoke Test` run IDs on that exact SHA).
