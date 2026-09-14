## Context

The audit's own note framed the remaining gap narrowly: "no dedicated `inference-components`-crate-level integration test drives `LoadedInferenceComponent::load` end to end." Building that test required a completely-specified real bundle (config.json/tokenizer.json/model.safetensors with every tensor the real Qwen Component's graph resolves) -- `HuggingFaceIngestor`'s existing tests use deliberately minimal bundles (single-value tensors, no tokenizer) that prove ingestion-shape correctness but could never actually drive a real generation. Once the test was written and passing locally, the natural next step -- confirm it actually runs in CI -- found it structurally could not: `inference-components` is never checked out with the submodules it depends on in any CI job.

## Goals / Non-Goals

**Goals:**
- A real, passing, orchestration-level test proving `LoadedInferenceComponent::load` -> `invoke_payload` works end to end through this crate's own code, not just the `magnetar-runtime` layer underneath it.
- Close the CI-coverage gap using the exact precedent already established for the structurally identical case (`magnetar-cli`), rather than inventing a new pattern.

**Non-Goals:**
- A GGUF-format equivalent of this test. The audit's own note said "for either format," and the Hugging Face path is the one exercised here; a GGUF-shaped equivalent would repeat the same bundle-construction effort for the other ingestor and is left for a future chantier if it turns out to matter (the underlying `is_gguf` branch itself is already covered by `loaders/gguf`'s own tests and the `magnetar-runtime`-level singleton/named-component tests).
- Wiring `inference-components` into `magnetar-cli` or any other consumer. Nothing in this repository currently calls `LoadedInferenceComponent::load` outside tests -- that is a separate, larger scope decision (how Tachyon or any embedder actually consumes this crate) this change does not make.

## Decisions

- **Build the bundle by hand, matching `qwen_expected_tensor_shape` exactly**, rather than trying to reuse `HuggingFaceIngestor`'s existing minimal test fixtures -- those are deliberately too small (single-value tensors) to drive a real graph. `weight_layout.rs`'s documented HF-stored-vs-internal shape convention (`[out_features, in_features]` on disk, transposed to `[in_features, out_features]` internally) is followed precisely so the ingestor's own transpose logic, not a hand-transposed shortcut, produces the shapes the Component's graph expects.
- **Reuse `loaders/huggingface`'s own tiny WordLevel tokenizer recipe** (`{"type": "WordLevel", "vocab": {...}, "unk_token": ...}` with a `Whitespace` pre-tokenizer) rather than a byte-level BPE vocabulary -- this session's own earlier `loaders/gguf` tokenizer test found a minimal byte-level-BPE vocabulary cannot round-trip arbitrary text; WordLevel with an exact vocabulary match avoids that failure mode entirely and deterministically tokenizes "hello world" to two token ids.
- **Add CI coverage by extending the existing `submodule-integration` job**, using `magnetar-cli`'s own steps as the direct template (same job, same checkout, same reasoning for why a first-party out-of-workspace crate needs its own fmt/clippy/test steps) -- not a new job, since the submodule checkout and toolchain setup are already exactly what this crate needs too.

## Risks / Trade-offs

- **Still only the Hugging Face format is exercised end to end at this crate's own boundary layer.** Documented as a Non-Goal, not silently dropped.
- **The new CI steps add runtime to `submodule-integration`.** Proportionate: the job already builds/tests six other submodule-dependent crates in the same run: this is one more crate of comparable size, using the same already-cached toolchain and dependency graph.
