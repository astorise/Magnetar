## Why

`reach-architecture-freeze-1` task group 8's remaining items (8.1 "build a real minimal Model Artifact," 8.2 "parse/read those bytes through Model Loading," 8.6 "remove `bind_qwen_fixture_weights()` from the production path") have stayed open because no byte-level Model Artifact format existed to build one from. `implement-model-format-parsers` closed that gap generally (real GGUF/Safetensors parsers), and `model-loading-materializes-weight-resources` already generalized the weight-materialization step itself (`materialize_model_instance_weights`, zero Qwen dependency, already spec-recognized as a Model Loading phase) -- but nothing yet connects the two. The E2E fixture's weights (`e2e_fixture_weights`) are still a pure in-memory `BTreeMap<String, HostTensor>`, never serialized to any byte format, and `e2e_fixture_manifest`'s declared weights digest is a placeholder (`sha256:000...0003`) that no real file's bytes are ever checked against. This Change closes that last connection: make the fixture's weights a real `.safetensors` file, parsed through the real parser, materialized through the existing generic phase -- the literal ask of 8.1/8.2/8.6.

## What Changes

- Serialize the E2E fixture's deterministic weight tensors into a real `.safetensors` file (bytes generated at test-fixture-build time, checked in as a fixture, the same pattern the real Qwen wasm Component binary already uses in this repository).
- Add a new, fully generic `magnetar-runtime` function (no format-crate dependency, per `externalize-runtime-extension-modules`) that turns a caller-supplied `&[ModelTensorMetadata]` plus the raw file bytes those tensors' `offset_bytes`/`size_bytes` index into into a `BTreeMap<String, HostTensor>` -- the missing "read the actual tensor bytes" step `SafetensorsArtifact`/`GgufArtifact` deliberately leave to their caller (`implement-model-format-parsers` design.md).
- `e2e_fixture_manifest`'s weights digest becomes the real sha256 of the real file's bytes, checked against it the same way `bind_qwen_fixture_weights`'s existing digest gate already checks the in-memory path today.
- The first-native production path (`bind_qwen_fixture_weights` / whatever currently supplies weights to `materialize_model_instance_weights`) switches from the in-memory `BTreeMap` bypass to: read the real `.safetensors` file bytes -> `magnetar_format_safetensors::parse` -> the new generic bytes-to-`HostTensor` bridge -> the existing, unchanged `materialize_model_instance_weights`.
- New parity/golden test: the same prompt through the same fixture produces identical generated logits whether weights arrived via the old in-memory path or the new real-file path, before the old path is removed -- proving the new path is correct, not just present.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-loading`: adds a requirement that Model Loading's weight-materialization phase SHALL be able to source tensor data from real Model Artifact bytes (via a format parser's generic tensor inventory), not only a pre-materialized in-memory source.

## Impact

- `magnetar-runtime`: new generic bytes-to-`HostTensor` bridge function (`model_loading.rs`, no format-crate dependency); `first_native_runtime.rs`'s fixture-construction and production weight-loading call sites updated to use it.
- New fixture: a real `.safetensors` file for the E2E fixture's weights, checked in.
- `formats/safetensors`: no changes required for the read path itself (its existing `parse` is sufficient); gains a small `serialize` function symmetric with `parse`, used by its own test suite to generate and round-trip-verify the checked-in fixture file -- `magnetar-runtime` cannot depend on `formats/safetensors` even as a dev-dependency (empirically verified, see design.md), so the writer and its round-trip proof both live on the format-crate side of the boundary, not `magnetar-runtime`'s.
- Closes `reach-architecture-freeze-1` tasks 8.1, 8.2, and 8.6 (the remainder of task group 8).
- **Explicitly out of scope:** `magnetar-cli` support for loading an arbitrary real-world checkpoint by path (needs tokenizer.json/config.json/generation_config support -- real, separate, larger work already tracked as roadmap-level in `model-format-roadmap`, not attempted here); GGUF fixture parity (the same mechanism applies, not duplicated here since Safetensors alone proves it); any change to `ModelLoadingCoordinator::load()`'s signature (already decided against by `model-loading-materializes-weight-resources`'s design.md Decision 1 -- materialization stays a distinct step, not a phase inside `load()`).
