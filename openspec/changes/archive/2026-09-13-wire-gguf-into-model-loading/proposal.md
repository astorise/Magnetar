## Why

The Tachyon scope charter names "GGUF wired into Model Loading" as future work: `formats/gguf` already parses the full GGUF container into Magnetar's generic Model Artifact types, but nothing turns that into a `ProductionModelArtifactIngestor` Model Loading can actually load and generate from -- an embedder with a GGUF checkpoint has no supported path today, unlike Hugging Face bundles (`loaders/huggingface`). Closing this lets Magnetar serve the single most common local-inference distribution format for Qwen2 checkpoints without any new Runtime-side code, following the exact same externalization boundary `production-model-ingestion` already defines.

## What Changes

- New external, pinned submodule `loaders/gguf` (`Magnetar-loader-GGUF`), implementing `ProductionModelArtifactIngestor` for a single local `model.gguf` file: architecture-config extraction from GGUF's own key-value metadata, tensor discovery/renaming into the same canonical Model Artifact names `loaders/huggingface` already produces, and a real byte-level-BPE tokenizer built programmatically from the GGUF file's own embedded vocabulary (no `tokenizer.json`).
- Scoped to `general.architecture == "qwen2"` and unquantized `F32`/`F16`/`BF16` tensors only; a quantized tensor (`Q4_K`/`Q5_K`/`Q8_0`) is rejected structurally with a named error, not silently accepted or approximated -- dequantization-at-load and quantized compute kernels remain a separate, not-yet-implemented Magnetar chantier.
- Small, additive change to `formats/gguf`: exposes `GgufArtifact.tensor_data_start` (already computed internally during parsing, not previously returned), the one piece of information a caller needs to read raw tensor bytes back out of the original file without re-deriving the parser's own alignment/header-walk logic.
- **BREAKING**: none. No existing public API changes signature or behavior; `loaders/huggingface`, `magnetar-runtime`, and every existing production entry point are unaffected.

## Capabilities

### New Capabilities
- `gguf-model-ingestion`: the concrete GGUF-specific ingestion requirements (architecture scope, quantization fencing, tensor shape/layout normalization, tokenizer construction from embedded vocabulary) `loaders/gguf` implements against the existing generic `production-model-ingestion` contract -- mirrors how `gguf-format` already documents the parser layer below it.

### Modified Capabilities
- `gguf-format`: adds `GgufArtifact.tensor_data_start` as a new field on the existing parse result (additive, no existing behavior changes).

## Impact

- New repository/submodule: `loaders/gguf` (`Magnetar-loader-GGUF`), pinned at `loaders/gguf`.
- `formats/gguf`: additive `tensor_data_start` field.
- `integration-tests/production-loading`: new `magnetar-loader-gguf`/`magnetar-format-gguf` dev-dependencies, new test files proving real end-to-end GGUF ingestion, cross-format numerical parity against `loaders/huggingface` for identical weight values, and a manual/nightly real-checkpoint smoke test against the public `Qwen2.5-0.5B-Instruct-GGUF` checkpoint.
- `.github/workflows/quality.yml`: `submodule-integration` job's build/test loop and fixture-manifest source guard extended to cover `loaders/gguf`.
- `.gitmodules`/`SUBMODULES.md`: new submodule registration and compatibility-matrix entry.
- Tests: 28 unit tests in `loaders/gguf` (config normalization, tensor naming, tokenizer construction including a real discovered bos/eos-id-conflict fix, projection transpose, tied-`lm_head` derivation), 5 integration tests (synthetic end-to-end ingestion+generation, cross-format numerical parity, architecture rejection), 1 real-checkpoint nightly test (all passing against the real downloaded checkpoint).
