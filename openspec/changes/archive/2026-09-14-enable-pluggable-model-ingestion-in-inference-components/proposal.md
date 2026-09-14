## Why

The Tachyon integration audit's remaining P0 finding, MAG-01, is that `inference-components::LoadedInferenceComponent` -- despite its generic name -- hardcoded `HuggingFaceIngestor`/`HuggingFaceTokenizer` unconditionally, so it could only ever load Hugging Face-shaped bundles. This codebase already has a real, independent alternative ingestor (`loaders/gguf`, closed by `wire-gguf-into-model-loading` and its follow-ups) implementing the exact same format-neutral `ProductionModelArtifactIngestor` contract -- it was simply never wired in as a pluggable choice at the embedder layer. This change closes that gap.

## What Changes

- `LoadedInferenceComponent::load` (and `local_bundle_manifest_digest`) detect bundle format by presence of `loaders/gguf`'s own single expected file (`model.gguf`, now `pub const GGUF_FILE_NAME`) and select between `GgufIngestor`/`HuggingFaceIngestor` accordingly -- both already implement the identical `ProductionModelArtifactIngestor` contract, so no other ingestion-consuming code needed to change.
- Tokenizer construction and chat-template loading are similarly branched: the GGUF path uses two new `loaders/gguf` helpers (`load_gguf_tokenizer`, mirroring the pre-existing `load_gguf_chat_template`'s "re-read the file, return a ready-to-use value" convention) instead of Hugging Face's separate `tokenizer.json`/`tokenizer_config.json` file reads. Chat template *rendering* needed no format-specific code at all: `loaders/gguf::load_gguf_chat_template` already returns raw template text specifically so a caller can feed it to whatever `ChatTemplateFormatter` it has -- the existing (despite its name) format-neutral `HuggingFaceChatTemplateFormatter::new(text: impl Into<String>)` handles both formats' raw text identically.
- **BREAKING**: none. A Hugging Face-shaped bundle (the only kind this facade previously accepted) is still detected and handled exactly as before; GGUF-shaped bundles are newly accepted.

## Capabilities

### New Capabilities
(none -- extends the existing `production-model-ingestion` embedder-selection surface)

### Modified Capabilities
- `production-model-ingestion`: adds "An Embedder Selects Among Multiple Format-Specific Ingestors" -- an embedder handling more than one real ingestor SHALL select among them by inspecting the bundle itself, never by hardcoding one format's ingestor as the only path.

## Impact

- `loaders/gguf/src/lib.rs`: `GGUF_FILE_NAME` made `pub`; new `load_gguf_tokenizer` helper (mirrors `load_gguf_chat_template`'s existing pattern exactly).
- `inference-components/src/lib.rs`: `LoadedInferenceComponent::load`/`local_bundle_manifest_digest` branch on bundle shape instead of hardcoding `HuggingFaceIngestor`.
- Tests: `loaders/gguf` gains `load_gguf_tokenizer_and_chat_template_read_the_real_file`, verifying both new/existing helpers against the same real GGUF fixture bytes `GgufIngestor`'s own ingestion test already uses (35 tests total, up from 34, zero regressions). `inference-components`'s pre-existing 3 tests (`InvocationPayload` parsing, unaffected) still pass. Both crates: `clippy --all-targets -- -D warnings` and `fmt --check` clean.
- **Still open** (see `design.md`): no dedicated `inference-components`-crate-level integration test drives `LoadedInferenceComponent::load` end to end for *either* format -- a pre-existing gap (documented by the prior `wire-inference-component-to-generic-registry` change) this change does not close, now applying equally to the new GGUF path.
