## Why

There is no real Model Artifact byte format anywhere in the Magnetar workspace today: `formats/gguf` and `formats/safetensors` are empty `cargo new --lib` templates, and every fixture (including the first-native E2E path) supplies weights as a plain in-memory `BTreeMap<String, HostTensor>` that was never serialized. `magnetar-runtime`'s generic Model Artifact type system (`ModelManifest`, `ModelTensorMetadata`, `ModelDType`, `ModelQuantization`, `ModelQuantizationFormat` -- the latter already anticipating GGUF's `Q4K`/`Q5K` quantization) already exists and is exercised by `ModelManifest::validate()`, but nothing produces those types from a real file today. This blocks `reach-architecture-freeze-1` task group 16 (external formats without type leakage) outright, and blocks task group 8's remaining 8.1/8.2 (build a real minimal Model Artifact; parse it through Model Loading) on the same missing piece. Both task groups independently flagged this as real work large enough to warrant its own OpenSpec Change given the safety requirements involved (parsers accept untrusted input: malformed headers, adversarial offsets, absurd dimensions) -- this is that Change.

## What Changes

- Implement a real Safetensors parser in `formats/safetensors`: JSON header parsing, tensor inventory extraction (name, shape, dtype, byte range), and normalization into `magnetar-runtime`'s existing generic `ModelTensorMetadata`/`ModelManifest` types -- zero Safetensors-specific types crossing into `magnetar-runtime`.
- Implement a real GGUF parser in `formats/gguf`: chunked key-value metadata and tensor-info parsing, alignment-padding handling, and the same normalization into generic types, including GGUF's quantized dtypes (`Q4K`/`Q5K`/`Q8` already modeled in `ModelDType`/`ModelQuantizationFormat`).
- Both parsers reject arithmetic overflow, out-of-bounds/overlapping tensor byte ranges, and absurd declared dimensions before any allocation is sized from them; neither panics on malformed input, backed by a fuzz target and a checked-in malformed-input corpus regression suite.
- Add a static guard (extending the existing task-16.4 check) verifying `magnetar-runtime` still has zero dependency on either format crate once both are real.
- **Non-goal, explicitly out of scope:** extending `ModelLoadingCoordinator::load()`'s API to consume parsed bytes and materialize weight resources from inside `load()` itself (`reach-architecture-freeze-1` tasks 8.1-8.6's deeper half). The audit that scoped that work already flagged it as its own deliberate API/lifecycle decision (does materialization become a phase inside `load()`, or a strictly-sequenced follow-on call?) independent of whether a parser exists. This Change only makes a real parser available to consume; wiring it into `Model Loading`'s call graph is deliberately left for a follow-on.

## Capabilities

### New Capabilities

- `safetensors-format`: Real Safetensors file parsing (header, tensor inventory, byte ranges) normalized into Magnetar's generic Model Artifact types, with overflow/panic safety on untrusted input.
- `gguf-format`: Real GGUF file parsing (chunked metadata, tensor info, quantization metadata) normalized into the same generic types, with the same untrusted-input safety guarantees.

### Modified Capabilities

- `model-format-roadmap`: The existing "Safetensors Support" and "GGUF Support" requirements are SHOULD/MAY roadmap language describing intent with no implementation; this change makes Safetensors support real and required (not aspirational) and GGUF support real for the unquantized-and-`Q4K`/`Q5K`/`Q8`-quantized subset this change implements, superseding the roadmap-only framing for both.

## Impact

- New/affected crates: `formats/safetensors`, `formats/gguf` (currently empty `cargo new --lib` submodules, both become real implementations depending on `magnetar-runtime`'s public Model Artifact types, never the reverse -- same dependency direction as `providers/cpu`).
- `magnetar-runtime`: no new dependency (verified by static guard); its existing `ModelManifest`/`ModelTensorMetadata`/`ModelDType`/`ModelQuantization` types are the normalization target, unchanged.
- Unblocks `reach-architecture-freeze-1` task group 16 (16.1-16.3) directly, and unblocks (without itself completing) task group 8's 8.1/8.2.
- New test surface: fuzz targets and malformed-input corpora for both parsers, run in each submodule's own CI (the `submodule-integration` job already runs `cargo test` per submodule; fuzzing itself is a separate, not-yet-defined CI concern this change should address for these two crates).
