## 1. Crate scaffolding

- [ ] 1.1 Add a `magnetar-runtime` path dependency to `formats/safetensors/Cargo.toml` and `formats/gguf/Cargo.toml` (matching `providers/cpu`'s pattern: relative path, default features on, since `magnetar-runtime` is fail-closed without one of `wasmtime-component-engine`/`non-strict-fixture-fallback`).
- [ ] 1.2 Define `SafetensorsArtifact { tensors: Vec<ModelTensorMetadata>, metadata: BTreeMap<String, String> }` and `SafetensorsError` in `formats/safetensors/src/lib.rs` (structured error variants: header-too-short, header-not-utf8, header-not-json, unknown-dtype, out-of-bounds-range, overlapping-ranges, overflow).
- [ ] 1.3 Define `GgufArtifact { tensors: Vec<ModelTensorMetadata>, metadata: BTreeMap<String, GgufMetadataValue> }`, `GgufMetadataValue`, and `GgufError` in `formats/gguf/src/lib.rs` (structured error variants: bad-magic, unsupported-version, truncated-section, invalid-utf8, out-of-bounds-range, overlapping-ranges, overflow, quantization-unsupported).

## 2. Safetensors parser

- [ ] 2.1 Parse the 8-byte little-endian header-length prefix and the JSON header, with the header length itself bounds-checked against total file length before allocating a buffer for it.
- [ ] 2.2 Parse each tensor entry's dtype string (mapping onto `ModelDType::parse`, rejecting unknown dtype strings structurally) and shape array (`Vec<u64>`), and its `data_offsets: [start, end)`.
- [ ] 2.3 Validate every declared byte range: `start <= end`, `end <= file_length`, using checked arithmetic throughout (no unchecked `as` narrowing from `u64` to `usize`).
- [ ] 2.4 Reject overlapping tensor byte ranges (sort by `start`, check each range's `start >= previous end`).
- [ ] 2.5 Validate declared shape/dtype element-count and byte-size arithmetic does not overflow before comparing against the declared range length.
- [ ] 2.6 Parse the optional `__metadata__` key into the returned flat string map; reject non-string values in it structurally rather than silently coercing or dropping them.
- [ ] 2.7 Test: a well-formed multi-tensor fixture parses to the expected `ModelTensorMetadata` list and metadata map.
- [ ] 2.8 Test: every dtype `ModelDType::parse` supports round-trips correctly through a fixture using that dtype.

## 3. GGUF parser

- [ ] 3.1 Parse the fixed header: magic bytes (`"GGUF"`), version (reject unsupported versions structurally, not silently), tensor count, metadata KV count.
- [ ] 3.2 Parse the typed key-value metadata section (string/int8/16/32/64/uint8/16/32/64/float32/float64/bool/string/array-of-any-above), preserving every entry (recognized or not) in the returned metadata map per design.md's "unrecognized metadata preserved" decision.
- [ ] 3.3 Parse the tensor-info section: name, dimension count and dimensions (`Vec<u64>`), `ggml_type`, offset.
- [ ] 3.4 Map each tensor's `ggml_type` to `ModelDType`/`ModelQuantizationFormat` for the supported subset (`F32`/`F16`/`Bf16`/integer types, `Q4K`/`Q5K`/`Q8`); return a structured `gguf-quantization-unsupported` error naming the type for anything else.
- [ ] 3.5 Compute each tensor's byte range from its offset, dimensions, and dtype/quantization block size, honoring the file's declared (or default) alignment padding, with checked arithmetic throughout.
- [ ] 3.6 Validate every computed byte range against total file length and against every other tensor's range (out-of-bounds and overlap checks, same discipline as Safetensors task 2.3/2.4).
- [ ] 3.7 Test: a well-formed fixture using each supported dtype/quantization type parses to the expected `ModelTensorMetadata` list.
- [ ] 3.8 Test: a fixture using an unsupported `ggml_type` (e.g. a synthetic `Q2_K`-tagged tensor) is rejected with `gguf-quantization-unsupported`, not silently mapped.
- [ ] 3.9 Test: unrecognized metadata keys survive normalization unchanged and do not affect tensor parsing.

## 4. Overflow, bounds, and panic safety (both crates)

- [ ] 4.1 Audit every `u64`-from-file-bytes to `usize` narrowing in both crates; replace any bare `as` cast with `usize::try_from(..).map_err(..)`.
- [ ] 4.2 Audit every size/offset arithmetic expression in both crates; replace any unchecked `+`/`*` on untrusted values with `checked_add`/`checked_mul` and a structured overflow error.
- [ ] 4.3 Build a checked-in malformed-input corpus per crate (`formats/safetensors/tests/corpus/`, `formats/gguf/tests/corpus/`): truncated headers, invalid UTF-8, invalid JSON (safetensors)/wrong magic or unsupported version (GGUF), out-of-range offsets, overlapping ranges, overflow-inducing dimensions, inconsistent counts.
- [ ] 4.4 Add a corpus-replay `#[test]` per crate asserting every corpus entry returns a structured error and none panics; runs under plain `cargo test` (stable toolchain, no `cargo-fuzz` requirement).
- [ ] 4.5 Add a `cargo-fuzz` target per crate (`fuzz/fuzz_targets/parse.rs`) exercising the same public parse entry point the corpus test uses, documented as a local/periodic (not per-PR) fuzzing entry point given the nightly-toolchain requirement.

## 5. Dependency boundary and CI

- [ ] 5.1 Extend the existing task-16.4 static guard (`magnetar-runtime` has zero GGUF/Safetensors dependency) to a real, automated check (e.g. `cargo tree -p magnetar-runtime` inspection, or an equivalent manifest-graph check) rather than the current "trivially true, unverified by tooling" state.
- [ ] 5.2 Wire that guard into the `submodule-integration` CI job (or a new job, whichever fits `.github/workflows/quality.yml`'s existing structure better) so it runs on every change touching these crates.
- [ ] 5.3 Verify `cargo test --locked` passes for both `formats/safetensors` and `formats/gguf` standalone, matching the pattern already used for `providers/cpu`.
- [ ] 5.4 Verify both crates build for `wasm32-unknown-unknown` (matching the "platform-neutral" expectation set for `providers/cpu`; format parsers have no native-library dependency either).
- [ ] 5.5 Run `cargo clippy --lib --tests -- -D warnings` and `cargo fmt --check` clean for both crates.

## 6. Documentation and OpenSpec closure

- [ ] 6.1 Update `formats/safetensors/README.md` and `formats/gguf/README.md` from "Empty template" status to describe the real implementation, its supported subset (GGUF quantization scope in particular), and point at the new `safetensors-format`/`gguf-format` OpenSpec capabilities as governing contracts.
- [ ] 6.2 Update `SUBMODULES.md`'s compatibility matrix with real rows for both crates (commit pin, supported format-version/quantization subset).
- [ ] 6.3 Update `reach-architecture-freeze-1`'s task group 16 status note and close 16.1-16.3 with honest notes referencing this change's commits, once implemented and verified.
- [ ] 6.4 Cross-reference `reach-architecture-freeze-1` task group 8's remaining 8.1/8.2 status note: a real parser now exists, but wiring it into `Model Loading` itself remains this change's explicit non-goal (design.md) and stays open there.
