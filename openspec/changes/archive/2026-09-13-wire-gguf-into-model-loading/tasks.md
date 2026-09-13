## 1. `formats/gguf`: expose the tensor data section offset

- [x] 1.1 Added `GgufArtifact.tensor_data_start: u64` (the already-internally-computed absolute file offset where tensor data begins), updated the parser's one construction site.
- [x] 1.2 Added a test assertion that `tensor_data_start + a tensor's own offset_bytes` addresses that tensor's real bytes in the file, by re-slicing the raw file and comparing.
- [x] 1.3 `cargo fmt`/`cargo clippy -D warnings`/`cargo test` all pass; all 18 pre-existing tests unaffected.

## 2. New submodule `loaders/gguf`

- [x] 2.1 Created GitHub repository `astorise/Magnetar-loader-GGUF` (public) and registered it as a git submodule at `loaders/gguf`.
- [x] 2.2 `src/config.rs`: normalizes GGUF `qwen2.*`/`tokenizer.ggml.{bos,eos}_token_id` key-value metadata into `ModelArchitectureConfig`, scoped to `general.architecture == "qwen2"`, with the same validation discipline (non-zero fields, GQA head-count divisibility, head_dim derivation, positive rope_theta) `loaders/huggingface::config` applies to `config.json`.
- [x] 2.3 `src/naming.rs`: normalizes GGUF's llama.cpp tensor naming (`token_embd.weight`, `blk.N.attn_q.weight`, `output_norm.weight`, `output.weight`, ...) into the exact same canonical Model Artifact names `loaders/huggingface::naming` produces from Hugging Face names.
- [x] 2.4 `src/weights.rs`: composes the real `magnetar-format-gguf` parser, rejects any tensor carrying quantization metadata with a structured error, and reverses every tensor's GGUF `ne[]`-order shape to the Hugging Face-equivalent row-major labeling (no byte reordering -- see design.md's sourced research on llama.cpp's real Qwen2 conversion applying neither a general transpose nor a Q/K permutation).
- [x] 2.5 `src/weight_layout.rs`/`src/derived_lm_head.rs`: ported the real projection-weight transpose and tied-`lm_head` derivation algorithms from `loaders/huggingface` (cannot depend on that crate directly -- independent externalized modules).
- [x] 2.6 `src/tokenizer.rs`: builds a real `tokenizers::Tokenizer` from the GGUF file's own embedded `tokenizer.ggml.tokens`/`merges` arrays via `BPE::builder().vocab_and_merges(...)` plus a `ByteLevel` pre-tokenizer/decoder, requiring `tokenizer.ggml.model == "gpt2"`.
- [x] 2.7 `src/lib.rs`: `GgufIngestor` implementing `ProductionModelArtifactIngestor`, orchestrating config/tensor/tokenizer normalization into a `ModelManifest` plus bounded payload source; `load_gguf_chat_template` helper exposing the raw declared template text (rendering left to the caller, matching design.md's Non-Goals).
- [x] 2.8 28 unit tests across all modules pass; `cargo fmt`/`cargo clippy -D warnings` clean.
- [x] 2.9 **Real bug found and fixed** running tokenizer construction against the real downloaded Qwen2.5-0.5B-Instruct-GGUF checkpoint: llama.cpp's real GGUF writer always declares `bos_token_id`, defaulting it to the same id as `eos_token_id` for a model with no distinct BOS; `TokenizerMetadata::validate` rejects two special tokens sharing an id regardless of kind. Fixed by registering eos first and skipping any later kind claiming an already-used id, plus two regression tests (`shared_bos_eos_id_does_not_conflict`, `distinct_bos_and_eos_ids_are_both_registered`).
- [x] 2.10 A prior local commit accidentally staged `target/`; caught before push, soft-reset, `.gitignore` added, recommitted cleanly.
- [x] 2.11 Added `README.md` documenting scope, real implementation status, and both verification methods (task 3 below).

## 3. Verification

- [x] 3.1 `integration-tests/production-loading`: added `magnetar-loader-gguf`/`magnetar-format-gguf` dev-dependencies.
- [x] 3.2 Added `tests_gguf_loading_e2e.rs`: `ingests_and_generates_from_a_real_gguf_file` (a hand-built synthetic GGUF bundle reaches `run_production_qwen_generation` through the real compiled Qwen Component and Reference CPU Provider -- the same production entry point Hugging Face bundles use); `rejects_a_non_qwen2_gguf_architecture`.
- [x] 3.3 Added `gguf_and_huggingface_ingestion_of_identical_weights_produce_identical_generation`: the exact same real weight values (via the existing fixture's `tensor_values` seeded by the Hugging Face-convention tensor name), written into both a Safetensors bundle and a hand-built GGUF file, ingested through both crates, driven by identical `PromptInput::TokenIds` (isolating weight/architecture correctness from the two bundles' deliberately different tokenizer implementations) -- asserts byte-for-byte identical generated token ids. This is the strongest available correctness proof without a real checkpoint, and it passed.
- [x] 3.4 Downloaded the real public `Qwen2.5-0.5B-Instruct-GGUF` checkpoint (`qwen2.5-0.5b-instruct-fp16.gguf`, unquantized F16, revision `9217f5d`, sha256-verified against the real downloaded bytes) and added `tests_gguf_real_checkpoint_smoke.rs`'s `#[ignore]`d `real_public_gguf_checkpoint_loads_and_generates_on_reference_cpu`, mirroring `tests_real_checkpoint_smoke.rs`'s established manual/nightly pattern.
- [x] 3.5 Ran it against the real downloaded file: architecture config matched the known real checkpoint dimensions (24 layers, 896 hidden, 14/2 attention/KV heads); generation produced `" 1000000"` for `"The capital of France is"` -- the exact same output the already-verified `loaders/huggingface` path produces for the same prompt against the equivalent real Safetensors checkpoint (task 2.9's tokenizer fix was required to get this far; also discovered this checkpoint's real GGUF export includes an explicit `output.weight` tensor despite `tie_word_embeddings: true`, unlike its Safetensors export -- corrected the test's assumption, not the ingestor's evidence-based logic, which was already correct).
- [x] 3.6 Full regression: `magnetar-runtime` (1250+173), `integration-tests/production-loading` (19 passed + 4 pre-existing `#[ignore]`d, now 5 including the new GGUF real-checkpoint test), `loaders/gguf` (28), `formats/gguf` (18) -- all pass. `cargo fmt --check`/`cargo clippy -D warnings` clean across every touched crate (including catching pre-existing fmt drift in `tests_production_loading_cuda_e2e.rs`/`tests_production_loading_e2e.rs` from earlier changes this session, fixed alongside).

## 4. CI and documentation

- [x] 4.1 `.github/workflows/quality.yml`: added `loaders/gguf/Cargo.toml` to `submodule-integration`'s build/test loop; extended the fixture-manifest source guard to also scan `loaders/gguf/src/`. The generic `magnetar-(component|format|provider|loader)-` externalization dependency guard already covers `magnetar-loader-gguf` automatically, no change needed there.
- [x] 4.2 `.gitmodules`: registered `loaders/gguf`.
- [x] 4.3 `SUBMODULES.md`: new module table row, compatibility-matrix entries for both `formats/gguf` (pin bump, `tensor_data_start` note) and `loaders/gguf` (full narrative, matching the existing per-commit changelog style).
- [x] 4.4 `openspec validate wire-gguf-into-model-loading --strict` passes.
