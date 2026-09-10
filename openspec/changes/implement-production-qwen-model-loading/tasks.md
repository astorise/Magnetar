## 1. Production ingestion contract and externalization

- [x] 1.1 Define a format-neutral Runtime contract for production Model Artifact ingestion that returns normalized artifact/config/tokenizer metadata plus bounded payload access, without importing any concrete format/source implementation.
- [x] 1.2 Define structured ingestion/payload errors covering unsupported format, unauthorized source, missing part, malformed metadata, payload out-of-bounds, payload unavailable, integrity mismatch, and implementation unavailable.
- [x] 1.3 Ensure ingestion success never grants trust; Runtime trust evaluation remains mandatory before materialization.
- [x] 1.4 Add a static CI guard proving `magnetar-runtime` has no compile-time dependency on Hugging Face/Safetensors concrete loader crates.
- [x] 1.5 Add/modify `project-architecture` documentation so concrete production Model Artifact ingestors are externalized and pinned, analogous to Formats/Providers.
- [x] 1.6 Create/pin the first external production ingestor module (recommended `loaders/huggingface` / `astorise/Magnetar-loader-HuggingFace`) or document an equivalently external module layout before implementation begins.

## 2. Authorized Hugging Face bundle discovery

- [x] 2.1 Accept an already-authorized local/client-provided bundle source; do not accept arbitrary raw filesystem traversal from a model reference.
- [x] 2.2 Discover `config.json`, tokenizer/config/generation files, chat-template source, and either single-file or indexed-sharded Safetensors inside the authorized bundle boundary.
- [x] 2.3 Reject path traversal and bundle escapes; define and test symlink policy explicitly.
- [x] 2.4 Normalize source identity/provenance without treating local/HuggingFace/Tachyon source kind as trust.
- [x] 2.5 Add tests for missing required config, missing weights, duplicate candidate files, unauthorized path, and bundle escape.

## 3. Real `config.json` parsing and normalization

- [x] 3.1 Parse Qwen-compatible Hugging Face config JSON in the external ingestor/module.
- [x] 3.2 Normalize at least `model_type`, `architectures`, `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim` (or explicit validated derivation), `vocab_size`, `rms_norm_eps`, `rope_theta`, `rope_scaling`, `tie_word_embeddings`, `torch_dtype`, `bos_token_id`, `eos_token_id`.
- [x] 3.3 Reject missing/zero/overflow/internally-inconsistent architecture values with structured `model-config-invalid`/Qwen config errors.
- [x] 3.4 Preserve unknown source annotations without allowing them to become Runtime policy implicitly.
- [x] 3.5 Prove `torch_dtype` does not silently override requested compute dtype.
- [x] 3.6 Add corpus/negative tests for malformed JSON, wrong types, integer overflow, unsupported RoPE variants, invalid head/KV-head relationships, invalid vocab/token IDs.

## 4. Single-file and sharded Safetensors integration

- [x] 4.1 Reuse the real `Magnetar-format-safetensors` parser; do not introduce a second Safetensors parser in Core or the Qwen Component.
- [x] 4.2 Implement `model.safetensors.index.json` parsing and normalize tensor -> shard mapping into existing `ModelShard` / `ModelTensorMetadata` contracts.
- [x] 4.3 Resolve and parse every required shard inside the authorized bundle boundary.
- [x] 4.4 Reject missing shard, duplicate tensor, tensor mapped to multiple shards, unexpected shard escape, index/header inconsistency, and missing required tensor.
- [x] 4.5 Validate whole-part/shard/tensor digest bindings where declared.
- [x] 4.6 Add deterministic tests covering one-file, 2+ shard, missing shard, corrupt shard, duplicate tensor, wrong index mapping, and digest mismatch.

## 5. F16/BF16 storage materialization

- [x] 5.1 Extend production bytes-to-staging materialization to decode F16 and BF16 in addition to F32.
- [x] 5.2 Verify tensor digest against source storage bytes before dtype conversion.
- [x] 5.3 Perform checked element-count/byte-count validation for each storage dtype.
- [x] 5.4 Convert F16/BF16 explicitly to selected supported compute/staging dtype when the Provider lacks native half support; record conversion in residency plan/diagnostics.
- [x] 5.5 Reject unsupported storage dtypes structurally; no reinterpret cast or silent dequantization.
- [x] 5.6 Add numeric edge-case tests: ±0, subnormal policy, finite extrema, infinities/NaNs according to declared policy, odd/truncated byte counts, shape mismatch.
- [x] 5.7 Keep native CUDA F16/BF16 compute outside this change unless required by discovered correctness constraints.

## 6. Streaming transactional weight materialization

- [x] 6.1 Add a production materialization path that iterates required tensor metadata/payloads without first building a whole-model `BTreeMap<String, HostTensor>`.
- [x] 6.2 Reuse/extend `WeightMaterializationTransaction`; do not create a parallel non-transactional loader.
- [x] 6.3 For each tensor: read bounded payload -> validate -> convert if needed -> Provider stage -> release transient host staging before proceeding where possible.
- [x] 6.4 Commit only when all required weight names are successfully staged and bindings match the validated manifest.
- [x] 6.5 On any failure, release Provider-side resources, Memory Manager allocations/residency, pending bindings, and readiness evidence.
- [x] 6.6 Add failure-injection tests at first/middle/last tensor and Provider-write/completion failure; assert no orphan Provider tensor and no leaked Memory Manager allocation.
- [x] 6.7 Add a scale test proving host staging peak is bounded independently of total model size (within explicit buffering policy).

## 7. Versioned Runtime-authorized model config Capability

- [x] 7.1 Evolve `magnetar:model-component-graph` contract (recommended 1.1 if compatible, otherwise 2.0) so the Component can query Runtime-authorized normalized model configuration.
- [x] 7.2 Expose only architecture/config values needed by model semantics; do not expose raw JSON, arbitrary file paths, weight bytes, Provider/Device identities, native handles, or unrestricted annotation maps.
- [ ] 7.3 Extend portable tensor descriptor support beyond the current hard-coded contiguous-F32 assumption where required by production graph validation.
- [x] 7.4 Preserve strict first-native rule: production graphs still originate from the Component through graph-builder, never Runtime-side Qwen synthesis.
- [x] 7.5 Add WIT compatibility/version-mismatch tests and `wasm-tools component wit` validation.
- [x] 7.6 Update Runtime Component host adapters for the new config Capability with fail-closed authorization and structured errors.

## 8. Configurable production Qwen Component

- [x] 8.1 Update `components/qwen` to derive hidden size, layer count, attention heads, KV heads, head dimension, intermediate size, vocab size, RMSNorm epsilon, RoPE parameters, and tied-embedding behavior from Runtime-authorized config.
- [x] 8.2 Remove fixture architecture constants from the load-bearing production graph path.
- [x] 8.3 Build the decoder stack for `0..num_hidden_layers` and derive logical weight names/shapes from config.
- [x] 8.4 Validate required tensor inventory and shapes against the normalized Model Artifact before graph execution can produce a Ready instance.
- [x] 8.5 Preserve GQA/MQA semantics with distinct attention/KV head counts and explicit RoPE metadata.
- [x] 8.6 Preserve Provider/Device agnosticism: static guard for no CUDA/CPU/provider crate imports and no Provider/Device WIT values.
- [x] 8.7 Compile/regenerate the checked-in Component artifact and update digest/version evidence.
- [x] 8.8 Add Component tests for at least two materially different Qwen configurations (different layer/head/intermediate sizes), not only the existing tiny fixture dimensions.

## 9. Production tokenizer/config/generation/chat-template loading

- [x] 9.1 Provide a real `tokenizer.json`-backed implementation behind `RuntimeTokenizer`/Tokenizer Contract; keep implementation dependency out of `magnetar-runtime`.
- [x] 9.2 Parse/normalize `tokenizer_config.json`, special tokens, BOS/EOS/PAD, model max length, truncation/padding metadata.
- [x] 9.3 Validate tokenizer vocabulary size and special-token compatibility against the loaded Qwen config.
- [x] 9.4 Implement encode/decode and streaming decode through the real tokenizer implementation.
- [x] 9.5 Parse/normalize `generation_config.json` into overridable defaults; explicit generation request values always win.
- [x] 9.6 Load chat-template data only from the authorized artifact/config; inference must not fetch arbitrary filesystem/network template data.
- [x] 9.7 Add parity tests against known tokenizer vectors and negative tests for tokenizer/model mismatch.

## 10. Generic first-native loaded-model execution

- [x] 10.1 Add/complete a caller-facing path where generation targets a ready `ModelInstanceId` produced by production loading.
- [x] 10.2 Remove any production condition that requires `model_ref == "qwen-test"`.
- [x] 10.3 Keep `qwen-test` as fixture/demo/conformance only.
- [ ] 10.4 Remove `fixture_model_manifest` / hand-built tensor inventory from local production model loading.
- [ ] 10.5 Ensure tied `lm_head` derivation remains a load-time operation and is driven by real config/tensor metadata.
- [ ] 10.6 Verify Provider/Device selection remains Runtime-owned and the same loaded artifact can target Reference CPU or CUDA according to policy/capabilities.

## 11. Public embedder / Tachyon loading surface

- [ ] 11.1 Expose one supported public orchestration surface for authorized source -> normalized artifact -> Model Loading -> materialization -> warm/readiness -> `ModelInstanceId`.
- [ ] 11.2 Ensure callers cannot mint trust decisions, ready evidence, resource bindings, or Provider allocations.
- [ ] 11.3 Add a Tachyon-shaped integration test using only public APIs and a client-provided/Tachyon source identity.
- [ ] 11.4 Static/source guard: the integration test and sample embedder contain no Safetensors tensor parser, Qwen graph builder, dtype conversion loop, or TensorResourceId fabrication.
- [ ] 11.5 Document the minimal public integration recipe for Tachyon-Mesh and other embedders.

## 12. Production E2E and release gates

- [ ] 12.1 Add a tiny deterministic **production-layout** Qwen bundle for per-PR tests: real `config.json`, real tokenizer files, real Safetensors parser input, optional real shard index. It may be small, but it must enter through the production ingestor; no fixture manifest/inventory constructor is allowed.
- [ ] 12.2 E2E Reference CPU: authorized bundle -> ingestion -> trust/integrity -> configurable Component -> ModelInstance -> materialization -> prefill/decode -> output.
- [ ] 12.3 E2E CUDA on real hardware through the same normalized artifact and Component; assert CUDA Provider/Device residency and no silent Reference CPU fallback.
- [ ] 12.4 Add a real public Qwen-compatible checkpoint smoke test pinned by revision/digest (manual/nightly/hardware profile acceptable if size prevents per-PR execution).
- [ ] 12.5 Compare CPU/CUDA logits or deterministic output within a documented tolerance.
- [ ] 12.6 Assert unload/failure leaves no ModelInstance weight allocation or Provider tensor leak.
- [ ] 12.7 Make GPU CI fail if the required hardware test is skipped or selects `0 tests`.
- [ ] 12.8 Add source guards proving production loading/generation does not call `fixture_model_manifest`, fixture tensor inventory builders, or require literal `qwen-test`.
- [ ] 12.9 Run `cargo test`/clippy/fmt/doc/wasm checks for Runtime and every affected external module.
- [ ] 12.10 Run CUDA hardware suite and capture exact commit/runner evidence.
- [ ] 12.11 Run `openspec validate --all --strict` and archive only after every task above is complete with linked evidence.

## 13. Documentation / cutover declaration

- [ ] 13.1 Update `README.md` production status from "general model loading incomplete" only when tasks 1-12 meet their exit criteria.
- [ ] 13.2 Update `SUBMODULES.md` with the production ingestor/tokenizer module pins and compatibility requirements.
- [ ] 13.3 Document supported initial production profile precisely: Qwen + Hugging Face-style config/tokenizer + Safetensors F32/F16/BF16 storage + Reference CPU/CUDA Float32 compute path.
- [ ] 13.4 Document explicit non-support separately: remote hub/cache behavior not yet implemented, native half compute if deferred, GGUF quantized execution, other model families.
- [ ] 13.5 Publish a Tachyon cutover criterion: Tachyon may remove its fail-closed placeholder only after the public production E2E in task group 12 is green.
