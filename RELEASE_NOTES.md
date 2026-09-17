# Magnetar v0.1.0 Release Notes

Status: unreleased candidate evidence, not a stable publication.

## Included Baseline

- Rust workspace with `magnetar-runtime` and `magnetar-cli`, plus externalized
  Format, Component, Loader, and Provider submodules kept out of Runtime
  Core's compile-time dependency graph.
- Local-node runtime contract baseline covering Component, Capability,
  Provider, Device, Resource Affinity, Resolution Policy, Memory planning and
  Tensor Resource, Operator/Kernel/Kernel Registry/Dispatch, Model Artifact,
  Model Loading, Model Instance, Tokenizer, Generation, Sampling, Session, KV
  Cache, Prefix Cache, Continuous Batching, Runtime Inference API, CLI
  boundary, and E2E conformance.
- WebAssembly Component registration, contract validation, fail-closed import
  authorization, and a feature-gated Wasmtime Component Engine adapter, with
  fuel metering, epoch-based deadlines, and no ambient WASI authority.
- Real Reference CPU Provider execution and a real, hardware-verified external
  CUDA Provider: device-resident first-native chaining (MatMul, RMSNorm,
  multi-head/GQA-aware RoPE), device-resident multi-step decode, and
  cross-Provider result comparison against Reference CPU on real hardware.
- Real external WGPU Provider device discovery and one real compute kernel
  (`add`) over Vulkan/Metal/DX12, verified on real hardware -- not yet wired
  into the Kernel Registry dispatch contract.
- Real device-discovery-only external ROCm, NPU (Intel Level Zero), and TPU
  (Google Coral) Providers, gracefully unavailable wherever this
  repository's own tooling has no matching hardware; an honest,
  unconditionally-unavailable Metal Provider placeholder pending real macOS
  tooling.
- First-native Qwen generation driven by a real compiled Qwen Component,
  `ExecutionGraph`, real `ModelInstance` placement, and Provider-resolved
  weight materialization -- including `magnetar chat`'s persistent Runtime
  session and cancellation.
- Production Qwen model artifact ingestion: an external, pinned
  `loaders/huggingface` parses real `config.json`, `tokenizer.json`/
  `tokenizer_config.json`, and single-file or sharded Safetensors
  (F32/F16/BF16) into the same generic Model Loading contracts, including
  tied-embedding derivation and Qwen2/2.5 attention bias -- verified end to
  end, including against a real, publicly downloaded
  `Qwen/Qwen2.5-0.5B-Instruct` checkpoint, on both Reference CPU and real CUDA
  hardware, with real per-token generation-usage metadata and streaming
  events. A real `tokenizer.json`-backed tokenizer and real
  artifact-declared chat template rendering back this path; `magnetar model
  load --file` uses it for local bundles.
- A real, external, pinned GGUF ingestor (`loaders/gguf`) for
  `general.architecture: "qwen2"`, including real `Q8_0`/`Q4_K`/`Q5_K`
  block-quantized dequantization, verified against the real public
  `Qwen2.5-0.5B-Instruct-GGUF` checkpoint. `loaders/huggingface` separately
  gained real GPTQ, AWQ, and BitsAndBytes 4-bit dequantization, each verified
  element-by-element against real downloaded checkpoint weights.
- Fail-closed artifact trust for digest policy, rejected/revoked/quarantined
  digests, and explicit local-development policy.
- Quality gates documented in [docs/quality.md](docs/quality.md) and enforced
  in [.github/workflows/quality.yml](.github/workflows/quality.yml).

## Preview Or Contract-Only

- Production model source/hub download flows: a caller supplies an
  already-authorized local bundle; Magnetar does not fetch or discover one
  itself.
- Production continuous batching, prefix-cache reuse, LoRA adapters, and
  multi-device inference beyond the real Reference-CPU-plus-CUDA proof
  described above.
- Complete Component host adapters and end-to-end WIT host-call coverage for
  every intended production capability.
- Wiring `magnetar-cli` to generate from a production-loaded instance at all:
  `magnetar model load --file` proves real ingestion and loading, but
  `magnetar run`/`magnetar chat` remain bound to the separate `qwen-test`
  Component fixture, not a production-loaded one.
- A full, real, downloaded-checkpoint end-to-end generation proof for GPTQ,
  AWQ, or BitsAndBytes (dequantization correctness and ingestion-layer wiring
  are verified; GGUF's own quantized formats already have a full generation
  proof).
- Native CUDA `F16`/`bfloat16` compute exists for elementwise `add`/`mul`
  only, dispatchable through the Kernel Registry but not device-resident
  between calls and not requested by any production graph; `matmul`/
  `rmsnorm`/`rope`/`attention` still compute in `F32`.
- `magnetar run`, `magnetar chat`, `magnetar model ...`, `magnetar
  providers`, `magnetar devices`, and `magnetar serve` as fully stabilized
  production service interfaces.

## Deferred

- OpenVINO and QNN Providers.
- Native Metal FFI compute (no macOS environment in this repository's own
  tooling to implement or verify it).
- ROCm, NPU, and TPU compute kernels (device discovery only -- no matching
  hardware in this repository's own tooling to implement or verify compute).
- Production server/API transport and an OpenAI-compatible facade.
- General model hub downloads, OCI distribution, and credential/retry
  handling.
- Agent and tool execution inside the Runtime (owned by `magnetar-cli`'s CLI
  boundary instead, per its own documented scope).
- LoRA adapters and non-Qwen architecture families.
- A concrete Component distribution protocol and a stable Provider ABI.

## Unsupported Claims

- v0.1 does not provide cryptographic artifact signing: Component and Model
  Artifact signatures carry no cryptographic material and are not verified.
  The design is recorded in
  [docs/cryptographic-artifact-signatures.md](docs/cryptographic-artifact-signatures.md);
  implementation is separate, not-yet-scheduled follow-up work.
- v0.1 does not claim hardened production sandboxing for native Providers
  (Providers are trusted native code by architectural definition; see
  [SECURITY.md](SECURITY.md)).
- v0.1 does not claim general production large-model execution: the real,
  hardware-verified profile is specifically Qwen2/2.5-family decoder models,
  as described above and in
  [docs/production-model-loading-integration.md](docs/production-model-loading-integration.md).

## Verification Snapshot

- `cargo fmt --all -- --check`: pass.
- `cargo clippy --locked --workspace --all-targets --all-features -- -D warnings`: pass.
- `cargo test --locked --workspace --all-targets`: pass.

These are the root-workspace subset of the full CI gate list (rustfmt,
check/clippy/test matrices, MSRV, `cargo-deny`, Wasmtime and `wasm32`
Component Engine, Provider and E2E conformance, docs, WIT validation,
OpenSpec validation, model-family isolation, Component/Format/Provider
submodule integration, and the coverage ratchet) --
[docs/quality.md](docs/quality.md) lists every command, and
[.github/workflows/quality.yml](.github/workflows/quality.yml) is the
authoritative gate definition. `magnetar-cli` and the other
submodule-dependent crates (each its own Cargo workspace; see the root
`Cargo.toml`'s own comment) are covered by the `submodule-integration` job,
not by the root-workspace commands above.

Final checksums, SBOM, provenance, OpenSpec validation report, WIT validation
report, and release tag evidence must be generated from the final release
commit before stable publication.
