## Why

Magnetar has crossed the provider-backed execution milestone: the first-native Runtime can execute a real Component-produced Qwen graph through Reference CPU and CUDA, and the external Safetensors/GGUF format modules now parse real files. The remaining blocker to a production caller such as Tachyon-Mesh is no longer Kernel execution; it is the missing causal chain from a real model bundle to a ready `ModelInstance`.

Today that chain is still fixture-shaped in several load-bearing places:

- the caller-facing first-native path is still privileged around `qwen-test`;
- CLI local-file loading canonicalizes a path but still builds fixture manifest metadata rather than normalizing the actual bundle;
- `formats/safetensors` returns a real tensor inventory but is deliberately not wired into Runtime Model Loading;
- Hugging Face `config.json`, `model.safetensors.index.json`, tokenizer/config/generation files are not yet composed into one production `ModelManifest`;
- the Qwen Component is real but its architecture dimensions remain hard-coded to the tiny E2E fixture;
- `host_tensors_from_artifact_bytes` only materializes F32 storage, while real Qwen checkpoints commonly store F16/BF16;
- production loading still lacks a bounded streaming payload path that avoids materializing the entire model into a `BTreeMap<String, HostTensor>` before Provider upload;
- Tokenizer Contract exists, but a real `tokenizer.json`-backed implementation is not yet load-bearing in this path.

This change closes those gaps without weakening the architecture boundary: `magnetar-runtime` remains format-, model-source-, and Provider-implementation agnostic; the Qwen Component remains Provider/Device agnostic; concrete Hugging Face/Safetensors ingestion stays external to Core and is composed through Runtime-owned generic contracts.

## What Changes

- Add a Runtime-owned, format-neutral production Model Artifact ingestion contract for normalized manifests plus bounded tensor-payload access. Concrete ingestors do not gain trust authority, Provider authority, or arbitrary filesystem/network authority.
- Add the first external production ingestor for Hugging Face-style Qwen bundles, pinned into Magnetar as an external module. It composes the existing real Safetensors parser and parses/normalizes `config.json`, `model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, and authorized chat-template metadata.
- Wire normalized ingested artifacts into the existing Model Loading trust/integrity/residency/Model Instance lifecycle rather than creating a parallel loader.
- Support single-file and sharded Safetensors bundles, including missing/duplicate/inconsistent shard/tensor rejection and digest validation.
- Extend storage materialization to F16/BF16 checkpoints with explicit, tested conversion to a Runtime/Provider-supported compute dtype. Native F16/BF16 Provider compute is not required by this change.
- Add bounded streaming transactional weight materialization so production loading does not require all model tensors to coexist as host `HostTensor`s before Provider upload.
- Version the Model Component graph/config contract so a Component can read Runtime-authorized normalized architecture configuration without receiving raw files, raw weight bytes, Provider/Device identities, or native handles.
- Make the external Qwen Component configurable from that normalized configuration and remove fixture architecture constants from the load-bearing production path.
- Add a real Hugging Face tokenizer implementation behind Tokenizer Contract and normalize tokenizer/generation/chat-template metadata through existing Runtime contracts.
- Generalize first-native caller-facing execution so a ready loaded Qwen `ModelInstance` is authoritative; `qwen-test` remains only a fixture/demo/conformance identity.
- Add a public embedder path suitable for Tachyon: authorized source -> ingestion -> Model Loading -> transactional materialization -> ready ModelInstance, with no Tachyon-owned model parsing or weight/resource management.
- Add per-PR production-shaped E2E plus real-checkpoint Reference CPU/CUDA smoke evidence.

## Capabilities

### New Capabilities

- `production-model-ingestion`: format-neutral Runtime contract and external implementation boundary for converting an authorized model source into normalized Model Artifact/config/tokenizer metadata plus bounded payload access.

### Modified Capabilities

- `model-loading`: production ingestion becomes load-bearing; sharded artifacts, F16/BF16 storage materialization, streaming transactional materialization, and caller-facing source-to-ModelInstance loading are required for the Qwen production profile.
- `model-format-roadmap`: Hugging Face config, sharded Safetensors index, tokenizer/config/generation/chat-template normalization move from roadmap-only intent into the first production Qwen loading path.
- `model-component-graph-contract`: adds versioned Runtime-authorized model configuration access for configurable Components.
- `qwen-model-component`: the production Qwen Component becomes checkpoint-configurable rather than fixture-dimension-specific.
- `tokenizer`: a real `tokenizer.json`-backed implementation is required for the production Qwen profile.
- `conformance`: adds non-fixture production-bundle and real-checkpoint CPU/CUDA acceptance gates.
- `project-architecture`: concrete production Model Artifact ingestors are externalized like format implementations and may depend on Core contracts, never the reverse.

## Impact

- `magnetar-runtime`: new generic ingestion/payload/config contracts and production orchestration; no dependency on Hugging Face, Safetensors, CUDA, or any concrete external implementation.
- `components/qwen`: versioned Component contract update and configurable graph construction.
- Existing `formats/safetensors`: reused as the real tensor parser; no duplicate parser is introduced in Core.
- New external module, recommended name `Magnetar-loader-HuggingFace` and pin `loaders/huggingface`: production Hugging Face bundle normalization and bounded payload access.
- Tokenizer implementation: external or separately composable implementation behind the existing Tokenizer Contract; the exact repository name is an implementation decision, but no tokenizer implementation dependency is added to `magnetar-runtime`.
- `magnetar-cli`: replaces fixture-manifest local loading with the same production path an embedder uses.
- `integration-tests`: adds production-shaped local bundle tests and a real-checkpoint CPU/CUDA smoke package.
- Tachyon-Mesh can integrate the resulting public loading surface without parsing model formats or managing model weight resources itself.

## Non-Goals

- Native CUDA F16/BF16 compute kernels; explicit F16/BF16 storage -> supported compute conversion is sufficient for this change.
- Full persistent model hub/cache implementation, remote Hugging Face download UX, OCI distribution, credential storage, retry/range-resume behavior.
- Full GGUF quantized execution, GPTQ/AWQ/BitsAndBytes, LoRA production, or additional model families.
- Giving Components filesystem/network, Provider/Device, raw buffer, or native-handle authority.
- Moving concrete format parsers into `magnetar-runtime`.
