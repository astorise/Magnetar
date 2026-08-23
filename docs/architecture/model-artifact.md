# Model Artifact

Model Artifacts are inference data. They cover model weights, model
configuration, tokenizer data, tokenizer configuration, chat and prompt
templates, generation defaults, quantization metadata, sharding metadata,
adapter metadata, license metadata, provenance, and optional signature metadata.

They are intentionally separate from executable Component Artifacts, Provider
binaries, Device metadata, Runtime configuration, execution plans, inference
session state, KV cache, and loaded Model Instances.

## Identity

Model Artifact identity is content-addressed. The canonical identity includes
the artifact kind, logical model name, revision, optional variant, digest
algorithm, digest value, optional source identity, and optional shard identity.

Names, aliases, paths, registry tags, and user-friendly labels are metadata.
They are not sufficient identity without a verified digest. The Runtime supports
SHA-256 digest identity and rejects mismatches before loading.

## Manifest

A Model Artifact manifest uses the `magnetar-model-artifact` schema. It records
the schema version, artifact kind, model identity, architecture metadata, parts,
digests, storage dtype, compute dtype preferences, tensors where available,
tokenizer and template associations, generation defaults, quantization metadata,
sharding metadata, Runtime and Memory Manager feature requirements, Provider
Capability requirements, optional Component requirements, license metadata,
provenance, and optional signatures.

The manifest is not proof of trust. Trust is decided by Runtime policy.

## Bundle Structure

A `model-bundle` groups related Model Artifact parts such as `model-weights`,
`model-config`, `tokenizer`, `tokenizer-config`, `chat-template`,
`generation-config`, and `quantization-config`. Required bundle parts must
validate before the bundle is complete.

## Architecture

Architecture metadata identifies how model data is interpreted, for example
`llama`, `qwen`, `gemma`, `mistral`, `phi`, or `custom`. It does not create a
Provider. Providers remain execution implementations such as CPU, CUDA, Metal,
OpenVINO, QNN, or Candle.

## DTypes And Quantization

Model Artifacts distinguish storage dtype from compute dtype. Quantization
metadata records the format, group or block size, scale dtype, zero-point dtype,
per-channel behavior, dequantization workspace, and any required Provider
Capabilities. Quantized storage does not imply every Provider can execute it
directly.

## Sharding And Tensor Metadata

Sharded artifacts record shard identity, digest, size, ordering, and optional
tensor-to-shard mapping. Tensor metadata can record tensor name, shape, storage
dtype, layout, shard reference, offset, size, quantization metadata, and
expected compute dtype. It never exposes raw memory handles.

## Tokenizers And Templates

Tokenizers, tokenizer configuration, vocabularies, special tokens, chat
templates, and prompt templates may be Model Artifact parts. This model records
identity and association only; tokenizer execution and template rendering are
future contracts.

## Memory Manager Relationship

Model Artifact loading uses the Runtime Memory Manager for feasibility,
residency, pressure, transfer staging, quantization workspace, dtype conversion,
and sharded placement. Artifact bytes are distinct from resident model memory.
Resident memory may be host-owned, device-owned, Provider-owned, or
Runtime-owned staging.

## Trust And Provenance

Trust policy may consider digest, source, publisher, signature metadata,
provenance, license metadata, revocation, and local administrator policy. A
Model Artifact cannot declare itself trusted. Provenance and source identity are
diagnostic and policy inputs, not trust by themselves.

## Non-Goals

This model does not define model inference, generation APIs, tokenizer
execution, sampling, KV cache semantics, the full Model Instance lifecycle,
model registry protocols, model download protocols, Hugging Face integration,
Tachyon model distribution, license enforcement, full adapter loading, LoRA
merge behavior, distributed model sharding, cross-node placement, or
Provider-specific model loading ABI.
