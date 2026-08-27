# Magnetar v0.1.0 Release Notes

Status: unreleased candidate evidence, not a stable publication.

## Included Baseline

- Rust workspace with `magnetar-runtime` and `magnetar-cli`.
- CPU-local runtime contract baseline covering Component, Capability, Provider,
  Device, Memory, Tensor, Operator, Kernel Registry/Dispatch, Reference CPU,
  Model Artifact, Model Loading, Model Instance, Tokenizer, Generation,
  Sampling, Session, KV Cache, Prefix Cache, Continuous Batching, Runtime
  Inference API, CLI boundary, and E2E conformance.
- Deterministic fixture-backed local inference and conformance coverage.
- Fail-closed artifact trust for digest policy, rejected/revoked/quarantined
  digests, and explicit local-development policy.

## Preview Or Contract-Only

- Runtime-owned Provider-backed full generation path.
- Incremental prefill/decode backed by KV cache.
- Production model parsing, residency, tokenizer integration, batching,
  adapters, quantization, and multi-device execution.
- CLI commands beyond boundary/conformance harness behavior.

## Deferred

- CUDA, ROCm, Metal, OpenVINO, QNN, Vulkan, and WebGPU Providers.
- Production server API and OpenAI-compatible facade.
- Model hub downloads and remote registry authentication.
- Agent/tool execution inside the Runtime.
- Concrete Component distribution protocol and stable Provider ABI.

## Unsupported Claims

- v0.1 does not provide cryptographic artifact signing.
- v0.1 does not claim hardened production sandboxing for native Providers.
- v0.1 does not claim production large-model execution.

## Verification Snapshot

- `cargo fmt --all -- --check`: pass.
- `cargo clippy --workspace --all-targets -- -D warnings`: pass.
- `cargo test --workspace`: pass.

Final checksums, SBOM, provenance, OpenSpec validation report, WIT validation
report, and release tag evidence must be generated from the final release
commit before stable publication.
