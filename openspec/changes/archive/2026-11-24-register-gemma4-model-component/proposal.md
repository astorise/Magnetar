## Why

Issue #187: the Gemma 4 Model Component (`components/gemma`) builds graphs
for every Gemma 4 text decoder, and its side of the contract is merged
(astorise/Magnetar-component-Gemma#2, `9186a0c`). The Runtime had pinned only
its scaffold, shipped no artifact for it, and ran none of its tests. The
Component's own checks ran against a local Magnetar checkout, outside CI.

## What Changes

- `components/gemma` pins `9186a0c`.
- **The Component artifact.** `magnetar-runtime/fixtures/components/
  gemma-real.component.wasm` is built from that commit with the pinned
  toolchain (Rust 1.100, `cargo build --release --target
  wasm32-unknown-unknown`, `wasm-tools component new` 1.262). Its
  manifest declares:
  - the digest `sha256:aae4961d…`, which two builds reproduce;
  - WIT `model-component-graph` `1.6.0` imports, `graph-builder` and
    `model-config` only;
  - the families `gemma4`, `gemma4_text` and `gemma4_unified`;
  - the formats `huggingface` and `gguf`.
- **Engine tests** (`magnetar-engine`, adopted from the Component's
  `docs/engine-smoke-test/`). The artifact runs in the Wasmtime engine with
  the real `graph-builder` and `model-config` Capabilities and a layered
  configuration, so the Runtime's validation checks every graph it builds:
  - dense, MoE and per-layer-input layouts;
  - a decode graph and a segment;
  - a segment splitting a shared-KV layer from its producer, which is
    refused;
  - the published 31B, 26B-A4B and E2B shapes.

  Their node counts are pinned.
- **Logit comparison with `transformers`** (`integration-tests/
  production-loading`, adopted from the Component's
  `tools/qualification/`).
  - Six tiny random checkpoints and their `transformers` 5.23 reference
    logits are checked in under `fixtures/gemma4/`, about 3 MB. They are a
    dense and a `gemma4_unified` Hugging Face bundle, and four quantized
    `gemma4` GGUF files: dense, shared KV, per-layer inputs and MoE.
  - Each is ingested by its loader, loaded with the Component its family
    selects, and generates 12 greedy tokens on the Reference CPU. Every
    step's top-8 logits are compared with the reference.
- `test_support`'s real Components include `gemma-real`, so a checkpoint of
  a Gemma 4 family selects it.

## Evidence

| Checkpoint | Max abs logit difference |
|---|---|
| `dense` (Hugging Face) | 3.481e-5 |
| `unified` (`gemma4_unified`, Hugging Face) | 3.481e-5 |
| `gguf-dense` (`Q4_0`/`Q8_0`) | 3.338e-5 |
| `gguf-shared` | 1.144e-4 |
| `gguf-ple` | 2.718e-5 |
| `gguf-moe` | 3.004e-5 |

Tokens are identical in all six, and the values match the Component
side's (issue #187). The comparison fails when it should: the quantized
`gguf-dense` file compared with the unquantized reference generates
different tokens from the first step.

The engine tests build these node counts: dense 81, MoE 61, per-layer
inputs 101, segment `[0, 2)` 44, 31B 1314, 26B-A4B 869, E2B 1094.

**Limits.** The checkpoints are tiny (6 layers, 64 hidden, 8 experts). No
real Gemma 4 weights are run here. The 128-expert router, the 512-wide
heads and `Q6_K` tables are only checked for graph acceptance.

## Impact

- `magnetar-runtime/fixtures/components/`: the artifact and its manifest.
- `magnetar-engine`: `component_wasmtime/tests/gemma_component_tests.rs`.
- `integration-tests/production-loading`: `fixtures/gemma4/`,
  `tests_gemma4_logits.rs`, `test_support.rs`.
- `components/gemma`, `SUBMODULES.md`.
- No Runtime behaviour changes: an embedder registers the artifact like
  any other.
