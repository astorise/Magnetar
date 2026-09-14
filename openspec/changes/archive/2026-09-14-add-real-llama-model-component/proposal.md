## Why

The original Tachyon scope charter names additional Model Components (Llama, Mistral, Gemma) as future work; `components/llama` has existed as a pinned git submodule since early in this repository's history but was, until now, an empty `cargo new --lib` scaffold with "not yet scheduled against any OpenSpec change" in its own README. This change gives Magnetar its first real second production model architecture family, proving the `model-component-graph-contract` genuinely generalizes beyond Qwen rather than only claiming to.

## What Changes

- `components/llama` (submodule, commit `70692e2`) implements the real `magnetar:model-component-graph@1.2.0` `model-component-graph-producer` world. Its graph-building logic is structurally identical to `components/qwen`'s own: Llama (1/2/3) and Qwen2/2.5 are both real instances of the same pre-norm, RoPE, grouped-query-attention, SwiGLU-gated decoder block -- Qwen2's architecture is Llama's with an added QKV bias term, not a different block shape. Every axis that actually differs between the two families (bias presence, tied embeddings, dimensions, RoPE base, GQA group size) is already expressed generically by `model-config`'s `architecture-config` record, so there is no separate Llama-specific graph shape to add.
- Built and validated through the same real toolchain `components/qwen`'s own checked-in fixture uses (`cargo build --target wasm32-unknown-unknown --release`, `wasm-tools component new`, `wasm-tools validate`). Its WIT world is structurally identical to the real Qwen Component's own (confirmed via `wasm-tools component wit` on both).
- `magnetar-runtime/fixtures/components/llama-real.component.wasm` (+ manifest) is a new checked-in test fixture, mirroring `qwen-real.component.wasm`'s own.
- `magnetar-runtime`'s test suite gains `build_first_native_graphs_from_named_component_serves_a_real_second_architecture_family`, registering the real Llama Component alongside the real Qwen Component under the generic registry (`wire-generic-inference-component-runtime`) and proving: (1) for the same `architecture-config`, the independently-compiled Llama and Qwen binaries produce byte-identical graphs (node counts and operator-sequence hashes) -- the contract's genericity is real; (2) the same Llama binary produces a genuinely different graph for a bias-bearing config, driven entirely by `model-config`, never a hardcoded per-family branch.
- `qwen_operator_kind_code` (a `#[cfg(test)]`-only helper mapping graph operator names to hash-able kind codes) gains an entry for `"add"` (the QKV bias-add operator) -- a real, previously-latent gap: no prior test ever hashed a bias-bearing graph, since every existing fixture config defaults `attention_bias: false`.
- **BREAKING**: none. New submodule content, a new checked-in test fixture, and additive test-only changes; no production code path changes to `magnetar-runtime`'s non-test surface beyond the one-line operator-kind-code table addition (itself `#[cfg(test)]`-only).

## Capabilities

### New Capabilities
- `llama-model-component`: the Llama Model Component baseline -- implements the same generic graph contract as `qwen-model-component`, is not a Provider, and is config-driven (no hardcoded architecture dimensions).

### Modified Capabilities
- `model-component-graph-contract`: "Multiple Registered Components May Coexist Under Caller-Supplied Trust" gains a scenario proving the registry serves a genuine second real architecture family, not only a caller-supplied digest that always resolves to one hardcoded family.

## Impact

- `components/llama`: real implementation (submodule commit `70692e2`), pinned in this repository.
- `magnetar-runtime/fixtures/components/`: two new checked-in files (`llama-real.component.wasm`, `llama-real.component.wasm.magnetar-component.yaml`).
- `magnetar-runtime/src/first_native_runtime.rs`: two new `#[cfg(test)]`-gated `include_bytes!` constants; one new `"add"` entry in the test-only `qwen_operator_kind_code` table.
- `magnetar-runtime/src/first_native_runtime/tests.rs`: one new cross-architecture proof test.
- `SUBMODULES.md`: `components/llama` row updated from "empty template" to real, with a new compatibility-matrix row.
