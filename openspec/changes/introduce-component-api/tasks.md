## 1. The repository

- [x] 1.1 `wit/<version>/` for 1.2.0 to 1.14.0, recovered from Magnetar's history and cross-checked against the Components' copies; `SHA256SUMS`, `PROVENANCE.md`.
- [x] 1.2 The crate: one cargo feature per version, the bindings generated once, exactly one version per build with a clear refusal otherwise.
- [x] 1.3 Checks: a release only adds to the one before it (with negative tests), a released file never changes, every version has its feature and module; `scripts/sync.py`.
- [x] 1.4 `examples/minimal-component`, built for `wasm32-wasip2` and validated as a Component.
- [x] 1.5 CI: every version (Clippy, tests, build), the immutability check on pull requests, documentation, the example.
- [x] 1.6 Checked on a copy of the Mistral Component: 13 tests pass (golden graphs included), the Component validates, its interface is unchanged (+2.1 % in size).

## 2. Components move to it, each under its golden graphs

- [x] 2.1 Mistral (contract 1.8.0): `wit/` and the `generate!` call replaced by the dependency and the `v1_8` feature; the golden graphs pass unchanged, and so do the Component Model validation and Magnetar's own build path (`wasm32-unknown-unknown` then `wasm-tools component new`). Magnetar's independent-logits check runs on this PR.
- [ ] 2.2 OLMo and Gemma (1.8.0; done, with their SDK moves, same golden graphs), Llama (1.8.0), SmolLM (1.9.0), DeepSeek (1.10.0), ModernBERT (1.12.0), Qwen (1.14.0), each with its own golden graphs.
- [ ] 2.3 Each Component's `Cargo.lock` records the pinned commit, and no Component carries a `wit/model-component-graph.wit`.

## 3. The Runtime

- [x] 3.1 A Magnetar CI check (`tools/check_component_contract_copy.py`, in the `wit` job) that `magnetar-runtime/wit/model-component-graph.wit` is byte-identical to the repository's file of the same version at a pinned commit.
- [ ] 3.2 Decide D6 (submodule or comparison) and write its delta to `wit` and `model-component-graph-contract`.
- [ ] 3.3 The Runtime serves, and a CI test loads a Component built on each contract version it declares it serves.

## 4. Towards publication

- [ ] 4.1 Publish `magnetar-component-api` to crates.io once the Components have moved, and move them to a version.
- [ ] 4.2 A `cargo generate` template for a new Component, from `examples/minimal-component`.
- [ ] 4.3 Typed operator constructors in the SDK, once the operator catalog is readable outside the core (follow-up of `introduce-component-sdk`).
