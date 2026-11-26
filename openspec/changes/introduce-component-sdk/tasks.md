## 1. The crate

- [x] 1.1 `graph`: `Phase`, `KvRole`, `Value`, `Attr`, `Kv`, `Node` and the `GraphSink` trait, the shapes Mistral, OLMo and Gemma already define.
- [x] 1.2 `wit_graph_sink!`: the adapter of Mistral and OLMo, tested against a stand-in of the generated bindings (every call, every attribute kind, the KV involvement, a bare node).
- [x] 1.3 `testing::RecordingSink`: Gemma's recorder, with the edge ids it answers documented.
- [x] 1.4 Standalone CI: format, Clippy, tests, documentation with warnings denied, build for `wasm32-wasip2`.
- [x] 1.5 Checked against the real bindings: Mistral's types and adapter replaced by the crate's on a scratch copy, its tests pass with the `bindings` feature.

## 2. Dependency form

- [x] 2.1 A `main` branch for the crate's repository (its default branch is still the session's; the owner switches it in the repository's settings).
- [ ] 2.2 Each migrated Component names the crate as a git dependency pinned by commit, and its `Cargo.lock` records it (Mistral and OLMo: done; the others as they move).
- [ ] 2.3 Publish `magnetar-component-sdk` to crates.io once its API has been through the Components, and move the Components to a version.

## 3. Components move to it, each with its golden graphs

- [x] 3.1 Mistral: golden graphs recorded (7 scenarios, before the change), `plan.rs` types and `component.rs` adapter replaced; `main` of `magnetar-component-mistral` carries it, pinned by commit 4e2e556.
- [x] 3.2 OLMo, same: seven golden graphs recorded before the change (YaRN on full-attention layers only, the published 7B geometry), then moved; pinned on its branch `claude/charming-lamport-1b9m74` (its `main` does not carry the code yet: Codex's pull request is open).
- [ ] 3.3 Gemma, same (its adapter differs from the other two in names only).
- [ ] 3.4 SmolLM and Llama, which share a graph builder (`llama_graph`): one builder, in the crate or in a Component both depend on.
- [ ] 3.5 `GraphSink` with nodes of several outputs and recurrent state, as methods that default to a refusal.
- [ ] 3.6 Qwen and DeepSeek, which call the bindings directly, move onto the sink.

## 4. The shared decoder blocks

- [ ] 4.1 Normalization, rotary embedding, attention, gated MLP and LM head as functions over `GraphSink`, from the Components that build them identically; each Component states its variants.
- [ ] 4.2 Mixture of experts, per-layer layouts and latent attention, from the Components that have them.
- [ ] 4.3 The configuration helpers every Component rewrites (`from_huggingface_config`'s common fields).
- [ ] 4.4 One copy of the contract's WIT file per contract version, not one per Component.

## 5. Follow-ups (each its own change)

- [ ] 5.1 Weights and KV layout declared by the Component.
- [ ] 5.2 An opaque model configuration.
- [ ] 5.3 Operators a Component brings.
- [ ] 5.4 The isolation check over the whole core.
