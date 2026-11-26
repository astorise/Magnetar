## Context

### What adding a family costs today

Measured on the seven Magnetar pull requests of the Qwen3.8-Flash-Next work after Qwen3.5 (#234, #237, #243 to #247), tests and fixtures included:

| Zone | Lines changed |
|---|---|
| `magnetar-runtime` | 4,674 (30 files) |
| `magnetar-provider-api` | 450 |
| `magnetar-engine` | 234 |
| `magnetar-model` | 207 |
| `magnetar-core` | 111 |
| `components/` (pointers) | 4 |

The Component's own code lives in its repository and is not in this table. A new primitive (an n-gram embedding, a token selection) legitimately needs a catalog entry, reference semantics and kernels in every Provider. What the table also holds is family knowledge in the core:

- `magnetar-runtime/src/qwen_model_component.rs`: 1,795 lines, 228 mentions of family names. It computes the KV row widths, the indexed layers and the KV layer count of a Qwen configuration.
- `first_native_runtime/production.rs`: 89 mentions. Per-architecture weight shape tables (`insert_ngram_shapes`, the indexer's `index_q_proj` and `index_k_proj`) and the names of the tables read by row (`layers.{L}.ple.table`).
- The `model-config` WIT interface: each family feature (`hyper-connections`, `ngram-embeddings`, `token-selection`, `linear-attention-gate`) is a function of the contract and an encoder in the host.
- The `model-family isolation` check guards a list of 25 files, none of the above.

### What Components copy

- `mistral/src/plan.rs`, `olmo/src/plan.rs` and `gemma/src/plan.rs` each define `Phase`, `KvRole`, `Value`, `Attr`, `Kv`, `Node` and the `GraphSink` trait. Mistral's and OLMo's `component.rs` carry the same adapter to the bindings, byte for byte. Gemma's differs only in argument names (`logical_name` for `name`), which is how a copy drifts.
- Mistral's and OLMo's `plan.rs` differ by 119 lines out of about 480.
- By function name: `attr` is defined eight times, `from_huggingface_config` nine, and `begin_graph`, `declare_input`, `add_node`, `finish_graph` seven (tests included).
- Each Component copies the contract's WIT file, at five versions: 1.8.0 in Gemma, Llama, Mistral and OLMo, 1.9.0 in SmolLM, 1.10.0 in DeepSeek, 1.12.0 in ModernBERT and 1.14.0 in Qwen, the Runtime's own being 1.14.0. SmolLM copies Llama's graph (`llama_graph`).
- Qwen, Llama and DeepSeek call the bindings directly instead of through a sink, so their graph code cannot run without the bindings.

## Goals / Non-Goals

**Goals**
- A Component keeps only what is specific to its family.
- What Components share is written once, with one set of tests, and a contract change is a change to one place.
- Moving a Component to the shared crate changes nothing it emits.

**Non-Goals**
- Removing the family knowledge from the core: it is the follow-up list below, each its own change, since each changes a contract.
- Choosing a contract version for every Component: the contract is additive, and each Component keeps targeting the version it needs.
- Publishing the crate to crates.io before the API has been through the Components (it is published once stable).

## Decisions

### D1. A crate with no dependency, whose scope is the graph description

`magnetar-component-sdk` holds the graph description, the sink, the adapter macro and the recording double, and nothing that names a family or depends on the Runtime. A Component for `wasm32-wasip2` pulls in nothing it does not use, and the crate can be tested without a WebAssembly host.

### D2. The sink is the 1.8.0 subset of the `graph-builder`

`GraphSink` has the seven operations every contract version since 1.8.0 has: `begin_graph`, `declare_input`, `weight_edge`, `alias_weight_edge`, `kv_resource`, `add_node` and `finish_graph`. A node has one output. Later contract versions add nodes with several outputs and recurrent state (Qwen uses them): they arrive in the crate as methods with a default that refuses, so that a Component which does not use them keeps compiling and one that does says so.

### D3. The adapter is a macro, not a function

The generated bindings are types of the Component's own crate: the same interface generated twice gives two unrelated sets of types, so a function in the crate cannot take them. `wit_graph_sink!(Host, gb)` expands in the Component's crate, against its `gb` alias, and relies only on the items present since 1.8.0. A contract change that touches one of them fails the compile at every call, which is the intended alarm.

Checked against the real bindings: Mistral's types and adapter replaced by the crate's, on a scratch copy, pass its five tests with the `bindings` feature.

### D4. A git dependency pinned by commit, then crates.io

A Component's repository runs its own checks without a Magnetar checkout (its graph and Component Model build, validated with `wasm-tools`), so a relative path to a sibling crate would not resolve there. While the API settles, a Component names the crate as `magnetar-component-sdk = { git = "https://github.com/astorise/magnetar-component-sdk", rev = "<commit>" }`, its `Cargo.lock` records the commit, and moving to a newer one is a pull request of the Component's own. A branch or a tag is refused: they move. Once the crate has been through the Components and is stable, it is published to crates.io and Components depend on it by version.

Not a submodule: nothing builds the crate beside a Component, so Magnetar has nothing to pin. `cargo-deny` runs on the root workspace and three Providers, none of which depends on a Component, so `deny.toml` needs no `allow-git` today; a graph it covers that came to include the crate would need one for that repository alone.

### D5. A migration is proved by a golden graph

Before a Component changes, the graphs its tests build are recorded with `RecordingSink` (nodes, ids, operators, attributes, inputs, shapes, KV involvements, in order). After the change the test compares. A Component's existing numerical tests against Transformers stay as the second check. The order is Mistral, OLMo and Gemma (same adapter), SmolLM and Llama, then Qwen and DeepSeek, which first need D2's multi-output methods.

### D6. The shared decoder blocks follow the types

Normalization, rotary embedding, attention, gated MLP, mixture of experts and the LM head are built the same way in Mistral, OLMo, Llama and SmolLM, with variants (sliding window, per-head norms, partial rotary). They move into the crate as functions over `GraphSink` once the types are shared, under D5, so that a Component states its variants instead of its nodes.

## Follow-ups, each its own change

1. **Weights and KV layout declared by the Component.** The Component lists its weights (names, shapes, which are read by row) and the shape of each KV store; the Runtime validates and sizes from that. It removes `qwen_model_component.rs` and the per-architecture tables of `production.rs` from the core.
2. **An opaque model configuration.** A typed tree the loader fills and the Component reads, which the core does not interpret, instead of a WIT function and a host encoder per feature.
3. **Operators a Component brings.** A composite operator is a subgraph with a fallback, and a Provider advertises the fused kernels it has (the kernel-artifact RFC), so a new primitive no longer needs a catalog edit for the first Provider to run it. The catalog keeps the true primitives.
4. **The isolation check over the whole core.** The check goes from a list of 25 files to every file of the core, with a short exception list that only shrinks, and a pull request reports how many files it changes outside Component and loader repositories.

## Risks / Trade-offs

- **A change to the crate touches every Component.** Mitigated by additive changes (defaults on new methods), the pinned revision (a Component takes a change when it chooses to), and the golden graphs, which fail where a change alters a graph.
- **Components on different crate revisions.** They can be, since each pins its own. The crate keeps its changes additive (defaults on new methods) so an older revision keeps working, and a Component moves to a newer one in a pull request of its own.
- **The macro hides the adapter.** It is the point, and its tests (against a stand-in of the bindings) and the real-bindings check above show what it expands to.
