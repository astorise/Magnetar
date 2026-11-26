## Why

Adding a model family should be adding a Model Component. It is not. The Qwen3.8-Flash-Next work (seven Magnetar pull requests, without the Qwen3.5 lot) changed 4,674 lines in 30 files of `magnetar-runtime`, 450 in `magnetar-provider-api`, 234 in `magnetar-engine`, 207 in `magnetar-model` and 111 in `magnetar-core`, against four lines of Component pointers. Part of that is unavoidable, a new primitive needs its catalog entry and kernels. Part of it is a core that knows model families: `qwen_model_component.rs` (1,795 lines, 228 mentions of family names) sizes the KV stores of a Qwen config, `production.rs` holds per-architecture weight shape tables, and each family feature adds a function to the `model-config` WIT interface and its host encoder.

The other half of the problem is on the Component side. There is no crate for what Components share, so each carries its own copy. The Mistral and OLMo adapters to the WIT bindings are byte-identical, Gemma's differs from them only in argument names, and the three `plan.rs` files declare the same `Phase`, `KvRole`, `Value`, `Attr`, `Kv`, `Node` and `GraphSink`. `attr` is defined eight times and `from_huggingface_config` nine; Mistral and OLMo's `plan.rs` differ by 119 lines out of about 480. The WIT file is copied into each Component, at five contract versions (1.8.0 in four of them, 1.9.0, 1.10.0, 1.12.0 and 1.14.0).

Without a shared crate, a core that lacks what a Component needs absorbs it instead. This change starts with the crate.

## What Changes

1. **`magnetar-component-sdk`**, a new repository and crate with no dependency, holding what Components share: the graph description (`Phase`, `KvRole`, `Value`, `Attr`, `Kv`, `Node`), the `GraphSink` trait a Component's host-independent graph code is written against, the `wit_graph_sink!` macro that implements it on the generated WIT bindings, and `RecordingSink`, the test double.
2. **Mounted as a submodule at `components/sdk`**, beside the Components that depend on it by relative path.
3. **Components move to it one at a time**, each emitting the same graph as before, proved by a golden graph recorded from its current code before it changes. Mistral, OLMo and Gemma first (their adapters are the same code), then SmolLM and Llama, then Qwen and DeepSeek, which call the bindings directly and need the sink to carry nodes with several outputs and recurrent state.
4. **The decoder blocks they share** (normalization, rotary embedding, attention, gated MLP, mixture of experts, LM head) and the configuration helpers move into the crate after the types, under the same golden-graph discipline.

The follow-ups that take family knowledge out of the core are changes of their own, listed in the design: weights and KV layout declared by the Component, an opaque model configuration, operators a Component can bring, and a family-isolation check over the whole core.

## Capabilities

### New Capabilities

- `component-sdk`: what the shared crate holds and promises, and the rules for a Component that uses it.

### Modified Capabilities

None. `project-architecture`'s requirement that Components live in their own repositories is unchanged.

## Impact

- New repository `magnetar-component-sdk`; new submodule `components/sdk`; `SUBMODULES.md`.
- Component repositories, one pull request each: their `plan.rs` and `component.rs` lose their copies of the types, the trait and the adapter.
- No change to the Runtime, the contract or any Provider.
