## Why

The contract between a Model Component and the Runtime, the `magnetar:model-component-graph` WIT package, has no home. Its source is `magnetar-runtime/wit/model-component-graph.wit`, and every Component carries its own copy: nine files at five versions (1.8.0 in Gemma, Llama, Mistral and OLMo, 1.9.0 in SmolLM, 1.10.0 in DeepSeek, 1.12.0 in ModernBERT, 1.14.0 in Qwen and in the Runtime). Each Component also runs the bindings generator on its copy, so the bindings are generated eight times from files that are, version for version, the same.

That costs the people who write Components most. A team outside this repository, adopting Magnetar for a model family, has to clone Magnetar to read the contract, copy a file from it, and pick the right one of five. Nothing tells them which versions a Runtime accepts, and nothing but a reviewer's attention keeps a new version additive over the last. The contract's own rule, that an earlier version keeps linking, is enforced by tests inside the engine, far from the file a Component's author edits.

The contract belongs in a repository of its own, which a Component depends on instead of copying, and which holds the checks that make the contract's promise mechanical. `magnetar-component-sdk` (change `introduce-component-sdk`) is the Components' shared code; this is the contract they are written against.

## What Changes

1. **`magnetar-component-api`**, a new repository (https://github.com/astorise/magnetar-component-api), holding every released contract version, 1.2.0 to 1.14.0, one immutable directory each, recovered from Magnetar's history and cross-checked against the copies Components carry; the guest bindings of each version, generated once, behind one cargo feature per version; the checks that a release only adds to the one before it and that a released file never changes; and a template Component.
2. **Used as a git dependency pinned by commit**, like the SDK, until it is stable and published: `magnetar-component-api = { git = "https://github.com/astorise/magnetar-component-api", rev = "…", features = ["v1_8"] }`. A Component drops its `wit/` copy and its call to the bindings generator.
3. **Components move to it one at a time**, each emitting the same graph as before (the golden graphs of the SDK migration), then the Runtime takes its versions from the repository instead of keeping its own.
4. The SDK keeps no dependency on the API: its `wit_graph_sink!` macro takes the Component's `gb` alias and works on the shared bindings as it did on a Component's own.

## Capabilities

### New Capabilities

- `component-api`: what the contract repository holds and promises, and what a Component that targets it does.

### Modified Capabilities

None yet. `wit` (release versions, release gate) and `model-component-graph-contract` (where a version is defined) change when the Runtime stops keeping its own copy; that cutover is its own step in the tasks, with its own delta, so that nothing is modified here before it is true.

## Impact

- New repository `magnetar-component-api` (not a submodule of Components: they depend on it by pinned revision, then by version).
- Component repositories, one pull request each: `wit/` and the `generate!` call go, a dependency and a feature come.
- No change to the contract, the Runtime or any Provider in this change.
