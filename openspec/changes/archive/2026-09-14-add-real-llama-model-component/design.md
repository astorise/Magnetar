## Context

`components/llama`'s own README had stood, since early in this repository's history, with the honest status "Empty template... not yet scheduled against any OpenSpec change." The original Tachyon scope charter names additional Model Components as future work; the `model-component-graph-contract` (`magnetar:model-component-graph@1.2.0`) was designed generic from the start (its own WIT doc comments already frame it as architecture-general, not Qwen-specific), and `wire-generic-inference-component-runtime`/`add-second-component-fixture-for-registry-multiplicity-proof` already proved the *registry* serves multiple Components -- but only ever using Qwen bytes (under different digests) or a deliberately synthetic, degenerate fixture. Neither proved the contract serves a genuine *second real architecture family*.

## Goals / Non-Goals

**Goals:**
- A real, working Llama Model Component, built through the identical toolchain and validated the identical way `components/qwen`'s own checked-in fixture is.
- Proof that the generic contract's genericity is real: an independently-compiled second-family binary produces byte-identical graphs to Qwen's for the same config, and correctly reacts to a real per-family config difference (QKV bias) without any hardcoded branching in either Component.

**Non-Goals:**
- Real Llama checkpoint *ingestion* end-to-end (a real public Llama checkpoint downloaded and generating real text). `loaders/huggingface`'s existing `normalize_tensor_name`/`config.rs` already accept a Llama-shaped bundle unmodified (identical tensor-naming convention, bias tensors simply absent), so this is plausible without further loader changes, but proving it against a real downloaded checkpoint is a separate, larger verification effort (network access to a real gated/ungated Llama checkpoint, license considerations) left for a future chantier.
- Mistral, Gemma, or any other additional architecture family. This change closes the "at least one second real family" gap; further families are additional, separate future work.
- A `llama-model-component` capability spec anywhere near `qwen-model-component`'s own ~40-requirement depth (built up over many chantiers across this repository's history). This change adds a small, honestly-scoped set of requirements for what is actually verified today; the spec can grow the same way Qwen's did, incrementally, as real Llama-specific work (ingestion, tokenizer family, adapters) actually lands.
- Rewriting `components/qwen`'s graph-building code to be shared between the two Components (e.g. a common crate). Each Model Component is its own independently-versioned, independently-compiled WASM binary by design (`externalize-runtime-extension-modules`); sharing Rust source between two separate repositories is a packaging question for those repositories' own maintainers, out of scope for this Magnetar-repository-side change.

## Decisions

- **Structurally identical graph-building code, documented as a deliberate consequence of real architectural fact**, not introduced as an unexplained duplication. `components/llama/src/lib.rs`'s own module doc comment states the reasoning directly (Qwen2's architecture is Llama's with an added QKV bias term), matching how HuggingFace's own `transformers` library documents the same relationship.
- **Verify via cross-binary graph comparison, not a new interim Rust-native oracle.** Qwen's own interim `magnetar_runtime::qwen_model_component` Rust-native graph builder predates the Component Model path entirely and is not something this change replicates for Llama (`components/llama`'s README already states this explicitly, matching the pre-existing status quo the empty template's README noted). Instead, the real Qwen Component itself serves as the cross-check oracle: same config in, same graph out, for two independently-compiled binaries.
- **Fix the test-only operator-kind-code gap in place, minimally.** `qwen_operator_kind_code` is `#[cfg(test)]`-only and purely additive here (`"add" => Some(9)`); no prior graph-hash comparison ever needed to hash a bias-bearing graph, since the shared fixture config defaults `attention_bias: false`. Found and fixed while writing the cross-architecture test itself (a real `GraphValidationFailed` locally, not a hypothetical).

## Risks / Trade-offs

- **No real end-to-end Llama generation proof yet** (ingestion + tokenizer + a real downloaded checkpoint). Documented as a Non-Goal; the graph-production-layer proof this change adds is real and meaningful on its own (the same layer `qwen-model-component`'s own graph requirements govern), but a reader should not mistake it for "Llama inference works end to end today."
- **The new `llama-model-component` capability spec is intentionally thin.** It states what this change actually verifies, not an aspirational full spec; it will need real extension once ingestion/tokenizer/adapter work for Llama actually lands, the same way `qwen-model-component`'s own spec grew.
