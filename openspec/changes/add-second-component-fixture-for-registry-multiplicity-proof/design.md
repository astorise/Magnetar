## Context

`wire-generic-inference-component-runtime` built a real, digest-keyed, trust-store-driven registry replacing the single hardcoded Qwen singleton. Every test exercising it so far, though, registers the *same* underlying bytes (`QWEN_REAL_COMPONENT_BYTES`) under different digests, trust decisions, or call paths -- proving the registry's mechanics (trust enforcement, idempotence, matching the pre-existing singleton path) but never proving its actual multiplicity claim: that two genuinely different Components can be registered and used at the same time, with no leftover model-family branching anywhere between registration and graph production.

## Goals / Non-Goals

**Goals:**
- A second real Component artifact, structurally different enough that its graph output can never accidentally coincide with Qwen's, built through the same real toolchain (`cargo build --target wasm32-unknown-unknown`, `wasm-tools component new`) as the real Qwen Component -- not a hand-edited `.wat` stub standing in for a real Component.
- Prove both artifacts are usable *concurrently* within one process/registry, and that using one does not corrupt or evict the other.

**Non-Goals:**
- Building a second production Model Component (a real architecture family beyond Qwen). That is a much larger, separate chantier (still open in the original Tachyon scope charter); this fixture exists solely to exercise the registry's multiplicity, and is explicitly documented as test-only, never production.
- Modifying `components/qwen` itself. Too risky -- it is the real production submodule and the actual graph source for real generation; a synthetic fixture lives entirely outside it, in `magnetar-runtime/fixtures/components/`, matching where `qwen-real.component.wasm` itself already lives as a checked-in test fixture.

## Decisions

- **Reuse the real WIT contract, degenerate the graph, not the contract.** The synthetic Component implements the identical `model-component-graph-producer` world (same imports/exports) as the real Qwen Component -- so it passes the same WIT-consistency validation any real Component would -- but its graph omits every decoder layer (`embedding -> rmsnorm -> matmul` only), making its node count and operator-sequence hash structurally impossible to coincide with Qwen's, rather than relying on incidental parameter differences that could theoretically collide.
- **Extend the existing consolidated test, not a new `#[test]`.** `register_inference_component_artifact_enforces_trust_is_idempotent_and_matches_the_singleton_path` already documents, from a real caught CI failure, that splitting assertions sharing `QWEN_REAL_COMPONENT_BYTES`'s digest across separate test functions races under Rust's default parallel test execution (a later test's trusted registration can land before an earlier test's untrusted-rejection check runs, silently passing through the idempotent-return fast path). A first attempt at a standalone MAG-07 test reproduced this exact race locally (confirmed via 4 repeated full-suite runs, 2 of 4 failing) before being folded into the existing consolidated test, which eliminates it by construction.
- **Manifest mirrors the real fixture's format exactly**, including `component.role: "qwen-model-component"` (the only role WIT-validated today) but with an honestly-labeled `component.name`/`description` making clear it is test-only.

## Risks / Trade-offs

- **Still only one real production Model Component family (Qwen).** MAG-07 asked specifically whether the registry supports multiple *Components*, which this closes; it does not by itself close the separate, much larger question of a second production *model architecture*, which remains open in the broader scope charter.
- **The synthetic Component's graph is not runnable end-to-end** (no attention/MLP/KV-cache nodes at all) -- it proves registry multiplicity and graph-shape independence, not that a degenerate graph can itself execute a real generation. That was never the claim MAG-07 asked to be proven.
