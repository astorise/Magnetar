## Context

MAG-06 asked for the concurrency model to be validated explicitly, listing several candidate shapes (one generation per instance; an internal scheduler; batching; concurrent sessions; controlled instance duplication) without mandating which. Investigating the actual code before writing anything down (this session's established practice) found the answer is already settled, and more strongly than a documentation convention: `ProductionQwenLoadedModel::generate`/`generate_streaming` both take `&mut self`. Rust's own borrow checker already refuses to compile two concurrent calls on the same value without an external synchronization wrapper -- `inference-components`'s `Mutex<ProductionQwenLoadedModel>` is not an arbitrary choice serializing something that could otherwise run concurrently; it is close to the *only* legal way to share a `&mut self`-requiring value across threads at all.

## Goals / Non-Goals

**Goals:**
- State the concurrency model explicitly, in the one place (`LoadedInferenceComponent`, the Tachyon-facing type) the audit actually reviewed.
- Record the real, compiler-enforced reason it holds, not just assert a policy.

**Non-Goals:**
- Building a scheduler, batching, or an instance-pool abstraction. The audit listed these as *candidate* shapes to consider, not requirements to build; investigating the `&mut self` constraint settles which one is actually true today (one generation per instance, serialized) without needing to build anything new to enforce it -- it's already enforced.
- Changing `ProductionQwenLoadedModel`'s `&mut self` signatures to something that would *allow* concurrent generation (e.g. interior mutability). That would be a real, large design decision (multi-step decode already threads KV cache state through `&mut self.runtime` throughout `execute_generation_step`) explicitly out of scope for a documentation-closure change.

## Decisions

- **Document at `LoadedInferenceComponent` (the type), not just at each method.** Both `invoke_payload`/`invoke_payload_streaming` get a one-line pointer back to the type-level doc rather than duplicating the full explanation twice.
- **State it as "blocks, never rejects, never interleaves"** -- the actual, verified behavior of `Mutex::lock()` (not `try_lock()`) on the pre-existing code, not a description of some different behavior this change does not actually implement.
- **No production behavior change.** This is the rare chantier in this session's sequence that touches zero non-comment lines -- verified by `cargo build`/`clippy --all-targets -- -D warnings`/`fmt --check`/`doc --no-deps` all clean and the pre-existing 3 tests passing unmodified.

## Risks / Trade-offs

- **Still no built-in scheduler/batching/instance-pool**, if an embedder eventually wants one. Explicitly out of scope (see Non-Goals) -- MAG-06 asked for the model to be *validated and stated*, not for new infrastructure to be built on spec.
