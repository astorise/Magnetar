## Context

`reach-architecture-freeze-1` and `implement-model-format-parsers` already did the real extraction work this Change formalizes: `components/qwen`, `providers/cpu`, `formats/gguf`, and `formats/safetensors` are real, independent crates in their own repositories, each depending on `magnetar-runtime`'s public contracts and never the reverse, pinned into this repository as `160000` git submodules (`SUBMODULES.md`). What those Changes did *not* do is state this as a requirement anywhere in `openspec/specs/` — it is currently true by construction and by discipline, not by a checked rule. `reach-architecture-freeze-1`'s design.md flagged exactly this gap as an open question and deferred it. This Change closes that question: yes, formalize it.

## Goals / Non-Goals

**Goals:**
- State the externalization requirement normatively in `project-architecture`, the capability that already owns Component/Provider/Device boundary statements.
- Name the one real, deliberate exception (Reference CPU's in-crate generic test double) precisely enough that the requirement does not contradict `reach-architecture-freeze-1` task group 14's already-shipped architecture.
- Generalize the dependency-boundary CI check from "two format crates" to "every externalized module crate," since the requirement being formalized covers all three module kinds (Component, Provider, Format), not just Formats.

**Non-Goals:**
- No new extraction work. `components/llama` and `providers/cuda` remain empty templates; this Change does not require them to become real, only states that *if* they become real, they live externally (which is already where they are pinned).
- No change to how trust/compatibility is established for a module (`SUBMODULES.md`'s existing versioning/pinning model is unaffected).
- Does not decide whether *future* module kinds (e.g. a Tokenizer plugin family, should one ever become a separate extension point) fall under this requirement — scoped to Components, Providers, and Formats, the three kinds that exist today.

## Decisions

**Requirement lives in `project-architecture`, not a new capability.** That spec already owns "Component and Provider Boundary," "Components and Providers Are Distinct Extension Mechanisms," and the Magnetar/Tachyon repository-dependency-direction requirements — this is the same kind of statement (a repository/packaging boundary, not a runtime behavior contract), so it belongs beside them rather than fragmenting architecture statements across capabilities.

**The Reference CPU exception is named explicitly, not left implicit.** A requirement that simply said "Providers SHALL live outside the Core" would be false as written the moment `magnetar-runtime/src/reference_cpu.rs` is read: it defines a real, working `ReferenceCpuExecutor` in-crate today, deliberately, because `magnetar-runtime`'s own ~1000-test suite needs a concrete Provider double and migrating that suite onto an external crate dependency was already judged (task group 14) not worth the churn. Rather than word the requirement to technically-not-apply to this case, or ignore the contradiction, the requirement names the exception directly: one minimal, generic, in-crate reference implementation may exist for the Core's own test suite, distinct from and never substituting for the real externally-distributable implementation.

*Alternative considered:* phrase the requirement narrowly enough to only cover "the Provider a production build actually uses," sidestepping the in-crate double entirely. Rejected as less honest — a future reader grepping for "does the Core contain a real Provider implementation" would get a misleading "no" from the narrow phrasing when `reference_cpu.rs` visibly says otherwise.

**The CI guard generalizes by naming convention, not by an enumerated allowlist.** Every module crate today is named `magnetar-component-*`/`magnetar-format-*`/`magnetar-provider-*` (verified directly against all six submodules' `Cargo.toml` `name` fields, not assumed). A single pattern (`magnetar-(component|format|provider)-`) checked against both `magnetar-runtime/Cargo.toml` and its resolved `cargo tree` output catches any current or future externalized module automatically, without needing the guard's own list of names to be kept in sync as new submodules are added — the previous format-only guard would have needed a matching new grep pattern (or manual update) for every new module kind, which this generalization removes as an ongoing maintenance burden.

## Risks / Trade-offs

- [Risk] A future, genuinely justified reason to add a second built-in Provider (not just Reference CPU) would need either a spec amendment or would read as a violation. → Mitigation: acceptable — that would be exactly the kind of new architectural decision this repository's governance already says belongs in its own Change, not something to pre-authorize speculatively here.
- [Trade-off] The naming-convention-based CI guard is precise today but silently stops enforcing anything if a future module crate is named outside the `magnetar-(component|format|provider)-` convention. Accepted: the convention is already established and documented (`SUBMODULES.md`), and a naming-convention check is simpler and lower-maintenance than an enumerated list that must be updated by hand for every new submodule — the trade-off favors the common case (new modules follow the existing convention) over the rare one (a differently-named module slips through unnoticed).
