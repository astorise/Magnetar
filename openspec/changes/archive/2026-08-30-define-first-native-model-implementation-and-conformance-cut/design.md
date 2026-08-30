# Design

## Goal

Define and enforce the first native implementation cut for Magnetar by making the
normal CLI generation path depend on Runtime-owned model execution evidence
instead of caller-supplied placeholder logits.

## Scope

This change covers the first-native fixture implementation path used for local
conformance and CLI smoke flows. It does not claim production-quality model
output. The fixture is intentionally tiny and deterministic so that component
boundaries, evidence, cancellation, and failure modes can be tested without a
large model artifact.

## Runtime Boundary

The runtime owns prompt tokenization, execution graph validation, kernel
selection, kernel dispatch, provider execution, tensor ownership, sampling, and
decode. The CLI can pass prompt text and render output, but it cannot provide
logits or bypass runtime execution.

The public entry point for this cut is
`magnetar_runtime::run_first_native_fixture_generation`. It returns generated
text together with the `InferenceApiObserver` evidence trail used by tests and
CLI callers to prove that the runtime path executed.

## CLI Boundary

`magnetar-cli` routes `run`, `chat`, `serve`, and bounded `agent` smoke flows
through the runtime fixture generation entry point. CLI transcript formatting
remains CLI-owned, while all generation evidence comes from the runtime.

Cancellation keeps a runtime session boundary in place and makes the chat
session unusable after cancellation, preserving the expected CLI lifecycle
contract.

## Freeze and Migration Inventory

The architecture freeze is represented in code by
`architecture_freeze_1()`. Reopening is limited to blockers, security defects,
and strict conformance failures.

The Phase 0 migration inventory is represented by
`phase_0_migration_inventory()` and mirrored in
`migration-inventory.md`. The CLI placeholder logits path is marked as removed
from the normal path. Remaining bypass classes are intentionally retained only
as bounded fixtures, test harnesses, or deprecated migration inventory entries.

## Verification

The implementation is verified by workspace formatting, compilation, tests,
clippy, strict OpenSpec validation, WIT/WAT validation through `wasm-tools`, and
`cargo deny --all-features check`. Detailed command evidence is recorded in
`implementation-evidence.md`.
