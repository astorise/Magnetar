## Context

Magnetar already defines Components as portable WebAssembly Component Model
artifacts that import WIT Capabilities through Runtime-owned Link Plans. The
missing piece is an executable local engine implementation that preserves the
existing Runtime boundary: Components request Capabilities, the Runtime owns
authorization and linking, and Providers/Devices remain native implementation
details selected only by Runtime policy.

The first engine integration is implemented in `magnetar-runtime` behind the
optional `wasmtime-component-engine` Cargo feature. The default feature set
remains engine-neutral.

## Goals / Non-Goals

**Goals:**

- Add a concrete Wasmtime Component Model adapter behind `ComponentEngine`.
- Validate, prepare, instantiate, invoke, interrupt, and destroy local
  Components through Magnetar-owned APIs.
- Translate approved Link Plans into private Wasmtime linker entries.
- Preserve fail-closed authority semantics for absent imports, unauthorized
  WASI, unsupported signatures, async host adapters without typed support, and
  WIT resource imports without Runtime mappings.
- Provide reproducible WAT fixtures and CI validation for real Component
  execution.
- Keep Wasmtime-native objects out of public Magnetar APIs.

**Non-Goals:**

- Component artifact trust, signatures, digest addressing, registries, and
  Tachyon distribution.
- Scoped authorized WASI authority beyond the fail-closed rule.
- Complete Compute inference execution from a Component fixture.
- Typed async Runtime host adapters for long-running Provider work.
- WIT resource ownership transfer between Components and Runtime resources.
- A generic dynamic invocation ABI.

## Decisions

1. **Use Wasmtime behind a feature-gated adapter.**

   Wasmtime provides Component Model validation, linking, typed invocation,
   limits, and interruption support. It is kept private to
   `component_wasmtime.rs`, and public exports remain Magnetar-owned types.
   Alternatives considered were a no-op engine or making Wasmtime part of the
   public API; both would either avoid real execution or bind the architecture
   to one engine.

2. **Keep public invocation synchronous for the first implementation.**

   The current `ComponentManager` and `ComponentEngine` boundary uses `&mut
   self` and synchronous calls. The adapter enables Wasmtime Component Model
   support, but only unit-shaped synchronous host adapters are linked today.
   Async-required host signatures fail closed until typed Runtime adapters are
   introduced, avoiding exposure of a concrete async runtime or accidental
   thread blocking semantics.

3. **Use WAT fixtures instead of generated checked-in WASM binaries.**

   WAT fixtures are small, reviewable, reproducible, and validated in CI with
   `wasm-tools`. Tests compile them through Wasmtime's text support, so no
   network access or Tachyon dependency is required.

4. **Treat WIT resources as unsupported without explicit Runtime mappings.**

   Engine resource tables remain private and are not stable Magnetar handles.
   The first adapter rejects resource imports without an explicit Runtime
   mapping, which prevents cross-instance resource forgery and avoids premature
   ownership semantics.

5. **Use epoch interruption as a private implementation detail.**

   Deadlines and Runtime-requested interruption are normalized to stable
   `ComponentError::Interrupted` variants. Epoch deadlines are bounded to avoid
   overflow and are not visible through public APIs.

6. **Normalize traps and host-adapter failures separately.**

   Component traps map to stable redacted trap errors. Host adapter failures
   reached through an authorized import map to invocation failures so they are
   not misclassified as engine traps or Provider resolution failures.

7. **Introduce a local-path artifact reference as the extension point.**

   `ComponentArtifactReference::LocalPath` records the current local artifact
   source without adding trust, digest, publisher, or registry semantics. The
   dedicated artifact/trust change can extend this boundary later.

## Risks / Trade-offs

- **Async host adapters are scoped down** -> unsupported async signatures fail
  closed until a typed adapter exists.
- **WIT resources are not transferred** -> resource imports fail closed rather
  than creating unstable Runtime resource identities.
- **Wasmtime adds build cost when enabled** -> the dependency is feature-gated
  and disabled by default.
- **Coverage denominator changes with the new adapter** -> the coverage
  baseline is updated to the measured workspace result after the new feature
  code and fixtures are included.

## Migration Plan

The change is additive. Existing engine-neutral Component APIs continue to
compile without the Wasmtime feature. CI validates engine-neutral tests by
default and concrete Wasmtime tests in a feature-enabled job. Rollback is the
removal of the optional feature, adapter module, fixtures, and CI fixture
validation.

## Open Questions

- Which typed Runtime adapter should introduce true async host-call execution?
- Which change owns scoped authorized WASI value-level policy?
- Which artifact/trust model fields extend `ComponentArtifactReference` first:
  digest, publisher identity, registry locator, or signature bundle?
