# Tasks

## 1. OpenSpec Contract

- [x] 1.1 Move the capability delta to `specs/capability/spec.md` with valid OpenSpec headings.
- [x] 1.2 Add `design.md` covering versioning, ownership, lifecycle, errors, and exclusions.
- [x] 1.3 Replace planning-only checklist entries with verifiable implementation tasks.

## 2. WIT Contract

- [x] 2.1 Publish `magnetar:compute@1.1.0` in `magnetar-runtime/wit/compute.wit`.
- [x] 2.2 Define fixed-width shape, dtype, view, and tensor descriptors.
- [x] 2.3 Define opaque tensor, graph, and operation resources.
- [x] 2.4 Define submit, status, await, cancel, output retrieval, and structured error semantics.

## 3. Runtime Metadata and Documentation

- [x] 3.1 Update canonical Compute capability metadata and tests to advertise `1.1.0`.
- [x] 3.2 Document the stabilized coarse Compute boundary in the architecture taxonomy.

## 4. Verification

- [x] 4.1 Validate the WIT package syntax with `wasm-tools component wit`.
- [x] 4.2 Run Rust formatting, clippy, and tests for the workspace.
- [x] 4.3 Run strict OpenSpec validation for `stabilize-compute-run-boundary`.
