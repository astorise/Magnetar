# Define Compute Error Model

## Why

`magnetar:compute/run` now has a coarse graph submission model, tensor
descriptors, opaque tensor resources and explicit data movement.

These contracts need a shared error model.

Provider errors are often backend-specific, unstable and not portable.

Magnetar must expose stable structured errors to Components while keeping native
backend diagnostics as optional debug information.

The error model must distinguish validation failures, resolution failures,
affinity failures, unsupported features, execution failures, cancellation and
interruption.

It must also avoid promising automatic failover or live state migration.

## What Changes

This proposal introduces a Compute Error Model shared by compute-related
contracts.

The model includes:

- stable error categories
- error phases
- structured error payloads
- optional diagnostics
- retry and recovery hints
- resource affinity error reporting
- Provider and Device error reporting through stable identifiers

The model applies to:

- tensor descriptor validation
- compute graph submission
- data movement
- Provider selection
- Provider execution
- operation completion
- cancellation
- interruption

This proposal does not expose native backend error types.

This proposal does not make backend diagnostic strings part of the stable
contract.

This proposal does not introduce automatic fallback, migration or replay.

## Impact

Components can handle compute failures predictably.

Providers can map native errors to stable Magnetar error categories.

The Runtime can report failures without leaking backend internals.

Future operation schemas can reuse the same error model.