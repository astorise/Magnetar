# Derive Capability Contracts from Existing Runtimes

## Why

Magnetar's Compute capability is currently only a marker, so future contracts
still need stable boundaries for device execution, model inference, and
application-facing AI operations. Deriving those boundaries from pinned Candle
and Crane interfaces reduces the risk of exposing provider-specific or
stateful implementation details through WIT.

## What Changes

- Analyze pinned revisions of Candle and Crane and record the source symbols
  that motivate each responsibility family.
- Publish a three-layer taxonomy covering low-level execution, model
  execution, and application-facing AI abilities.
- Map every source responsibility to Magnetar's existing Provider, Capability,
  Component, and Device roles instead of assuming that every source interface
  becomes a Capability.
- Document dependencies, candidate WIT packages, native-versus-Component
  boundaries, and fallback semantics.
- Keep all package names provisional; this change does not expand the Compute
  marker, add final WIT contracts, or change runtime behavior.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `capability`: Require proposed Capability families to be traceable to source
  evidence and to document responsibilities, dependencies, portability
  boundaries, and fallback behavior before a WIT contract is introduced.

## Impact

- Adds an architecture document under `docs/architecture/`.
- Extends the existing capability specification with contract-derivation
  requirements.
- Does not change Rust APIs, dependencies, the existing
  `magnetar:compute/run@1.0.0` marker contract, or runtime behavior.
