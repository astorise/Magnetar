## Context

See proposal.md: investigation found the capability the tracking issue
asked about already exists and is already correct. This is a
documentation-only Change (plus one regression test); none of the
usual reasons for a design document (cross-cutting change, new
dependency, security/performance/migration complexity, real ambiguity
to resolve before coding) apply, since there is no code to design.

## Goals / Non-Goals

**Goals:**
- Record, canonically, that a `Loading` (or any other non-terminal)
  Model Instance can be explicitly failed and then unloaded, and that
  doing so before anything was ever bound produces a clean, empty
  release report rather than an error.

**Non-Goals:**
- Adding a new lifecycle transition or a dedicated "cancel" entrypoint
  -- `fail_instance` plus `unload_model_instance` already does this
  correctly today; inventing a second, narrower path for the same
  outcome would be pure duplication.
- Changing whether/when an instance enters `Loading`, or how long it
  may remain there -- explicitly out of scope per the tracking issue.

## Decisions

### D1: Document the existing `fail` + `unload` path rather than add a new one

`ModelInstance::fail`/`invalidate` already bypass
`allows_transition_to` entirely (by design -- they are the "something
went wrong, force a terminal state" escape hatch, not a validated
transition), and `(Failed, Unloading)` is already valid. Adding a
dedicated `Loading -> Unloading` transition or a new
`cancel_model_instance` entrypoint would be a second way to reach the
same place, not a capability that does not exist today. The spec
requirement added by this Change describes the real, already-taken
path (fail, then unload) rather than a new API surface.

## Risks / Trade-offs

- **[Risk]** A reader could still expect a single dedicated
  "cancel" call rather than two calls (fail, then unload). → Mitigated
  by the new requirement's own scenario spelling out the two-step
  sequence explicitly, and by `ModelInstanceManager::fail_instance`'s
  and `Runtime::unload_model_instance`'s own doc comments (unchanged by
  this Change, already accurate).

## Migration Plan

None -- no code changes, no behavior changes. Add the test, add the
spec requirement, archive.
