# Model Instance Lifecycle

Model Instance is the Runtime-owned loaded inference context between Model
Loading and inference use. It is not a Model Artifact, not a residency record,
not a session, not a Provider handle, and not KV cache state.

Runtime owns Model Instance identity, registry lookup, lifecycle, readiness,
usage accounting, policy, cache dependency tracking, unload, reload, failure
mapping, and redacted observations.

## Identity

`ModelInstanceId` is opaque and Runtime-issued. It does not encode Provider
handles, Device handles, memory pointers, model weights, or raw native
resources. Possessing an ID does not grant authority; Runtime lookup and policy
checks remain authoritative.

## Definition

A Model Instance definition binds:

- Model Artifact identity
- architecture implementation identity
- one or more Model Residency records
- tokenizer compatibility metadata
- Provider/Device placement metadata
- Resource Affinity
- Runtime policy
- active adapter state
- associated sessions
- KV Cache and Prefix Cache dependencies
- usage counters and mutation version

The public status view is redacted and never exposes raw prompts, raw weights,
raw cache contents, Provider handles, Device handles, or memory pointers.

## Lifecycle And Readiness

Lifecycle and readiness are separate. Lifecycle tracks where the instance is in
the Runtime state machine: creating, loading, warming, ready, active, idle,
draining, suspended, reloading, unloading, unloaded, failed, invalid, or
removed.

Readiness tracks whether inference can be admitted: not-ready, ready,
read-only, draining, suspended, or failed. Readiness checks consider residency,
Provider readiness, Device readiness, adapter readiness, memory pressure,
Runtime policy, and browser/native support.

## Creation

Creation requires successful Model Loading or an explicit policy-controlled load
path. Creation checks validate artifact identity and trust, architecture
availability, residency plan validity, Memory Manager admission,
Provider/Device compatibility, tokenizer compatibility, Runtime policy, and
browser/native constraints. Runtime does not publish a ready instance until
these checks pass.

## Warmup

Warmup is policy-controlled. Supported warmup steps include Provider
initialization, kernel preparation placeholder, operator graph preparation
placeholder, shape plan preparation placeholder, tokenizer/model metadata
validation, small test execution placeholder, memory residency verification,
and adapter readiness verification. Warmup failure leaves the instance failed or
not ready according to policy.

## Usage References

Generation, Sessions, and Scheduler acquire Runtime-managed usage before
execution and release it on completion, failure, or cancellation. Normal unload
is blocked while active usage exists unless forced policy applies.

## Sharing

Sharing is explicit and policy-controlled. Runtime-local sharing is allowed only
when adapter state, cache privacy, Prefix Cache privacy, and Resource Affinity
are compatible. Tenant-isolated sharing requires matching tenant metadata.
Private instances are never shared.

## Adapter Relationship

Adapter activation records active adapter set, activation scope, merge state,
and semantic mutation version. Adapter changes can invalidate dependent KV cache
and Prefix Cache entries and affect batching compatibility and determinism
metadata.

## KV Cache And Prefix Cache

Model Instance tracks dependent KV caches and Prefix Cache entries. Unload,
incompatible reload, and semantic mutation invalidate dependent cache state
according to policy and return a structured invalidation report.

## Generation Relationship

Generation uses `GenerationModelReference::ModelInstance` only when the
instance is ready, or follows an explicit policy-controlled implicit load path.
Draining, suspended, failed, invalid, unloading, unloaded, and removed instances
return structured Model Instance errors.

## Batching Relationship

Continuous Batching compatibility includes Model Instance identity, readiness,
active adapter set, Resource Affinity, Provider/Device placement, and Provider
pressure. Incompatible Model Instances do not share one execution step.

## Provider And Device Relationship

Provider health, readiness, pressure, and admission affect instance readiness
and lifecycle. Provider failures map to failed or invalid instance state.
Device loss, reset, pressure, or unavailability can suspend, invalidate, reload,
or unload an instance according to policy.

## Memory Manager Relationship

All instance residency is tracked through Memory Manager records. Unload
releases tracked memory allocations and Provider-owned opaque model resources.
Memory pressure may suspend idle instances when policy allows. Browser targets
can apply reduced behavior with explicit memory limits and structured
unsupported-feature errors.

## Suspension And Draining

Suspension can be triggered by memory pressure, Provider pressure, Device
pressure, administrative policy, browser lifecycle events, or temporary
resource loss. Suspended instances reject new operations and may resume, reload,
unload, or fail according to policy.

Draining rejects new operations while allowing active operations to complete
according to policy. Draining may be triggered by unload, reload, policy
change, Provider drain, Device pressure, Runtime shutdown, adapter mutation, or
failure isolation.

## Unload

Unload stops new operation admission, drains or rejects active operations,
invalidates dependent KV caches and Prefix Cache entries, releases adapter
associations, releases Memory Manager residency, releases Provider-owned
resources, updates lifecycle, emits observations, and avoids dangling session
references.

## Reload

Reload is treated as a new validated loading process. It may create a
replacement instance with updated residency, Provider/Device placement, compute
dtype, quantization handling, adapter compatibility, and Resource Affinity.
Runtime rejects silent active semantic mutation unless explicit policy allows
it. Session migration is policy-controlled.

## Non-Goals

Model Instance does not define operator graph semantics, kernel ABI, distributed
serving, persistent instances across restarts, model download protocol, adapter
math, sampling behavior, chat conversation storage, or raw weight access.
