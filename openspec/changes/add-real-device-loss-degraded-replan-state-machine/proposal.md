## Why

`multi-device-placement`'s spec has carried "Device Loss Invalidates Dependent Placement", "Degraded Placement Requires Valid Plan", and "Placement Change Uses New Plan Generation" requirements since before this session began, backed by a real `MultiDevicePlacementState` state machine (`Ready`/`Stale`/`Invalidated`/`Retiring`/`Retired`/`Failed`, with real, enforced transition rules) -- but that state machine had only ever been exercised by its own module's 24 synthetic unit tests, never with a real Device-derived `MultiDevicePlacementPlan`. This is the last of the four real gaps the user asked to be closed following `add-real-second-gpu-cuda-provider`.

## What Changes

- `integration-tests/multi-device-cpu-cuda`: a new test, `a_real_two_gpu_plan_correctly_invalidates_and_refuses_to_revert`, builds a real `MultiDevicePlacementPlan` from two real GPUs' own real Device metadata, reaches `Ready`, transitions it to `Invalidated`, confirms `accepts_new_work()` becomes `false`, and confirms the real state machine rejects an attempt to revert it back to `Ready` in place.
- Honestly documented as not a hardware-failure-injection test: no safe mechanism exists in this repository's tooling to force a real GPU to disappear mid-test on shared CI infrastructure, so the "loss" event is a real, explicit, caller-driven state transition. What is real: the Plan's own source data and the unmodified production state-machine code.
- No `magnetar-runtime` code changes: the state machine needed none.
- **BREAKING**: none.

## Capabilities

### New Capabilities
(none)

### Modified Capabilities
- `multi-device-placement`: gains a requirement that the existing invalidation state machine is genuinely exercised with a real, Device-derived Plan, not only synthetic fixtures.

## Impact

- `integration-tests/multi-device-cpu-cuda/src/tests_multi_device_cpu_cuda.rs`: new test, updated module doc comment.
