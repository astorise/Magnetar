## Why

`multi-device-placement`'s spec has carried a "Per Device Memory Feasibility" requirement since before this session began ("Every Device binding SHALL satisfy its own Memory Manager capacity policy"), backed by a real, working `PlacementCandidate`/`select_lowest_cost_eligible` implementation in `magnetar_runtime::multi_device_placement` -- but that implementation had only ever been exercised by its own module's synthetic unit test fixtures, never by a real Device's own real capacity numbers. With real peer-to-peer movement now closed (`add-real-peer-to-peer-gpu-movement`) and per the user's instruction to close the remaining real gaps, this change drives that existing logic with real data for the first time.

## What Changes

- `integration-tests/multi-device-cpu-cuda`: a new test, `per_device_memory_feasibility_ranking_rejects_an_infeasible_real_gpu_and_accepts_a_feasible_one`, builds two real `PlacementCandidate`s -- one using a real GPU's real, full `DeviceMetadata.memory_capacity`, one using a deliberately, honestly constrained artificial budget (this repository's own two real GPUs are identical, so no genuinely heterogeneous real budget exists to test infeasibility against) -- and feeds both through the existing, unmodified `select_lowest_cost_eligible`, verifying it correctly selects the feasible real candidate and rejects the constrained one specifically for `MemoryInfeasible`. The constrained candidate is also given a deliberately lower cost than the feasible one, proving eligibility is checked before cost ranking.
- No `magnetar-runtime` code changes: `PlacementCandidate`/`select_lowest_cost_eligible` needed no changes, only real data.
- **BREAKING**: none.

## Capabilities

### New Capabilities
(none)

### Modified Capabilities
- `multi-device-placement`: gains a requirement that the existing memory-feasibility logic is genuinely exercised with a real Device's real capacity data, not only synthetic fixtures.

## Impact

- `integration-tests/multi-device-cpu-cuda/src/tests_multi_device_cpu_cuda.rs`: new test, updated module doc comment.
