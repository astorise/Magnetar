## Why

`qualify-real-checkpoint-performance`'s CUDA campaign (GPU run 41041038682, RTX 3060) showed every request on one resident `ProductionLoadedModel` slower than the last: TPOT 1.48 s on the cold request, 5.34 s five requests later, and +18.8 ms per position inside a request. The cost of a decode step grew with every token the instance had ever processed. The same drift reproduces without a GPU on the tiny bundle (TPOT 7 ms to 45 ms over 12 requests of 32 tokens).

Four causes, each confirmed by a regression test that records every bookkeeping table after each request:

1. `MemoryManager` kept every released allocation in the map that `allocate` walks several times (`find_reusable_allocation`, `active_bytes`, class pressure), and its observation logs grew without bound.
2. A graph edge both pre-admitted by the Runtime (`admit_kernel_output`) and explicitly written through the Provider was then self-admitted again by the Provider on the next step, overwriting the Runtime's residency record and orphaning one allocation per edge per step (Reference CPU, `providers/cpu`, `providers/cuda`).
3. KV commit and abort released superseded generations with `release_tensor`, leaving one stale Provider tracking entry per layer and role per step.
4. A finished request left its committed KV tensors, `.pending` staging tensors, closed session and released cache records behind.

## What Changes

- `MemoryManager` keeps only live (active, reusable, pending) allocations in its working map. Released, evicted, cancelled and failed ones move to a history bounded by `RETIRED_ALLOCATION_HISTORY_CAPACITY` (4096), still returned by `allocations()` in id order. `is_live_allocation` is added. Observation logs are bounded by `MEMORY_OBSERVATION_LOG_CAPACITY`.
- Providers treat an output as pre-admitted whenever its residency record names an allocation other than the one they track for the id, and drop their stale tracking entry instead of overwriting the caller's record.
- KV commit/abort release superseded generations with `release_admitted_tensor`.
- The model execution engine gains a crate-private `release_request_state`; `ProductionLoadedModel` calls it before closing each request's session, then forgets the closed session and its released KV caches through the new `Runtime::forget_closed_inference_session` and `KvCacheManager::forget_released`.

## Capabilities

### Modified Capabilities

- `memory`: bookkeeping proportional to live memory.

## Impact

`magnetar-runtime` (memory, Reference CPU, KV commit, production loaded model), `providers/cpu`, `providers/cuda`. Additive public API only (`is_live_allocation`, two capacity constants, `forget_closed_inference_session`, `forget_released`). Allocations older than the retired history are no longer listed by `allocations()`; releasing one again still fails with `InvalidAllocationHandle`, as a double release always did.

## Non-goals

- The ~1.2 s fixed cost of a CUDA decode step and its per-step host work are measured by `real_checkpoint_decode_cost_profile` and belong to `attribute-redacted-inference-costs` (A06) and the preparation/replay changes.
