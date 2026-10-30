## Decisions

### Live and retired allocations

`allocate` and its pressure checks walk the allocation map, so that map holds only live allocations. Terminal allocations move into `retired` (with an insertion-order queue for eviction), capped at 4096. `allocations()` merges both back in allocation-id order, so callers inspecting a recent release still see it as `Released`. `record_tensor_residency` and `record_resource_residency` accept any known handle, live or retired. `Runtime::release_kv_cache_memory` treats any non-live allocation as already released, which also covers one that has aged out of the history.

The observation logs drop all but the most recent capacity once they reach twice the capacity, keeping trimming O(1) amortized while `observations()` still returns a contiguous slice.

### Pre-admission is decided by ownership, not by first sight

A Provider used to treat an output as pre-admitted only if it had never self-admitted that id. An edge first written explicitly (tracked by the Provider) and later pre-admitted per step therefore looked self-admitted forever. The Provider now compares the residency record's allocation with the one it tracks: a different one means the caller re-admitted it, so the caller's record wins and the Provider releases and forgets its stale entry.

### Per-request release on resident instances

`ProductionLoadedModel` outlives its requests, so it releases the engine's per-request KV state (committed bindings and `.pending` staging resources, through the Provider that wrote them) before closing the session, then forgets the closed session and its released caches. Caches still named by a prefix cache entry are kept. One-shot entry points build a fresh Runtime per call and are unaffected.

## Verification

`resident_production_model_bookkeeping_stays_bounded_across_requests` serves six requests from one resident instance and requires every table (live allocations, residency, Provider storage and tracking, sessions, KV caches) not to grow after the second request, the bounded logs to stay within their caps, and no live allocation to be orphaned. Unit tests cover the live/retired split, history and log bounds, and cache eviction.
