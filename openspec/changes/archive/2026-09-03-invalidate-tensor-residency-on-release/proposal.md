## Why

A revalidation audit of `transactional-weight-materialization` (commit
`633e942`) found that `MemoryManager::release(allocation)` only changes the
`MemoryAllocation`'s own state -- it never removes the corresponding entry
from `MemoryManager`'s `tensor_residency` map. Weight materialization
rollback (`WeightMaterializationTransaction::abort`) and Model Instance
unload (`Runtime::unload_model_instance`) both already release a weight's
Provider-owned tensor and its Memory Manager allocation, but neither removes
its `TensorResidency` record. `tensor_residency()` therefore keeps reporting
a resource as resident indefinitely after both the Provider storage and the
allocation it described are gone -- a metadata leak that grows on every
failed materialization attempt and every load/unload cycle, and a real
divergence risk if `tensor_residency()` is ever relied on as proof a
resource actually exists.

## What Changes

- Add `MemoryManager::remove_tensor_residency(&TensorResourceId) ->
  Option<TensorResidency>`.
- `WeightMaterializationTransaction::abort` (rollback) now removes each
  staged weight's residency record, after releasing its Provider tensor and
  before releasing its Memory Manager allocation.
- `Runtime::unload_model_instance` now removes each released weight
  resource's residency record, after resolving its owning Provider through
  that same record and releasing its Provider tensor, and before releasing
  its Memory Manager allocation.
- Strengthen the `memory` spec's existing "Memory Manager Releases Model
  Residency" requirement so "releases associated memory records" is
  unambiguous about removing the `TensorResidency` entry itself, not only
  changing the underlying allocation's state.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `memory`: "Memory Manager Releases Model Residency" gains an explicit
  scenario requiring `tensor_residency()` to return `None` for a resource
  after its residency is released, not only that "associated memory
  records" are released in some unspecified sense.

## Impact

- `magnetar-runtime/src/memory.rs`: new `remove_tensor_residency` method.
- `magnetar-runtime/src/first_native_runtime.rs`:
  `WeightMaterializationTransaction::abort` wiring.
- `magnetar-runtime/src/runtime.rs`: `unload_model_instance` wiring.
- Test coverage: the existing rollback test
  (`check_weight_materialization_failure_never_reaches_ready`), the existing
  unload test (`check_unload_releases_weight_resource_allocations`), and the
  existing repeated load/unload test
  (`check_repeated_load_unload_does_not_accumulate_weight_storage`) each
  gain a `tensor_residency(id).is_none()` assertion.
