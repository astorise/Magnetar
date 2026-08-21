# Define Compute Operation Catalog

## Why

`magnetar:compute/run` is the coarse compute submission boundary.

The previous capability taxonomy identified several tensor operation families
from Candle that are useful coverage areas for future Compute revisions.

These operation families should be documented as a catalog before concrete WIT
operation schemas are introduced.

The catalog defines what kinds of work `magnetar:compute/run` is expected to
cover over time.

It does not make each operation family a separate Capability.

It does not introduce one WIT function per eager tensor primitive.

It does not yet define exact numerical semantics.

## What Changes

This proposal introduces the Compute Operation Catalog.

The catalog groups supported and future compute work into semantic families:

- tensor descriptors and views
- construction and allocation requests
- data movement and conversion
- elementwise operations
- comparisons and selection
- reductions
- linear algebra
- convolution and spatial transforms
- indexing and updates
- random generation
- synchronization and completion

The catalog also records exclusions:

- autograd
- training graphs
- arbitrary Rust custom operations
- raw backend storage
- backend-specific kernel names
- native device queue synchronization from Components

Future changes will define concrete operation schemas, validation rules,
numerical semantics and WIT representation.

## Impact

The Compute Capability gains a documented scope.

Providers can advertise which operation families they support.

The Runtime can validate operation family coverage before execution.

Future operation-specific changes can evolve `magnetar:compute/run` without
fragmenting the Capability model.