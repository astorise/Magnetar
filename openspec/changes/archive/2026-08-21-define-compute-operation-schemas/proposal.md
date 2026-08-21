# Define Compute Operation Schemas

## Why

`magnetar:compute/run` now has:

- a coarse graph submission boundary
- a tensor descriptor model
- an explicit data movement model
- a shared compute error model
- a compute operation catalog

The next step is to define concrete portable operation schemas that can appear
inside a Compute Graph.

These operation schemas describe semantic operations.

They are not individual WIT functions.

They are not separate Capabilities.

They are graph node variants interpreted by the Runtime and executed by a
compatible Provider.

The goal is to make compute graphs validatable before Provider execution while
keeping native kernel selection, memory planning and hardware optimization
inside the Provider.

## What Changes

This proposal introduces the initial Compute Operation Schema model.

Each operation schema defines:

- operation identifier
- operation family
- input value rules
- attribute rules
- output descriptor rules
- dtype compatibility rules
- shape validation rules
- Provider support requirements
- structured error behavior

The initial schema set covers foundational inference operations:

- descriptor and view operations
- elementwise unary operations
- elementwise binary operations
- comparison operations
- conditional selection
- reductions
- linear algebra
- indexing
- concatenation
- random generation

This proposal intentionally excludes:

- convolution schemas
- pooling schemas
- spatial transform schemas
- attention-specific fused schemas
- quantized operation schemas
- custom kernels
- autograd
- training graphs

Those require dedicated follow-up changes because their numerical, layout and
precision semantics need more detailed specification.

## Impact

Compute Graph validation becomes more precise.

Providers can advertise support at the operation-schema level.

The Runtime can reject unsupported or malformed compute graphs before invoking a
Provider.

Future specialized operation sets can extend the same schema model without
fragmenting the Capability model.