# Compute Operation Schemas

`magnetar:compute/run` uses Compute Operation Schemas to describe portable
graph node semantics. A schema is not a WIT function and is not a standalone
Capability. It is metadata used by the Runtime to validate a submitted Compute
Graph before a compatible Provider executes it with native kernels.

## Schema Format

Each schema has:

- a stable Provider-independent operation identifier
- one Compute Operation Family from the catalog
- typed attributes
- input arity and descriptor rules
- output descriptor rules
- dtype and shape compatibility rules
- Provider support requirements

Operation attributes are portable values such as booleans, integers, dtypes,
shapes and axes. Unknown attributes are rejected unless a later schema revision
explicitly defines an extension point.

## Graph Integration

Compute Graph nodes carry operation descriptors. A descriptor may reference a
concrete schema identifier and includes operation attributes. Graph validation
resolves all node inputs and outputs to tensor descriptors, validates the
schema, then checks whether the selected Provider advertises support for that
schema. Providers may still choose native kernels, memory plans and hardware
execution strategies internally.

## Not WIT Functions

Operation schemas intentionally remain inside the coarse graph submission
boundary. Components do not call one WIT function per tensor primitive. This
keeps batching, kernel fusion, scheduling, memory planning and hardware
optimization Provider-owned while preserving a portable validation contract.

## Initial Schemas

The initial schema set includes:

- descriptor and view: `tensor.reshape`, `tensor.transpose`, `tensor.permute`,
  `tensor.slice`, `tensor.broadcast`, `tensor.squeeze`, `tensor.unsqueeze`
- unary elementwise: `abs`, `neg`, `exp`, `log`, `sqrt`, `recip`, `sin`, `cos`,
  `tanh`, `relu`, `silu`, `gelu`, `erf`, `floor`, `ceil`, `round`
- binary elementwise: `add`, `sub`, `mul`, `div`, `maximum`, `minimum`
- comparison: `eq`, `ne`, `lt`, `le`, `gt`, `ge`
- selection: `selection.where`
- reduction: `sum`, `mean`, `min`, `max`, `argmin`, `argmax`
- linear algebra: `linalg.matmul`, `linalg.batched-matmul`
- indexing and update: `tensor.gather`, `tensor.index-select`,
  `tensor.scatter`, `tensor.scatter-add`, `tensor.concat`
- random generation: `random.uniform`, `random.normal`

## Exclusions

This change excludes convolution, pooling, spatial transforms,
attention-specific fused operations, normalization fused operations, quantized
operation schemas, backend-specific kernel names, arbitrary custom kernels,
autograd and training graphs. These areas require follow-up changes with more
specific numerical, layout and precision semantics.
