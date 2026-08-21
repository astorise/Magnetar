# Provider Compute Advertisement

Providers that implement `magnetar:compute/run` publish a
`ProviderComputeAdvertisement`. The advertisement is stable runtime metadata:
it uses capability versions, operation families, operation schema identifiers,
portable dtypes, layout categories, tensor limits, data movement kinds and
Device identifiers. It does not expose native handles, kernel symbols, queues,
streams, backend storage or Provider-private APIs.

## Format

An advertisement contains:

- `ComputeCapabilitySupport`: supported `magnetar:compute/run` versions,
  operation catalog revision, operation schema revision and experimental
  extensions.
- `OperationFamilySupport`: coarse support by operation family.
- `OperationSchemaSupport`: precise support by portable operation schema.
- `DTypeSupport`: portable dtypes plus explicitly named Provider-specific
  dtypes.
- `LayoutSupport`: accepted input and output layouts, view consumption and
  materialization requirements.
- `ShapeLimitSupport`: maximum rank, dimension, element count, byte size,
  broadcasting and batch limits.
- `PrecisionSupport`: precision modes, accumulation dtypes, approximate math
  and deterministic behavior declarations.
- `DataMovementSupport`: upload, download, copy, transfer, materialize,
  dtype-conversion, placement-conversion and host-staging support.
- `DeviceComputeSupport`: per-Device overrides for memory, operation and data
  movement constraints.

The runtime treats an explicit unsupported operation schema as authoritative.
That allows a Provider to advertise broad family support while excluding a
specific schema that it cannot execute correctly.

## Portable and Provider-Specific Support

Portable Components rely only on portable operation schemas, portable dtypes,
portable layout categories and the `magnetar:compute/run` capability contract.
Provider-specific operation schemas and dtypes are advertised separately and
are not part of the portable contract.

Provider-specific support can be selected only when the submitted graph
explicitly uses that extension and the selected Provider advertises it. Native
kernel names, function pointers and backend object references remain outside
the advertisement.

## Device-Specific Advertisements

A multi-Device Provider may attach constraints to stable `DeviceId` values.
Device-specific entries can narrow memory limits, dtype support, operation
schema support or data movement support for one Device without changing the
Provider-wide defaults.

Resource Affinity remains authoritative. A tensor pinned to one Provider or
Device cannot be moved to another merely because the other target advertises
compatible compute support. Transfers, placement conversions and
materialization must be explicit data movement operations.

## Selection Flow

During compute resolution and validation, the runtime checks:

1. compatible `magnetar:compute/run` version;
2. Resolution Policy and live Resource Affinity constraints;
3. operation schema support, falling back to operation family support when no
   schema-specific entry exists;
4. dtype, layout, tensor shape, size, precision and determinism constraints;
5. explicit data movement support for upload, download, copy, transfer,
   materialization and conversions.

Failures are reported with structured compute errors such as unsupported
operation schema, unsupported dtype, unsupported layout, unsupported data
movement or incompatible Resource Affinity.

## Examples

CPU Provider:

- advertises `magnetar:compute/run@1.1.0`;
- supports dense and strided layouts for common elementwise, reduction and
  linear algebra schemas;
- supports host upload and download without host staging;
- may support deterministic execution for exact and default precision modes.

CUDA Provider:

- advertises the same portable capability version with CUDA-specific extension
  schemas marked non-portable;
- supports dense GPU layouts broadly and may require materialization for some
  strided views;
- attaches lower or higher memory limits per `DeviceId`;
- advertises transfer support and whether host staging is required.

OpenVINO Provider:

- advertises portable inference-oriented operation schemas;
- may omit or explicitly reject unsupported random generation or update-like
  schemas;
- may support provider-opaque layouts for compiled graph execution;
- uses Device-specific entries when CPU, GPU and NPU targets differ.
