# Compute Data Movement

`magnetar:compute/run` treats tensor movement as explicit compute work. The
Runtime does not silently upload, download, copy across Providers or Devices,
materialize views, or stage through host memory without a represented data
movement request.

## Operations

The portable movement model contains seven operation kinds:

- `upload`: creates a Provider-owned `TensorResourceDescriptor` from host data.
- `download`: copies a tensor resource into a host-visible representation.
- `copy`: creates a distinct tensor resource with equivalent contents.
- `materialize`: turns a tensor view into a distinct materialized resource.
- `transfer`: moves or copies data between Provider or Device placements.
- `dtype-conversion`: changes dtype only when Provider support is advertised.
- `placement-conversion`: requests a different host, Provider, Device, or
  affinity-group placement.

These are validation and contract objects. Native buffer handles, queues,
streams, locks, backend storage and Provider-private handles remain outside the
WIT surface.

## Host Buffers

Host data is described with `HostBufferDescriptor`: stable byte length plus a
portable encoding. Upload validation compares byte length with the target
`TensorDescriptor` byte size. Download returns host-visible data described by
the same stable metadata rather than exposing Provider-owned storage directly.

Supported encodings are raw bytes, native endian, little endian and big endian.
Providers advertise which encodings they accept for each movement kind.

## Affinity

Every produced tensor resource receives `ResourceAffinity` for the selected
Provider and Compute capability version. Target Device and affinity group are
attached when requested. Non-transfer operations preserve strict compatibility
with the selected Provider. Cross-Provider or cross-Device changes require an
explicit `transfer` or `placement-conversion` descriptor.

Affinity validation happens before Provider execution. Incompatible Provider,
Device or group usage is rejected as a structured data movement failure instead
of being recovered through fallback.

## Provider Advertisement

Providers advertise data movement support separately from compute operation
family support. An advertisement can constrain operation kind, dtypes,
provider-specific dtypes, layouts, host encodings, descriptor limits and whether
explicit host staging is supported.

Runtime validation checks both the source and output tensor descriptors against
that advertisement. Unsupported dtype or layout conversion is reported before
Provider code runs.

## CPU Staging

CPU staging is not an invisible fallback. If a transfer needs host staging, the
request must opt into it and the selected Provider must advertise support. The
Runtime otherwise rejects the movement request.

## Relationship To Graph Submission

Graph submission consumes tensor resources that already carry affinity. If a
graph needs a tensor on a different Provider or Device, the caller must submit
an explicit movement operation first and pass the produced tensor resource into
the graph. `magnetar:compute/run` therefore remains auditable: placement,
materialization, conversion and synchronization costs are represented in the
compute model rather than hidden behind graph validation.
