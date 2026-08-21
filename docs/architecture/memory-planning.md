# Memory Planning

Memory planning is a Runtime responsibility in Magnetar.

Components describe compute work with portable tensor descriptors, compute
graphs and explicit data movement. Providers execute native work. The Runtime
plans tensor resource lifetimes, intermediate buffers, materialization buffers,
transfer buffers, output ownership and memory pressure before Provider
execution.

The model is host-side only. It does not add a WIT allocator, raw memory
handle, GPU pointer, queue, stream, native buffer or Provider storage object to
portable Component contracts.

## Tensor Resources

Tensor Resources remain opaque Provider-owned values. The Runtime tracks their
portable descriptor and Resource Affinity metadata separately from the native
value.

A Tensor Resource remains live while a compute graph output, later operation,
download, transfer, materialization or dependent resource can still observe it.
Intermediate buffers can be reused only after their last required use and only
when layout and Resource Affinity constraints remain compatible.

Produced output resources inherit Provider, Capability, execution context and
Device affinity when those bindings are known.

## Provider Advertisements

Memory planning uses Provider Compute Advertisements and legacy compute support
maps through the same effective advertisement path as compute validation.

The Runtime validates tensor byte sizes, Provider descriptor limits, Device
memory capacity, supported layouts, explicit materialization requirements and
explicit transfer requirements before submission.

Provider-native allocation remains internal to the Provider. A Provider may use
backend-specific allocators, pools, queues or device APIs after the Runtime has
validated the portable plan.

## Data Movement

Upload, download, copy, materialize, transfer, dtype conversion and placement
conversion contribute explicit memory requirements to the Memory Plan.

Host staging is not hidden. If a transfer requires host staging, the operation
must request it explicitly and the Provider must advertise support. Otherwise
the Runtime rejects the plan instead of silently changing placement, cost or
synchronization behavior.

## Errors And Diagnostics

Memory planning failures use stable structured categories for planning failure,
out of memory, resource exhaustion, size overflow, incompatible Resource
Affinity, unsupported layout, materialization required, transfer required,
Provider memory limit exceeded and Device memory limit exceeded.

Memory pressure diagnostics may include required bytes, peak bytes, selected
Provider, selected Device, rejected memory limits, materialization cost and
transfer buffer cost. Diagnostics must not expose native addresses, pointers,
allocator internals or backend storage handles.
