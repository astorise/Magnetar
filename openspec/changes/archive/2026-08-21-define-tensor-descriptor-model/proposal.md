# Define Tensor Descriptor Model

## Why

`magnetar:compute/run` needs a portable way to describe tensors without
exposing native tensor objects, backend storage, device handles or memory
pointers.

Candle provides useful evidence for tensor concepts such as shape, dtype,
layout, views and storage.

However, Candle's eager Tensor API is not a portable WIT boundary.

Magnetar therefore needs its own Tensor Descriptor Model.

A Tensor Descriptor describes tensor metadata.

A Tensor Resource represents opaque Provider-owned storage.

Components may inspect and pass portable descriptors.

Components must not access raw storage, native layout objects, backend queues,
GPU pointers, locks or Provider-owned memory.

This change defines the descriptor model required before concrete compute
operation schemas can be introduced.

## What Changes

This proposal introduces:

- TensorDescriptor
- ShapeDescriptor
- DTypeDescriptor
- LayoutDescriptor
- ViewDescriptor
- TensorResourceDescriptor
- TensorSize validation rules

Tensor descriptors use fixed-width values.

Shape dimensions, strides and offsets must not use platform-sized integers.

The model distinguishes:

- logical tensor metadata
- opaque tensor resources
- tensor views
- materialized tensor copies

This proposal does not define concrete compute operations.

This proposal does not define memory allocation APIs.

This proposal does not expose native tensor storage.

This proposal does not define autograd or training metadata.

## Impact

Compute operation schemas can reference a stable tensor descriptor model.

Providers can validate descriptor support before execution.

The Runtime can reject invalid or unsupported tensor shapes, dtypes and layouts
before Provider execution begins.

Future data movement, graph execution and memory planning changes can reuse the
same tensor resource model.