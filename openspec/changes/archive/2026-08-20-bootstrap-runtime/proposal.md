# Bootstrap the magnetar Runtime

## Why

magnetar aims to become a hardware-agnostic execution runtime for AI workloads.

Before introducing graph compilation, model loaders or hardware backends, the project needs a minimal runtime architecture defining the execution boundaries.

This change establishes the initial runtime contracts while deliberately avoiding any hardware-specific implementation.

The runtime will become the stable entry point for every future capability.

## What Changes

This proposal introduces:

- a runtime crate
- an execution context
- device abstraction
- backend abstraction
- runtime lifecycle
- stable public API

This proposal intentionally excludes:

- graph compilation
- tensor implementation
- kernels
- model loading
- plugins
- CUDA
- Metal
- Vulkan
- NPU support

These capabilities will be introduced by later changes.

## Impact

This creates the architectural foundation of magnetar without committing to any specific accelerator or model format.

Future changes can extend the runtime without breaking the public execution interface.