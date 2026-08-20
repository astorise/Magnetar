# Define Compute Capability

## Why

Compute is the primary capability provided by Magnetar.

Rather than exposing hardware-specific APIs, Providers expose a portable
compute interface defined through WIT contracts.

Components consume compute capabilities without knowledge of the underlying
hardware implementation.

This abstraction enables execution portability, automatic provider
selection and runtime fallback.

## What Changes

This proposal introduces the Compute Capability.

The Compute Capability defines the contract used by Components to execute
mathematical operations.

The capability is independent from:

- CUDA
- Metal
- Vulkan
- CPU
- NPUs

Providers become responsible for mapping Compute operations onto native
hardware implementations.

This proposal intentionally excludes concrete mathematical operations.

Those operations will be introduced by future capability revisions.

## Impact

Magnetar gains its first portable execution capability.

Future Components can target Compute without depending on hardware-specific
implementations.