# Define the Device Model

## Why

Providers expose hardware resources to the Runtime.

The Runtime schedules execution on Devices, not on Providers.

A Provider may expose one or more Devices.

For example:

- CUDA Provider → multiple NVIDIA GPUs
- OpenVINO Provider → CPU, GPU and NPU
- CPU Provider → multiple NUMA nodes
- Metal Provider → integrated GPU

Separating Providers from Devices allows the Runtime to reason about
execution targets independently from their implementation.

## What Changes

This proposal introduces the Device abstraction.

Devices become first-class runtime objects.

Every Device describes:

- identity
- capabilities
- memory
- topology
- execution queues

Providers become responsible for discovering Devices.

The Runtime becomes responsible for scheduling work on Devices.

## Impact

Execution planning becomes independent from Provider implementations.

Future scheduling strategies can target Devices directly.