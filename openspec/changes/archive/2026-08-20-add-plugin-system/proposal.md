# Add Dynamic Plugin System

## Why

Magnetar is designed as a hardware-agnostic runtime with an extensible core.

Execution backends must evolve independently from the runtime itself and should be loadable without recompiling Magnetar. They are, however, only one kind of extension: future plugins may provide model loaders, kernel providers, compiler passes, scheduler extensions, or telemetry providers.

Introducing a general plugin system early establishes a stable extension mechanism without coupling the extension layer to hardware.

## What Changes

This proposal introduces:

- a general `Plugin` interface that exposes metadata and registers contributions into a `Registry`
- a separate `Backend` interface for hardware-specific execution
- plugin discovery
- plugin loading
- plugin lifecycle
- plugin metadata
- plugin version compatibility

This proposal intentionally excludes:

- CPU backend
- CUDA backend
- device enumeration
- kernels
- graph execution

These capabilities will be introduced in future changes.

## Impact

Magnetar gains a stable extension mechanism that enables independent extension development while keeping the runtime implementation hardware-agnostic and open to non-hardware plugins.
