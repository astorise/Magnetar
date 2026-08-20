# Refactor Plugin Architecture into Capability Providers

## Why

During the implementation of the runtime, it became clear that the term
"Plugin" describes a loading mechanism rather than an architectural role.

Magnetar is fundamentally capability-driven.

Components should never know which native implementation executes a
capability. They should depend exclusively on WIT contracts.

Native implementations expose one or more capabilities and become
Providers.

This terminology better reflects the architecture and enables future
features such as automatic provider selection, runtime fallback,
multi-provider execution and heterogeneous scheduling.

## What Changes

This proposal replaces the Plugin abstraction with Providers.

The following concepts are renamed:

- Plugin → Provider
- Plugin API → Provider API
- Plugin Registry → Provider Registry
- Plugin Loader → Provider Loader
- Plugin Metadata → Provider Metadata
- Plugin Descriptor → Provider Descriptor

This proposal also introduces the Capability Registry.

Providers register the capabilities they implement.

Components request capabilities without referencing any Provider.

The runtime becomes responsible for resolving Providers capable of
satisfying each requested capability.

This proposal does not introduce new Providers.

CPU, CUDA, Metal, Vulkan, OpenVINO and QNN Providers will be introduced
by future changes.

## Impact

This refactoring simplifies the architecture.

Providers become responsible for hardware integration.

Components become completely portable.

The runtime owns capability resolution and future scheduling decisions.

No functional behavior changes are introduced.